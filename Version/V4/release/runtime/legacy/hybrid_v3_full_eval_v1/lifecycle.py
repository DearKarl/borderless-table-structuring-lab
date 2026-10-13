"""Owned Docker lifecycle, explicit mount policy and absolute deadlines (Linux)."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import time
from .core import ContractError, DeadlineError, positive, write

def terminal_reason(state, now, phase_start, phase_limit):
    """An exited/OOM process takes precedence over the live watchdog."""
    s = state["State"]
    if s.get("OOMKilled"):
        return "oom"
    if not s.get("Running"):
        return "exited_ok" if s.get("ExitCode") == 0 else "system_exit"
    if now >= phase_start + phase_limit:
        return "live_timeout"
    return None

class Leases:
    def __init__(self, directory, keys, recovery=False):
        self.directory, self.keys, self.stack = Path(directory), sorted(keys), ExitStack()
        self.recovery=recovery
    def __enter__(self):
        import fcntl
        if not self.directory.is_dir():
            raise ContractError("Provision an explicit shared lease directory first")
        try:
            for key in self.keys:
                if "/" in key or "\\" in key:
                    raise ContractError("Unsafe lease key")
                handle = self.stack.enter_context((self.directory / (key+".lease")).open("a"))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not self.recovery:
                from .gpu_admission import assert_unblocked
                assert_unblocked(self.directory,self.keys)
            return self
        except BaseException:
            self.stack.close()
            raise
    def __exit__(self, *args):
        self.stack.close()

class OwnedDocker:
    def __init__(self, token, deadline, output, runner=subprocess.run):
        self.token, self.deadline, self.output, self.runner = token, deadline, Path(output), runner
        self.cid = None
        self.creation_attempted = False
        self.last_verified = None
        self.admission = None

    def command(self, args, cleanup=False):
        return self.runner(args, text=True, capture_output=True, check=True,
                           timeout=self.deadline.bound(10, cleanup)).stdout.strip()

    def inspect(self, cleanup=False):
        if not self.cid:
            raise ContractError("No owned CID")
        obj = json.loads(self.command(["docker","inspect",self.cid], cleanup))[0]
        if obj["Id"] != self.cid or obj["Config"].get("Labels",{}).get("hybrid-v3-owner") != self.token:
            raise ContractError("Container ownership mismatch; refuse cleanup")
        self.last_verified = obj
        if self.admission is not None:self.admission.observe(self)
        return obj

    def create(self, args):
        self.creation_attempted = True
        if self.admission is not None:self.admission.observe(self)
        self.cid = self.command(["docker","create","--label","hybrid-v3-owner="+self.token,
                                 "--name","hybrid-v3-"+self.token, *args])
        if self.admission is not None:self.admission.observe(self)
        write(self.output/"CONTAINER_CREATED.json", {"cid":self.cid, "owner":self.token,
                                                    "arguments":args})
        return self.cid

    def start(self):
        self.command(["docker","start",self.cid])

    def cleanup(self, release=None, clock=None):
        from .bounded_cleanup import finalize
        return finalize(self, release, clock)

def gpu_idle(uuids, ceiling_mib, deadline, runner=subprocess.run):
    positive(ceiling_mib, "idle memory", zero=True)
    def run(args):
        return runner(args, check=True, capture_output=True, text=True,
                      timeout=deadline.bound(5)).stdout
    rows = run(["nvidia-smi","--query-gpu=uuid,memory.used","--format=csv,noheader,nounits"])
    found = {}
    for line in rows.splitlines():
        uid, memory = [v.strip() for v in line.split(",")]
        found[uid] = int(memory)
    apps = run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid","--format=csv,noheader,nounits"])
    busy = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    for uid in uuids:
        if uid not in found or uid in busy or found[uid] > ceiling_mib:
            raise ContractError("GPU missing/busy: " + uid)
    return {uid:found[uid] for uid in uuids}

def mount_args(mounts, output_target):
    args, targets = [], set()
    for src, dst, writable in mounts:
        if "," in str(src) or "," in dst:
            raise ContractError("Comma in bind path")
        if not Path(src).exists() or not dst.startswith("/") or dst in targets:
            raise ContractError("Missing or duplicate mount")
        if writable and dst != output_target:
            raise ContractError("Only current output may be writable")
        targets.add(dst)
        args += ["--mount", f"type=bind,src={Path(src).resolve()},dst={dst}"+
                 ("" if writable else ",readonly")]
    return args

def limits(b, evaluation=False):
    if evaluation:
        for key, cap in (("cpus",24),("ram_gib",128),("shm_gib",8),("total_seconds",259200)):
            positive(b[key], key, cap)
        positive(b["swap_gib"],"swap",128,zero=True)
        if b["ram_gib"] + b["swap_gib"] > 128:
            raise ContractError("Evaluation RAM plus swap exceeds 128 GiB")
    return ["--pull","never","--network","none","--read-only","--cap-drop","ALL","--security-opt","no-new-privileges",
            "--cpus",str(b["cpus"]),"--memory",str(b["ram_gib"])+"g",
            "--memory-swap",str(b["ram_gib"]+b["swap_gib"])+"g",
            "--shm-size",str(b["shm_gib"])+"g","--pids-limit","4096",
            "--tmpfs","/tmp:rw,size=1g","--env","PYTHONDONTWRITEBYTECODE=1"]

def verify_isolation(state, mounts, b, image, gpus, image_identity, backend_binding=None):
    """Verify Docker's effective configuration, not merely the requested flags."""
    from .image_identity import verify_container_image
    verify_container_image(state, image, image_identity)
    hc=state["HostConfig"]
    expected=sorted((str(Path(src).resolve()),dst,writable) for src,dst,writable in mounts)
    actual=sorted((m["Source"],m["Destination"],m["RW"])
                  for m in state["Mounts"] if m["Type"]=="bind")
    if expected!=actual or any(m["Type"] not in ("bind","tmpfs") for m in state["Mounts"]):
        raise ContractError("Effective mounts differ")
    if (hc["NetworkMode"]!="none" or not hc["ReadonlyRootfs"] or hc.get("Privileged")
        or (hc.get("Devices") and backend_binding is None) or hc["Memory"]!=int(b["ram_gib"]*1024**3)
        or hc["MemorySwap"]!=int((b["ram_gib"]+b["swap_gib"])*1024**3)
        or hc["NanoCpus"]!=int(b["cpus"]*1e9) or hc["ShmSize"]!=int(b["shm_gib"]*1024**3)
        or state["Config"]["Image"]!=image):
        raise ContractError("Effective isolation/resources differ")
    requests=hc.get("DeviceRequests") or []
    if backend_binding is not None:
        from .gpu_backend import verify_container
        if gpus!=[backend_binding["uuid"]]:raise ContractError("Manual GPU role request differs")
        verify_container(state,backend_binding)
    elif gpus:
        if len(requests)!=1 or requests[0].get("DeviceIDs")!=gpus:
            raise ContractError("Effective GPU UUID request differs")
    elif requests:
        raise ContractError("Evaluator unexpectedly has GPUs")
