"""One owned Ovis service on the second leased GPU; sequential requests."""
from pathlib import Path
import time
from .core import ContractError, read, write
from .lifecycle import OwnedDocker, limits, mount_args, verify_isolation, terminal_reason
from . import gpu_backend

class ExpertService:
    def __init__(self,control,clock,out,mounts,image_identity,backend_binding=None):
        self.control,self.clock,self.out=control,clock,Path(out)
        r=control["runtime"];self.root=self.out/"experts"
        self.root.mkdir(exist_ok=False);(self.root/"ipc").mkdir()
        self.docker=OwnedDocker(control["run_id"]+"-ovis",clock,self.root)
        self.mounts=mounts
        self.image_identity=image_identity
        self.backend_binding=backend_binding
        if (gpu_backend.kind(r)=="manual")!=(backend_binding is not None):
            raise ContractError("Expert backend binding missing/unexpected")
        if backend_binding is not None and (backend_binding["run_id"]!=control["run_id"] or backend_binding["role"]!="ovis"):
            raise ContractError("Expert backend run/role differs")
        self.env=r["environments"]["ovis"]
        from .v32_cluster import mounts as cluster_mounts
        self.mounts=list(self.mounts)+cluster_mounts(self.env,clock)
        self.budget={**control["budget"],"cpus":8,"ram_gib":16,"swap_gib":0}
        self.gpus=[r["gpu_uuids"][-1]]
    def start(self):
        args=limits(self.budget)+gpu_backend.launch_flags(self.control["runtime"],self.gpus[0],self.backend_binding)+[
              "--env","PYTHONPATH=/framework","--env","CUDA_DEVICE_ORDER=PCI_BUS_ID",
              "--env","HF_HUB_OFFLINE=1","--env","TRANSFORMERS_OFFLINE=1",
              "--env","HOME=/output/experts","--env","HF_HOME=/output/experts/.cache",
              "--workdir","/output/experts","--entrypoint",self.env["python"]]
        from .v32_cluster import environment
        for key,value in environment(self.env).items():args += ["--env",key+"="+value]
        args+=mount_args(self.mounts,"/output")
        args+=[self.env["image"],"-B","-m","hybrid_v3_full_eval_v1.v32_worker"]
        fresh=gpu_backend.fresh_role(self.control,self.backend_binding,self.clock) if self.backend_binding is not None else None
        self.docker.create(args)
        verify_isolation(self.docker.inspect(),self.mounts,self.budget,self.env["image"],self.gpus,self.image_identity,**({"backend_binding":self.backend_binding} if self.backend_binding is not None else {}))
        from .v32_cluster import mount_evidence
        write(self.root/"CLUSTER_MOUNTS.json",mount_evidence(self.docker.inspect(),self.env))
        if fresh is not None:gpu_backend.seal_fresh(self.root,self.control,self.backend_binding,fresh,self.docker.cid,self.docker.token,self.out)
        self.docker.start();start=time.monotonic()
        while not (self.root/"READY.json").exists():
            state=self.docker.inspect()
            p=read(self.root/"PROGRESS.json") if (self.root/"PROGRESS.json").exists() else {"monotonic":start}
            reason=terminal_reason(state,time.monotonic(),p["monotonic"],self.control["budget"]["load_seconds"])
            if reason:raise ContractError("Expert initialization failed: "+reason)
            self.clock.remaining();time.sleep(.1)
        ready=read(self.root/"READY.json")
        if ready.get("host_environment_binding_sha256")!=self.env["host_environment_binding"]["sha256"]:
            raise ContractError("Resident cluster binding differs")
        if ready["run_id"]!=self.control["run_id"] or ready["gpu_uuid"].lower()!=self.gpus[0].lower():
            raise ContractError("Resident ready identity differs")
        if ready["components"]!=self.control["runtime"]["v32"]["components"] or ready["model_loads"]!=len(ready["components"]):
            raise ContractError("Unexpected expert model loads")
    def healthy(self):
        state=self.docker.inspect()
        if not state["State"]["Running"] or state["State"].get("OOMKilled") or (self.root/"ERROR.json").exists():
            raise ContractError("Expert resident system failure")
    def close(self):
        write(self.root/"CLOSE.json",{"run_id":self.control["run_id"]})
        end=time.monotonic()+15
        while True:
            state=self.docker.inspect()
            if not state["State"]["Running"]:
                if self.backend_binding is not None:verify_isolation(state,self.mounts,self.budget,self.env["image"],self.gpus,self.image_identity,backend_binding=self.backend_binding)
                if state["State"].get("OOMKilled") or state["State"]["ExitCode"]!=0:
                    raise ContractError("Expert resident did not exit normally")
                break
            if time.monotonic()>=end:raise ContractError("Expert normal close deadline")
            self.clock.remaining();time.sleep(.1)
    def cleanup(self,clock):
        return self.docker.cleanup(clock=clock)
