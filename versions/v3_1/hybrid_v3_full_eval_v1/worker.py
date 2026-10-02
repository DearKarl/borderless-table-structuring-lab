"""Container-only resident worker. Host owns the absolute deadline and cleanup."""
import argparse
from pathlib import Path
import signal
import time
import traceback
from .core import ContractError, DeadlineError, read, write, progress, sha

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--control", required=True)
    a=p.parse_args()
    control=read(a.control); out=Path("/output")
    b=control["budget"]
    started=time.monotonic()
    absolute=control["absolute_stop_monotonic"]
    def available(cap):
        left=min(cap, absolute-time.monotonic())
        if left <= 0:
            raise DeadlineError("Host absolute deadline")
        return left
    def expired(*_):
        raise DeadlineError("Bounded worker phase expired")
    signal.signal(signal.SIGALRM, expired)
    ledger_path=out/"LEDGER.jsonl"
    def ledger(event, **data):
        with ledger_path.open("a",encoding="utf-8") as f:
            import json
            f.write(json.dumps({"event":event,"time":time.time(),**data})+"\n")
            f.flush()
    def phase(name, **data):
        progress(out/"PROGRESS.json",{"phase":name,"monotonic":time.monotonic(),**data})
    result={"processing_complete":False,"runtime_verified":False,"integration_verified":False,
            "evaluation_complete":False,"quality_claim":False,"completed":[],"system_error":None}
    page=None
    try:
        if control["profile"] not in ("native","v31-layout","v31-formula","v31-both"):
            raise ContractError("PROFILE_UNBOUND; no native substitution")
        phase("load")
        signal.setitimer(signal.ITIMER_REAL, available(b["load_seconds"]))
        from .gpu_backend import worker_guard
        worker_guard(control,"native",out)
        from .native import Native
        experts=None
        if control.get("mode")=="on" and control["profile"]!="native":
            from .v31_protocol import ResidentClient
            experts=ResidentClient(control,out,ledger)
        native=Native(control["runtime"],out,b,ledger,control["run_id"],
            control["profile"],control.get("mode","off"),experts)
        # Original load sets/cancels its alarm; host independently enforces load+total limits.
        signal.setitimer(signal.ITIMER_REAL, 0)
        result["runtime_verified"]=True
        for page in control["inputs"]["pages"]:
            folder=out/"pages"/page["page_id"]
            folder.mkdir(parents=True,exist_ok=False)
            source=Path("/inputs")/page["path"]
            if sha(source) != page["file_sha256"]:
                raise ContractError("Input changed at point of use")
            start={"run_id":control["run_id"],"page_id":page["page_id"],
                   "input_sha256":page["file_sha256"],"monotonic":time.monotonic()}
            write(folder/"PAGE_START.json",start)
            phase("page",page_id=page["page_id"])
            ledger("PAGE_START",page_id=page["page_id"])
            signal.setitimer(signal.ITIMER_REAL,available(b["page_seconds"]))
            md,truncated=native.parse(page,source,folder)
            data=md.read_bytes()
            (folder/"prediction.md").write_bytes(data)
            write(folder/"PAGE_RESULT.json",{
                **start,"start_sha256":sha(folder/"PAGE_START.json"),
                "status":"truncated" if truncated else "success",
                "prediction_sha256":sha(folder/"prediction.md"),
                "native_calls":native.count,"expert_calls":native.expert_calls,"audit_complete":True,
                "natural_adoptions":native.adoptions,"quality_claim":False,
                "elapsed_seconds":time.monotonic()-start["monotonic"]})
            ledger("PAGE_RESULT",page_id=page["page_id"])
            signal.setitimer(signal.ITIMER_REAL,0)
            result["completed"].append(page["page_id"])
        result["processing_complete"]=True
        result["integration_verified"]=True
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        result["system_error"]={"type":type(exc).__name__,"message":str(exc),
                                "traceback":traceback.format_exc()}
        ledger("SYSTEM_STOP",error=result["system_error"])
        # Started failure remains in the denominator, but unknown/system failure stops the session.
        if page is not None:
            folder=out/"pages"/page["page_id"]
            if (folder/"PAGE_START.json").exists() and not (folder/"PAGE_RESULT.json").exists():
                (folder/"prediction.md").write_bytes(b"")
                write(folder/"PAGE_RESULT.json",{
                    "run_id":control["run_id"],"page_id":page["page_id"],
                    "input_sha256":page["file_sha256"],"start_sha256":sha(folder/"PAGE_START.json"),
                    "status":"failed","prediction_sha256":sha(folder/"prediction.md"),
                    "error":result["system_error"],"audit_complete":False})
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result["elapsed_seconds"]=time.monotonic()-started
        write(out/"SESSION_RESULT.json",result)
        phase("complete" if result["processing_complete"] else "stopped")
    return 0 if result["processing_complete"] else 1

if __name__=="__main__":
    raise SystemExit(main())
