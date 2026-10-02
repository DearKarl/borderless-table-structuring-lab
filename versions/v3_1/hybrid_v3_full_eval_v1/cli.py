"""Unified commands: help/inspect never import GPU frameworks or access the network."""
import argparse
import json
from pathlib import Path
from .core import ContractError, read
from .contracts import PROFILES

def main(argv=None):
    p=argparse.ArgumentParser(description="Archived hybrid inference with explicit deployment contracts")
    sub=p.add_subparsers(dest="command",required=True)
    inspect=sub.add_parser("inspect-assets")
    inspect.add_argument("--runtime")
    make=sub.add_parser("manifest")
    make.add_argument("--source",required=True);make.add_argument("--output",required=True)
    bind=sub.add_parser("bind-assets")
    bind.add_argument("--plan",required=True);bind.add_argument("--output",required=True)
    prep=sub.add_parser("prepare-assets")
    prep.add_argument("--configuration",required=True);prep.add_argument("--output",required=True)
    req=sub.add_parser("environment-requirements");req.add_argument("--output",required=True)
    driver=sub.add_parser("bind-host-driver",help="Validate an explicit local driver descriptor; no GPU query or installation")
    driver.add_argument("--spec",required=True);driver.add_argument("--output",required=True)
    recovery=sub.add_parser("clear-gpu-block",help="Explicitly verify absence and GPU idleness before clearing a blocked run")
    for key in ("lease-directory","run-id","output"):
        recovery.add_argument("--"+key,required=True)
    recovery.add_argument("--gpu-uuid",action="append",required=True)
    recovery.add_argument("--idle-memory-mib",type=int,required=True)
    for name in ("preflight","infer"):
        s=sub.add_parser(name);s.add_argument("--profile",choices=PROFILES,required=True)
        s.add_argument("--mode",choices=("off","pass-through","on"),required=True)
        for key in ("inputs","runtime","budget"):
            s.add_argument("--"+key,required=True)
        if name=="infer":s.add_argument("--output",required=True)
    s=sub.add_parser("collect")
    s.add_argument("--run",required=True);s.add_argument("--output",required=True)
    s.add_argument("--diagnostic",action="store_true")
    s=sub.add_parser("evaluate")
    for key in ("run","predictions","gt","evaluator","output"):
        s.add_argument("--"+key,required=True)
    s.add_argument("--diagnostic",action="store_true")
    s=sub.add_parser("package-check");s.add_argument("--root",default=str(Path(__file__).resolve().parent.parent))
    a=p.parse_args(argv)
    try:
        if a.command=="inspect-assets":
            result={"version":"public-v3_1","providers":{'native': 'Archived implementation; validate external deployment; full benchmark Pending', 'v31-layout': 'Archived implementation; validate external deployment; full benchmark Pending', 'v31-formula': 'Archived implementation; validate external deployment; full benchmark Pending', 'v31-both': 'Archived implementation; validate external deployment; full benchmark Pending', 'v32-text': 'Not included in V3.1'},
                    "runtime":read(a.runtime) if a.runtime else None,
                    "network":False,"model_loads":0,"quality_claim":False}
        elif a.command=="bind-host-driver":
            from .gpu_backend import bind_host_driver
            result=bind_host_driver(a.spec,a.output)
        elif a.command=="clear-gpu-block":
            from .gpu_admission import clear
            result=clear(a.lease_directory,a.gpu_uuid,a.run_id,a.idle_memory_mib,a.output)
        elif a.command=="manifest":
            from .inputs import make_manifest
            result=make_manifest(a.source,a.output)
        elif a.command=="bind-assets":
            from .asset_binder import bind
            result=bind(a.plan,a.output)
        elif a.command=="prepare-assets":
            from .asset_binder import prepare
            result=prepare(a.configuration,a.output)
        elif a.command=="environment-requirements":
            from .asset_binder import requirements
            result=requirements(a.output)
        elif a.command in ("preflight","infer"):
            from .runner import preflight, infer
            result=(preflight(a.profile,a.inputs,a.runtime,a.budget,mode=a.mode) if a.command=="preflight"
                    else infer(a.profile,a.inputs,a.runtime,a.budget,a.output,mode=a.mode))
        elif a.command=="collect":
            from .collection import collect
            result=collect(a.run,a.output,a.diagnostic)
        elif a.command=="evaluate":
            from .evaluation import evaluate
            result=evaluate(a.run,a.predictions,a.gt,a.evaluator,a.output,a.diagnostic)
        else:
            from .core import verify_tree
            root=Path(a.root).resolve();lock=read(root/"PACKAGE_LOCK.json")
            # PACKAGE_LOCK itself is the externally hashed index.
            from .core import sha, closed
            files={x.relative_to(root).as_posix() for x in root.rglob("*") if x.is_file()
                   and x.name!="PACKAGE_LOCK.json" and "__pycache__" not in x.parts}
            if files!=set(lock["files"]):
                raise ContractError("Package file set differs")
            for name,h in lock["files"].items():
                if sha(closed(root,name))!=h:raise ContractError("Package bytes differ: "+name)
            result={"passed":True,"files":len(files),"lock_sha256":sha(root/"PACKAGE_LOCK.json")}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ContractError,KeyError,FileNotFoundError,ValueError) as exc:
        print(json.dumps({"status":"blocked","error_type":type(exc).__name__,"error":str(exc)},ensure_ascii=False))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
