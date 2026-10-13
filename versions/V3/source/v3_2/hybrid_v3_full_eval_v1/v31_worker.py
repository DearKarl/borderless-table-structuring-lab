"""Owned resident expert service; serial mailbox, no network and no process spawning."""
import ctypes
import json
import os
from pathlib import Path
import signal
import time
import traceback
import uuid
from .core import ContractError, DeadlineError, closed, read, sha, write, progress
from .asset_binding import load_boundary
from .request_binding import rgb_binding
from .v31_protocol import IDENTITY, validate
from .v31_rules import PROFILE_SHA

def actual_gpu_uuid(expected):
    from .gpu_backend import actual_gpu_uuid as query
    return query(expected)

def main():
    control=read("/control.json");out=Path("/output");root=out/"experts";ipc=root/"ipc"
    runtime=control["runtime"];providers={};events=[];loaded=0;resources=None
    def expired(*_):raise DeadlineError("Expert load/request/host deadline reached")
    signal.signal(signal.SIGALRM,expired)
    def arm(cap):
        left=min(cap,control["absolute_stop_monotonic"]-time.monotonic())
        if left<=0:raise DeadlineError("Host absolute deadline")
        signal.setitimer(signal.ITIMER_REAL,left)
    def event(name,**data):
        row={"event":name,"monotonic":time.monotonic(),**data};events.append(row)
        with (root/"LEDGER.jsonl").open("a",encoding="utf-8") as f:f.write(json.dumps(row)+"\n")
    try:
        from .gpu_backend import worker_guard,kind
        manual=kind(runtime)=="manual"
        load_start=time.monotonic()
        if manual:
            progress(root/"PROGRESS.json",{"phase":"load","monotonic":load_start})
            arm(control["budget"]["load_seconds"])
            actual=worker_guard(control,"paddle",out)["actual_uuid"]
        else:actual=actual_gpu_uuid(runtime["gpu_uuids"][1])
        import sys
        sys.path.insert(0,"/assets/native_code")
        from .expert_resource_audit import ExpertResourceAudit
        identities=[{"path":r["image_path"] if "image_path" in r else "/assets/"+r["asset"]+"/"+r["path"],
                     "sha256":r["sha256"]} for r in runtime["native"]["identity_files"]]
        # Layout static weights are additionally locked in this version.
        for component in runtime["v31"]["components"]:
            asset=runtime["v31"][component]["model_asset"]
            identities.extend({"path":"/assets/"+asset+"/"+name,"sha256":h}
                              for name,h in runtime["assets"][asset]["files"].items())
        identities.extend({"path":"/assets/auxiliary_models/"+name,"sha256":h}
                          for name,h in runtime["assets"]["auxiliary_models"]["files"].items())
        resources=ExpertResourceAudit(identities)
        from .v31_providers import Formula,Layout
        for component in runtime["v31"]["components"]:
            if loaded>=control["budget"]["model_loads"]:raise ContractError("Expert load budget")
            loaded+=1;event("LOAD_START",engine=component)
            progress(root/"PROGRESS.json",{"phase":"load","monotonic":load_start if manual else time.monotonic(),"engine":component})
            arm(control["budget"]["load_seconds"]-(time.monotonic()-load_start) if manual else control["budget"]["load_seconds"])
            with resources.integrity_boundary(runtime,component):
                identity=load_boundary(runtime,"paddle")
            provider=(Formula if component=="formula" else Layout)(runtime)
            providers[component]=provider
            event("LOAD_RESULT",engine=component,returned=True,identity=identity)
            if resources.result()["unknown_resources"]:raise ContractError("Unbound expert load resource")
            signal.setitimer(signal.ITIMER_REAL,0)
        write(root/"READY.json",{"run_id":control["run_id"],"pid":os.getpid(),"process_group":os.getpgrp(),
              "gpu_uuid":actual,"model_loads":loaded,"components":list(providers)})
        seen=set();counts={};sequence=1
        while True:
            if time.monotonic()>=control["absolute_stop_monotonic"]:raise DeadlineError("Host absolute deadline")
            if (root/"CLOSE.json").exists():
                if read(root/"CLOSE.json")!={"run_id":control["run_id"]}:raise ContractError("Shutdown identity")
                break
            request_file=ipc/f"{sequence:08d}.request.json"
            if not request_file.exists():time.sleep(.05);continue
            request=read(request_file);engine=request["engine"]
            if request["run_id"]!=control["run_id"] or request["profile_sha256"]!=PROFILE_SHA or engine not in providers:
                raise ContractError("Request run/profile/provider mismatch")
            page=next((p for p in control["inputs"]["pages"] if p["page_id"]==request["page_id"]),None)
            if page is None or page["file_sha256"]!=request["input_sha256"]:raise ContractError("Request input identity")
            cfg=runtime["v31"][engine]
            for key in ("source_sha256","config_sha256","model_sha256"):
                if cfg[key]!=request[key]:raise ContractError("Request frozen provider identity")
            if request["request_id"] in seen:raise ContractError("Request replay")
            seen.add(request["request_id"])
            key=(request["page_id"],engine);counts[key]=counts.get(key,0)+1
            cap=1 if engine=="layout" else min(32,control["budget"]["expert_calls_per_page"])
            if counts[key]>cap:raise ContractError("Resident request cap exceeded")
            png=closed(ipc,request["png_name"])
            if sha(png)!=request["png_sha256"]:raise ContractError("PNG transport changed")
            from PIL import Image
            with Image.open(png) as source:
                source.load();image=source.copy()
            try:
                if rgb_binding(image)!=request["crop"]:raise ContractError("RGB transport differs")
                if engine=="layout" and request["crop"]!=request["render"]:raise ContractError("Layout not actual render")
                event("CALL_START",engine=engine,request_id=request["request_id"],page_id=request["page_id"])
                arm(60 if engine=="layout" else 180)
                response={**{k:request[k] for k in IDENTITY},**providers[engine].call(request,image,png)}
                response["resource_audit"]=resources.result()
                if response["resource_audit"]["unknown_resources"]:raise ContractError("Unbound expert request resource")
                validate(request,response)
                pending=ipc/f"{sequence:08d}.reply-pending";write(pending,response)
                pending.rename(ipc/f"{sequence:08d}.response.json")
                event("CALL_RESULT",engine=engine,request_id=request["request_id"],returned=True)
                signal.setitimer(signal.ITIMER_REAL,0)
            finally:image.close()
            sequence+=1
        write(root/"SESSION_RESULT.json",{"normal_exit":True,"run_id":control["run_id"],
                                        "model_loads":loaded,"requests":sequence-1})
        return 0
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL,0)
        write(root/"ERROR.json",{"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        close_errors=[]
        for provider in reversed(list(providers.values())):
            try:provider.close()
            except BaseException as exc:close_errors.append({"type":type(exc).__name__,"error":str(exc)})
        write(root/"FINAL_AUDIT.json",{"resource_audit":None if resources is None else resources.result(),"close_errors":close_errors})
        if close_errors:raise ContractError("Expert provider close failed: "+str(close_errors))

if __name__=="__main__":raise SystemExit(main())
