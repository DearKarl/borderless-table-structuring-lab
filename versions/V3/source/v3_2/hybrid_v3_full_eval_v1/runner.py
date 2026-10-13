"""Host inference controller; no heavy model imports and no automatic installation."""
import os
from pathlib import Path
import time
import uuid
from .core import ContractError, Deadline, closed, new_dir, read, sha, write, verify_tree
from .contracts import inputs, runtime, budget
from .completion import lifecycle_lock, normal_completion
from .lifecycle import OwnedDocker, Leases, gpu_idle, limits, mount_args, terminal_reason, verify_isolation
from .gpu_admission import Admission
from .group_cleanup import finalize_group
from .image_identity import effective_references, resolve_identities
from . import gpu_backend

def source_check(deadline=None):
    package=Path(__file__).resolve().parent
    lock=read(package/"SOURCE_LOCK.json")
    # Lock includes everything except itself and the distribution-level package lock.
    actual={p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()
            and p.name != "SOURCE_LOCK.json"}
    if actual != set(lock["files"]):
        raise ContractError("Framework source file set differs")
    for name, expected in lock["files"].items():
        if sha(closed(package,name), deadline) != expected:
            raise ContractError("Framework source tamper: "+name)
    return {"sha256":sha(package/"SOURCE_LOCK.json"),"files":lock["files"]}

def preflight(profile, manifest_path, runtime_path, budget_path, deadline=None, mode="on"):
    m=inputs(manifest_path, deadline=deadline)
    b=budget(read(budget_path),len(m["pages"]))
    r=runtime(runtime_path,profile,b,deadline=deadline,mode=mode)
    s=source_check(deadline)
    return {"inputs":m,"budget":b,"runtime":r,"source":s,
            "bindings":{"inputs":sha(manifest_path),"runtime":sha(runtime_path),"budget":sha(budget_path)},
            "states":{"code_ready":True,"runtime_verified":False,"integration_verified":False,
                      "processing_complete":False,"evaluation_complete":False,"quality_claim":False},
            "preflight_kind":"CPU asset checks only; GPU/container not inspected"}

def infer(profile, manifest_path, runtime_path, budget_path, output, mode="on"):
    start=time.monotonic()
    b=read(budget_path)
    from .v32_cluster import bound_budget
    b=bound_budget(read(runtime_path),b,start)
    clock=Deadline(start,b["total_seconds"],b["cleanup_seconds"])
    plan=preflight(profile,manifest_path,runtime_path,budget_path,clock,mode)
    plan["budget"]=b
    if plan["runtime"].get("v32",{}).get("execution")=="crop":
        raise ContractError("Use dedicated v32-crop; bootstrap cannot enter formal page inference")
    clock.remaining()
    if os.name != "posix":
        raise ContractError("Real inference requires Linux Docker/NVIDIA; no local fallback")
    out=new_dir(output)
    run_id=uuid.uuid4().hex
    control={**plan,"profile":profile,"mode":mode,"run_id":run_id,"cleanup_schema":2,"image_identity_schema":1,
             "host_started_monotonic":start,
             "absolute_stop_monotonic":start+b["total_seconds"]-b["cleanup_seconds"]}
    manual=gpu_backend.kind(plan["runtime"])=="manual"
    if manual:control["gpu_binding_schema"]=1
    write(out/"CONTROL.json",control)
    write(out/"CLAIMS.json",{"run_id":run_id,"page_ids":[p["page_id"] for p in plan["inputs"]["pages"]]})
    write(out/"INPUT_MANIFEST.json",plan["inputs"])
    r=plan["runtime"]
    docker=OwnedDocker(run_id,clock,out)
    state=None; failure=None; cleanup=None; idle_after=None;expert=None;bindings=None;native_binding=None
    with Leases(r["lease_directory"],r["gpu_uuids"]):
        admission=Admission(r["lease_directory"],r["gpu_uuids"])
        try:
            identities=resolve_identities(run_id,effective_references(r,mode),clock)
            write(out/"IMAGE_IDENTITIES.json",identities)
            write(out/"GPU_BEFORE.json",gpu_idle(r["gpu_uuids"],r["idle_memory_mib"],clock))
            if manual:
                bindings=gpu_backend.resolve_manual(control,Path(runtime_path).resolve().parent,out,identities,clock)
                native_binding=bindings["roles"]["native"]
            mounts=[(out,"/output",True),(out/"CONTROL.json","/control.json",False),
                    (Path(__file__).resolve().parent,"/framework/hybrid_v3_full_eval_v1",False)]
            for name,a in r["assets"].items():
                from .v32_cluster import asset_root
                mounts.append((asset_root(r,Path(runtime_path).resolve().parent,name),"/assets/"+name,False))
            if manual:mounts.append(gpu_backend.driver_mount(native_binding))
            seen=set()
            for page in plan["inputs"]["pages"]:
                if page["path"] not in seen:
                    mounts.append((closed(Path(manifest_path).resolve().parent,page["path"]),
                                   "/inputs/"+page["path"],False))
                    seen.add(page["path"])
            active_expert=mode=="on" and bool(r.get("v31",{}).get("components") or r.get("v32",{}).get("components"))
            if active_expert:
                role="ovis" if profile=="v32-text" else "paddle"
                if role=="ovis":
                    from .v32_service import ExpertService
                else:
                    from .v31_service import ExpertService
                # Registered before create/start so partial initialization is always owned.
                expert_mounts=[m for m in mounts if not m[1].startswith("/inputs/") and m[1]!="/driver"]
                if manual:expert_mounts.append(gpu_backend.driver_mount(bindings["roles"][role]))
                expert=ExpertService(control,clock,out,expert_mounts,
                                     identities["roles"][role],**({"backend_binding":bindings["roles"][role]} if manual else {}))
            resources=([expert.docker] if expert is not None else [])+[docker]
            admission.reserve(run_id,[d.token for d in resources])
            for resource in resources:resource.admission=admission
            if expert is not None:
                expert.start()
            native_gpus=[r["gpu_uuids"][0]]
            args=limits(b)+gpu_backend.launch_flags(r,native_gpus[0],native_binding)+[
                  "--env","PYTHONPATH=/framework","--env","CUDA_DEVICE_ORDER=PCI_BUS_ID",
                  "--env","HF_HUB_OFFLINE=1","--env","TRANSFORMERS_OFFLINE=1",
                  "--env","HOME=/output","--env","HF_HOME=/output/.cache",
                  "--workdir","/output","--entrypoint",r["python"]]
            args+=mount_args(mounts,"/output")
            args += [r["image"],"-B","-m","hybrid_v3_full_eval_v1.worker","--control","/control.json"]
            fresh=gpu_backend.fresh_role(control,native_binding,clock) if manual else None
            docker.create(args)
            verify_isolation(docker.inspect(),mounts,b,r["image"],native_gpus,identities["roles"]["native"],**({"backend_binding":native_binding} if manual else {}))
            if manual:gpu_backend.seal_fresh(out,control,native_binding,fresh,docker.cid,docker.token,out)
            docker.start()
            fallback_start=time.monotonic()
            while True:
                state=docker.inspect(cleanup=True)
                progress_file=out/"PROGRESS.json"
                progress=read(progress_file) if progress_file.exists() else {"phase":"load","monotonic":fallback_start}
                phase_cap=(b["load_seconds"] if progress["phase"]=="load" else
                           b["page_audit_seconds"] if progress["phase"]=="page_audit" else b["page_seconds"])
                reason=terminal_reason(state,time.monotonic(),progress["monotonic"],phase_cap)
                if reason:
                    if reason != "exited_ok":
                        raise ContractError("Container terminal: "+reason)
                    break
                clock.remaining()
                if expert is not None:expert.healthy()
                time.sleep(min(0.25,clock.remaining()))
            if expert is not None:expert.close()
            if manual:verify_isolation(state,mounts,b,r["image"],native_gpus,identities["roles"]["native"],backend_binding=native_binding)
            if source_check(clock)!=plan["source"]:
                raise ContractError("Framework source changed during inference")
        except BaseException as exc:
            failure={"type":type(exc).__name__,"message":str(exc)}
        finally:
            from .bounded_cleanup import CleanupClock
            cleanup_clock=CleanupClock(clock)
            try:
                def release(cleanup_clock):
                    memory=gpu_idle(r["gpu_uuids"],r["idle_memory_mib"],cleanup_clock)
                    return {"verified":True,"gpu_uuids":r["gpu_uuids"],"memory_mib":memory,
                            "observed_monotonic":time.monotonic()}
                resources=([expert.docker] if expert is not None else [])+[docker]
                group=finalize_group(resources,release,clock=cleanup_clock)
                cleanup=group["resources"][-1]
                # Physical release and audit success are deliberately distinct.
                # Pending durable markers already exist before any create call.
                admission_result=admission.finish(group)
                write(out/"GROUP_CLEANUP.json",group)
                write(out/"ADMISSION_RESULT.json",admission_result)
                if (not group["complete"] or not admission_result["released"]) and failure is None:
                    failure={"type":"CleanupFailure","message":"Group cleanup incomplete; inspect admission result"}
                if (cleanup.get("state") or {}).get("OOMKilled"):
                    failure={"type":"ContainerOOM","message":"Owned container OOM; supersedes live timeout",
                             "prior_error":failure}
                idle_after=cleanup.get("release")
            except BaseException as exc:
                cleanup={"error_type":type(exc).__name__,"error":str(exc),
                         "unresolved_owned_cid":docker.cid,"previous":cleanup}
    finished=time.monotonic()
    write(out/"HOST_EXIT.json",{"failure":failure,"cleanup":cleanup,"gpu_after":idle_after,
            "started_monotonic":start,"finished_monotonic":finished,
            "budget_exceeded":finished>start+b["total_seconds"] or bool(cleanup and cleanup.get("budget_exceeded")),
            "elapsed_seconds":finished-start})
    session=read(out/"SESSION_RESULT.json") if (out/"SESSION_RESULT.json").exists() else {}
    results=[]; starts=[]; unresolved=[]
    for page in plan["inputs"]["pages"]:
        folder=out/"pages"/page["page_id"]
        if (folder/"PAGE_START.json").exists():
            starts.append(page["page_id"])
        if (folder/"PAGE_RESULT.json").exists():
            results.append(page["page_id"])
        else:
            unresolved.append(page["page_id"])
    states={**plan["states"],"runtime_verified":session.get("runtime_verified",False),
            "integration_verified":session.get("integration_verified",False),
            "processing_complete":False}
    record={"schema":1,"image_identity_schema":1,"run_id":run_id,"profile":profile,"states":states,
            "inputs":plan["inputs"],"bindings":plan["bindings"],"source":plan["source"],
            "claims":[p["page_id"] for p in plan["inputs"]["pages"]],
            "actual_page_starts":starts,"actual_page_results":results,"unresolved":unresolved,
            "all_slots_terminal":not unresolved,
            "host_failure":failure,"cleanup":cleanup,"lifecycle":lifecycle_lock(out)}
    if manual:record["gpu_binding_schema"]=1
    states["processing_complete"]=not unresolved and normal_completion(out,record,strict=False)
    write(out/"RUN_MANIFEST.json",record)
    if not states["processing_complete"]:
        raise ContractError("Run stopped or cleanup unresolved; inspect RUN_MANIFEST/HOST_EXIT")
    return record
