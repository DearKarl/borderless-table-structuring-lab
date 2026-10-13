"""Normal completion is a sealed lifecycle, not merely N terminal records."""
from pathlib import Path
from .core import ContractError, read, sha

LIFECYCLE_FILES = ("CONTROL.json","CLAIMS.json","SESSION_RESULT.json","HOST_EXIT.json",
                   "LEDGER.jsonl","CONTAINER_CREATED.json","CONTAINER_EXIT.json",
                   "CLEANUP.json","GPU_BEFORE.json","GPU_RELEASE.json")

EXPERT_FILES=tuple("experts/"+name for name in ("READY.json","LEDGER.jsonl","SESSION_RESULT.json",
    "CONTAINER_CREATED.json","CONTAINER_EXIT.json","CLEANUP.json","FINAL_AUDIT.json"))

def required_lifecycle(root):
    control=read(Path(root)/"CONTROL.json")
    active=control.get("mode")=="on" and bool(control["runtime"].get("v31",{}).get("components"))
    required=set(LIFECYCLE_FILES)
    from .gpu_backend import required_evidence
    required.update(required_evidence(control))
    if control.get("image_identity_schema")==1:
        required.add("IMAGE_IDENTITIES.json")
    if control.get("cleanup_schema")==2:
        required.update(("GROUP_CLEANUP.json","ADMISSION_RESULT.json"))
    if control["runtime"].get("schema")==2:
        required.update(("NATIVE_LOAD_BOUNDARY.json","LOADED_PARAMETERS.json","PROCESSOR.json","TELE_LOAD.json"))
    if active:
        required.update(EXPERT_FILES)
        required.update(p.relative_to(root).as_posix() for p in (Path(root)/"experts").rglob("*") if p.is_file())
    return tuple(sorted(required))

def lifecycle_lock(root):
    root=Path(root)
    return {name:sha(root/name) for name in required_lifecycle(root) if (root/name).is_file()}

def normal_completion(root, run, strict=True):
    root=Path(root)
    def reject(message):
        if strict:
            raise ContractError("Formal lifecycle rejected: "+message)
        return False
    locks=run.get("lifecycle",{})
    if set(locks)!=set(required_lifecycle(root)):
        return reject("missing lifecycle evidence")
    for name,h in locks.items():
        if not (root/name).is_file() or sha(root/name)!=h:
            return reject("lifecycle evidence changed: "+name)
    control=read(root/"CONTROL.json");claims=read(root/"CLAIMS.json")
    session=read(root/"SESSION_RESULT.json");host=read(root/"HOST_EXIT.json")
    created=read(root/"CONTAINER_CREATED.json");exit_state=read(root/"CONTAINER_EXIT.json")
    cleanup=read(root/"CLEANUP.json");release=read(root/"GPU_RELEASE.json")
    from .gpu_backend import validate_evidence
    try:validate_evidence(root,control,run)
    except (ContractError,KeyError,TypeError,ValueError,OSError) as exc:
        return reject("manual GPU evidence differs: "+str(exc))
    if control.get("image_identity_schema") is not None or run.get("image_identity_schema") is not None:
        if (type(control.get("image_identity_schema")) is not int or type(run.get("image_identity_schema")) is not int
                or control.get("image_identity_schema") != 1 or run.get("image_identity_schema") != 1):
            return reject("image identity audit version differs")
        from .image_identity import effective_references, validate_identities, verify_container_image
        try:
            references=effective_references(control["runtime"],control["mode"])
            evidence=validate_identities(read(root/"IMAGE_IDENTITIES.json"),run["run_id"],references)
            for role,reference in references.items():
                state=exit_state if role=="native" else read(root/"experts/CONTAINER_EXIT.json")
                verify_container_image(state,reference,evidence["roles"][role])
        except (ContractError,KeyError,TypeError,ValueError,OSError) as exc:
            return reject("image identity evidence differs: "+str(exc))
    if control.get("cleanup_schema")==2:
        group=read(root/"GROUP_CLEANUP.json");admission=read(root/"ADMISSION_RESULT.json")
        expected=[run["run_id"]]
        if control.get("mode")=="on" and control["runtime"].get("v31",{}).get("components"):
            expected.insert(0,run["run_id"]+"-paddle")
        if (not group.get("complete") or not group.get("resource_release_verified") or group.get("budget_exceeded")
            or not admission.get("released") or admission.get("blocked") or admission.get("run_id")!=run["run_id"]
            or group.get("release")!=release or admission.get("gpu_release")!=release
            or [x.get("owner") for x in group["resources"]]!=expected
            or group["resources"][-1]!=cleanup
            or any(not x.get("complete") for x in group["resources"])):
            return reject("group cleanup/admission evidence differs")
        if len(expected)==2 and group["resources"][0]!=read(root/"experts/CLEANUP.json"):
            return reject("expert group cleanup binding differs")
    ids=[p["page_id"] for p in run["inputs"]["pages"]]
    if strict and not all(run["states"].get(k) is True for k in
                          ("processing_complete","runtime_verified","integration_verified")):
        return reject("run state is not formally eligible")
    if control["run_id"]!=run["run_id"] or claims!={"run_id":run["run_id"],"page_ids":ids}:
        return reject("run/claim identity")
    if control["inputs"]!=run["inputs"] or control["profile"]!=run["profile"]:
        return reject("control/input/profile identity")
    if not all(session.get(k) is True for k in ("processing_complete","runtime_verified","integration_verified")):
        return reject("session incomplete/unverified")
    if session.get("system_error") is not None or session.get("completed")!=ids:
        return reject("session error or completed coverage")
    if host.get("failure") is not None or host.get("budget_exceeded") is not False:
        return reject("host failure/budget")
    if run.get("host_failure")!=host.get("failure") or run.get("cleanup")!=cleanup:
        return reject("run/host lifecycle binding differs")
    if not cleanup.get("complete") or not cleanup.get("removed") or not cleanup.get("absence_verified"):
        return reject("owned release unconfirmed")
    if cleanup.get("errors") or cleanup.get("budget_exceeded") or not cleanup.get("final_nonrunning_pid0"):
        return reject("cleanup audit/budget/state")
    if host.get("cleanup")!=cleanup:
        return reject("host cleanup binding")
    if not release.get("verified") or release.get("gpu_uuids")!=control["runtime"]["gpu_uuids"]:
        return reject("GPU release absent/mismatched")
    if host.get("gpu_after")!=release or cleanup.get("release")!=release:
        return reject("fresh release evidence binding differs")
    uuids=control["runtime"]["gpu_uuids"]
    for memory in (read(root/"GPU_BEFORE.json"),release.get("memory_mib",{})):
        if set(memory)!=set(uuids) or any(type(v) not in (int,float) or not 0<=v<=control["runtime"]["idle_memory_mib"] for v in memory.values()):
            return reject("GPU memory/idleness evidence differs")
    if release.get("observed_monotonic",0)<cleanup.get("removed_monotonic",float("inf")):
        return reject("GPU release is not fresh")
    if created["owner"]!=run["run_id"] or created["cid"]!=exit_state["Id"]:
        return reject("CID ownership")
    if exit_state["Config"].get("Labels",{}).get("hybrid-v3-owner")!=run["run_id"]:
        return reject("container owner label")
    state=exit_state["State"]
    if state.get("Running") or state.get("Pid")!=0 or state.get("OOMKilled") or state.get("ExitCode")!=0:
        return reject("container not normal exited/PID0")
    if not 0<=host["elapsed_seconds"]<=control["budget"]["total_seconds"]:
        return reject("host duration exceeds locked budget")
    if host["started_monotonic"]!=control["host_started_monotonic"]:
        return reject("host clock identity")
    if abs(host["finished_monotonic"]-host["started_monotonic"]-host["elapsed_seconds"])>0.001:
        return reject("host elapsed evidence differs")
    if release["observed_monotonic"]>host["finished_monotonic"]:
        return reject("release timestamp is after host completion")
    import json
    if "experts/READY.json" in locks:
        ec=read(root/"experts/CONTAINER_CREATED.json");ee=read(root/"experts/CONTAINER_EXIT.json")
        clean=read(root/"experts/CLEANUP.json");ready=read(root/"experts/READY.json")
        es=read(root/"experts/SESSION_RESULT.json")
        final=read(root/"experts/FINAL_AUDIT.json")
        if final.get("close_errors") or final.get("resource_audit",{}).get("unknown_resources"):
            return reject("expert final resource/close audit")
        owner=run["run_id"]+"-paddle"
        if ec["owner"]!=owner or ec["cid"]!=ee["Id"] or ee["Config"].get("Labels",{}).get("hybrid-v3-owner")!=owner:
            return reject("expert owned CID identity")
        if (ee["State"].get("Running") or ee["State"].get("Pid")!=0 or ee["State"].get("OOMKilled") or ee["State"].get("ExitCode")!=0
            or not clean.get("complete") or clean.get("errors") or not clean.get("absence_verified") or clean.get("budget_exceeded")):
            return reject("expert exit/cleanup failed")
        if ready["run_id"]!=run["run_id"] or not es.get("normal_exit") or es["run_id"]!=run["run_id"]:
            return reject("expert session identity")
        if clean["removed_monotonic"]>release["observed_monotonic"]:
            return reject("GPU lease released before all experts removed")
        el=[json.loads(line) for line in (root/"experts/LEDGER.jsonl").read_text().splitlines()]
        if len([x for x in el if x["event"]=="LOAD_START"])!=ready["model_loads"]:
            return reject("expert model load ledger differs")
    ledger=[json.loads(line) for line in (root/"LEDGER.jsonl").read_text().splitlines()]
    if not ledger or any(x["event"]=="SYSTEM_STOP" for x in ledger):
        return reject("missing/failed ledger")
    for name in ("PAGE_START","PAGE_RESULT"):
        if [x["page_id"] for x in ledger if x["event"]==name]!=ids:
            return reject("ledger page coverage differs")
    return True
