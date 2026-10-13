"""Host-wide cleanup for at most two owned containers within one 240-second clock."""
import json
from .core import ContractError, write
from .bounded_cleanup import CleanupClock, OwnershipError, inspect_absent

# Sum at full caps: 2*(8+8+20+8+5+12+6) + 10 GPU = 144 seconds.
# All stop/wait/state phases precede optional logs; removal retains its own allowance.
CAPS={"inspect":8,"kill":8,"wait":20,"final":8,"logs":5,"remove":12,"absence":6}

def finalize_group(resources,release,clock=None):
    if not 1<=len(resources)<=2:raise ContractError("One or two owned resources required")
    clock=clock or CleanupClock(resources[0].deadline)
    rows=[]
    for owned in resources:
        target=owned.cid or ("hybrid-v3-"+owned.token if owned.creation_attempted else None)
        rows.append({"owned":owned,"target":target,"confirmed":getattr(owned,"last_verified",None),
                     "pre":None,"final":None,"removed":False,"absent":target is None,
                     "errors":[],"events":[],"removed_at":None,"foreign":False})
    def error(row,stage,exc):
        item={"stage":stage,"type":type(exc).__name__,"message":str(exc)}
        row["errors"].append(item);return item
    def command(row,args,stage):
        before=clock.clock();cap=CAPS[stage]
        try:
            p=row["owned"].runner(args,text=True,capture_output=True,check=False,timeout=clock.bound(cap))
            row["events"].append({"stage":stage,"args":args,"returncode":p.returncode,
                "stdout":"(logs)" if stage=="logs" else p.stdout,
                "stderr":"(logs)" if stage=="logs" else p.stderr,"elapsed_seconds":clock.clock()-before})
            return p
        except BaseException as exc:
            row["events"].append({"stage":stage,"args":args,"error":str(exc),"elapsed_seconds":clock.clock()-before})
            raise
    def inspect(row,target,stage):
        p=command(row,["docker","inspect",target],stage)
        if p.returncode:
            if inspect_absent(p,target):
                row["absent"]=True;row["removed_at"]=clock.clock()
                return None
            raise ContractError("Container inspection failed: "+p.stderr)
        value=json.loads(p.stdout)[0];owned=row["owned"]
        if (value["Config"].get("Labels",{}).get("hybrid-v3-owner")!=owned.token
            or owned.cid and value["Id"]!=owned.cid):
            row["foreign"]=True;row["confirmed"]=None
            raise OwnershipError("Foreign identity; destructive action refused")
        return value
    def confirmed(row):
        owned=row["owned"];obj=row["confirmed"]
        return (not row["foreign"] and not row["absent"] and obj is not None
                and obj["Id"]==owned.cid and obj["Config"].get("Labels",{}).get("hybrid-v3-owner")==owned.token)

    # Discover every possible partial create before spending time on one resource.
    for row in rows:
        if row["target"] is None:continue
        try:
            obj=inspect(row,row["target"],"inspect")
            if obj is not None:
                row["pre"]=row["confirmed"]=obj;row["owned"].cid=obj["Id"]
                row["owned"].last_verified=obj
                if row["owned"].admission is not None:row["owned"].admission.observe(row["owned"])
        except BaseException as exc:error(row,"pre_inspect",exc)
        if not row["absent"] and not confirmed(row):
            error(row,"ownership",ContractError("Unresolved partial create/ownership; admission remains blocked"))
    for row in rows:
        if not confirmed(row):continue
        if row["pre"] is None or row["pre"]["State"].get("Running"):
            try:
                p=command(row,["docker","kill",row["owned"].cid],"kill")
                if p.returncode:raise ContractError(p.stderr)
            except BaseException as exc:error(row,"kill",exc)
    for row in rows:
        if not confirmed(row):continue
        try:
            p=command(row,["docker","wait",row["owned"].cid],"wait")
            if p.returncode:raise ContractError(p.stderr)
        except BaseException as exc:error(row,"wait",exc)
    for row in rows:
        if not confirmed(row):continue
        try:
            row["final"]=inspect(row,row["owned"].cid,"final")
            if row["final"] is not None:
                state=row["final"]["State"]
                if state.get("Running") or state.get("Pid")!=0:raise ContractError("Final state still running")
                if state.get("ExitCode")==0:
                    for e in row["errors"]:
                        if e["stage"]=="kill":e["recovered_kill_race"]=True
        except BaseException as exc:error(row,"final_inspect",exc)

    # Diagnostics have a small fixed cap and may never consume reserved removal/release time.
    remaining_critical=sum(CAPS["remove"]+CAPS["absence"] for r in rows if confirmed(r))+10
    for row in rows:
        if not confirmed(row):continue
        if clock.end-clock.clock()<=remaining_critical+CAPS["logs"]:
            error(row,"logs",ContractError("Logs skipped to preserve group removal/release budget"))
            continue
        try:
            p=command(row,["docker","logs",row["owned"].cid],"logs")
            if p.returncode:raise ContractError(p.stderr)
            (row["owned"].output/"stdout.log").write_text(p.stdout,encoding="utf-8")
            (row["owned"].output/"stderr.log").write_text(p.stderr,encoding="utf-8")
        except BaseException as exc:error(row,"logs",exc)

    # Independent removal attempts for EVERY confirmed owned resource, even after wait/log errors.
    for row in rows:
        if not confirmed(row):continue
        try:
            final=row["final"]
            args=["docker","rm"]
            if final is None or final["State"].get("Running"):args.append("--force")
            p=command(row,[*args,row["owned"].cid],"remove")
            if p.returncode:raise ContractError(p.stderr)
            row["removed"]=True;row["removed_at"]=clock.clock()
        except BaseException as exc:error(row,"remove",exc)
    for row in rows:
        if row["target"] is None or row["foreign"] or row["absent"]:continue
        # Even a failed rm may have succeeded in the daemon; verify without assuming that it did.
        try:
            p=command(row,["docker","inspect",row["owned"].cid or row["target"]],"absence")
            row["absent"]=inspect_absent(p,row["owned"].cid or row["target"])
            if not row["absent"]:raise ContractError("Container absence not verified")
            row["removed_at"]=clock.clock();row["owned"].cid=None
        except BaseException as exc:error(row,"absence",exc)
    released=all(r["absent"] and not r["foreign"] for r in rows)
    release_record=None;group_errors=[]
    if released:
        try:
            release_record=release(clock)
            if not release_record.get("verified"):raise ContractError("Fresh GPU release not verified")
        except BaseException as exc:
            group_errors.append({"stage":"gpu_release","type":type(exc).__name__,"message":str(exc)})
            released=False
    exhausted=clock.budget_exceeded or clock.clock()>clock.host_end or clock.clock()>clock.end
    results=[]
    for row in rows:
        owned=row["owned"];final=row["final"]
        errors=[e for e in row["errors"] if not e.get("recovered_kill_race")]
        item={"owner":owned.token,"cid":(row["confirmed"] or {}).get("Id",row["target"]),"created":row["target"] is not None,
              "removed":row["removed"],"absence_verified":row["absent"],
              "final_nonrunning_pid0":bool(final and not final["State"].get("Running") and final["State"].get("Pid")==0),
              "pre_kill_state":row["pre"],"final_state":final,"state":None if final is None else final["State"],
              "errors":errors,"recovered_errors":[e for e in row["errors"] if e.get("recovered_kill_race")],
              "events":row["events"],"budget_exceeded":exhausted,"emergency_cleanup":exhausted,
              "cleanup_started_monotonic":clock.started,"cleanup_finished_monotonic":clock.clock(),
              "removed_monotonic":row["removed_at"],"release":release_record,
              "unresolved_owned_cid":None if row["absent"] else owned.cid or row["target"],
              "resource_release_verified":row["absent"] and not row["foreign"]}
        item["complete"]=bool(item["removed"] and item["absence_verified"] and item["final_nonrunning_pid0"]
                              and not errors and not exhausted and released)
        try:
            if final is not None:write(owned.output/"CONTAINER_EXIT.json",final)
            if release_record is not None:write(owned.output/"GPU_RELEASE.json",release_record)
        except BaseException as exc:
            item["errors"].append({"stage":"persist","type":type(exc).__name__,"message":str(exc)});item["complete"]=False
        try:write(owned.output/"CLEANUP.json",item)
        except BaseException as exc:
            item["errors"].append({"stage":"persist_cleanup","type":type(exc).__name__,"message":str(exc)});item["complete"]=False
        results.append(item)
    return {"resources":results,"resource_release_verified":released,
            "release":release_record,"errors":group_errors,"budget_exceeded":exhausted,
            "started_monotonic":clock.started,"finished_monotonic":clock.clock(),
            "deadline_monotonic":clock.end,"maximum_cleanup_seconds":240,
            "complete":released and not group_errors and not exhausted and all(r["complete"] for r in results)}
