"""Single bounded owned-resource finalization, including emergency stop after overrun."""
import json
import re
from pathlib import Path
from .core import ContractError, write

def inspect_absent(result, target):
    """Recognize only a complete Docker absence response for this exact target."""
    if type(result.returncode) is not int or result.returncode != 1:
        return False
    if not isinstance(target, str) or not target or not isinstance(result.stdout, str) or not isinstance(result.stderr, str):
        return False
    if result.stdout.strip():
        try:
            if json.loads(result.stdout) != []:
                return False
        except (ValueError, TypeError):
            return False
    match = re.fullmatch(r"(?i:(?:error: |error response from daemon: )?no such (?:object|container): )([^\r\n]+)", result.stderr.strip())
    return match is not None and match.group(1) == target

class OwnershipError(ContractError):
    pass

class CleanupClock:
    def __init__(self, deadline):
        self.clock=deadline.clock
        self.started=self.clock()
        self.host_end=deadline.start+deadline.total
        self.end=self.started+240
        self.budget_exceeded=self.started>=self.host_end
    def bound(self, cap, cleanup=False):
        now=self.clock()
        if now>=self.host_end:
            self.budget_exceeded=True
        remaining=self.end-now
        if remaining<=0:
            raise ContractError("Finite emergency cleanup path exhausted")
        # Cleanup alone may pass host deadline to stop an already-owned resource.
        return min(cap,remaining)

def finalize(owned, release=None, clock=None):
    clock=clock or CleanupClock(owned.deadline)
    errors=[];events=[];pre=None;final=None;removed=False;absent=False
    release_record=None;kill_error=None;confirmed=getattr(owned,"last_verified",None)
    def error(stage,exc):
        item={"stage":stage,"type":type(exc).__name__,"message":str(exc)}
        errors.append(item);return item
    def call(args,cap=20):
        before=clock.clock()
        try:
            result=owned.runner(args,text=True,capture_output=True,check=False,timeout=clock.bound(cap))
            events.append({"args":args,"returncode":result.returncode,
                           "stdout":result.stdout if args[1]!="logs" else "(saved separately)",
                           "stderr":result.stderr if args[1]!="logs" else "(saved separately)",
                           "elapsed_seconds":clock.clock()-before})
            return result
        except BaseException as exc:
            events.append({"args":args,"error":str(exc),"elapsed_seconds":clock.clock()-before})
            raise
    def inspect(target):
        p=call(["docker","inspect",target])
        if p.returncode:
            raise ContractError("inspect failed: "+p.stderr)
        obj=json.loads(p.stdout)[0]
        if obj["Config"].get("Labels",{}).get("hybrid-v3-owner")!=owned.token:
            raise OwnershipError("Foreign owner; no destructive cleanup")
        if owned.cid and obj["Id"]!=owned.cid:
            raise OwnershipError("Foreign CID; no destructive cleanup")
        return obj
    target=owned.cid or ("hybrid-v3-"+owned.token if owned.creation_attempted else None)
    if target:
        try:
            pre=inspect(target)
            owned.cid=pre["Id"]
            confirmed=pre
        except BaseException as exc:
            error("pre_inspect",exc)
            if isinstance(exc,OwnershipError):
                confirmed=None
        # A previous exact CID+label verification may authorize a bounded stop
        # even when Docker inspect is now slow/unavailable. Unknown identity never does.
        if confirmed and confirmed["Id"]==owned.cid and confirmed["Config"].get("Labels",{}).get("hybrid-v3-owner")==owned.token:
            cid=owned.cid
            if pre is None or pre["State"].get("Running"):
                try:
                    k=call(["docker","kill",cid])
                    if k.returncode:
                        raise ContractError(k.stderr)
                except BaseException as exc:
                    kill_error=error("kill",exc)
            try:
                w=call(["docker","wait",cid],45)
                if w.returncode:raise ContractError(w.stderr)
            except BaseException as exc:
                error("wait",exc)
            try:
                final=inspect(cid)
                if final["State"].get("Running") or final["State"].get("Pid")!=0:
                    raise ContractError("Owned container still running or PID not zero")
                if kill_error and not final["State"].get("Running") and final["State"].get("ExitCode")==0:
                    kill_error["recovered_kill_race"]=True
            except BaseException as exc:
                error("final_inspect",exc)
            # Logs and disk failures never bypass the independent removal attempt.
            try:
                logs=call(["docker","logs",cid],10)
                if logs.returncode:raise ContractError(logs.stderr)
                (owned.output/"stdout.log").write_text(logs.stdout,encoding="utf-8")
                (owned.output/"stderr.log").write_text(logs.stderr,encoding="utf-8")
            except BaseException as exc:
                error("logs",exc)
            try:
                args=["docker","rm"]
                if final is None or final["State"].get("Running"):
                    args.append("--force")
                r=call([*args,cid])
                if r.returncode:raise ContractError(r.stderr)
                removed=True
                removed_at=clock.clock()
                absent_result=call(["docker","inspect",cid],10)
                absent=inspect_absent(absent_result,cid)
                if not absent:raise ContractError("Container absence not verified")
                owned.cid=None
            except BaseException as exc:
                error("remove",exc)
        else:
            error("ownership",ContractError("No confirmed owned identity; refuse kill/remove"))
    else:
        absent=True
    try:
        if release is not None:
            release_record=release(clock)
            if not release_record.get("verified"):
                raise ContractError("Resource release unverified")
    except BaseException as exc:
        error("release",exc)
    def persist(name,data):
        try:write(owned.output/name,data)
        except BaseException as exc:error("persist:"+name,exc)
    if final is not None:persist("CONTAINER_EXIT.json",final)
    if release_record is not None:persist("GPU_RELEASE.json",release_record)
    exhausted=clock.clock()>clock.host_end or clock.budget_exceeded
    result={"created":target is not None,"removed":removed,"absence_verified":absent,
            "final_nonrunning_pid0":bool(final and not final["State"].get("Running") and final["State"].get("Pid")==0),
            "pre_kill_state":pre,"final_state":final,"events":events,
            "state":None if final is None else final["State"],"errors":errors,
            "budget_exceeded":exhausted,"emergency_cleanup":exhausted,
            "cleanup_started_monotonic":clock.started,"cleanup_finished_monotonic":clock.clock(),
            "removed_monotonic":locals().get("removed_at"),
            "release":release_record,"unresolved_owned_cid":owned.cid}
    fatal=[x for x in errors if not x.get("recovered_kill_race")]
    result["complete"]=bool(removed and absent and result["final_nonrunning_pid0"] and not fatal
                            and not exhausted and (release is None or release_record and release_record.get("verified")))
    # Recovered race stays visible; it is not a missing audit error.
    result["errors"]=[x for x in errors if not x.get("recovered_kill_race")]
    result["recovered_errors"]=[x for x in errors if x.get("recovered_kill_race")]
    persist("CLEANUP.json",result)
    if any(x["stage"]=="persist:CLEANUP.json" for x in errors):
        result["complete"]=False
    return result
