"""Immutable, exact-N collection; failures remain empty predictions."""
import shutil
from pathlib import Path
from .core import ContractError, closed, digest, new_dir, read, sha, write, canonical, identifier
from .completion import normal_completion

def terminals(run_path):
    run_path=Path(run_path).resolve(); run=read(run_path); root=run_path.parent
    pages=run["inputs"]["pages"]
    ids=[identifier(p["page_id"]) for p in pages]
    if not ids or len(ids)!=len({x.casefold() for x in ids}):
        raise ContractError("Run page identity invalid")
    if run["claims"] != ids:
        raise ContractError("Claim/input order mismatch")
    records=[]; unresolved=[];started_ids=[]
    for page in pages:
        folder=closed(root,"pages/"+page["page_id"])
        terminal=folder/"PAGE_RESULT.json"
        start=folder/"PAGE_START.json"
        if start.exists():started_ids.append(page["page_id"])
        if not terminal.exists():
            unresolved.append(page["page_id"])
            continue
        if not start.exists():
            raise ContractError("Result without actual PAGE_START")
        t=read(terminal); s=read(start)
        for obj in (s,t):
            if (obj["run_id"],obj["page_id"],obj["input_sha256"]) != (run["run_id"],page["page_id"],page["file_sha256"]):
                raise ContractError("Terminal/start binding differs")
        if t["start_sha256"] != sha(start) or sha(folder/"prediction.md") != t["prediction_sha256"]:
            raise ContractError("Prediction/start tamper")
        if t["status"] not in ("success","truncated","failed"):
            raise ContractError("Unknown terminal status")
        if t["status"] == "failed":
            if (folder/"prediction.md").read_bytes() != b"":
                raise ContractError("Failed page must be empty")
        else:
            for name in ("INPUT.json","RENDER.json","GEOMETRY.json","CONTENT.json",
                         "GENERATION.json","REQUEST_BINDINGS.json","RESOURCE_AUDIT.json","AUDIT_STATUS.json"):
                if not (folder/name).is_file():
                    raise ContractError("Successful page audit missing: "+name)
            a=read(folder/"AUDIT_STATUS.json")
            if not a["complete"] or not a["patches_restored"] or a["error"]:
                raise ContractError("Success with incomplete audit")
            control=read(root/"CONTROL.json")
            if control.get("profile")=="v32-text":
                from .v32_dispatch import verify
                verify(folder)
            elif control["runtime"].get("schema")==2:
                for name in ("V31_TRANSACTIONS.json","V31_FINAL_RAW.json","PRIMARY_CONTENT.json"):
                    if not (folder/name).is_file():raise ContractError("V31 page transaction evidence missing")
                from .selection_audit import verify
                verify(folder)
        records.append((page,t,folder))
    if run.get("all_slots_terminal") != (not unresolved):
        raise ContractError("Terminal coverage contradicts actual results")
    if run["states"]["processing_complete"] and unresolved:
        raise ContractError("Processing state contradicts actual results")
    if run.get("actual_page_starts")!=started_ids or run.get("actual_page_results")!=[p["page_id"] for p,_,_ in records]:
        raise ContractError("Actual start/result coverage differs")
    if run.get("unresolved")!=unresolved:
        raise ContractError("Unresolved coverage differs")
    return run,records,unresolved

def collect(run_path, output, diagnostic=False):
    run,records,unresolved=terminals(run_path)
    if unresolved and not diagnostic:
        raise ContractError("Unstarted/unresolved pages: primary collection refused")
    if not diagnostic:
        if not run["states"]["processing_complete"]:
            raise ContractError("Formal processing is incomplete")
        normal_completion(Path(run_path).resolve().parent,run)
    out=new_dir(output); pred=out/"predictions";pred.mkdir()
    by_id={p["page_id"]:(t,f) for p,t,f in records}
    rows=[]; evidence=dict(run.get("lifecycle",{}))
    root=Path(run_path).resolve().parent
    for p in run["inputs"]["pages"]:
        item=by_id.get(p["page_id"])
        target=pred/(p["page_id"]+".md")
        if item:
            t,folder=item
            shutil.copyfile(folder/"prediction.md",target)
            status=t["status"]
            for path in folder.rglob("*"):
                if path.is_file():
                    evidence[path.relative_to(root).as_posix()]=sha(path)
        else:
            target.write_bytes(b"");status="unresolved_diagnostic_placeholder"
        rows.append({"page_id":p["page_id"],"path":"predictions/"+target.name,
                     "sha256":sha(target),"status":status})
    lock={"schema":1,"run_id":run["run_id"],"run_sha256":sha(run_path),
          "input_binding":canonical(run["inputs"]),"page_count":len(rows),
          "processing_complete":run["states"]["processing_complete"],"all_slots_terminal":not unresolved,"diagnostic":diagnostic,
          "predictions":rows,"run_evidence":evidence,"unresolved":unresolved}
    write(out/"COLLECTED_LOCK.json",lock)
    return lock

def verify_collection(run_path, lock_path, diagnostic=False):
    run,records,unresolved=terminals(run_path)
    lock=read(lock_path);root=Path(lock_path).resolve().parent
    if lock["run_sha256"] != sha(run_path) or lock["run_id"] != run["run_id"]:
        raise ContractError("Run lock mismatch")
    if lock["input_binding"] != canonical(run["inputs"]):
        raise ContractError("Locked input identity mismatch")
    expected=[p["page_id"] for p in run["inputs"]["pages"]]
    if [p["page_id"] for p in lock["predictions"]] != expected or lock["page_count"] != len(expected):
        raise ContractError("Prediction coverage/order differs")
    if (unresolved or lock["diagnostic"] or not lock["processing_complete"]) and not diagnostic:
        raise ContractError("Formal evaluation rejects incomplete/diagnostic input")
    if not diagnostic:
        normal_completion(Path(run_path).resolve().parent,run)
    for name,h in run.get("lifecycle",{}).items():
        if lock["run_evidence"].get(name)!=h:
            raise ContractError("Collected lifecycle evidence list incomplete")
    expected_evidence=set(run.get("lifecycle",{}))
    for _,_,folder in records:
        expected_evidence.update(p.relative_to(Path(run_path).resolve().parent).as_posix()
                                 for p in folder.rglob("*") if p.is_file())
    if set(lock["run_evidence"])!=expected_evidence:
        raise ContractError("Collected full run evidence list differs")
    if {p.name for p in (root/"predictions").iterdir()} != {x+".md" for x in expected}:
        raise ContractError("Extra/missing prediction")
    for p in lock["predictions"]:
        if p["path"] != "predictions/"+p["page_id"]+".md" or sha(closed(root,p["path"])) != digest(p["sha256"]):
            raise ContractError("Collected prediction tamper")
    for rel, expected_sha in lock["run_evidence"].items():
        if sha(closed(Path(run_path).resolve().parent,rel)) != digest(expected_sha):
            raise ContractError("Run evidence tamper")
    return run,lock
