"""Independent evaluator process only; GT is never accepted by infer."""
import difflib
import os
from pathlib import Path, PureWindowsPath
import shutil
import unicodedata
import time
import uuid
from .core import ContractError, Deadline, closed, digest, identifier, new_dir, read, sha, verify_tree, write
from .collection import verify_collection
from .image_identity import resolve_identities, validate_identities, verify_container_image
from .lifecycle import OwnedDocker, Leases, limits, mount_args, terminal_reason, verify_isolation

PINS={
    "revision":"f133a71e9e91c3621c7ce8994200a7b394a06eb3",
    "image":"ghcr.io/zeng-weijun/omnidocbench-eval@sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617",
    "config_sha256":"56db52659910273376223a5646eabfaff21b25dc30f434e1c90aab1a233d335a",
    "entry_sha256":"79dd63ae8b0bd19301374f170ab95569af69fe68a65416799a24a83e2390f0bd",
    "original_gt_sha256":"a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496",
    "source_receipt_sha256":"eb4d96c259827e7bb99b693b225466e21ac375897daf2b1ace2e5b27b38babdf",
}

def gt_subset(rows, ids):
    """Exact page stem join, duplicate IDs anywhere in source GT are rejected."""
    indexed={}
    for row in rows:
        key=Path(row["page_info"]["image_path"]).stem
        if key in indexed:
            raise ContractError("Duplicate GT page_id: "+key)
        indexed[key]=row
    if len(ids)!=len(set(ids)) or any(x not in indexed for x in ids):
        raise ContractError("Duplicate/missing GT join")
    return [indexed[x] for x in ids]

def benchmark_ids(pages):
    """Use the locked original stem without changing portable internal page IDs."""
    result=[]
    for page in pages:
        identifier(page['page_id'])
        key=page['original_page_id'] if 'original_page_id' in page else page['page_id']
        if (not isinstance(key,str) or not key or key in ('.','..')
            or '/' in key or '\\' in key or PureWindowsPath(key).drive
            or any(unicodedata.category(c) in ('Cc','Cs') for c in key)):
            raise ContractError('Unsafe or empty original page stem')
        result.append(key)
    if len(result)!=len(set(result)):
        raise ContractError('Duplicate original page stem')
    return result

def prediction_page_map(run,collection,gt_rows):
    pages=run['inputs']['pages'];keys=benchmark_ids(pages)
    subset=gt_subset(gt_rows,keys)
    predictions=collection['predictions']
    if len(predictions)!=len(pages) or [r['page_id'] for r in predictions]!=[p['page_id'] for p in pages]:
        raise ContractError('Evaluation mapping prediction order/count differs')
    rows=[];names=set()
    for page,key,gt_row,pred in zip(pages,keys,subset,predictions):
        # f133 end2end_dataset._resolve_prediction_path first uses basename[:-4].
        basename=Path(gt_row['page_info']['image_path']).name
        if not basename.endswith(('.png','.jpg')) or basename[:-4]!=key:
            raise ContractError('Unsupported official prediction filename mapping')
        name=basename[:-4]+'.md'
        if name in names:raise ContractError('Duplicate staged prediction name')
        names.add(name)
        rows.append({'safe_page_id':page['page_id'],'benchmark_id':key,
                     'original_page_id':page.get('original_page_id'),
                     'gt_image_path':gt_row['page_info']['image_path'],
                     'input_file_sha256':page['file_sha256'],
                     'collection_path':pred['path'],'prediction_sha256':pred['sha256'],
                     'staged_basename':name,'staged_sha256':pred['sha256']})
    return subset,rows

def stage_predictions(out,run_path,predictions,run,collection,gt_rows):
    subset,rows=prediction_page_map(run,collection,gt_rows)
    stage=Path(out)/'evaluation_predictions';stage.mkdir(exist_ok=False)
    root=Path(predictions).resolve().parent
    for row in rows:
        source=closed(root,row['collection_path']);target=closed(stage,row['staged_basename'])
        if sha(source)!=row['prediction_sha256']:raise ContractError('Original prediction changed before staging')
        shutil.copyfile(source,target)
        if target.is_symlink() or sha(target)!=row['staged_sha256']:
            raise ContractError('Staged prediction copy differs')
    mapping={'schema':1,'run_sha256':sha(run_path),'collection_sha256':sha(predictions),
             'page_count':len(rows),'pages':rows}
    write(Path(out)/'EVALUATION_PAGE_MAP.json',mapping)
    return subset,{'page_map_sha256':sha(Path(out)/'EVALUATION_PAGE_MAP.json'),
                   'staged_prediction_files':{r['staged_basename']:r['staged_sha256'] for r in rows}}

def verify_prediction_stage(out,run_path,predictions,gt,binding,diagnostic=False):
    run,collection=verify_collection(run_path,predictions,diagnostic)
    out=Path(out);stage=out/'evaluation_predictions'
    if sha(run_path)!=binding['run_sha256'] or sha(predictions)!=binding['collection_sha256'] or sha(gt)!=binding['gt_source_sha256']:
        raise ContractError('Evaluation mapping source changed')
    if sha(out/'EVALUATION_PAGE_MAP.json')!=binding['page_map_sha256']:
        raise ContractError('Evaluation page map changed')
    subset,rows=prediction_page_map(run,collection,read(gt))
    mapping=read(out/'EVALUATION_PAGE_MAP.json')
    expected={'schema':1,'run_sha256':binding['run_sha256'],'collection_sha256':binding['collection_sha256'],'page_count':len(rows),'pages':rows}
    if mapping!=expected or len(rows)!=binding['page_count']:
        raise ContractError('Evaluation mapping differs from locked inputs')
    files={r['staged_basename']:r['staged_sha256'] for r in rows}
    if files!=binding['staged_prediction_files'] or any(p.is_symlink() or not p.is_file() for p in stage.iterdir()) or {p.name for p in stage.iterdir()}!=set(files):
        raise ContractError('Evaluation prediction stage set differs')
    for name,h in files.items():
        if sha(closed(stage,name))!=h:raise ContractError('Evaluation staged bytes changed')
    if sha(out/'OmniDocBench.json')!=binding['gt_subset_sha256'] or read(out/'OmniDocBench.json')!=subset:
        raise ContractError('Evaluation GT subset changed')

def evaluate(run_path, predictions, gt, evaluator, output, diagnostic=False):
    started=time.monotonic()
    e=read(evaluator); base=Path(evaluator).resolve().parent
    b=e["budget"]; limit_args=limits(b,evaluation=True)
    if b["cleanup_seconds"]!=240:
        raise ContractError("Official host cleanup reserve must be 240 seconds")
    clock=Deadline(started,b["total_seconds"],b["cleanup_seconds"])
    for key in ("revision","image","config_sha256","entry_sha256"):
        if e.get(key)!=PINS[key]:
            raise ContractError("Official evaluator identity mismatch: "+key)
    run, collection=verify_collection(run_path,predictions,diagnostic)
    source=closed(base,e["source"]);config=closed(base,e["config"])
    original_entry=closed(base,e["original_entry"])
    source_receipt=closed(base,e["source_receipt"])
    if sha(source_receipt,clock)!=PINS["source_receipt_sha256"]:
        raise ContractError("Original evaluator source receipt changed")
    ready=read(source_receipt)
    if ready["source_revision"]!=PINS["revision"] or {x["path"]:x["sha256"] for x in ready["files"]}!=e["source_files"]:
        raise ContractError("Evaluator source file lock is not original")
    verify_tree(source,e["source_files"],clock)
    if sha(config)!=PINS["config_sha256"] or sha(original_entry)!=PINS["entry_sha256"]:
        raise ContractError("Official config/entry tamper")
    if sha(gt,clock)!=digest(e["gt_sha256"]):
        raise ContractError("GT identity mismatch")
    ids=[p["page_id"] for p in run["inputs"]["pages"]]
    subset=gt_subset(read(gt),benchmark_ids(run['inputs']['pages']))
    clock.remaining()
    if os.name!="posix":
        raise ContractError("Official runtime requires Linux Docker")
    from .runner import source_check
    framework=source_check()
    out=new_dir(output); work=out/"work";work.mkdir()
    subset,page_binding=stage_predictions(out,run_path,predictions,run,collection,read(gt))
    write(out/"OmniDocBench.json",subset)
    wrapper=Path(__file__).with_name("official_entry.py")
    (out/"OFFICIAL_WRAPPER.diff").write_text("".join(difflib.unified_diff(
        original_entry.read_text().splitlines(True),wrapper.read_text().splitlines(True),
        fromfile="original_entry.py",tofile="official_entry_n.py")),encoding="utf-8")
    binding={"run_id":run["run_id"],"profile":run["profile"],"page_count":len(ids),
        "run_sha256":sha(run_path),"collection_sha256":sha(predictions),
        "gt_source_sha256":sha(gt),"gt_subset_sha256":sha(out/"OmniDocBench.json"),
        "same_original_benchmark_gt":sha(gt)==PINS["original_gt_sha256"],
        "evaluator_lock_sha256":sha(evaluator),"wrapper_sha256":sha(wrapper),
        "framework":framework,"diagnostic":diagnostic,
        "failed_pages":[p["page_id"] for p in collection["predictions"] if p["status"]=="failed"],
        "unresolved":collection["unresolved"],**page_binding}
    write(out/"BINDING.json",binding)
    token=uuid.uuid4().hex; docker=OwnedDocker(token,clock,out)
    failure=None; cleanup=None; image_evidence_sha=None
    with Leases(e["lease_directory"],["official-scorer"]):
        try:
            identities=resolve_identities(run["run_id"],{"official":e["image"]},clock)
            write(out/"IMAGE_IDENTITIES.json",identities)
            image_evidence_sha=sha(out/"IMAGE_IDENTITIES.json")
            mounts=[(source,"/source",False),(config,"/config/end2end-full.yaml",False),
                    (out/"OmniDocBench.json","/gt/OmniDocBench.json",False),
                    (out/"evaluation_predictions","/pred",False),
                    (wrapper,"/entry.py",False),(work,"/work",True)]
            mounts.append((wrapper.with_name("official_schema.py"),"/official_schema.py",False))
            args=limit_args+["--workdir","/work","--entrypoint",e["python"],
                "--env","PYTHONPATH=/source","--env","CUDA_VISIBLE_DEVICES=","--env","HOME=/work",
                "--env","OMP_NUM_THREADS=1","--env","OPENBLAS_NUM_THREADS=1",
                "--env","MKL_NUM_THREADS=1","--env","NUMEXPR_NUM_THREADS=1"]
            args+=mount_args(mounts,"/work")
            args += [e["image"],"/entry.py","--pages",str(len(ids))]
            verify_prediction_stage(out,run_path,predictions,gt,binding,diagnostic)
            docker.create(args)
            verify_isolation(docker.inspect(),mounts,b,e["image"],[],identities["roles"]["official"])
            docker.start()
            while True:
                state=docker.inspect(cleanup=True)
                reason=terminal_reason(state,time.monotonic(),started,b["total_seconds"])
                if reason:
                    if reason!="exited_ok":
                        raise ContractError("Official evaluator terminal: "+reason)
                    break
                clock.remaining()
                time.sleep(min(.5,clock.remaining()))
            verify_container_image(state,e["image"],identities["roles"]["official"])
            if sha(out/"IMAGE_IDENTITIES.json")!=image_evidence_sha:
                raise ContractError("Official image identity evidence changed")
            validate_identities(read(out/"IMAGE_IDENTITIES.json"),run["run_id"],{"official":PINS["image"]})
            verify_prediction_stage(out,run_path,predictions,gt,binding,diagnostic)
            verify_tree(source,e["source_files"],clock)
            if sha(config)!=PINS["config_sha256"] or sha(gt)!=e["gt_sha256"]:
                raise ContractError("Evaluator input changed during run")
            score=read(work/"OFFICIAL_SCORE.json")
            if score.get("validated_result_schema") is not True or score.get("validation_sha256")!=sha(work/"RESULT_VALIDATION.json"):
                raise ContractError("Official schema validation missing or changed")
            validation=read(work/"RESULT_VALIDATION.json")
            required={"pred_quick_match_metric_result.json","pred_quick_match_run_summary.json"}|{
                "pred_quick_match_"+c+"_result.json" for c in ("text_block","display_formula","table","reading_order")}
            if set(validation["files"])!=required:
                raise ContractError("Official result evidence missing")
            for name,h in validation["files"].items():
                if sha(work/"result"/name)!=h:
                    raise ContractError("Official result bytes changed")
            if score["official_match_debug"]["page_count"]!=len(ids):
                raise ContractError("Official page denominator mismatch")
        except BaseException as exc:
            failure={"type":type(exc).__name__,"message":str(exc)}
        finally:
            try:
                cleanup=docker.cleanup(lambda c:{"verified":True,"kind":"cpu",
                    "observed_monotonic":c.clock()})
                if (cleanup.get("state") or {}).get("OOMKilled"):
                    failure={"type":"ContainerOOM","message":"Official container OOM; supersedes live timeout",
                             "prior_error":failure}
            except BaseException as exc:
                cleanup={"error":str(exc),"unresolved_owned_cid":docker.cid}
    if failure is None:
        try:
            if sha(out/"IMAGE_IDENTITIES.json")!=image_evidence_sha:
                raise ContractError("Official image identity evidence changed during cleanup")
            validate_identities(read(out/"IMAGE_IDENTITIES.json"),run["run_id"],{"official":PINS["image"]})
            verify_container_image(read(out/"CONTAINER_EXIT.json"),e["image"],identities["roles"]["official"])
        except (ContractError,KeyError,TypeError,ValueError,OSError) as exc:
            failure={"type":type(exc).__name__,"message":str(exc)}
    elapsed=time.monotonic()-started
    finished_ok=failure is None and bool(cleanup and cleanup.get("complete")) and elapsed<=b["total_seconds"]
    result={"evaluation_complete":finished_ok,
            "formal_result":not diagnostic and finished_ok,
            "quality_claim":False,"failure":failure,"cleanup":cleanup,
            "elapsed_seconds":elapsed,"budget_exceeded":elapsed>b["total_seconds"],
            "binding":binding,"image_identities_sha256":image_evidence_sha}
    write(out/"EVALUATION_RESULT.json",result)
    if not result["evaluation_complete"]:
        raise ContractError("Official evaluation failed; original logs preserved")
    return result
