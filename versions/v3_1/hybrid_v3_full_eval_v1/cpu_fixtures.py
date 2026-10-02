"""Synthetic CPU records only; never model, GPU or official-score evidence."""
import json
from pathlib import Path
from .core import write,sha
from .completion import lifecycle_lock
from .official_schema import EXPECTED

def lifecycle_fixture(root,manifest,failed=(),unrun=()):
    root=Path(root);ids=[p["page_id"] for p in manifest["pages"]]
    normal=not failed and not unrun
    control={"run_id":"fixture","profile":"native","inputs":manifest,
             "host_started_monotonic":10,"budget":{"total_seconds":1000},
             "runtime":{"gpu_uuids":["GPU-fixture"],"idle_memory_mib":0}}
    write(root/"CONTROL.json",control)
    write(root/"CLAIMS.json",{"run_id":"fixture","page_ids":ids})
    write(root/"SESSION_RESULT.json",{"processing_complete":normal,"runtime_verified":True,
        "integration_verified":normal,"system_error":None if normal else {"type":"synthetic failure"},
        "completed":[x for i,x in enumerate(ids) if i not in failed and i not in unrun]})
    state={"Id":"cid","Config":{"Labels":{"hybrid-v3-owner":"fixture"}},
           "State":{"Running":False,"Pid":0,"OOMKilled":False,"ExitCode":0 if normal else 1}}
    write(root/"CONTAINER_CREATED.json",{"cid":"cid","owner":"fixture"})
    write(root/"CONTAINER_EXIT.json",state)
    cleanup={"complete":True,"removed":True,"absence_verified":True,"errors":[],
             "budget_exceeded":False,"final_nonrunning_pid0":True,"removed_monotonic":20}
    release={"verified":True,"gpu_uuids":["GPU-fixture"],"observed_monotonic":21,"memory_mib":{"GPU-fixture":0}}
    cleanup["release"]=release
    write(root/"CLEANUP.json",cleanup)
    write(root/"GPU_BEFORE.json",{"GPU-fixture":0});write(root/"GPU_RELEASE.json",release)
    write(root/"HOST_EXIT.json",{"failure":None if normal else {"type":"system exit"},
        "budget_exceeded":False,"cleanup":cleanup,"gpu_after":release,"started_monotonic":10,"finished_monotonic":22,
        "elapsed_seconds":12})
    events=[]
    for i,key in enumerate(ids):
        if i in unrun:continue
        events.append({"event":"PAGE_START","page_id":key})
        events.append({"event":"PAGE_RESULT","page_id":key})
    if not normal:events.append({"event":"SYSTEM_STOP"})
    (root/"LEDGER.jsonl").write_text("".join(json.dumps(x)+"\n" for x in events))
    return lifecycle_lock(root)

def official_fixture(root,empty=()):
    root=Path(root);metrics={"match_debug":{"page_count":2}};denoms={}
    for category,names in EXPECTED.items():
        absent=category in empty
        sample={"img_id":"p0.png","gt":"abc","pred":"abd","upper_len":3,
                "metric":{name:.25 for name in names}}
        write(root/("pred_quick_match_"+category+"_result.json"),[] if absent else [sample])
        metrics[category]={"all":{},"group":{},"page":{}}
        denoms[category]={}
        for name in names:
            value="NaN" if absent else .25
            key="ALL_page_avg" if name=="Edit_dist" else "all"
            metrics[category]["all"][name]={key:value}
            if not absent:metrics[category]["page"][name]={"ALL":.25}
            denoms[category][name]={"ALL":0 if absent else 1}
    write(root/"pred_quick_match_run_summary.json",{"save_name":"pred_quick_match",
          "page_denominators":denoms,"notebook_metric_summary":{"source":"tools/generate_result_tables.ipynb#cell-2","metrics":{}}})
    return metrics
