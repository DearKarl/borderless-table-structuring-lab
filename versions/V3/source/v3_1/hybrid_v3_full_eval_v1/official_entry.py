"""N-page wrapper; official source/config/matcher/metrics remain unchanged."""
import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

def summarize(metrics, result_dir, n):
    # File execution in the evaluator imports the adjacent, locked validator.
    try:
        from .official_schema import validate_results
    except ImportError:
        from official_schema import validate_results
    observed,coverage,undefined,report=validate_results(metrics,result_dir,n)
    keys=("text_block_Edit_dist","display_formula_CDM","table_TEDS",
          "table_TEDS_structure_only","reading_order_Edit_dist")
    values={k:(v*100 if k in keys[1:4] else v) for k,v in observed.items() if k in keys and v is not None}
    raw=None;rounded=None;notebook=None
    if all(k in values for k in keys[:3]):
        raw=((1-values[keys[0]])*100+values[keys[1]]+values[keys[2]])/3
        import pandas as pd
        display=pd.DataFrame([values]).round(3)
        rounded=display.iloc[0].to_dict()
        notebook=float((((1-display[keys[0]])*100+display[keys[1]]+display[keys[2]])/3).iloc[0])
    return {"raw_metrics":values,"undefined_metrics":undefined,"raw_overall":raw,
            "notebook_rounded_components":rounded,"notebook_overall_after_component_rounding":notebook,
            "overall_defined":raw is not None,"matching_coverage":coverage,
            "official_match_debug":metrics["match_debug"],
            "official_page_denominators":report["page_denominators"],
            "validated_result_schema":True,
            "notebook_source":"tools/generate_result_tables.ipynb#cell-2",
            "warnings_preserved_in_original_logs":True,"external_publication":False}

def main():
    p=argparse.ArgumentParser();p.add_argument("--pages",required=True,type=int)
    a=p.parse_args()
    import src.core.pipeline
    if Path(inspect.getfile(src.core.pipeline)).resolve()!=Path("/source/src/core/pipeline.py"):
        raise ValueError("Evaluator source import differs")
    freeze=subprocess.check_output([sys.executable,"-m","pip","freeze","--all"],text=True)
    Path("/work/runtime-freeze.txt").write_text(freeze)
    command=[sys.executable,"/source/pdf_validation.py","--config","/config/end2end-full.yaml"]
    record={"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "cwd":str(Path.cwd()),"pipeline_source":inspect.getfile(src.core.pipeline),
            "sys_path":sys.path,"freeze_sha256":hashlib.sha256(freeze.encode()).hexdigest(),
            "command":command}
    Path("/work/RUNTIME.json").write_text(json.dumps(record,indent=2))
    result=subprocess.run(command)
    Path("/work/OFFICIAL_PROCESS_EXIT.json").write_text(json.dumps({"exit_code":result.returncode}))
    if result.returncode==0:
        metrics=json.loads(Path("/work/result/pred_quick_match_metric_result.json").read_text())
        summary=summarize(metrics,"/work/result",a.pages)
        names=["pred_quick_match_metric_result.json","pred_quick_match_run_summary.json"]+[
            "pred_quick_match_"+c+"_result.json" for c in ("text_block","display_formula","table","reading_order")]
        validation={"files":{name:hashlib.sha256((Path("/work/result")/name).read_bytes()).hexdigest() for name in names}}
        target=Path("/work/RESULT_VALIDATION.json")
        target.write_text(json.dumps(validation,indent=2))
        summary["validation_sha256"]=hashlib.sha256(target.read_bytes()).hexdigest()
        Path("/work/OFFICIAL_SCORE.json").write_text(json.dumps(summary,indent=2,allow_nan=False))
    return result.returncode

if __name__=="__main__":
    raise SystemExit(main())
