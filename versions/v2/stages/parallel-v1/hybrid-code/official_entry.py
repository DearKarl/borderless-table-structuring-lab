"""Evaluator container provenance plus the unmodified official command."""
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
import src.core.pipeline

assert Path(inspect.getfile(src.core.pipeline)).resolve()==Path('/source/src/core/pipeline.py')
freeze=subprocess.check_output([sys.executable,'-m','pip','freeze','--all'],text=True)
Path('/work/runtime-freeze.txt').write_text(freeze)
record={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'cwd':str(Path.cwd()),'pipeline_source':inspect.getfile(src.core.pipeline),'sys_path':sys.path,'freeze_sha256':hashlib.sha256(freeze.encode()).hexdigest(),'command':[sys.executable,'/source/pdf_validation.py','--config','/config/end2end-full.yaml']}
Path('/work/RUNTIME.json').write_text(json.dumps(record,indent=2))
result=subprocess.run(record['command'])
Path('/work/OFFICIAL_PROCESS_EXIT.json').write_text(json.dumps({'exit_code':result.returncode}))
if result.returncode==0:
    import math
    import pandas as pd
    result_path=Path('/work/result/pred_quick_match_metric_result.json')
    metrics=json.loads(result_path.read_text())
    assert metrics['match_debug']['page_count']==1651
    values={}
    for category,metric in [('text_block','Edit_dist'),('display_formula','CDM'),('table','TEDS'),('table','TEDS_structure_only'),('reading_order','Edit_dist')]:
        page_metric=metric!='Edit_dist'
        value=metrics[category]['page' if page_metric else 'all'][metric]['ALL' if page_metric else 'ALL_page_avg']
        assert isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1
        values[category+'_'+metric]=value*100 if page_metric else value
    raw=((1-values['text_block_Edit_dist'])*100+values['display_formula_CDM']+values['table_TEDS'])/3
    display=pd.DataFrame([values]).round(3)
    notebook=((1-display['text_block_Edit_dist'])*100+display['display_formula_CDM']+display['table_TEDS'])/3
    coverage={}
    for category in ['text_block','display_formula','table','reading_order']:
        samples=json.loads(Path('/work/result/pred_quick_match_'+category+'_result.json').read_text())
        assert isinstance(samples,list) and samples
        coverage[category]={'matched_samples':len(samples)}
    summary={'raw_metrics':values,'raw_overall':raw,'notebook_rounded_components':display.iloc[0].to_dict(),'notebook_overall_after_component_rounding':float(notebook.iloc[0]),'notebook_source':'tools/generate_result_tables.ipynb#cell-2','matching_coverage':coverage,'official_match_debug':metrics['match_debug'],'warnings_preserved_in_original_logs':True,'external_publication':False}
    Path('/work/OFFICIAL_SCORE.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
sys.exit(result.returncode)
