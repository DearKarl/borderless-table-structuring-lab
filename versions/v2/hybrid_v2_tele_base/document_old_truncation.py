"""Read old frozen evidence and write a separate explanatory receipt; never change old results."""
from pathlib import Path
import hashlib
import json
import zipfile
from .full_state import sha,read,exclusive_json,exclusive_bytes

W=Path(__file__).resolve().parents[1]
R=Path('/srv/hybrid-research')
OUT=R/'receipts/HYBRID_V2_TELE_BASE/attempt-001'
OLD=R/'receipts/HYBRID_V1/DELIVERY_20260923_v1'


def main():
    archive=OLD/'FROZEN_CODE.zip';proof={}
    excerpts={'native-code/core.py':(23,25),'native-code/native_controller.py':(79,107),
        'native-code/native_stop_audit.py':(4,32),'app/hybrid/hybrid_v1/assembly.py':(59,62)}
    with zipfile.ZipFile(archive) as z:
        for name,(start,end) in excerpts.items():
            raw=z.read(name);lines=raw.decode().splitlines()
            proof[name]={'sha256':hashlib.sha256(raw).hexdigest(),'start_line':start,'end_line':end,
                'excerpt':'\n'.join(f'{i}: {lines[i-1]}' for i in range(start,end+1))}
    controller=W/'scripts/remote_hybrid_official_eval_v1_run_v2.py'
    report=OLD/'REPORT.md';operations=OLD/'OPERATIONS.md'
    remote=OUT/'OLD_TRUNCATION_REMOTE.json'
    result={'scope':'Read-only old evidence; separate v2 addendum; no rescoring or modification of old outputs',
        'old_archive':{'path':str(archive),'sha256':sha(archive)},'frozen_source_excerpts':proof,
        'old_score_controller':{'path':str(controller),'sha256':sha(controller),'line':40,
            'behavior':'Checks every non-success primary is empty; this evaluator controller does not perform emptying'},
        'generation_classification':'Any audited generation length stop makes the page truncated; independent of official quick-match timeout',
        'disclosure':{'report_path':str(report),'report_sha256':sha(report),'operations_path':str(operations),'operations_sha256':sha(operations),
            'counts_disclosed':True,'failed_empty_primary_disclosed':True,'truncated_empty_primary_explicitly_disclosed':False},
        'actual_primary_check':None,'new_policy':'Both Tele raw and v2 preserve native truncated Markdown. Do not attribute cross-policy old/new score differences solely to model or Hybrid routing.'}
    if remote.exists():
        audit=read(remote)
        if not audit.get('read_only') or set(audit['arms'])!={'M','P','H'}:raise ValueError('Unexpected remote receipt')
        result['actual_primary_check']={'path':str(remote),'sha256':sha(remote),'arms':{
            arm:{k:row[k] for k in ('lock_sha256','counts','all_1651_actual_primary_hashes_verified',
                    'truncated_empty_primary','truncated_nonempty_primary')} for arm,row in audit['arms'].items()}}
    target=OUT/('OLD_TRUNCATION_VERIFIED.json' if remote.exists() else 'OLD_TRUNCATION_SOURCE_AUDIT.json')
    exclusive_json(target,result)
    print(json.dumps({'path':str(target),'sha256':sha(target),'actual_primary_checked':remote.exists()},indent=2))


if __name__=='__main__':main()
