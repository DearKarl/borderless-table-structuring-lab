"""Preserve faulted attempted pages in a separate, non-replaying continuation."""
import copy
import hashlib
import json
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_prior(root, binding, sources):
    expected=binding.get('prior_attempt_manifest_sha256')
    if not expected:return {}, 0
    path=Path(root)/'PRIOR_ATTEMPTS.json'
    if sha(path)!=expected:raise ValueError('Prior-attempt manifest changed')
    value=json.loads(path.read_text());parent=Path(value['parent_root'])
    if parent.parent!=Path(root).parent or parent==Path(root):raise ValueError('Invalid prior-stage location')
    for key in ['selection_sha256','checkpoint_sha256','model_run_id']:
        if value[key]!=binding[key]:raise ValueError('Prior-stage model/selection differs')
    for name,key in [(value['parent_report_file'],'parent_report_sha256'),('CALLS.jsonl','parent_calls_sha256'),('EXECUTION.json','parent_execution_sha256')]:
        if Path(name).name!=name or sha(parent/name)!=value[key]:raise ValueError('Prior evidence changed')
    execution=json.loads((parent/'EXECUTION.json').read_text())
    if not execution['worker_exited'] or not execution['lease_released']:
        raise ValueError('Parent consumer still running')
    if {p.name for p in (parent/'pages').iterdir()}!=set(value['attempted_page_ids']):
        raise ValueError('Parent attempted page set changed')
    rows={r['page_id']:r for r in value['rows']};source_map={r['page_id']:r for r in sources}
    if len(rows)!=len(value['rows']) or set(rows)!=set(value['attempted_page_ids']) or not set(rows)<=set(source_map):
        raise ValueError('Prior-attempt membership differs')
    for page,row in rows.items():
        if row['status']=='complete' or any(row[k]!=source_map[page][k] for k in ['original_page_id','input_sha256']):
            raise ValueError('Prior failure/source identity differs')
        if row.get('outputs'):
            out=row['outputs']['candidate'];path=parent/'pages'/page/'candidate.md'
            if Path(out['file'])!=path or sha(path)!=out['sha256'] or path.read_bytes()!=b'':
                raise ValueError('Prior empty failure output changed')
    return rows,value['parent_model_calls']


def preserve_failure(prior, folder):
    """No inference/cache lookup; immutable prior evidence stays at its old root."""
    row=copy.deepcopy(prior);target=Path(folder)/'candidate.md'
    with target.open('xb') as stream:stream.write(b'')
    row['outputs']={'candidate':dict(file=str(target),sha256=sha(target),status=row['status'],empty=True)}
    row['inherited_attempted_failure']=True
    row['inference_replayed']=False
    (Path(folder)/'RECEIPT.json').write_text(json.dumps(row,indent=2))
    return row
