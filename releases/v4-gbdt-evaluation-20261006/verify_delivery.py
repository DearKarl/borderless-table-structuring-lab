"""Read-only integrity check; no model deserialization, imports from runtime or network."""
import hashlib
import json
from pathlib import Path, PurePosixPath

root = Path(__file__).resolve().parent
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024*1024): h.update(block)
    return h.hexdigest()
def read(path): return json.loads(path.read_bytes())

manifest = read(root/'DELIVERY_MANIFEST.json')
expected = manifest['files']
excluded = {'DELIVERY_MANIFEST.json', 'V4-GBDT-evaluation-20261006-public.zip', 'SHA256SUMS.txt'}
paths = list(root.rglob('*'))
if any(p.is_symlink() for p in paths): raise ValueError('Linked entry found')
actual = {p.relative_to(root).as_posix() for p in paths if p.is_file()}-excluded
if actual != set(expected): raise ValueError('Delivery inventory differs')
for name, row in expected.items():
    rel = PurePosixPath(name)
    if rel.is_absolute() or '..' in rel.parts or '\\' in name: raise ValueError('Unsafe manifest path')
    path = root/name
    if path.stat().st_size != row['bytes'] or digest(path) != row['sha256']: raise ValueError('Hash differs: '+name)
locks = read(root/'provenance/SOURCE_LOCKS.json')
for field, base in [('gpu_code','runtime/gpu/code'),('cpu_code','runtime/cpu/code'),('legacy','runtime/legacy/hybrid_v3_full_eval_v1')]:
    for name, h in locks[field].items():
        if digest(root/base/name) != h: raise ValueError('Public source lock differs: '+name)
cpu = read(root/'runtime/cpu/CPU_MANIFEST.json')
for name,h in cpu['files'].items():
    if digest(root/'runtime/cpu'/name) != h: raise ValueError('Public CPU package differs')
metrics = read(root/'results/METRICS.json'); report=read(root/'results/REPORT.json')
if metrics != report['official_components']: raise ValueError('Metric projections differ')
if sum(report['local_protocol']['page_status_counts'].values()) != 1651: raise ValueError('Page denominator differs')
print(json.dumps({'status':'verified','files':len(expected),'gpu_source_files':len(locks['gpu_code']),
 'cpu_source_files':len(locks['cpu_code']),'overall':metrics['overall'],'execution':'none'},ensure_ascii=False))
