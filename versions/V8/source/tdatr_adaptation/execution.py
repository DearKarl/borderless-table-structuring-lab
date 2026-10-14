"""Frozen-input access, runtime identity, and entropy provenance for one run."""
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import signal
import sqlite3

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
def checked_json(path,expected):
    if sha(path)!=expected:raise ValueError('Frozen JSON changed: '+str(path))
    return json.loads(Path(path).read_text())
def verify_runtime(binding):
    observed={name:version(name) for name in binding['runtime_packages']}
    if observed!=binding['runtime_packages']:raise ValueError('Pinned runtime packages differ')
    import torch
    from versions.generator_pilot.supervision import gpu_uuid_key
    if torch.cuda.device_count()!=1 or gpu_uuid_key(torch.cuda.get_device_properties(0).uuid)!=gpu_uuid_key(binding['selected_gpu']['uuid']):
        raise ValueError('Fresh GPU binding differs')
    if binding['selected_gpu']['physical_index'] not in [1,2,3,5,6]:raise ValueError('Forbidden physical GPU')
    torch.set_num_threads(4)
    def timeout(signum,frame):raise TimeoutError('One example exceeded the frozen 1200-second limit')
    signal.signal(signal.SIGALRM,timeout)
    return observed
def load_base(binding):
    import torch
    from transformers import AutoModelForImageTextToText
    model=AutoModelForImageTextToText.from_pretrained(binding['model_root'],local_files_only=True,
        trust_remote_code=True,dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    model.config.use_cache=False;model.config.max_position_embeddings=16384
    return model
def rng_snapshot(path):
    import torch
    from safetensors.torch import save_file
    tensors={'torch_cpu':torch.get_rng_state()}
    if torch.cuda.is_available():tensors.update({'cuda_'+str(i):x for i,x in enumerate(torch.cuda.get_rng_state_all())})
    save_file(tensors,str(path))
    return dict(torch_initial_seed=torch.initial_seed(),state_sha256=sha(path),
        initialization='Framework OS entropy; no user-selected fixed seed or seed sweep')

class Inputs:
    def __init__(self,binding):
        self.common=checked_json(binding['common_inputs'],binding['common_inputs_sha256'])
        if sha(binding['detail_database'])!=binding['detail_database_sha256']:raise ValueError('Detail database changed')
        self.db=sqlite3.connect('file:'+str(Path(binding['detail_database']).resolve())+'?mode=ro',uri=True)
        self.db.row_factory=sqlite3.Row
        self.records={r['id']:r for r in self.common['train']+self.common['dev']}
        if len(self.common['train'])!=20000 or len(self.common['dev'])!=256:raise ValueError('Original input denominator changed')
        self.verified=set()
    def example(self,identifier):
        row=self.db.execute('SELECT * FROM examples WHERE id=?',(identifier,)).fetchone()
        if row is None:raise ValueError('Missing audited example')
        return dict(row)
    def native(self,identifier):
        import torch
        from safetensors.torch import load_file
        record=self.records[identifier]
        if identifier not in self.verified:
            if sha(record['cache_path'])!=record['cache_sha256']:raise ValueError('Native cache changed')
            self.verified.add(identifier)
        return {k:v.to('cuda:0',dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k,v in load_file(record['cache_path']).items()}
    def batch(self,identifier,task):
        from .inputs import task_batch
        native=self.native(identifier)
        if task=='native_otsl':return native
        row=self.db.execute('SELECT * FROM tasks WHERE id=? AND task=?',(identifier,task)).fetchone()
        if row is None:raise ValueError('Assigned detail task unavailable')
        return {k:v.to('cuda:0') for k,v in task_batch(native,row).items()}
    def close(self):self.db.close()
