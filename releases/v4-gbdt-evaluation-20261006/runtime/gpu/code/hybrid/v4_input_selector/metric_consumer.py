"""Evaluator-only binding to the accepted metric sources and stable identity.

Construction verifies small local files only. open_metrics is an explicit future
runtime operation: it invokes the existing observer once and never trusts a saved
"current" runtime as a substitute. No inference import uses this module.
"""
from copy import deepcopy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

from .supervisor import work_alarm
from .worker_contract import file_sha, digest

METRIC_SHA='394dd0c1b146a3996fa6ed4c454bd23073f6e2786813213dd75025bbfb78440c'
STABLE_SHA='0918cb9f243dc24f9c5514eb93f91aacfb329e047848a2a367ac216722b68f20'
IDENTITY_SHA='9b3428d7286af32c262d919b99fa0bc833f025de7d392a08497533d231edb514'
SUPPORT_SHA='815f077b703e192c6dd1a119e6f1d79373fcb9ed7e533b24960ce7721e45a4be'
ACCEPTED_SHA='dcb023bf01dcdba83286e9281246aaaecb4ee8ba7ad7929e1b49c2c9f34c2d0e'
QUALITY_SHA='3dd7ae7d0eaf4245d357f042bc2cc81db10a5ce517572b36762fa9e1c3afc58d'
SCHEME='v4-runtime-identity-exclude-four-version-probe-elapsed-v1'

def require(value,message):
    if not value: raise ValueError(message)

def canonical_sha(value):
    return digest(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode())

def read_bound(path,expected,cap=2*1024*1024):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'Metric file unavailable/excessive/link')
    raw=path.read_bytes()
    require(digest(raw)==expected,'Metric file hash differs: '+str(path))
    return raw

def load_locked(path,expected,name):
    source=read_bound(path,expected)
    require(name not in sys.modules,'Metric consumer namespace already loaded')
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    try:
        # Execute exactly the bytes already checked; no path-search import.
        exec(compile(source,str(path),'exec'),module.__dict__)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module

def stable_evidence(identity_module,accepted,current):
    require(identity_module.SCHEME==SCHEME,'Unapproved stable identity scheme')
    evidence=identity_module.require_compatible(accepted,current,expected_stable_sha256=STABLE_SHA)
    return dict(identity=dict(metric_source_sha256=METRIC_SHA,metric_runtime_config_sha256=STABLE_SHA),
                comparison=evidence)

class StableMetricConsumer:
    def __init__(self,binding_path,binding_sha256):
        binding=json.loads(read_bound(binding_path,binding_sha256))
        require(set(binding)=={'schema','bundle_root','files'} and binding['schema']=='v4_metric_consumer_binding_v1',
                'Unknown metric consumer binding')
        self.root=Path(binding['bundle_root']).resolve()
        require(self.root.is_absolute() and str(self.root)==binding['bundle_root'],'Canonical metric bundle required')
        self.files=binding['files']
        sources={n:h for n,h in self.files.items() if n.startswith('source/')}
        require(len(sources)==21 and sources.get('source/local_quality.py')==QUALITY_SHA,'Exact 21-source metric set required')
        require(canonical_sha({k:v for k,v in sources.items() if k!='source/local_quality.py'})==METRIC_SHA,
                'Frozen metric source identity differs')
        support={'frozen_numeric_support.py':SUPPORT_SHA,'runtime_identity.py':IDENTITY_SHA,'ACCEPTED_RUNTIME.json':ACCEPTED_SHA}
        require(set(self.files)==set(sources)|set(support)|{'OBSERVER_CONFIG.json'} and
                all(self.files.get(k)==h for k,h in support.items()),'Metric support identity differs')
        for name,h in self.files.items():
            path=self.root/name
            require(not Path(name).is_absolute() and '..' not in Path(name).parts and path.resolve().is_relative_to(self.root),
                    'Metric source escapes bundle')
            read_bound(path,h)
        allowed_dirs={str(parent) for name in self.files for parent in Path(name).parents}
        observed=set()
        for directory,folders,files in os.walk(self.root,followlinks=False):
            relative=Path(directory).relative_to(self.root)
            require(str(relative) in allowed_dirs and not any((Path(directory)/n).is_symlink() for n in folders+files),
                    'Unexpected metric bundle directory/link')
            observed.update((relative/name).as_posix() for name in files)
            require(len(observed)<=len(self.files),'Unexpected metric bundle files')
        require(observed==set(self.files),'Metric bundle is not the closed frozen inventory')
        self.accepted=json.loads(read_bound(self.root/'ACCEPTED_RUNTIME.json',ACCEPTED_SHA))
        self.observer=json.loads(read_bound(self.root/'OBSERVER_CONFIG.json',self.files['OBSERVER_CONFIG.json']))
        self.lock=dict(files=self.files,metric_source_sha256=METRIC_SHA)
        self.opened=False
        self.evidence=None
        self.deadline=None

    def open_metrics(self,output_root,deadline):
        """Call only inside a separately authorized bounded evaluator context."""
        require(not self.opened and type(deadline) in (int,float) and math.isfinite(deadline) and deadline>time.monotonic(),
                'One finite evaluator binding per consumer')
        out=Path(output_root).resolve()
        require(not out.is_relative_to(self.root) and not self.root.is_relative_to(out),'Evaluator output overlaps source bundle')
        self.opened=True
        self.deadline=deadline
        with work_alarm(deadline):
            out.mkdir(parents=True,exist_ok=False)
            identity=load_locked(self.root/'runtime_identity.py',IDENTITY_SHA,'_v4_consumer_identity')
            support=load_locked(self.root/'frozen_numeric_support.py',SUPPORT_SHA,'_v4_consumer_support')
            # Existing observer checks actual imported modules/tools/fonts/env.
            # No fixture or historical-current receipt is accepted in this path.
            support.preflight(out,self.lock,self.observer)
            current=json.loads((out/'runtime_lock.json').read_bytes())
            evidence=stable_evidence(identity,self.accepted,current)
            evidence.update(accepted_runtime_file_sha256=ACCEPTED_SHA,current_runtime_file_sha256=file_sha(out/'runtime_lock.json'),
                                 identity_module_sha256=IDENTITY_SHA)
            support.save(out/'METRIC_CONSUMER_IDENTITY.json',evidence)
            self.quality,table,kernel,_,_=support.load_metrics(self.lock)
            self.table=self.quality.checked_teds(table.TEDS(structure_only=False,n_jobs=1))
            self.formula=self.quality.StrictCDM(kernel,out/'strict_tmp',identity=evidence['identity'])
            self.evidence=evidence
        return deepcopy(self.evidence)

    def score_bound(self,host_row,reference,prediction,*,opaque_page_id,status,deadline):
        """Host-only identity projection; never return source/split to inference."""
        require(self.evidence is not None and deadline<=self.deadline and deadline>time.monotonic(),'Metric runtime unbound/expired')
        frozen_reference=json.loads(read_bound(host_row['reference']['path'],host_row['reference']['sha256'],16*1024*1024))
        require(reference==frozen_reference,'Reference bytes differ from frozen host binding')
        require(reference.get('page_id')==host_row['page_id'] and reference.get('source_family')==host_row['source_family']
                and reference.get('split')==host_row['split'] and reference.get('input_sha256')==host_row['input']['sha256'],
                'Host reference/roster identity differs')
        bound=deepcopy(prediction)
        # Failure statuses do not require a fabricated prediction. For successful
        # or partial results, verify opaque identity before restoring host page ID.
        if status in ('success','partial'):
            require(isinstance(bound,dict) and bound.get('page_id')==opaque_page_id
                    and bound.get('input_sha256')==host_row['input']['sha256'],'Unbound opaque prediction')
            bound['page_id']=host_row['page_id']
        with work_alarm(deadline):
            return self.quality.score_page(reference,bound,table_metric=self.table,formula_metric=self.formula,status=status)
