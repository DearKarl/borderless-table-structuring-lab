"""Complete official CPU scoring of seven frozen paired outputs, failures included."""
import argparse
import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path

from .io_utils import atomic_bytes,atomic_json,digest,process_identity,read_json,utc
from .supervisor import docker

SCORER_PYTHON='/opt/miniconda310/envs/omnidocbench_v16_smoke_20260408_py310/bin/python'
VERSIONS=('D0','GBDT_V6.1.1','GBDT_V6.1.2','GBDT_V6.1.3','MLP_V6.2.1','MLP_V6.2.2','MLP_V6.2.3')
METRIC_PATHS={
    'text_edit':('text_block','all','Edit_dist','ALL_page_avg'),
    'formula_cdm':('display_formula','page','CDM','ALL'),
    'formula_edit':('display_formula','all','Edit_dist','ALL_page_avg'),
    'table_teds':('table','page','TEDS','ALL'),
    'table_teds_structure':('table','page','TEDS_structure_only','ALL'),
    'table_edit':('table','all','Edit_dist','ALL_page_avg'),
    'reading_order_edit':('reading_order','all','Edit_dist','ALL_page_avg')}


def metrics(raw,pages=1651):
    if raw.get('match_debug',{}).get('page_count')!=pages:raise ValueError('Official scoring denominator differs')
    values={};missing={}
    for name,path in METRIC_PATHS.items():
        value=raw
        try:
            for key in path:value=value[key]
        except (KeyError,TypeError):value=None
        if value is None:
            missing[name]='Pinned evaluator did not emit '+'.'.join(path)
        elif not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            missing[name]='Pinned evaluator emitted a nonfinite or out-of-range value';value=None
        values[name]=value
    if all(values[x] is not None for x in ('text_edit','formula_cdm','table_teds')):
        values['overall']=100*((1-values['text_edit'])+values['formula_cdm']+values['table_teds'])/3
    else:
        values['overall']=None;missing['overall']='At least one required official component is unavailable'
    return {'values':values,'missing_reasons':missing,'page_denominator':pages,
            'units':{name:('0-100' if name=='overall' else '0-1') for name in values},
            'direction':{name:('lower' if name.endswith('_edit') else 'higher') for name in values},
            'category_breakdowns':{kind:{key:raw[kind][key] for key in ('all','group','page') if key in raw[kind]}
                                   for kind in ('text_block','display_formula','table','reading_order') if kind in raw},
            'denominator_note':'All 1651 prediction pages retained; component metrics use the pinned evaluator\'s applicable matched samples/pages.'}


def guard(deadline):
    command=[SCORER_PYTHON,'/source/pdf_validation.py','--config','/config/end2end-full.yaml']
    started=time.time();deadline=min(deadline,started+14400)
    child=subprocess.Popen(command,start_new_session=True)
    atomic_json('/work/SCORER_PROCESS.json',{'identity':process_identity(child.pid),'command':command,
                                          'absolute_deadline_unix':deadline,'started_at':utc(),'device':'cpu'})
    error=None
    try:code=child.wait(timeout=max(.001,deadline-time.time()))
    except subprocess.TimeoutExpired:
        error='four_hour_or_task_deadline';os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=5)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
    atomic_json('/work/SCORER_TERMINAL.json',{'exit_code':code,'error':error,'finished_at':utc(),
                                           'elapsed_seconds':time.time()-started,'absolute_deadline_unix':deadline})
    return 0 if code==0 and error is None else 1


def prepare_predictions(manifest,version,target):
    target.mkdir(parents=True,exist_ok=False);rows=[]
    for page in manifest['pages']:
        identifier=page['page_id'];source=Path(page['source_run_root'])/'shared/output/arms'/version
        receipt=read_json(source/'receipts'/(identifier+'.json'));file=source/'markdown'/(identifier+'.md')
        if receipt['run_id']!=page['source_run_id'] or receipt['version']!=version or digest(file)!=receipt['prediction_sha256']:
            raise ValueError('Shared prediction lineage differs')
        name=page['original_page_id']+'.md'
        if Path(name).name!=name or (target/name).exists():raise ValueError('Duplicate or unsafe prediction filename')
        atomic_bytes(target/name,file.read_bytes())
        rows.append({'page_id':identifier,'prediction_file':name,'sha256':digest(file),'status':receipt['status'],
                     'empty_prediction':receipt['empty_prediction'],'retained_stage':receipt['retained_stage'],
                     'source_run_id':page['source_run_id']})
    if len(rows)!=1651 or len(list(target.glob('*.md')))!=1651:raise ValueError('Full denominator must be 1651')
    return rows


def launch(manifest,binding,version):
    root=Path(manifest['round_root'])/version/'evaluation';root.mkdir(parents=True,exist_ok=False)
    rows=prepare_predictions(manifest,version,root/'predictions');atomic_json(root/'PREDICTION_BINDING.json',rows)
    work=root/'work';work.mkdir();deadline=min(manifest['absolute_deadline_unix']-10,time.time()+14400)
    if deadline-time.time()<60:raise RuntimeError('Insufficient time within the task ceiling')
    name=manifest['run_id']+'-'+version.lower().replace('.','-')+'-score'
    args=['create','--name',name,'--label','native-score-owner='+manifest['run_id'],'--network','none','--runtime','runc',
          '--restart','no','--cpus','8','--memory','32g','--memory-swap','32g','--pids-limit','512',
          '--env','CUDA_VISIBLE_DEVICES=','--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONDONTWRITEBYTECODE=1',
          '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
          '--env','PYTHONPATH=/source:/code','--workdir','/work']
    mounts=[(str(work),'/work',False),(str(root/'predictions'),'/pred',True),
            (binding['source']['path'],'/source',True),(binding['config']['path'],'/config/end2end-full.yaml',True),
            (binding['gt']['path'],'/gt/OmniDocBench.json',True),(manifest['code_root'],'/code',True)]
    for src,dst,ro in mounts:args+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    args+=['--entrypoint',SCORER_PYTHON,binding['image'],'-B','-m','versions.v5_native.evaluate','--guard','--deadline',str(deadline)]
    cid=docker(*args);info=json.loads(docker('inspect',cid))[0]
    if info['HostConfig'].get('Devices') or info['HostConfig'].get('DeviceRequests'):raise RuntimeError('Scoring must be CPU only')
    actual={(x['Source'],x['Destination'],not x['RW']) for x in info['Mounts'] if x['Type']=='bind'}
    if actual!=set(mounts):raise RuntimeError('Scorer mount identity mismatch')
    row={'version':version,'cid':cid,'name':name,'created_at':utc(),'deadline':deadline,'command':['docker',*args],
         'phase':'scoring','evaluation':binding,'prediction_binding_sha256':digest(root/'PREDICTION_BINDING.json'),
         'input_failures':sum(r['status']!='success' for r in rows),'empty_predictions':sum(r['empty_prediction'] for r in rows)}
    atomic_json(root/'LAUNCH.json',row);docker('start',cid);state=json.loads(docker('inspect',cid))[0]['State']
    row.update(started_at=state['StartedAt'],pid=state['Pid'],identity=process_identity(state['Pid']))
    atomic_json(root/'STATUS.json',row)
    with open(root/'scorer.log','w') as log:subprocess.Popen(['docker','logs','--follow',cid],stdout=log,stderr=subprocess.STDOUT)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest');p.add_argument('--binding');p.add_argument('--guard',action='store_true')
    p.add_argument('--deadline',type=float);p.add_argument('--concurrency',type=int,default=3);a=p.parse_args()
    if a.guard:return guard(a.deadline)
    if not 1<=a.concurrency<=3:raise ValueError('CPU scoring concurrency must be 1-3')
    m=read_json(a.manifest);binding=read_json(a.binding);root=Path(m['round_root'])
    if m['phase']!='full_paired_composite' or len(m['pages'])!=1651:raise ValueError('Full paired output manifest required')
    for key in ('gt','config'):
        if digest(binding[key]['path'])!=binding[key]['sha256']:raise ValueError('Frozen evaluator input changed')
    for name,h in binding['source']['files'].items():
        if digest(Path(binding['source']['path'])/name)!=h:raise ValueError('Frozen evaluator source changed')
    if json.loads(docker('inspect',binding['image']))[0]['Id']!=binding['image']:raise ValueError('Evaluator image changed')
    read_json(root/'INFERENCE_COMPLETE.json');pending={};results={};queue=list(VERSIONS)
    def emit(event,**details):
        with open(root/'PHASE_EVENTS.jsonl','a',encoding='utf-8') as stream:stream.write(json.dumps({'at':utc(),'event':event,**details})+'\n')
    while queue or pending:
        while queue and len(pending)<a.concurrency:
            version=queue.pop(0);pending[version]=launch(m,binding,version)
            emit('official_scoring_started',version=version,device='cpu',started_at=pending[version]['started_at'])
        for version,row in list(pending.items()):
            state=json.loads(docker('inspect',row['cid']))[0]
            if state['Config']['Labels'].get('native-score-owner')!=m['run_id']:raise RuntimeError('Scorer ownership differs')
            if state['State']['Running']:continue
            location=root/version/'evaluation';terminal_path=location/'work/SCORER_TERMINAL.json'
            terminal=read_json(terminal_path) if terminal_path.exists() else {'error':'missing_terminal_receipt'}
            row.update(exit_code=state['State']['ExitCode'],finished_at=state['State']['FinishedAt'],terminal=terminal)
            try:
                if row['exit_code']!=0 or terminal.get('exit_code')!=0 or terminal.get('error'):
                    raise RuntimeError('Official CPU scoring did not complete: '+str(terminal.get('error')))
                raw_path=location/'work/result/pred_quick_match_metric_result.json'
                result=metrics(read_json(raw_path));result.update(raw_result_sha256=digest(raw_path),raw_result_path=str(raw_path))
                row['phase']='evaluation_complete'
            except Exception as exc:
                reason=type(exc).__name__+': '+str(exc);row['phase']='evaluation_failed'
                result={'values':{k:None for k in (*METRIC_PATHS,'overall')},'missing_reasons':{k:reason for k in (*METRIC_PATHS,'overall')},'page_denominator':1651,'error':reason}
            result.update(input_failures=row['input_failures'],empty_predictions=row['empty_predictions'])
            results[version]=result;atomic_json(location/'METRICS.json',result);atomic_json(location/'STATUS.json',row)
            emit('official_scoring_completed',version=version,device='cpu',status=row['phase']);del pending[version]
        atomic_json(root/'EVALUATION_STATUS.json',{'updated_at':utc(),'running_versions':list(pending),'queued_versions':queue,'completed':results})
        if queue or pending:time.sleep(10)
    complete={'completed_at':utc(),'versions':results,'success':all('error' not in r for r in results.values())}
    atomic_json(root/'EVALUATION_COMPLETE.json',complete);return 0 if complete['success'] else 1


if __name__=='__main__':raise SystemExit(main())
