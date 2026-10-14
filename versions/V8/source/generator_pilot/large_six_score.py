"""Assemble all1651 pages and run the pinned official evaluator on CPU only."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime,timezone

SCORER_PYTHON=os.environ.get('BTSL_SCORER_PYTHON', '/opt/miniconda310/envs/omnidocbench_v16_smoke_20260408_py310/bin/python')

METRIC_PATHS={
    'text_edit':('text_block','all','Edit_dist','ALL_page_avg'),
    'formula_cdm':('display_formula','page','CDM','ALL'),
    'formula_edit':('display_formula','all','Edit_dist','ALL_page_avg'),
    'table_teds':('table','page','TEDS','ALL'),
    'table_teds_structure':('table','page','TEDS_structure_only','ALL'),
    'table_edit':('table','all','Edit_dist','ALL_page_avg'),
    'reading_order_edit':('reading_order','all','Edit_dist','ALL_page_avg')}


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def docker(*args):return subprocess.check_output(['docker',*args],text=True,timeout=60).strip()


def metrics(raw):
    if raw.get('match_debug',{}).get('page_count')!=1651:raise ValueError('Official page denominator differs')
    values={};missing={}
    for name,keys in METRIC_PATHS.items():
        value=raw
        try:
            for key in keys:value=value[key]
        except (KeyError,TypeError):value=None
        if value is None:missing[name]='Pinned evaluator omitted '+'.'.join(keys)
        elif not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            missing[name]='Nonfinite or out-of-range evaluator value';value=None
        values[name]=value
    if all(values[x] is not None for x in ['text_edit','formula_cdm','table_teds']):
        values['overall']=100*((1-values['text_edit'])+values['formula_cdm']+values['table_teds'])/3
    else:values['overall']=None;missing['overall']='Required official component missing'
    return dict(values=values,missing_reasons=missing,page_denominator=1651,
        units={key:'0-100' if key=='overall' else '0-1' for key in values},
        direction={key:'lower' if key.endswith('_edit') else 'higher' for key in values},
        category_breakdowns={kind:{key:raw[kind][key] for key in ['all','group','page'] if key in raw[kind]}
            for kind in ['text_block','display_formula','table','reading_order'] if kind in raw},
        denominator_note='All1651 pages retained; component denominators follow the pinned official matching/evaluation protocol')


def guard():
    start=time.monotonic();command=[SCORER_PYTHON,'/source/pdf_validation.py','--config','/config/end2end-full.yaml']
    child=subprocess.Popen(command,start_new_session=True);write('/work/SCORER_PROCESS.json',dict(pid=child.pid,command=command,started_at_utc=now(),timeout_seconds=14400,device='cpu'))
    error=None
    try:code=child.wait(timeout=14400)
    except subprocess.TimeoutExpired:
        error='Four-hour CPU evaluator timeout';os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=10)
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
    write('/work/SCORER_TERMINAL.json',dict(exit_code=code,error=error,elapsed_seconds=time.monotonic()-start,finished_at_utc=now()))
    return 0 if code==0 and error is None else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--guard',action='store_true');args=parser.parse_args()
    if args.guard:return guard()
    root=Path(__file__).resolve().parent;binding=read(root/'STAGE_BINDING.json')
    config_path=root/'SCORING_CONFIG.json'
    if digest(config_path)!=binding['scoring_config_sha256']:raise ValueError('Scoring configuration changed')
    config=read(config_path);policies=config['policies'];evaluator=config['evaluator'];full=config['full_input_freeze'];expected={r['page_id']:r for r in full['pages']}
    if policies not in [['base','legacy_simple_rule'],['candidate']]:raise ValueError('Invalid amended policy set')
    if len(expected)!=1651 or len(full['pages'])!=1651:raise ValueError('Complete frozen page set required')
    for key in ['gt','config']:
        if digest(evaluator[key]['path'])!=evaluator[key]['sha256']:raise ValueError('Pinned evaluator asset changed')
    for name,h in evaluator['source']['files'].items():
        if digest(Path(evaluator['source']['path'])/name)!=h:raise ValueError('Pinned evaluator source changed')
    if json.loads(docker('image','inspect',evaluator['image']))[0]['Id']!=evaluator['image']:raise ValueError('Pinned evaluator image changed')
    pages={};shard_hashes={}
    for name in config['shard_roots']:
        stage=Path(name);receipt=read(stage/'BENCHMARK.json')
        if not (stage/'EXECUTION.json').exists() or receipt['status'] not in ['complete','failed']:raise ValueError('Benchmark shard still running')
        if receipt['selection_sha256']!=config['selection_sha256']:raise ValueError('Six-model selection changed')
        shard_hashes[name]=digest(stage/'BENCHMARK.json')
        for row in receipt['rows']:
            rid=row['page_id']
            if rid not in expected or rid in pages or row['original_page_id']!=expected[rid]['original_page_id']:raise ValueError('Duplicate/unexpected benchmark page')
            if set(row['outputs'])!=set(policies):raise ValueError('Missing paired policy output')
            for policy,item in row['outputs'].items():
                path=Path(item['file'])
                if path!=stage/'pages'/rid/(policy+'.md') or digest(path)!=item['sha256']:raise ValueError('Prediction provenance differs')
            pages[rid]=row
    if set(pages)!=set(expected):raise ValueError('Not all1651 paired pages are retained')
    if args.preflight:
        write(root/'CPU_PREFLIGHT.json',dict(status='passed',pages=1651,policies=policies,prediction_and_evaluator_hashes_verified=True,gpu_allocations=0));return 0
    if not read(root/'CPU_PREFLIGHT.json')['prediction_and_evaluator_hashes_verified']:raise ValueError('CPU gate missing')
    with (root/'SCORING_STARTED.json').open('x') as stream:json.dump(dict(at_utc=now(),selection_sha256=config['selection_sha256']),stream)
    launches={};results={}
    for policy in policies:
        folder=root/policy;folder.mkdir(exist_ok=False);predictions=folder/'predictions';predictions.mkdir();work=folder/'work';work.mkdir();records=[]
        for rid in expected:
            row=pages[rid];item=row['outputs'][policy];target=predictions/(expected[rid]['original_page_id']+'.md')
            target.write_bytes(Path(item['file']).read_bytes());records.append(dict(page_id=rid,sha256=digest(target),status=item['status'],empty=item['empty']))
        if len(list(predictions.glob('*.md')))!=1651:raise ValueError('Prediction packaging lost pages')
        write(folder/'PREDICTION_BINDING.json',records)
        name=binding['phase']+'-'+policy.replace('_','-')
        command=['create','--name',name,'--label','large-lora-score-owner='+binding['phase'],'--network','none','--runtime','runc','--restart','no',
            '--cpus','8','--memory','32g','--memory-swap','32g','--pids-limit','512','--env','CUDA_VISIBLE_DEVICES=',
            '--env','NVIDIA_VISIBLE_DEVICES=void','--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1',
            '--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1','--env','PYTHONPATH=/source:/code','--workdir','/work']
        mounts=[(str(work),'/work',False),(str(predictions),'/pred',True),(evaluator['source']['path'],'/source',True),
            (evaluator['config']['path'],'/config/end2end-full.yaml',True),(evaluator['gt']['path'],'/gt/OmniDocBench.json',True),(str(root),'/code',True)]
        for src,dst,ro in mounts:command+=['--mount',f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
        command+=['--entrypoint',SCORER_PYTHON,evaluator['image'],'-B','/code/large_six_score.py','--guard']
        cid=docker(*command);inspect=json.loads(docker('inspect',cid))[0]
        if inspect['HostConfig'].get('DeviceRequests') or inspect['HostConfig'].get('Devices'):raise ValueError('CPU scorer unexpectedly requests GPUs')
        observed={(r['Source'],r['Destination'],not r['RW']) for r in inspect['Mounts'] if r['Type']=='bind'}
        if observed!=set(mounts):raise ValueError('Evaluator mount identity differs')
        launch=dict(cid=cid,policy=policy,command=['docker',*command],at_utc=now(),input_failures=sum(r['status']!='complete' for r in records),empty_predictions=sum(r['empty'] for r in records))
        write(folder/'LAUNCH.json',launch);docker('start',cid);launches[policy]=launch
    while launches:
        for policy,launch in list(launches.items()):
            state=json.loads(docker('inspect',launch['cid']))[0]
            if state['Config']['Labels'].get('large-lora-score-owner')!=binding['phase']:raise ValueError('Evaluator ownership changed')
            if state['State']['Running']:continue
            folder=root/policy;terminal=read(folder/'work/SCORER_TERMINAL.json') if (folder/'work/SCORER_TERMINAL.json').exists() else {}
            raw=folder/'work/result/pred_quick_match_metric_result.json'
            if state['State']['ExitCode']==0 and terminal.get('exit_code')==0 and not terminal.get('error') and raw.exists():
                value=metrics(read(raw));value.update(status='complete',raw_result_sha256=digest(raw),cpu_seconds=terminal['elapsed_seconds'])
            else:value=dict(status='failed',terminal=terminal,container_exit_code=state['State']['ExitCode'],page_denominator=1651)
            value.update(input_failures=launch['input_failures'],empty_predictions=launch['empty_predictions'])
            log=subprocess.run(['docker','logs',launch['cid']],capture_output=True,text=True,timeout=60)
            (folder/'scorer.log').write_text(log.stdout+'\n'+log.stderr)
            write(folder/'METRICS.json',value);results[policy]=value;del launches[policy]
        write(root/'SCORING_STATUS.json',dict(at_utc=now(),running=list(launches),results=results))
        if launches:time.sleep(15)
    report=dict(status='complete' if all(r['status']=='complete' for r in results.values()) else 'failed',at_utc=now(),policies=results,
        denominator=1651,selection_sha256=config['selection_sha256'],shard_result_hashes=shard_hashes,gpu_allocations=0,leaderboard_submission=False)
    write(root/'BENCHMARK_SCORES.json',report);print(json.dumps(dict(status=report['status'],policies=list(results),pages=1651,gpu_allocations=0)))
    return 0 if report['status']=='complete' else 1


if __name__=='__main__':raise SystemExit(main())
