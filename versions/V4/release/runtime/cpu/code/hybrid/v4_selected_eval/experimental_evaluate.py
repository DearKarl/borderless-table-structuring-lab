"""One frozen experimental prediction group and one bounded official CPU scorer."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import uuid
from hybrid.v4_input_selector.worker_contract import atomic_json,bound_file,file_sha
from hybrid.v4_input_selector.supervisor import process_identity,work_alarm
from .contracts import checked_json,verify_code
from .dependencies import need,inventory
from .experimental import validate_auth,LIMITS
from .experimental_host import page_receipt
from .experimental_state import State
from .host_state import boot
from .host_manifest import read
from .absence import inspect_absent

IMAGE='sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617'
PYTHON='/opt/miniconda310/envs/omnidocbench_v16_smoke_20260408_py310/bin/python'
COMMIT='147cd5ac9472002f5751221d390bf00abdbc0d2f'
ENTRY_FILES={'pdf_validation.py':'7603a81d473f723d6b520773e0742ecd9259cdfffcd81d720c6a772acc8dfe68',
    'src/core/pipeline.py':'078a538bb39803932c58468680871f1927cc181d951e9631044505fe77e3e26c',
    'src/core/pipeline_eval.py':'623406c80d2d99a1cbdecf5539a3031fe4fa8919e51a6b0eae67c6bdd8a6a376',
    'src/runtime/eval_report.py':'dd2dd74965220783399ba840bc7dae8937c2794b40a6ff75e3fba7538f9069ad'}

def metrics(raw):
    need(raw['match_debug']['page_count']==1651,'Official evaluator denominator differs')
    paths=dict(text_edit=('text_block','all','Edit_dist','ALL_page_avg'),cdm=('display_formula','page','CDM','ALL'),
        teds=('table','page','TEDS','ALL'),structure_teds=('table','page','TEDS_structure_only','ALL'),reading_order=('reading_order','all','Edit_dist','ALL_page_avg'))
    values={}
    for name,path in paths.items():
        value=raw
        for key in path:value=value[key]
        need(type(value) in (int,float) and math.isfinite(value) and 0<=value<=1,'Missing/nonfinite/out-of-range official metric '+name)
        values[name]=value
    values['overall']=((1-values['text_edit'])*100+values['cdm']*100+values['teds']*100)/3
    return values

def immutable_json(path,value):
    if path.exists():need(json.loads(path.read_bytes())==value,'Frozen artifact changed: '+str(path))
    else:atomic_json(path,value)

def prepare_predictions(m,s,root):
    need(s['active'] is None and s['terminal'] is True and not s['stopped'],'Inference not terminal or unknown failure')
    need(len(m['pages'])==len(s['slots'])==1651 and set(s['slots'])=={p['page_id'] for p in m['pages']},'Full1651 ledger required')
    need(all(a.get('reconciled') and a['release']['container_absent'] and a['release']['lease_released'] and not a['release'].get('error') for a in s['allocations']),
        'Every allocation requires positive release and reconciliation')
    rows=[];expected={};pred=root/'predictions';pred.mkdir(parents=True,exist_ok=True)
    by_id={i['item_id']:i for i in m['items']}
    for page in m['pages']:
        key=page['page_id'];slot=s['slots'][key];status=slot['status']
        need(status in ('completed','failed','timeout','not_run_budget_exhausted'),'Unresolved page cannot be scored')
        dest=bound_file(pred,page['original_page_id']+'.md');raw=b''
        row=dict(page_id=key,original_page_id=page['original_page_id'],status=status,prediction_file=dest.name)
        if status=='not_run_budget_exhausted':need(slot['reservation'] is None,'Consumed page mislabeled unstarted')
        else:
            need(slot['reservation'] is not None,'Terminal page was never reserved')
            out=Path(m['root'])/'runs'/slot['reservation']['allocation']/'output/queue'
            if status=='completed':
                need(Path(slot['output']).resolve()==out.resolve(),'Output outside bound allocation')
                receipt,selected,audit=page_receipt(m,by_id[key],slot['result'],out)
                folder=out/key;raw=(folder/'native.md').read_bytes()
                for rel,h in audit['native_image_files'].items():
                    src=bound_file(folder,rel);need(file_sha(src)==h,'Native image changed')
                    name=(Path('images')/key/Path(rel).relative_to('native_images')).as_posix();target=bound_file(pred,name)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists():need(file_sha(target)==h,'Frozen image changed')
                    else:shutil.copyfile(src,target)
                    expected[name]=h
                row.update(end_to_end_seconds_including_export=receipt['end_to_end_seconds_including_export'],
                    measurement=audit.get('measurement'),native_stage_diagnostics=audit.get('native_stage_diagnostics'),
                    native_page_pipeline_calls=selected['native_page_pipeline_calls'],experimental_receipt_sha256=file_sha(out/'_experimental'/f'{key}.json'))
            elif (out/'_selected'/f'{key}.json').exists():
                selected=json.loads((out/'_selected'/f'{key}.json').read_bytes())
                need(selected['selection']['policy_identity']==m['effective_policy']['effective_policy_identity'],'Failure selection identity differs')
                row['native_page_pipeline_calls']=selected['native_page_pipeline_calls']
            decision_path=out/'_decisions'/f'{key}.json'
            if decision_path.exists():
                decision=json.loads(decision_path.read_bytes())
                need(decision['item_id']==key and decision['input_sha256']==by_id[key]['input_sha256'] and
                    decision['policy_identity']==m['effective_policy']['effective_policy_identity'],'Decision differs from frozen run')
                row.update(action=decision['selection']['action'],fallback_reason=decision['fallback_reason'],
                    estimator_predict_calls=decision['prediction']['estimator_predict_calls'],candidate_aliases=decision['preparation']['candidate_aliases'],
                    decision_sha256=file_sha(decision_path))
        if dest.exists():need(dest.read_bytes()==raw,'Frozen prediction changed')
        else:dest.write_bytes(raw)
        expected[dest.name]=file_sha(dest);row['prediction_sha256']=expected[dest.name];rows.append(row)
    need(len({r['prediction_file'] for r in rows})==1651 and inventory(pred)==expected,'One official Markdown per original ID required')
    frozen=dict(schema='v4_experimental_predictions_frozen_v1',manifest_sha256=s['manifest_sha256'],
        state_sha256=file_sha(Path(m['root'])/'STATE.json'),files=expected,items=rows,
        full_run=s['item_starts']==1651 and all(r['status']!='not_run_budget_exhausted' for r in rows))
    immutable_json(root/'PREDICTIONS_FROZEN.json',frozen)
    return frozen

def verify_protocol_after_freeze(m,root):
    frozen=json.loads((root/'PREDICTIONS_FROZEN.json').read_bytes())
    need(inventory(root/'predictions')==frozen['files'],'Predictions changed before scoring')
    cfg=m['evaluation'];need(cfg['image']==IMAGE and cfg['commit']==COMMIT,'Official image/commit changed')
    for name in ('gt','config'):read_ref=cfg[name];need(file_sha(read_ref['path'])==read_ref['sha256'],'Official '+name+' bytes changed')
    source=Path(cfg['source']['path'])
    actual={p.relative_to(source).as_posix():file_sha(p) for p in source.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    need(actual==cfg['source']['files'] and all(actual.get(n)==h for n,h in ENTRY_FILES.items()),'Official source closure changed')
    config=Path(cfg['config']['path']).read_text()
    need('/pred' in config and '/gt/OmniDocBench.json' in config and 'quick_match' in config,'Locked local protocol paths/matcher differ')
    gt=json.loads(Path(cfg['gt']['path']).read_bytes());ids=[Path(x['page_info']['image_path']).stem for x in gt]
    need(len(ids)==len(set(ids))==1651 and set(ids)=={p['original_page_id'] for p in m['pages']},'GT/input original-ID mapping differs')
    immutable_json(root/'PROTOCOL_VERIFIED.json',dict(predictions_frozen_sha256=file_sha(root/'PREDICTIONS_FROZEN.json'),evaluation=cfg))

def docker(args,deadline,*,cap=5,check=False):
    remaining=deadline-time.time();need(remaining>0,'Original CPU deadline exhausted; no reset')
    return subprocess.run(['docker',*args],capture_output=True,text=True,timeout=min(cap,remaining),check=check)

def inspect_owned(identity,deadline):
    q=docker(['inspect',identity['name']],deadline)
    if q.returncode:
        need(inspect_absent(q,identity['name']),'CPU inspect outcome unknown');return None
    info=json.loads(q.stdout)[0]
    need(info['Name']=='/'+identity['name'] and info['Image']==IMAGE and
        info['Config']['Labels'].get('v4-experimental-score-owner')==identity['owner'] and
        (identity['cid'] is None or info['Id']==identity['cid']),'Foreign CPU identity; refuse cleanup')
    return info

def mounts(m,root,code):
    cfg=m['evaluation']
    return [(str(root/'score'),'/work',True),(str(root/'predictions'),'/pred',False),
        (cfg['source']['path'],'/source',False),(cfg['config']['path'],'/config/end2end-full.yaml',False),
        (cfg['gt']['path'],'/gt/OmniDocBench.json',False),(str(code),'/cpu-code',False)]

def verify_container(info,expected,command):
    hc=info['HostConfig'];env=info['Config']['Env']
    need(hc['NetworkMode']=='none' and hc.get('Runtime')=='runc' and not hc.get('Privileged') and not hc.get('Devices') and not hc.get('DeviceRequests')
        and hc['NanoCpus']==8_000_000_000 and hc['Memory']==32*1024**3 and hc['MemorySwap']==32*1024**3 and hc['PidsLimit']==512,
        'CPU scorer isolation/resources differ')
    need(not hc.get('AutoRemove') and hc.get('RestartPolicy',{}).get('Name') in ('','no') and hc.get('PidMode','')!='host'
        and hc.get('IpcMode','')!='host' and not hc.get('Binds') and not hc.get('VolumesFrom'),'CPU lifecycle/mount scope differs')
    need('NVIDIA_VISIBLE_DEVICES=void' in env and 'CUDA_VISIBLE_DEVICES=' in env,'CPU-only environment differs')
    need(sorted((r['Source'],r['Destination'],r['RW']) for r in info['Mounts'])==sorted(expected),'CPU mount identity differs')
    need(info['Path']==PYTHON and info['Args']==command,'CPU entrypoint differs')

def release_owned(identity,deadline):
    info=inspect_owned(identity,deadline)
    need(identity['cid'] is not None,'Creation outcome unknown; retain identity for investigation')
    code=None
    if info is not None:
        if info['State']['Running']:
            docker(['kill',identity['cid']],deadline,check=True);info=inspect_owned(identity,deadline)
        need(info is not None and not info['State']['Running'],'CPU stop unconfirmed');code=info['State']['ExitCode']
        docker(['rm',identity['cid']],deadline,check=True)
    need(inspect_owned(identity,deadline) is None,'Owned CPU name remains')
    absent=docker(['inspect',identity['cid']],deadline)
    need(inspect_absent(absent,identity['cid']),'Exact owned CPU CID absence not confirmed')
    return dict(container_id=identity['cid'],container_absent=True,exit_code=code,finished_wall=time.time())

def official_score(m,root,code,deadline,*,allow_create,collection_deadline=None):
    work=root/'score';work.mkdir(exist_ok=True);identity_path=work/'OWNED_CONTAINER.json';release_path=work/'RELEASE.json'
    result_path=work/'result/pred_quick_match_metric_result.json'
    if release_path.exists():
        release=json.loads(release_path.read_bytes())
        need(release['container_absent'] and release['status']=='completed' and release['deadline_wall']==deadline,'Prior scorer failed or unresolved; no rerun')
        return metrics(checked_json(result_path,release['result_sha256']))
    score_deadline=deadline-45;expected=mounts(m,root,code)
    # This allowance can only collect/clean up an existing attempt. It never
    # changes the deadline passed to PID1 or the terminal-success checks below.
    need(collection_deadline is None or not allow_create,'Collection allowance cannot authorize scoring')
    io_deadline=deadline if collection_deadline is None else collection_deadline
    entry=['-B','-m','hybrid.v4_selected_eval.experimental_score_guard','--deadline',str(score_deadline),'--',PYTHON,'/source/pdf_validation.py','--config','/config/end2end-full.yaml']
    if identity_path.exists():
        identity=json.loads(identity_path.read_bytes());need(identity['deadline_wall']==deadline,'CPU deadline reset refused')
    else:
        need(allow_create and deadline-time.time()>120,'No authorized first CPU creation/time remaining')
        name='v4-exp-score-'+uuid.uuid4().hex;identity=dict(name=name,owner=name,cid=None,image=IMAGE,deadline_wall=deadline)
        atomic_json(identity_path,identity)
        command=['create','--name',name,'--label','v4-experimental-score-owner='+name,'--network','none','--runtime','runc',
            '--cpus','8','--memory','32g','--memory-swap','32g','--pids-limit','512','--restart','no','--init=false',
            '--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--env','PYTHONDONTWRITEBYTECODE=1',
            '--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1','--env','MKL_NUM_THREADS=1',
            '--env','PYTHONPATH=/source:/cpu-code','--workdir','/work']
        for src,dst,rw in expected:command+=['--mount',f'type=bind,src={src},dst={dst}'+('' if rw else ',readonly')]
        command+=['--entrypoint',PYTHON,IMAGE,*entry];atomic_json(work/'COMMAND.json',dict(argv=['docker',*command],deadline_wall=deadline))
        try:docker(command,deadline,cap=20,check=True)
        except BaseException:
            # Never repeat ambiguous create; inspect the exact saved name below.
            info=inspect_owned(identity,deadline)
            if info is None:raise
    collection_info=None
    if not allow_create:
        need(isinstance(identity.get('cid'),str) and bool(identity['cid']),'Collection requires a registered exact CID')
        collection_info=inspect_owned(identity,io_deadline)
        if collection_info is not None and collection_info['State']['Running'] and time.time()<score_deadline:
            raise RuntimeError('Original scorer still running; collect later without restarting or interrupting it')
    error=None;values=None;exit_code=None
    try:
        info=inspect_owned(identity,io_deadline) if allow_create else collection_info
        need(info is not None,'Owned scorer absent without release proof')
        identity['cid']=info['Id'];atomic_json(identity_path,identity);verify_container(info,expected,entry)
        if info['State']['Status']=='created':
            need(allow_create,'Collection cannot start an unstarted container')
            need(not (work/'START_INTENT.json').exists(),'Ambiguous start; no duplicate start')
            need(score_deadline-time.time()>30,'Original scorer deadline exhausted')
            atomic_json(work/'START_INTENT.json',dict(container_id=identity['cid'],deadline_wall=deadline))
            docker(['start',identity['cid']],deadline,cap=10,check=True)
        while info['State']['Running'] or info['State']['Status']=='created':
            need(time.time()<score_deadline,'Original scorer work deadline reached')
            time.sleep(min(1,max(.01,score_deadline-time.time())));info=inspect_owned(identity,io_deadline)
            need(info is not None,'CPU container disappeared during scoring')
        exit_code=info['State']['ExitCode']
        log=docker(['logs',identity['cid']],io_deadline,cap=5);(work/'container.log').write_text(log.stdout+log.stderr,encoding='utf-8')
        terminal=json.loads((work/'SCORER_TERMINAL.json').read_bytes())
        need(exit_code==0 and terminal['returncode']==0 and terminal['error'] is None and terminal['reaped'] is True
            and terminal['absolute_deadline']==score_deadline and terminal['finished_wall']<=score_deadline,'Official scorer exit/deadline failed')
        values=metrics(json.loads(result_path.read_bytes()));atomic_json(work/'METRICS.json',values)
    except BaseException as exc:error=repr(exc)
    finally:
        try:
            release=release_owned(identity,io_deadline);release.update(status='completed' if values is not None and error is None else 'failed',
                deadline_wall=deadline,error=error,scorer_exit_code=exit_code,result_sha256=file_sha(result_path) if values is not None else None)
            if collection_deadline is not None:release.update(collection_only=True,collection_deadline_wall=collection_deadline)
            atomic_json(release_path,release)
        except BaseException as exc:
            atomic_json(work/'RELEASE_UNCERTAIN.json',dict(identity=identity,error=repr(exc),prior_error=error));raise
    need(error is None,'Official scorer failed; positive release saved, no automatic rerun: '+str(error))
    return values

def reports(m,s,root,frozen,values,leaderboard,start,package_sha):
    rows=frozen['items'];counts=dict(Counter(r['status'] for r in rows))
    lat=[r['end_to_end_seconds_including_export'] for r in rows if 'end_to_end_seconds_including_export' in r]
    release=json.loads((root/'score/RELEASE.json').read_bytes())
    internal=dict(schema='v4_experimental_internal_report_v1',experiment_auth=m['experiment_auth'],
        experimental_only=True,adaptive_eligible=False,calibration_accepted=False,training_allowed=False,
        calibration_report=m['calibration_report'],effective_policy=m['effective_policy'],
        predictions_frozen_sha256=file_sha(root/'PREDICTIONS_FROZEN.json'),cpu_package_sha256=package_sha,
        inference_state_sha256=file_sha(Path(m['root'])/'STATE.json'),counts=counts,full_run=frozen['full_run'],
        actual_action_counts=dict(Counter(r['action'] for r in rows if 'action' in r)),
        estimator_predict_calls=sum(r.get('estimator_predict_calls',0) for r in rows),
        decision_records=sum('decision_sha256' in r for r in rows),
        observed_native_page_pipeline_calls=sum(r.get('native_page_pipeline_calls',0) for r in rows),
        consumed_pages_without_native_call_receipt=sum(r['status']!='not_run_budget_exhausted' and 'native_page_pipeline_calls' not in r for r in rows),
        fallback_counts=dict(Counter(r['fallback_reason'] for r in rows if r.get('fallback_reason'))),
        candidate_alias_pages=sum(bool(r.get('candidate_aliases')) for r in rows),
        completed_e2e_mean=statistics.mean(lat) if lat else None,completed_e2e_median=statistics.median(lat) if lat else None,
        completed_e2e_p95=sorted(lat)[max(0,math.ceil(.95*len(lat))-1)] if lat else None,
        completed_e2e_scope='Preparation, prediction, recognition, export and experimental receipt; terminal failures remain in original allocation/queue records',
        gpu_allocation_seconds=s['gpu_seconds'],wall_inference_seconds=max((a['started_wall']+a['elapsed_seconds'] for a in s['allocations']),default=s['started_wall'])-s['started_wall'],
        scorer_elapsed_through_release=release['finished_wall']-start['started_wall'],scorer_release=release,
        allocations=[dict(id=a['id'],elapsed_seconds=a['elapsed_seconds'],release=a['release']) for a in s['allocations']],
        per_page_costs='PREDICTIONS_FROZEN.json',metrics=values,fit_calls=0,leaderboard_submitted=False)
    immutable_json(root/'INTERNAL_REPORT.json',internal)
    comparison=[dict(model=r['model'],published_overall=r['overall'],local_minus_published=values['overall']-r['overall']) for r in leaderboard['rows']]
    report=dict(schema='v4_experimental_main_report_v1',version='GBDT+Tele experimental-001',run_id=m['run_id'],
        configuration=dict(margin=0.0,fallback='B',model_sha256=m['effective_policy']['model_sha256'],candidate_actions=['A','B'],max_native_calls_per_page=1),
        local_protocol=dict(evaluator='OmniDocBench v1.6',commit=COMMIT,image=IMAGE,config_sha256=m['evaluation']['config']['sha256'],
            gt_sha256=m['evaluation']['gt']['sha256'],page_count=1651,full_inference=frozen['full_run'],page_status_counts=counts,
            failed_timeout_predictions='empty Markdown retained in1651 denominator',unstarted_budget_pages=counts.get('not_run_budget_exhausted',0)),
        official_components=values,leaderboard_comparison=dict(source=leaderboard['source'],snapshot_date=leaderboard['snapshot_date'],
            published_protocol=leaderboard['published_protocol'],scope_note=leaderboard['scope_note'],rows=comparison),leaderboard_submitted=False)
    immutable_json(root/'REPORT.json',report)
    text=[f"# {report['version']} — {m['run_id']}",'',f"Overall: **{values['overall']:.4f}**",'',
        '| Official component | Score |','|---|---:|',f"| Text Edit ↓ | {values['text_edit']:.6f} |",
        f"| Formula CDM ↑ | {values['cdm']*100:.4f} |",f"| Table TEDS ↑ | {values['teds']*100:.4f} |",
        f"| Table TEDS-S ↑ | {values['structure_teds']*100:.4f} |",f"| Reading order Edit ↓ | {values['reading_order']:.6f} |",'',
        '| Official reference | Published Overall | Local minus published |','|---|---:|---:|']
    text += [f"| {r['model']} | {r['published_overall']:.2f} | {r['local_minus_published']:+.4f} |" for r in comparison]
    text += ['',f"Source: [official {leaderboard['published_protocol']} table]({leaderboard['source']}), snapshot {leaderboard['snapshot_date']}. {leaderboard['scope_note']}",'',
        f"Local protocol: frozen v1.6 evaluator `{COMMIT}`, 1651 pages, single prediction group. Full inference: {frozen['full_run']}; statuses: {json.dumps(counts,sort_keys=True)}.",
        'Configuration: original saved GBDT, margin 0.0, fallback B, raster candidates A/B, at most one Tele call per page. This is a local experimental result; no leaderboard submission.',
        f"Evaluator image: `{IMAGE}`. GT SHA256: `{m['evaluation']['gt']['sha256']}`. Config SHA256: `{m['evaluation']['config']['sha256']}`.",'']
    body='\n'.join(text);path=root/'REPORT.md'
    if path.exists():need(path.read_text(encoding='utf-8')==body,'Main report changed')
    else:path.write_text(body,encoding='utf-8')
    return report

def bound_context(args):
    package=checked_json(args.cpu_lock,args.cpu_lock_sha256);packet=Path(args.cpu_lock).resolve().parent
    for name,h in package['files'].items():need(file_sha(bound_file(packet,name))==h,'CPU package changed')
    code=packet/'code';need(Path(__file__).resolve()==code/'hybrid/v4_selected_eval/experimental_evaluate.py','Execute exact CPU staging')
    need(inventory(code)==package['code_files'],'Unbound CPU code files')
    m=checked_json(args.manifest,args.manifest_sha256);need(m['schema']=='v4_experimental_host_v1' and m['limits']==LIMITS,'Experimental manifest required')
    authority=validate_auth(read(m['experiment_auth']));verify_code(Path(m['provider']['code_root']),authority['code_files'])
    need(inventory(m['provider']['code_root'])==m['provider']['code_files']==package['gpu_code_files'],'Frozen GPU closure differs')
    need(read(m['evaluation_binding'])==m['evaluation'] and m['evaluation_binding']['sha256']==authority['evaluation_binding_sha256'],'Evaluation authority differs')
    return m,code,read(dict(path=str(packet/'LEADERBOARD_SNAPSHOT.json'),sha256=package['files']['LEADERBOARD_SNAPSHOT.json']))

def call_argv(args,command):
    return [sys.executable,'-B','-m','hybrid.v4_selected_eval.experimental_evaluate',command,
        '--manifest',str(Path(args.manifest).resolve()),'--manifest-sha256',args.manifest_sha256,
        '--cpu-lock',str(Path(args.cpu_lock).resolve()),'--cpu-lock-sha256',args.cpu_lock_sha256]

def submit(args,m,code):
    import fcntl
    root=Path(m['root'])/'official';root.mkdir(exist_ok=True)
    with (root/'submit.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'STARTED.json').exists():return json.loads((root/'STARTED.json').read_bytes())
        s=State(m['root'],args.manifest_sha256).snapshot()
        need(s['terminal'] and not s['stopped'] and s['active'] is None,'Inference must finish before scoring submit')
        wall=time.time();mono=time.monotonic();start=dict(status='intent_persisted',started_wall=wall,started_monotonic=mono,
            deadline_wall=wall+86400,deadline_monotonic=mono+86400,boot_id=boot(),manifest_sha256=args.manifest_sha256,cpu_lock_sha256=args.cpu_lock_sha256)
        atomic_json(root/'STARTED.json',start)
        try:
            with (root/'CONTROLLER_STDOUT.txt').open('xb') as out,(root/'CONTROLLER_STDERR.txt').open('xb') as err:
                p=subprocess.Popen(call_argv(args,'controller'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                    cwd=code,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            observation=dict(status='spawned',identity=process_identity(p.pid),boot_id=start['boot_id'])
        except BaseException as exc:observation=dict(status='uncertain_no_relaunch',error=repr(exc))
        atomic_json(root/'SUBMISSION.json',observation);return dict(start=start,submission=observation)

def run(args,m,code,leaderboard):
    import fcntl
    root=Path(m['root'])/'official'
    with (Path(m['root'])/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=json.loads((root/'STARTED.json').read_bytes())
        need(start['boot_id']==boot() and start['manifest_sha256']==args.manifest_sha256 and start['cpu_lock_sha256']==args.cpu_lock_sha256,
            'Scoring identity/boot differs; deadline cannot reset')
        need(start['deadline_wall']==start['started_wall']+86400 and start['deadline_monotonic']==start['started_monotonic']+86400,'CPU budget differs')
        if args.command=='controller':
            with (root/'CONTROLLER_STARTED.json').open('x',encoding='utf-8') as f:
                json.dump(dict(identity=process_identity(os.getpid()),boot_id=boot()),f);f.flush();os.fsync(f.fileno())
        s=State(m['root'],args.manifest_sha256).snapshot()
        # Preparation and GT verification consume the same original24h interval.
        end=min(start['deadline_monotonic'],time.monotonic()+start['deadline_wall']-time.time())
        collection=args.command=='collect'
        if collection or (root/'score/RELEASE.json').exists():
            # Read-only collection of already-released results needs no new scorer budget.
            frozen=json.loads((root/'PREDICTIONS_FROZEN.json').read_bytes())
            verified=json.loads((root/'PROTOCOL_VERIFIED.json').read_bytes())
            need(verified['predictions_frozen_sha256']==file_sha(root/'PREDICTIONS_FROZEN.json') and verified['evaluation']==m['evaluation'],'Frozen protocol changed')
        else:
            with work_alarm(end-90):
                frozen=prepare_predictions(m,s,root)
                verify_protocol_after_freeze(m,root)
        if collection:
            # A fresh, bounded observation/cleanup window is independent of the
            # exhausted scoring budget. No predictions or GT are regenerated.
            collection_wall=time.time();collection_end=time.monotonic()+60
            with work_alarm(collection_end):
                values=official_score(m,root,code,start['deadline_wall'],allow_create=False,collection_deadline=collection_wall+60)
        else:values=official_score(m,root,code,start['deadline_wall'],allow_create=True)
        need(file_sha(Path(m['root'])/'STATE.json')==frozen['state_sha256'] and inventory(root/'predictions')==frozen['files'],'Inference/predictions changed while scoring')
        return reports(m,s,root,frozen,values,leaderboard,start,args.cpu_lock_sha256)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('submit','status','controller','collect'))
    for k in ('manifest','manifest-sha256','cpu-lock','cpu-lock-sha256'):p.add_argument('--'+k,required=True)
    a=p.parse_args()
    if a.command=='status':
        m=checked_json(a.manifest,a.manifest_sha256);root=Path(m['root'])/'official'
        result={name:json.loads((root/name).read_bytes()) for name in ('STARTED.json','SUBMISSION.json','score/OWNED_CONTAINER.json','score/RELEASE.json','score/RELEASE_UNCERTAIN.json') if (root/name).exists()}
    else:
        need(sys.platform=='linux','Linux CPU controller required');signal.signal(signal.SIGHUP,signal.SIG_IGN)
        m,code,leaderboard=bound_context(a)
        result=submit(a,m,code) if a.command=='submit' else run(a,m,code,leaderboard)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
