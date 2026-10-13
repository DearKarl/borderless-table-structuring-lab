"""Independent1651 controller; no old A/B queue, state, predictions or scores."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from hybrid.v4_input_selector.worker_contract import atomic_json,file_sha,bound_file,validate_contract
from hybrid.v4_input_selector.supervisor import run_queue,validated_result,process_identity,work_alarm
from hybrid.v4_input_selector.provider_wire import Client
from hybrid.v4_input_selector.linux_provider import load_legacy,SocketTransport
from hybrid.v4_input_selector.selector_input import verify_selected_render
from .contracts import checked_json
from .dependencies import need
from .host_state import boot
from .host_manifest import GPU,read
from .experimental_manifest import load,bind,batch,read_batch
from .experimental_state import State
from .experimental_monitor import spawn_host
from .experimental_timeout import normal_timeout
from .guardian import CalibrationGuardian
from .runtime import Budget as CalibrationBudget,facade
from .co_use import install_co_use_query
from .absence import inspect_absent

class ExperimentalGuardian(CalibrationGuardian):
    metadata_keys=('experiment_auth','projection','rounding','overlay_manifest','dependency_receipt')

def page_receipt(m,item,result,out):
    validated_result(dict(item,action='A'),result,out)
    receipt=json.loads(bound_file(out,'_experimental/'+item['item_id']+'.json').read_bytes())
    need(receipt['experiment']==m['effective_policy'] and receipt['result']==result,'Experimental result/policy differs')
    selected=checked_json(bound_file(out,receipt['selected_file']),receipt['selected_sha256'])
    need(selected['result']==result and selected['selection']['policy_identity']==m['effective_policy']['effective_policy_identity'],'Selected identity differs')
    decision=checked_json(bound_file(out,selected['decision_file']),selected['decision_sha256'])
    need(decision['page_id']==item['page_id'] and decision['input_sha256']==item['input_sha256']
        and decision['policy_identity']==m['effective_policy']['effective_policy_identity']
        and decision['margin']==0.0 and decision['learned_enabled'] is True
        and decision['preparation']['recognition_calls']==0,'Decision/zero-preparation-call binding differs')
    need(set(decision['candidate_features'])=={'A','B'} and selected['selection']['action'] in ('A','B'),'Raster A/B candidates required')
    calls=decision['prediction']['estimator_predict_calls'];reason=decision['selection']['reason']
    need(calls in (0,1) and (calls==1 or reason in ('missing_or_invalid_features','unreliable_features')),'Valid features must run saved estimator')
    need(selected['native_page_pipeline_calls']<=1,'Repeated native recognition')
    elapsed=receipt['end_to_end_seconds_including_export']
    need(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=selected['total_seconds'],'Missing end-to-end export timing')
    audit=checked_json(bound_file(out,result['audit_file']),result['audit_sha256'])
    if audit.get('render') is not None:
        verify_selected_render(decision['preparation']['actions'][selected['selection']['action']]['render'],audit['render'])
    if result['status']=='completed':
        need(selected['native_page_pipeline_calls']==1,'Completed page must have one native call')
        folder=Path(out)/item['item_id'];export=json.loads((folder/'export.json').read_bytes())
        need(file_sha(folder/'native.md')==export['markdown_sha256'] and export['native_middle_sha256']==audit['native_middle_sha256']
            and export['converter_sha256']==m['template']['vendor_files']['TeleOCR/src/vlm_middle_json_mkcontent.py'],'Native export changed')
    return receipt,selected,audit

def ordinary_model_failure(m,item,row):
    if row.get('status')!='failed' or 'result' not in row:return False
    _,selected,audit=page_receipt(m,item,row['result'],row['output'])
    # Only an explicit native OOM is classified automatically; other model errors
    # without a reviewed signature stay unknown instead of hiding integrity faults.
    return (audit.get('failure_stage')=='native_pipeline' and selected['native_page_pipeline_calls']==1
        and ('OutOfMemoryError(' in audit.get('error','') or 'CUDA out of memory' in audit.get('error','')))

class Budget(CalibrationBudget):
    def result(self,item,result,out):
        original=next(i for i in self.manifest['items'] if i['item_id']==item['item_id'])
        page_receipt(self.manifest,original,result,out)
        if result['status']!='completed':
            need(ordinary_model_failure(self.manifest,original,dict(status=result['status'],result=result,output=str(out))),
                'Unclassified native failure; stop all later allocations')
        self.state.result(self.allocation,item['item_id'],result,out)

def host(args,m):
    need(args.boot_id==boot(),'Host boot changed')
    state=State(m['root'],args.manifest_sha256);s=state.snapshot();state.require_allocation(s,args.allocation)
    a=s['active'];need(args.global_end==a['global_end'] and time.monotonic()<args.initial_work_end<=a['started_monotonic']+540,'Original batch clock differs')
    bm=read_batch(m,args.allocation,a['indices'],a['batch_refs']);run=Path(m['root'])/'runs'/args.allocation;out=run/'output/queue'
    path=run/'control/contract.json';contract=json.loads(path.read_bytes())
    need(contract['items']==[dict(m['items'][i],action='A') for i in a['indices']],'Transport batch differs')
    channel=socket.socket(fileno=args.provider_fd);rpc=Client(channel)
    proxy=SimpleNamespace(rpc=rpc,allocation=args.allocation,initial_work_end=args.initial_work_end,global_end=args.global_end)
    budget=Budget(state,args.allocation,rpc,args.global_end,args.initial_work_end+60,bm)
    try:
        rpc.call('lease_enter',dict(allocation=args.allocation),args.initial_work_end)
        rpc.call('fresh',dict(uuid=GPU),args.initial_work_end)
        observed=rpc.call('inspect_owned',dict(allocation=args.allocation),args.initial_work_end)
        image=run/'control/image.json';h=atomic_json(image,dict(schema='v4_external_image_v1',job_id=contract['job_id'],
            image=observed['image'],container_id=observed['container_id'],inspected_at_utc=observed['inspected_at_utc']))
        contract['runtime_binding']['external_image']['receipt']=dict(path=str(image),sha256=h)
        validate_contract(contract);ch=atomic_json(path,contract)
        def factory(owner,logs):return SocketTransport(proxy,dict(contract=str(path),contract_sha256=ch,owner=owner,logs=str(logs),output_root=str(out)))
        run_queue(contract,ch,out,factory,alarm=work_alarm,budget=budget)
        return 0  # Parent classifies any noncompleted item after physical release.
    except BaseException as exc:
        if state.snapshot()['active'] is not None:
            try:budget.failure(None,exc);budget.finish({},budget.hard_deadline,time.monotonic,work_alarm)
            except BaseException:pass
        raise
    finally:channel.close()

def reconcile(m,state,a,g):
    s=state.snapshot();need(s['active'] is None,'Unreleased allocation')
    release=s['allocations'][-1]['release']
    need(release==g.get('finalization') and release['container_absent'] and release['lease_released']
        and not release.get('error') and not g.get('error') and g.get('host_returncode')==0,'Guardian/host release failure')
    run=Path(m['root'])/'runs'/a['id'];out=run/'output';q=json.loads((out/'queue/ledger.json').read_bytes())
    need(len(q['attempts'])==1 and [r['item_id'] for r in q['items']]==a['items'],'Queue batch/attempt differs')
    need(q['contract_sha256']==file_sha(run/'control/contract.json'),'Queue contract hash differs')
    attempt=q['attempts'][0];need('startup' in attempt,'Worker never reached ready')
    classified={};timeout_proof=False
    watch=json.loads((out/'WATCH_DEADLINE.json').read_bytes()) if (out/'WATCH_DEADLINE.json').exists() else {}
    for row in q['items']:
        key=row['item_id'];slot=s['slots'][key];item=next(i for i in m['items'] if i['item_id']==key)
        need(row['dispatch_count']==(0 if row['status']=='pending' else 1),'Repeated/missing dispatch')
        if row['status']=='completed':
            need(slot['status']=='completed' and slot['result']==row['result'],'Completed result ledger differs')
            page_receipt(m,item,row['result'],out/'queue')
        elif row['status']=='pending':need(slot['status']=='pending' and slot['reservation'] is None,'Consumed page cannot become pending')
        elif normal_timeout(row,attempt,watch,g,slot.get('failure_monotonic')):
            if 'InterruptedError' in str(attempt.get('error')):
                sig=json.loads((out/'DEADLINE_SIGNAL.json').read_bytes())
                need(sig['identity']==g['host_identity'] and sig['work_end']==watch['work_end'] and sig['signal_monotonic']>=watch['work_end'],'Deadline signal not bound')
            classified[key]='timeout';timeout_proof=True
        elif ordinary_model_failure(m,item,slot) and attempt.get('error')=="RuntimeError('Worker page failed; stop further dispatch')":
            classified[key]='failed'
        else:raise ValueError('Unknown failure/early EOF/integrity fault; no further allocation')
    need(not attempt.get('error') or bool(classified),'Unclassified queue error')
    if q['cleanup'].get('group_absent') is not True or q['cleanup'].get('cleanup_error'):
        physical=json.loads((out/'PHYSICAL_RELEASE.json').read_bytes());inspect=json.loads((out/'CONTAINER_INSPECT.json').read_bytes())
        need(timeout_proof and physical['container_absent'] is True and physical['container_id']==inspect['Id']==release['container_id'],'No exact timeout namespace absence proof')
    state.reconcile(a['id'],classified)

def argv(args,mode):return [sys.executable,'-B','-m','hybrid.v4_selected_eval.experimental_host',mode,'--manifest',str(Path(args.manifest).resolve()),'--manifest-sha256',args.manifest_sha256]

def submit(args,m):
    need(args.once,'Only submit --once supported');state=State(m['root'],args.manifest_sha256);claimed,s=state.claim_submit()
    if not claimed:return s
    root=Path(m['root'])
    try:
        with (root/'CONTROLLER_STDOUT.txt').open('xb') as out,(root/'CONTROLLER_STDERR.txt').open('xb') as err:
            process=subprocess.Popen(argv(args,'controller'),stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,
                cwd=m['provider']['code_root'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        state.submission(dict(status='spawned',identity=process_identity(process.pid),boot_id=boot(),code_sha256=file_sha(__file__)))
    except BaseException as exc:state.submission(dict(status='uncertain_no_relaunch',error=repr(exc)));raise
    return state.snapshot()

def controller(args,m):
    import fcntl
    root=Path(m['root']);state=State(root,args.manifest_sha256)
    with (root/'controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (root/'CONTROLLER_STARTED.json').open('x',encoding='utf-8') as f:
            json.dump(dict(identity=process_identity(os.getpid()),boot_id=boot(),manifest_sha256=args.manifest_sha256),f);f.flush();os.fsync(f.fileno())
        legacy=load_legacy(m['provider']['legacy_root'],m['provider']['legacy_files'])
        install_co_use_query(legacy.gpu_backend,args.manifest_sha256);legacy.bounded_cleanup.inspect_absent=inspect_absent
        try:
            while True:
                a=state.begin()
                if a is None:return 0
                bm=batch(m,a['indices'],a['id'],a['seconds'])
                state.bind_batch(a['id'],dict(projection=bm['projection'],runtime=bm['runtime'],contract=bm['batch_contract']))
                g=ExperimentalGuardian(facade(m),state,m['provider'],legacy,root/'runs'/a['id']/'output',a['global_end'],manifest=bm)
                g.hard_end=min(g.global_end,a['started_monotonic']+600);g.work_end=g.hard_end-60
                need(g.work_end>time.monotonic(),'Original allocation startup budget exhausted')
                result=spawn_host(g,argv(args,'host')+['--allocation',a['id']])
                if state.snapshot()['active'] is not None:state.finish(a['id'],result.get('finalization') or {})
                reconcile(m,state,a,result)
        except BaseException as exc:state.stop_unknown(exc);raise

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('bind','submit','status','controller','host'))
    for key in ('request','request-sha256','manifest','manifest-sha256','allocation','boot-id'):p.add_argument('--'+key)
    p.add_argument('--once',action='store_true');p.add_argument('--provider-fd',type=int)
    p.add_argument('--global-end',type=float);p.add_argument('--initial-work-end',type=float);a=p.parse_args()
    if a.command=='status':
        m=checked_json(a.manifest,a.manifest_sha256);result=State(m['root'],a.manifest_sha256).snapshot()
    elif a.command=='bind':result=bind(a.request,a.request_sha256)
    else:
        need(sys.platform=='linux','Linux host required');signal.signal(signal.SIGHUP,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guardian deadline')))
        m=load(a.manifest,a.manifest_sha256,check_inputs=a.command!='host')
        result={'submit':submit,'controller':controller,'host':host}[a.command](a,m)
    if isinstance(result,int):return result
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
