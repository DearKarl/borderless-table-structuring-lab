"""probe -> bind -> submit --once -> status -> finalize. No resume or retry."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from hybrid.v4_input_selector.worker_contract import atomic_json,file_sha,validate_contract,bound_file
from hybrid.v4_input_selector.supervisor import run_queue,work_alarm,process_identity,validated_result
from hybrid.v4_input_selector.linux_provider import load_legacy,SocketTransport
from hybrid.v4_input_selector.provider_wire import Client
from hybrid.v4_input_selector.provider_host import spawn_host
from .contracts import checked_json,MODEL_SHA,MARGINS
from .dependencies import need,require_receipt
from .host_manifest import load,read,bind,GPU
from .host_state import State,boot
from .guardian import CalibrationGuardian
from .co_use import install_co_use_query
from .absence import inspect_absent

def facade(m):
    return SimpleNamespace(root=Path(m['root']),input_root=Path(m['input_root']),template=m['template'],
        master=dict(bank_id=m['task_id'],gpu_uuid=GPU,lease_directory=m['lease_directory']),
        private_paths=[Path(p) for p in m['private_paths']])

def validate_page(m,item,result,out):
    validated_result(item,result,out)
    if result['status']!='completed':return
    from hybrid.v4_input_selector.selector_input import verify_selected_render
    from hybrid.v4_input_selector.scale_policy import valid_vector
    out=Path(out);receipt=json.loads(bound_file(out,'_calibration_results/'+item['item_id']+'.json').read_bytes())
    need(receipt['result']==result and receipt['item_id']==item['item_id'] and receipt['native_page_pipeline_calls']==1,'Calibration/base result or call count differs')
    pre=checked_json(bound_file(out,receipt['predecision_file']),receipt['predecision_sha256'])
    need(pre['item']==item and pre['model_sha256']==MODEL_SHA and pre['diagnostic_only'] is True
        and pre['preparation']['recognition_calls']==0 and pre['selected_for_recognition']==item['action'],'Calibration predecision differs')
    need(set(pre['features'])=={'A','B'},'Only raster A/B features allowed')
    for action,vector in pre['features'].items():
        need(vector is None or (valid_vector(action,vector) and len(vector)==14 and vector[-1]==0),'Invalid raster feature record')
    need([p['margin'] for p in pre['predictions']]==list(MARGINS),'Three frozen diagnostic predictions required')
    audit=checked_json(bound_file(out,result['audit_file']),result['audit_sha256'])
    verify_selected_render(pre['preparation']['actions'][item['action']]['render'],audit['render'])
    folder=out/item['item_id'];export=json.loads((folder/'export.json').read_bytes())
    need(export['item_id']==item['item_id'] and export['markdown_sha256']==file_sha(folder/'native.md')
         and export['native_middle_sha256']==audit['native_middle_sha256']
         and export['converter_sha256']==m['template']['vendor_files']['TeleOCR/src/vlm_middle_json_mkcontent.py'],'Native export binding differs')

class Budget:
    def __init__(self,state,allocation,rpc,end,hard,m):
        self.state,self.allocation,self.rpc,self.end,self.hard_deadline,self.manifest=state,allocation,rpc,end,hard,m
    def absolute_deadline(self):return self.end
    def inner_cleanup_deadline(self,hard,now):return min(hard-35,now+25)
    def before_start(self):self.state.require_allocation(self.state.snapshot(),self.allocation)
    def reserve(self,item):self.state.reserve(self.allocation,item['item_id'])
    def result(self,item,result,out):
        validate_page(self.manifest,item,result,out)
        self.state.result(self.allocation,item['item_id'],result,out)
    def failure(self,current,error):self.state.failure(self.allocation,current['item_id'] if current else None,error)
    def finish(self,cleanup,hard,clock,alarm):
        if cleanup.get('group_absent') is not True or cleanup.get('cleanup_error'):
            self.state.failure(self.allocation,None,'Inner cleanup failed')
        release=self.rpc.call('release',dict(allocation=self.allocation),min(hard-5,clock()+30))
        self.state.finish(self.allocation,release)
        return release

def host(args,m):
    need(args.boot_id==boot(),'Host boot changed')
    state=State(m['root'],args.manifest_sha256);saved=state.snapshot();state.require_allocation(saved,args.allocation)
    need(args.global_end==saved['active']['global_end'] and time.monotonic()<args.initial_work_end<=saved['started_monotonic']+540,'Original host deadline differs')
    run=Path(m['root'])/'runs'/args.allocation;out=run/'output'/'queue'
    contract_path=run/'control'/'contract.json';contract=json.loads(contract_path.read_bytes())
    channel=socket.socket(fileno=args.provider_fd);rpc=Client(channel)
    proxy=SimpleNamespace(rpc=rpc,allocation=args.allocation,initial_work_end=args.initial_work_end,global_end=args.global_end)
    budget=Budget(state,args.allocation,rpc,args.global_end,args.initial_work_end+60,m)
    try:
        rpc.call('lease_enter',dict(allocation=args.allocation),args.initial_work_end)
        rpc.call('fresh',dict(uuid=GPU),args.initial_work_end)
        observed=rpc.call('inspect_owned',dict(allocation=args.allocation),args.initial_work_end)
        image=run/'control'/'image.json';image_sha=atomic_json(image,dict(schema='v4_external_image_v1',
            job_id=contract['job_id'],image=observed['image'],container_id=observed['container_id'],inspected_at_utc=observed['inspected_at_utc']))
        contract['runtime_binding']['external_image']['receipt']=dict(path=str(image),sha256=image_sha)
        validate_contract(contract);sha=atomic_json(contract_path,contract)
        def factory(owner,logs):
            return SocketTransport(proxy,dict(contract=str(contract_path),contract_sha256=sha,owner=owner,logs=str(logs),output_root=str(out)))
        queue=run_queue(contract,sha,out,factory,alarm=work_alarm,budget=budget)
        return 0 if queue['state']=='completed' else 1
    except BaseException as exc:
        if state.snapshot()['active'] is not None:
            try:budget.failure(None,exc);budget.finish({},budget.hard_deadline,time.monotonic,work_alarm)
            except BaseException:pass
        raise
    finally:channel.close()

def base_argv(args,mode):
    return [sys.executable,'-B','-m','hybrid.v4_selected_eval.runtime',mode,
        '--manifest',str(Path(args.manifest).resolve()),'--manifest-sha256',args.manifest_sha256]

def submit(args,m,*,popen=subprocess.Popen,identity=process_identity):
    need(args.once is True,'Only submit --once is supported')
    state=State(m['root'],args.manifest_sha256);claimed,s=state.claim_submit()
    if not claimed:return s  # Never spawn after either success or ambiguous creation.
    allocation=s['active']['id'];root=Path(m['root']);run=root/'runs'/allocation
    try:
        (run/'control').mkdir(parents=True,exist_ok=False);(run/'output').mkdir()
        import shutil
        shutil.copyfile(m['runtime']['path'],run/'control'/'SELECTOR_RUNTIME.json')
        contract=deepcopy(m['template']);contract.update(job_id='raster-cal-'+allocation,wall_seconds=14400,items=read(m['projection'])['items'])
        contract['runtime_binding']['external_image']['receipt']=dict(path=str(run/'control'/'image.json'),sha256='0'*64)
        validate_contract(contract);atomic_json(run/'control'/'contract.json',contract)
        argv=base_argv(args,'guardian')+['--allocation',allocation,'--boot-id',s['boot_id']]
        with (run/'output'/'GUARDIAN_STDOUT.txt').open('xb') as stdout,(run/'output'/'GUARDIAN_STDERR.txt').open('xb') as stderr:
            process=popen(argv,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,start_new_session=True,
                cwd=m['provider']['code_root'],env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        state.submission(dict(status='spawned',identity=identity(process.pid),argv=argv,
            runtime_sha256=m['runtime']['sha256'],code_sha256=file_sha(__file__),logs=str(run/'output')))
    except BaseException as exc:
        state.submission(dict(status='uncertain_no_relaunch',error=repr(exc)))
        raise
    return state.snapshot()

def guardian(args,m):
    import fcntl
    state=State(m['root'],args.manifest_sha256);s=state.snapshot();state.require_allocation(s,args.allocation)
    need(args.boot_id==s['boot_id']==boot(),'Original guardian boot differs')
    run=Path(m['root'])/'runs'/args.allocation
    with (run/'guardian.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Exclusive marker survives exit, preventing manual detached relaunch too.
        with (run/'GUARDIAN_STARTED.json').open('x',encoding='utf-8') as f:
            json.dump(dict(identity=process_identity(os.getpid()),boot_id=boot(),manifest_sha256=args.manifest_sha256),f);f.flush();os.fsync(f.fileno())
        legacy=load_legacy(m['provider']['legacy_root'],m['provider']['legacy_files'])
        install_co_use_query(legacy.gpu_backend,args.manifest_sha256)
        legacy.bounded_cleanup.inspect_absent=inspect_absent
        g=CalibrationGuardian(facade(m),state,m['provider'],legacy,run/'output',s['active']['global_end'],manifest=m)
        g.hard_end=min(g.global_end,s['started_monotonic']+600);g.work_end=g.hard_end-60
        need(g.work_end>time.monotonic(),'Original startup allowance expired')
        receipt=spawn_host(g,base_argv(args,'host')+['--allocation',args.allocation])
        release=receipt.get('finalization') or {}
        if state.snapshot()['active'] is not None:
            state.failure(args.allocation,None,'Host did not durably finish its collection')
            state.finish(args.allocation,release)
        return 0 if receipt.get('error') is None and receipt.get('host_returncode')==0 and not release.get('error') else 1

def collection_complete(m,state):
    s=state.snapshot();need(s['active'] is None and s['release'] is not None,'No positive release')
    allocation=s['submit']['allocation'];run=Path(m['root'])/'runs'/allocation
    if s['collection']!='completed':return False
    g=json.loads((run/'output'/'GUARDIAN_EXIT.json').read_bytes())
    need(g.get('error') is None and g['host_returncode']==0 and g['finalization']==s['release'],'Guardian collection/release differs')
    q=json.loads((run/'output'/'queue'/'ledger.json').read_bytes());items=read(m['projection'])['items']
    need(q['state']=='completed' and len(q['attempts'])==1 and [r['item_id'] for r in q['items']]==[i['item_id'] for i in items],'Exactly one complete sixteen-item queue required')
    need(q['cleanup']['group_absent'] is True and not q['cleanup']['cleanup_error'],'Inner cleanup unconfirmed')
    for item,row in zip(items,q['items']):
        need(row['status']=='completed' and row['dispatch_count']==1 and row['result']==s['slots'][item['item_id']]['result'],'Repeated/missing calibration result')
        validate_page(m,item,row['result'],run/'output'/'queue')
    return True

def finalize(args,m):
    state=State(m['root'],args.manifest_sha256)
    complete=collection_complete(m,state)
    claimed,s=state.claim_finalize()
    if not claimed:return s
    root=Path(m['root']);report=root/'calibration'
    if not complete:
        report.mkdir();atomic_json(report/'REPORT.json',dict(accepted=False,reason='partial_or_failed_collection',
            fit_calls=0,adaptive_bundle_created=False,manifest_sha256=args.manifest_sha256))
        state.finalized(dict(status='reported_failure',accepted=False));return state.snapshot()
    allocation=s['submit']['allocation'];seconds=s['finalize']['seconds']
    argv=[m['cpu_python']['path'],'-B','-m','hybrid.v4_selected_eval.recalibrate',
        '--plan',m['plan']['path'],'--plan-sha',m['plan']['sha256'],
        '--roster',m['roster']['path'],'--roster-sha',m['roster']['sha256'],
        '--model',m['gbdt_root'],'--data-root',m['data_root'],'--results-root',str(root/'runs'/allocation/'output'/'queue'),
        '--output',str(report),'--code-root',m['provider']['code_root'],'--seconds',str(seconds),'--bounded-child']
    try:
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with (root/'FINALIZE_STDOUT.txt').open('xb') as stdout,(root/'FINALIZE_STDERR.txt').open('xb') as stderr:
            result=subprocess.run(argv,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,env=env,
                cwd=m['provider']['code_root'],timeout=seconds)
        need(result.returncode==0,'Bounded calibration failed; no retry')
        value=json.loads((report/'REPORT.json').read_bytes())
        state.finalized(dict(status='finished',accepted=value['accepted'],report_sha256=file_sha(report/'REPORT.json'),exit_code=result.returncode))
    except BaseException as exc:
        state.finalized(dict(status='failed_no_retry',error=repr(exc)));raise
    return state.snapshot()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('probe','bind','submit','status','finalize','guardian','host'))
    for key in ('manifest','manifest-sha256','request','request-sha256','allocation','boot-id'):p.add_argument('--'+key)
    p.add_argument('--once',action='store_true');p.add_argument('--provider-fd',type=int)
    p.add_argument('--global-end',type=float);p.add_argument('--initial-work-end',type=float)
    a=p.parse_args()
    if a.command=='bind':result=bind(a.request,a.request_sha256)
    elif a.command=='probe':
        c=checked_json(a.request,a.request_sha256)
        result=require_receipt(read(c['dependency_receipt']),overlay_sha=c['overlay_manifest']['sha256'],fixture_sha=c['fixture_sha256'])
    elif a.command=='status':
        m=checked_json(a.manifest,a.manifest_sha256);result=State(m['root'],a.manifest_sha256).snapshot()
    else:
        need(sys.platform=='linux','Native host operations require Linux')
        signal.signal(signal.SIGHUP,signal.SIG_IGN)
        signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(InterruptedError('Guardian deadline')))
        m=load(a.manifest,a.manifest_sha256)
        result={'submit':submit,'guardian':guardian,'host':host,'finalize':finalize}[a.command](a,m)
    if isinstance(result,int):return result
    print(json.dumps(result,sort_keys=True));return 0

if __name__=='__main__':raise SystemExit(main())
