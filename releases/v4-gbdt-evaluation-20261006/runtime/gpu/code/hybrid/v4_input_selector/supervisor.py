"""Persistent load-once worker supervisor; no model imports or automatic retries."""
import argparse
from contextlib import contextmanager
import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

from .worker_contract import (CLEANUP_SECONDS,PAGE_SECONDS,WORK_SECONDS,atomic_json,
    bound_file,file_sha,read_contract,recv_message,send_message)


def process_details(pid):
    p=Path('/proc')/str(pid);tail=(p/'stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,pgid=int(tail[2]),sid=int(tail[3]),start_ticks=int(tail[19]),uid=p.stat().st_uid,
        ppid=int(tail[1]),state=tail[0])


def process_identity(pid):
    """An unreaped child PID cannot be reused; state/PPID are not identity."""
    info=process_details(pid)
    return {key:info[key] for key in ('pid','pgid','sid','start_ticks','uid')}


def enable_subreaper():
    """Dedicated serial Linux supervisor adopts orphans before any worker starts."""
    if sys.platform!='linux' or threading.current_thread() is not threading.main_thread() or threading.active_count()!=1:
        raise RuntimeError('Subreaper requires a dedicated single-threaded Linux supervisor')
    if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
        raise RuntimeError('Existing SIGCHLD handler/auto-reaper is incompatible with owned waits')
    libc=ctypes.CDLL(None,use_errno=True);prctl=libc.prctl
    prctl.argtypes=[ctypes.c_int,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong,ctypes.c_ulong]
    prctl.restype=ctypes.c_int
    def call(option,value):
        if prctl(option,value,0,0,0)!=0:
            code=ctypes.get_errno();raise OSError(code,os.strerror(code))
    before=ctypes.c_int();address=ctypes.cast(ctypes.byref(before),ctypes.c_void_p).value
    call(37,address)  # PR_GET_CHILD_SUBREAPER
    call(36,1)        # PR_SET_CHILD_SUBREAPER; not inherited by the worker
    after=ctypes.c_int();call(37,ctypes.cast(ctypes.byref(after),ctypes.c_void_p).value)
    if after.value!=1:raise RuntimeError('Kernel did not enable child subreaper')
    return dict(supervisor_pid=os.getpid(),previous=before.value,enabled=True,selective_wait='positive PID only')


def group_members(leader):
    """Inspect only; the unreaped leader reserves this PGID/session identity."""
    members=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:info=process_details(int(path.name))
        except FileNotFoundError:continue
        if info['pgid']!=leader['pgid']:continue
        if info['sid']!=leader['sid'] or info['uid']!=leader['uid'] or info['start_ticks']<leader['start_ticks']:
            raise RuntimeError('Owned group member identity differs')
        members.append(info)
    return members


def reap_adopted(members,leader,receipt):
    """Never consume an unrelated child or the leader owned by subprocess.Popen."""
    for info in members:
        pid=info['pid']
        if pid==leader['pid'] or info['ppid']!=os.getpid() or info['state']!='Z':continue
        if pid<=0 or info['pgid']!=leader['pgid'] or info['sid']!=leader['sid'] or info['uid']!=leader['uid'] or info['start_ticks']<leader['start_ticks']:
            raise RuntimeError('Adopted child is not bound to the owned worker')
        if process_details(pid)!=info:raise RuntimeError('Adopted child identity changed before targeted wait')
        # SIGCHLD is default and this supervisor has no competing wait thread;
        # a verified direct zombie child keeps its PID until this exact wait.
        waited,status=os.waitpid(pid,os.WNOHANG)
        if waited!=pid:raise RuntimeError('Verified adopted zombie was not reaped by exact PID')
        receipt['reaped_descendants'].append(dict(identity=info,wait_status=status,exit_code=os.waitstatus_to_exitcode(status)))


class OwnedWorker:
    def __init__(self,args,output_root,log_root,contract_sha,owner_token):
        self.args=args;self.output_root=output_root;self.log_root=log_root
        self.contract_sha=contract_sha;self.owner_token=owner_token
        self.process=None;self.sock=None;self.logs=[];self.identity=None;self.subreaper=None

    def start(self,deadline):
        self.subreaper=enable_subreaper()
        parent,child=socket.socketpair();self.sock=parent
        argv=[sys.executable,'-B','-m','hybrid.v4_input_selector.tele_worker']
        for key,value in [('contract',self.args.contract),('contract-sha256',self.contract_sha),
            ('vendor-root',self.args.vendor_root),('model-root',self.args.model_root),
            ('aux-root',self.args.aux_root),('input-root',self.args.input_root),
            ('output-root',self.output_root),('owner-token',self.owner_token),('socket-fd',child.fileno())]:
            argv.extend(['--'+key,str(value)])
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
        try:
            self.logs=[(self.log_root/'stdout.log').open('xb'),(self.log_root/'stderr.log').open('xb')]
            self.process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=self.logs[0],stderr=self.logs[1],
                start_new_session=True,pass_fds=(child.fileno(),),env=env)
            self.identity=process_identity(self.process.pid)
            if self.identity['pgid']!=self.process.pid or self.identity['sid']!=self.process.pid or self.identity['uid']!=os.getuid():
                raise RuntimeError('Worker is not in its own expected process group')
        finally:child.close()
        ready=recv_message(self.sock,deadline=deadline)
        if ready.get('kind')!='ready' or ready.get('owner_token')!=self.owner_token or ready.get('contract_sha256')!=self.contract_sha or ready.get('pid')!=self.process.pid or ready.get('pgid')!=self.process.pid:
            raise RuntimeError('Unbound worker handshake: '+repr(ready))
        return dict(process=self.identity,handshake=ready,subreaper=self.subreaper)

    def page(self,item_id,deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Page dispatch deadline')
        self.sock.settimeout(remaining)
        send_message(self.sock,dict(kind='page',item_id=item_id,deadline=deadline))
        return recv_message(self.sock,deadline=deadline)

    def stop(self,deadline):
        receipt=dict(group_absent=self.process is None,cleanup_error=None,signals=[],subreaper=self.subreaper,reaped_descendants=[])
        try:
            if self.process is None:return receipt
            if not self.subreaper or not self.subreaper['enabled']:raise RuntimeError('Owned child adoption was not established')
            pid=self.process.pid
            # Never poll/wait/reap before the final destructive signal. Retaining
            # the child reserves its PID while group ownership is checked.
            actual=process_identity(pid)
            if self.identity is None:self.identity=actual
            if actual!=self.identity or actual['pgid']!=pid or actual['sid']!=pid or actual['uid']!=os.getuid():
                raise RuntimeError('Cleanup refused: owned process identity differs')
            for sig in (signal.SIGTERM,signal.SIGKILL):
                if time.monotonic()>=deadline:raise TimeoutError('Owned cleanup deadline')
                if process_identity(pid)!=self.identity:raise RuntimeError('Worker identity changed before group signal')
                try:os.killpg(pid,sig);receipt['signals'].append(int(sig))
                except ProcessLookupError:break
                if sig==signal.SIGTERM:time.sleep(min(.25,max(0,deadline-time.monotonic())))
            # Reserve the leader PID until descendants are gone. Adoption after
            # parent death is asynchronous, so rescan inside the same deadline.
            while True:
                if time.monotonic()>=deadline:raise TimeoutError('Owned descendants remain after termination')
                if process_identity(pid)!=self.identity:raise RuntimeError('Leader identity changed during descendant cleanup')
                members=group_members(self.identity)
                receipt['last_group_members']=members
                reap_adopted(members,self.identity,receipt)
                if len(members)==1 and members[0]['pid']==pid and members[0]['state']=='Z':break
                time.sleep(min(.05,max(0,deadline-time.monotonic())))
            self.process.wait(timeout=max(.001,deadline-time.monotonic()))
            receipt['exit_code']=self.process.returncode
            # No destructive signal or broad wait is allowed after leader reap.
            try:os.killpg(pid,0)
            except ProcessLookupError:receipt['group_absent']=True
            if not receipt['group_absent']:raise RuntimeError('Owned group absence not confirmed after targeted reaping')
        except BaseException as exc:receipt['cleanup_error']=repr(exc)
        finally:
            if self.sock is not None:self.sock.close()
            for stream in self.logs:stream.close()
        return receipt


@contextmanager
def work_alarm(deadline):
    """Linux main-thread alarm also bounds parent-side output validation."""
    previous=signal.getsignal(signal.SIGALRM)
    def timeout(signum,frame):raise TimeoutError('Supervisor work deadline')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)


def validated_result(item,result,output_root):
    keys={'item_id','status','page_starts','audit_file','audit_sha256','prediction_file','prediction_sha256'}
    if not isinstance(result,dict) or set(result)!=keys or result['item_id']!=item['item_id'] or result['status'] not in ('completed','failed','timeout') or type(result['page_starts']) is not int or result['page_starts'] not in (0,1):
        raise ValueError('Malformed worker result')
    if result['audit_file']!=item['item_id']+'/audit.json':raise ValueError('Unbound audit path')
    audit_path=bound_file(output_root,result['audit_file'])
    if file_sha(audit_path)!=result['audit_sha256']:raise ValueError('Audit checksum mismatch')
    audit=json.loads(audit_path.read_bytes())
    if any(audit.get(k)!=result[k] for k in ('item_id','status','page_starts')) or audit.get('page_id')!=item['page_id']:raise ValueError('Audit identity differs')
    if result['status']=='completed':
        if result['page_starts']!=1 or result['prediction_file']!=item['item_id']+'/prediction.json':raise ValueError('Missing successful page')
        p=bound_file(output_root,result['prediction_file'])
        if file_sha(p)!=result['prediction_sha256'] or audit.get('prediction_sha256')!=result['prediction_sha256']:raise ValueError('Prediction checksum mismatch')
        prediction=json.loads(p.read_bytes())
        if prediction.get('page_id')!=item['page_id'] or prediction.get('input_sha256')!=item['input_sha256'] or prediction.get('frame')!='canonical_upright_original_page':raise ValueError('Prediction identity/frame differs')
        for name,key in [('final_native_blocks.json','final_native_sha256'),('native_middle.json','native_middle_sha256')]:
            if file_sha(bound_file(output_root,item['item_id']+'/'+name))!=audit.get(key):raise ValueError('Native artifact checksum mismatch')
        for name,h in audit.get('native_image_files',{}).items():
            if not name.startswith('native_images/') or file_sha(bound_file(output_root,item['item_id']+'/'+name))!=h:raise ValueError('Native image checksum mismatch')
    return result


def open_ledger(contract,contract_sha,output_root,resume):
    path=output_root/'ledger.json'
    if path.exists():
        if not resume:raise ValueError('Existing job requires explicit --resume-unstarted')
        ledger=json.loads(path.read_bytes())
        if ledger.get('contract_sha256')!=contract_sha or ledger.get('job_id')!=contract['job_id']:raise ValueError('Resume requires identical contract')
        cleanup=ledger.get('cleanup') or {}
        if ledger.get('state') not in ('completed','stopped') or cleanup.get('group_absent') is not True or cleanup.get('cleanup_error'):raise ValueError('Prior termination is not clean and confirmed')
        if [x.get('item_id') for x in ledger['items']]!=[x['item_id'] for x in contract['items']]:raise ValueError('Resume queue differs')
        for item,row in zip(contract['items'],ledger['items']):
            if row['status'] not in ('pending','completed','failed','timeout'):raise ValueError('Unknown/unresolved started item')
            if (row['status']=='pending' and row['dispatch_count']!=0) or (row['status']!='pending' and row['dispatch_count']!=1):raise ValueError('Prior dispatch count differs')
            if row['status']=='completed':validated_result(item,row['result'],output_root)
        return ledger
    if resume:raise ValueError('Cannot resume an absent ledger')
    return dict(schema='v4_worker_ledger_v1',job_id=contract['job_id'],contract_sha256=contract_sha,state='created',elapsed_seconds_total=0,attempts=[],items=[dict(item_id=x['item_id'],status='pending',dispatch_count=0) for x in contract['items']])


def run_queue(contract,contract_sha,output_root,transport_factory,*,resume=False,clock=time.monotonic,alarm=None,budget=None):
    start=clock()
    from contextlib import nullcontext
    guard=alarm or (lambda deadline:nullcontext())
    # BankSession owns preparation failures too. Never put this guard around
    # run_queue itself: nested SIGALRM guards would cancel the outer alarm.
    with guard(budget.hard_deadline-CLEANUP_SECONDS) if budget is not None else nullcontext():
        output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
        ledger=open_ledger(contract,contract_sha,output_root,resume)
        if not any(x['status']=='pending' for x in ledger['items']):return ledger
        remaining=contract['wall_seconds']-ledger['elapsed_seconds_total']
        if remaining<=CLEANUP_SECONDS:raise ValueError('No remaining job budget')
        job_end=min(start+remaining,budget.absolute_deadline()) if budget is not None else start+remaining
        work_end=job_end-CLEANUP_SECONDS
        owner=uuid.uuid4().hex;logs=output_root/('worker-'+owner);logs.mkdir(exist_ok=False)
        attempt=dict(owner_token=owner,elapsed_seconds=None,error=None);ledger['attempts'].append(attempt)
        ledger.update(state='running',cleanup=None,owner_token=owner)
        path=output_root/'ledger.json';atomic_json(path,ledger)
    startup_hard=min(job_end,budget.hard_deadline) if budget is not None else job_end
    worker=None;current=None;page_final=None;cleanup_deadline=startup_hard-5
    try:
        startup_deadline=min(work_end,startup_hard-CLEANUP_SECONDS) if budget is not None else min(work_end,clock()+PAGE_SECONDS)
        with guard(startup_deadline):
            if budget is not None:budget.before_start()
            worker=transport_factory(owner,logs)
            attempt['startup']=worker.start(startup_deadline)
            atomic_json(path,ledger)
        if clock()>=startup_deadline:raise TimeoutError('Worker initialization deadline')
        for item,row in zip(contract['items'],ledger['items']):
            if row['status']!='pending':continue
            if row['dispatch_count']!=0:raise RuntimeError('Never replay a started item')
            page_start=clock()
            if page_start>=work_end:raise TimeoutError('Job deadline before next dispatch')
            page_final=min(page_start+PAGE_SECONDS,job_end);page_work=min(page_start+WORK_SECONDS,work_end)
            cleanup_deadline=page_final-5;current=row
            if budget is not None:budget.hard_deadline=page_final
            with guard(page_work):
                if budget is not None:budget.reserve(item)
                row.update(status='started',dispatch_count=1,page_start_charge=1,page_budget_seconds=PAGE_SECONDS,work_budget_seconds=page_work-page_start)
                atomic_json(path,ledger)
                result=worker.page(item['item_id'],page_work)
                if clock()>=page_work:raise TimeoutError('Reply after page work deadline')
                validated_result(item,result,output_root)
                if budget is not None:budget.result(item,result,output_root)
                row.update(status=result['status'],result=result,work_elapsed_seconds=clock()-page_start)
                atomic_json(path,ledger)
                if clock()>=page_work:raise TimeoutError('Final ledger after page work deadline')
                if result['status']!='completed':raise RuntimeError('Worker page failed; stop further dispatch')
            current=None
            # Even a successful last page retains its original cleanup envelope.
            if budget is None:page_final=None;cleanup_deadline=job_end-5
        ledger['state']='completed' if all(x['status']=='completed' for x in ledger['items']) else 'stopped'
    except BaseException as exc:
        ledger['state']='stopped';attempt['error']=repr(exc)
        if current is not None:
            if current['status']=='started' or isinstance(exc,TimeoutError):current['status']='timeout' if isinstance(exc,TimeoutError) else 'failed'
            current['supervisor_error']=repr(exc)
        if budget is not None:budget.failure(current,exc)
    finally:
        tail_end=min(cleanup_deadline+5,clock()+CLEANUP_SECONDS)
        inner_deadline=budget.inner_cleanup_deadline(tail_end,clock()) if budget is not None else min(cleanup_deadline,clock()+CLEANUP_SECONDS-5)
        try:
            with guard(inner_deadline) if budget is not None and worker is not None else nullcontext():
                cleanup=worker.stop(inner_deadline) if worker else dict(group_absent=True,cleanup_error=None,signals=[])
            if budget is not None and clock()>inner_deadline:
                cleanup=dict(cleanup,group_absent=False,cleanup_error='Inner cleanup exceeded its allotted tail')
        except BaseException as exc:cleanup=dict(group_absent=False,cleanup_error=repr(exc),signals=[])
        attempt['cleanup']=cleanup;ledger['cleanup']=cleanup
        if cleanup.get('group_absent') is not True or cleanup.get('cleanup_error'):ledger['state']='stopped'
        if budget is not None:
            try:outer=budget.finish(cleanup,tail_end,clock,guard)
            except BaseException as exc:outer=dict(container_absent=False,lease_released=False,error=repr(exc))
            attempt['owned_finalization']=outer
            if outer.get('container_absent') is not True or outer.get('lease_released') is not True or outer.get('error') or outer.get('persistence_error'):
                ledger['state']='stopped'
        attempt['elapsed_seconds']=clock()-start;ledger['elapsed_seconds_total']+=attempt['elapsed_seconds']
        attempt['within_job_deadline']=clock()<=job_end
        if budget is not None:
            attempt['cleanup_hard_deadline']=tail_end
            attempt['cleanup_within_hard_deadline']=clock()<=tail_end
            if not attempt['cleanup_within_hard_deadline']:ledger['state']='stopped'
        if current is not None:
            current['failure_cleanup_within_page_deadline']=clock()<=page_final
            if not current['failure_cleanup_within_page_deadline']:ledger['state']='stopped'
        if not attempt['within_job_deadline']:ledger['state']='stopped'
        with guard(tail_end) if budget is not None else nullcontext():
            atomic_json(path,ledger)
            if budget is not None and clock()>tail_end:raise TimeoutError('Queue persistence exceeded original hard deadline')
    return ledger


def main():
    parser=argparse.ArgumentParser(description='Reviewed inference-only persistent job; submit inside an owned scheduler allocation.')
    for key in ('contract','contract-sha256','vendor-root','model-root','aux-root','input-root','output-root'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--execute-reviewed',action='store_true',required=True)
    parser.add_argument('--resume-unstarted',action='store_true');args=parser.parse_args()
    if sys.platform!='linux':parser.error('Owned process-group supervision requires Linux')
    contract=read_contract(args.contract,args.contract_sha256)
    roots=[Path(getattr(args,key)).resolve() for key in ('vendor_root','model_root','aux_root','input_root','output_root')]
    if any(a==b or a.is_relative_to(b) or b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]):raise ValueError('Staged roots must be disjoint')
    out=roots[-1];out.mkdir(parents=True,exist_ok=True)
    import fcntl
    def cancelled(signum,frame):
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        raise InterruptedError('Owned scheduler job cancelled')
    signal.signal(signal.SIGTERM,cancelled)
    with (out/'.supervisor.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        factory=lambda owner,logs:OwnedWorker(args,out,logs,args.contract_sha256,owner)
        ledger=run_queue(contract,args.contract_sha256,out,factory,resume=args.resume_unstarted,alarm=work_alarm)
    return 0 if ledger['state']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
