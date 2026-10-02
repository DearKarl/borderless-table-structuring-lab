"""Host-only GPU ownership recheck. Never signals a GPU PID."""
import json
import os
from pathlib import Path
import time
import uuid


def proc_identity(pid,proc_root=Path('/proc')):
    root=proc_root/str(pid)
    try:
        before=(root/'stat').read_text().rsplit(')',1)[1].split()
        record={'pid':int(pid),'start_ticks':before[19],'ppid':int(before[1]),
            'cgroup':(root/'cgroup').read_text(),'pid_namespace':os.readlink(root/'ns/pid'),
            'mount_namespace':os.readlink(root/'ns/mnt')}
        after=(root/'stat').read_text().rsplit(')',1)[1].split()
        if before[19]!=after[19]:return {'pid':int(pid),'error':'pid_reused_during_snapshot'}
        return record
    except OSError as exc:return {'pid':int(pid),'error':type(exc).__name__}


def stable(first,second):
    keys=('pid','start_ticks','cgroup','pid_namespace','mount_namespace')
    return not first.get('error') and not second.get('error') and all(first.get(k)==second.get(k) for k in keys)


def cgroups(text):
    return {(line.split(':',2)[0],line.split(':',2)[1]):line.split(':',2)[2] for line in text.splitlines() if len(line.split(':',2))==3}


def classify(pid,before,after,anchor_before,anchor_after,owned,active,cid):
    if pid not in active:return 'gone_from_gpu'
    if not stable(before,after) or not stable(anchor_before,anchor_after):return 'unknown'
    anchor=cgroups(anchor_after['cgroup']);candidate=cgroups(after['cgroup'])
    bound={k:v for k,v in anchor.items() if cid in v}
    if not bound:return 'unknown'
    same_group=all(k in candidate and (candidate[k]==v or candidate[k].startswith(v.rstrip('/')+'/')) for k,v in bound.items())
    same_ns=after['pid_namespace']==anchor_after['pid_namespace'] and after['mount_namespace']==anchor_after['mount_namespace']
    if pid in owned and same_group and same_ns:return 'owned_confirmed'
    # Mixed namespace/group evidence remains unknown, never silently accepted.
    if pid not in owned and all(k in candidate and candidate[k]!=v and not candidate[k].startswith(v.rstrip('/')+'/') for k,v in bound.items()) and after['pid_namespace']!=anchor_after['pid_namespace']:
        return 'external_confirmed'
    return 'unknown'


def active_pids(text,gpu_uuid):
    result={}
    for line in text.splitlines():
        cols=[c.strip() for c in line.split(',')]
        if len(cols)!=2:raise ValueError('GPU process query schema differs')
        if cols[0]==gpu_uuid:result[int(cols[1])]=line
    return result


def owned_pids(text):
    return {int(line.strip()) for line in text.splitlines()[1:] if line.strip()}


def scan(query,save,session,cid,state,gpu_uuid,snapshot=proc_identity):
    first_owned=query(['docker','top',cid,'-eo','pid'])
    first_active=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    active=active_pids(first_active,gpu_uuid);suspects=set(active)-owned_pids(first_owned)
    if not suspects:return None,[]
    anchor_pid=state['State']['Pid'];anchor_before=snapshot(anchor_pid)
    before={pid:snapshot(pid) for pid in suspects}
    second_owned=query(['docker','top',cid,'-eo','pid'])
    second_active=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
    owned=owned_pids(second_owned);active2=active_pids(second_active,gpu_uuid)
    # New unmatched PIDs in the second snapshot have no stable before identity: stop as unknown.
    candidates=suspects | (set(active2)-owned)
    after={pid:snapshot(pid) for pid in candidates};anchor_after=snapshot(anchor_pid)
    decisions={pid:classify(pid,before.get(pid,{'error':'new_second_snapshot_pid'}),after[pid],anchor_before,anchor_after,owned,active2,cid) for pid in candidates}
    reason='gpu_ownership_unresolved' if 'unknown' in decisions.values() else 'foreign_gpu_process' if 'external_confirmed' in decisions.values() else None
    evidence={'time':time.time(),'cid':cid,'gpu_uuid':gpu_uuid,'anchor_pid':anchor_pid,
        'first_owned':first_owned,'first_active':first_active,'second_owned':second_owned,'second_active':second_active,
        'anchor_before':anchor_before,'anchor_after':anchor_after,'before':before,'after':after,
        'decisions':decisions,'stop_reason':reason,'gpu_process_signals_sent':False}
    folder=Path(session)/'GPU_GUARD';folder.mkdir(exist_ok=True)
    save(folder/(str(time.time_ns())+'-'+uuid.uuid4().hex+'.json'),evidence)
    blocked=[active2[pid] for pid,decision in decisions.items() if decision in ('unknown','external_confirmed')]
    return reason,blocked
