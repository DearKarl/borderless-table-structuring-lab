"""Validate one exact external-PID event, not a generic resource-failure exception."""
def external_record(record,exit_record,pins,guard):
    pid=pins['event']['foreign_pid'];key=str(pid);cid=pins['cid'];gpu=pins['gpu_uuid']
    if record['cid']!=cid or record['gpu_uuid']!=gpu or record['stop_reason']!='foreign_gpu_process' or record['gpu_process_signals_sent'] is not False:
        raise ValueError('External guard identity differs')
    if record['decisions']!={key:'external_confirmed'}:raise ValueError('External decision differs')
    owned=guard.owned_pids(record['second_owned']);active=guard.active_pids(record['second_active'],gpu)
    first_active=guard.active_pids(record['first_active'],gpu)
    if pid not in first_active or pid not in active or pid in guard.owned_pids(record['first_owned']) or pid in owned:raise ValueError('External process snapshots differ')
    before=record['before'][key];after=record['after'][key]
    for value in (before,after):
        if value['start_ticks']!='50494325' or value['ppid']!=2387352 or value['pid_namespace']!='pid:[4026531836]' or value['mount_namespace']!='mnt:[4026531841]':raise ValueError('External host PID identity differs')
        if '/user.slice/user-0.slice/session-6186.scope' not in value['cgroup']:raise ValueError('External cgroup differs')
    if guard.classify(pid,before,after,record['anchor_before'],record['anchor_after'],owned,active,cid)!='external_confirmed':raise ValueError('External membership proof failed')
    if exit_record['cid']!=cid or exit_record['reason']!='foreign_gpu_process' or exit_record['supervisor_error'] is not None or exit_record['foreign_processes']!=[active[pid]]:
        raise ValueError('Specific external exit differs')
    if abs(exit_record['time']-1790244491.380297)>0.00001 or not 0<=exit_record['time']-record['time']<=10:raise ValueError('External event time differs')
    state=exit_record['container']['State'];labels=exit_record['container']['Config']['Labels']
    if exit_record['container']['Id']!=cid or state['Running'] or state['OOMKilled'] or state['ExitCode']!=137 or state['Status']!='exited':raise ValueError('External stopped container state differs')
    if labels.get('hybrid.arm')!='hybrid_v2' or labels.get('hybrid.freeze')!=pins['freeze_sha256']:raise ValueError('External arm binding differs')


def continuation_receipts(base,pins,read,sha,guard):
    root=base/'full-v2/hybrid_v2/sessions/session-0003'
    old_host=base/'hybrid-guard-recovery-v1/attempt-0001/HOST_RESULT.json'
    for path,digest in [(root/'EXIT.json',pins['interrupted_exit_sha256']),
                        (root/'GPU_GUARD'/pins['guard_record_name'],pins['guard_record_sha256']),
                        (old_host,pins['prior_host_result_sha256'])]:
        if sha(path)!=digest:raise ValueError('Pinned external continuation receipt changed')
    result=read(old_host)
    if result['exit_code']!=2 or result.get('original_failures_preserved') is not True:raise ValueError('Original guard host outcome differs')
    external_record(read(root/'GPU_GUARD'/pins['guard_record_name']),read(root/'EXIT.json'),pins,guard)
