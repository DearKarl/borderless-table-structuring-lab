"""Specific two-session foreign-GPU interruption policy; never a generic exit137 exception."""


def validate_foreign_exit(record,pin,arm,freeze_sha,gpu_uuid):
    container=record['container'];state=container['State'];labels=container['Config']['Labels']
    if record.get('reason')!='foreign_gpu_process' or record.get('supervisor_error'):
        raise ValueError('Not the pinned resource interruption')
    if state['Running'] or state['OOMKilled'] or state['ExitCode']!=137 or state['Status']!='exited':
        raise ValueError('Resource interruption is not a stopped non-OOM guard exit')
    if container['Id']!=record['cid'] or not record['cid'].startswith(pin['container_prefix']):raise ValueError('Wrong interrupted container')
    if labels.get('hybrid.arm')!=arm or labels.get('hybrid.freeze')!=freeze_sha:raise ValueError('Wrong interrupted arm binding')
    if abs(float(record['time'])-pin['exit_time'])>0.00001:raise ValueError('Different interruption event')
    rows=[]
    for value in record['foreign_processes']:
        columns=[x.strip() for x in value.split(',')]
        if len(columns)!=2:raise ValueError('Unexpected foreign process receipt schema')
        rows.append((columns[0],int(columns[1])))
    if rows!=[(gpu_uuid,pin['foreign_pid'])]:raise ValueError('Different foreign process event')


def completed_candidate(result,page,arm,freeze_sha,actual_sha,actual_bytes):
    for key in ('key','page_id','input_sha256'):
        if result.get(key)!=page[key]:raise ValueError('Completed page original identity differs')
    if result.get('arm')!=arm or result.get('freeze_sha256')!=freeze_sha:raise ValueError('Completed page runtime identity differs')
    if result.get('status') not in ('success','truncated','failed'):raise ValueError('Not a completed page')
    if result['prediction_sha256']!=actual_sha or result.get('bytes',actual_bytes)!=actual_bytes:raise ValueError('Completed page bytes differ')
    if result['status']=='failed' and actual_bytes:raise ValueError('Failed primary is not empty')
