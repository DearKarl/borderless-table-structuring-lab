"""Resource-only revision checks and serial-queue cancellation preconditions."""
import copy


def resource_only(parent,child,old_uuid,new_uuid):
    if parent['runtime_manifest']['gpu_uuid']!=old_uuid or child['runtime_manifest']['gpu_uuid']!=new_uuid:
        raise ValueError('Unexpected GPU identity')
    normalized=copy.deepcopy(child);normalized['runtime_manifest']['gpu_uuid']=old_uuid
    normalized.pop('resource_revision',None)
    allowed={'RUNTIME_MANIFEST.json','hybrid_v2_tele_base/run_smoke.py'}
    if set(parent['code_files'])!=set(child['code_files']):raise ValueError('Changed code inventory')
    for name,value in parent['code_files'].items():
        if name not in allowed and child['code_files'][name]!=value:raise ValueError('Scientific/control code changed: '+name)
    normalized['code_files']=parent['code_files']
    if normalized!=parent:raise ValueError('Non-resource freeze change')


def serial_command(command,expected):
    if command!=expected or '--resume' in command:raise ValueError('Original supervisor command changed')
    if command[-2:]!=['--arm','both']:raise ValueError('Serial guard requires original both without resume')


def gpu_pair(rows,old_uuid,new_uuid):
    by_uuid={r[1].strip():[x.strip() for x in r] for r in rows}
    old,new=by_uuid[old_uuid],by_uuid[new_uuid]
    if old[0]!='1' or new[0]!='2' or old[2:]!=new[2:] or 'A100' not in new[2] or int(new[3])<79000:
        raise ValueError('Expected same-model A100 80GB GPU1/GPU2')
    return {'tele_raw':old,'hybrid_v2':new}
