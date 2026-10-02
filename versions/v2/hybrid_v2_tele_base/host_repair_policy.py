"""Host-only short atomic temporary names and admission of the single documented host fault."""
import os
import re
from pathlib import Path
import uuid


def validate_binding_digests(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if key.endswith('sha256') and (not isinstance(item,str) or re.fullmatch('[0-9a-f]{64}',item) is None):
                raise ValueError('Invalid SHA256 binding: '+key)
            validate_binding_digests(item)
    elif isinstance(value,list):
        for item in value:validate_binding_digests(item)


def exclusive_bytes(path,encoded):
    path=Path(path);temporary=path.parent/('.atomic-'+uuid.uuid4().hex+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    os.link(temporary,path)
    temporary.unlink()


def specific_host_fault(record,freeze_sha,page_id):
    state=record['container']['State'];labels=record['container']['Config']['Labels']
    if record.get('reason')!='supervisor_error' or state['Running'] or state['OOMKilled'] or state['ExitCode']!=137:
        raise ValueError('Not the documented non-OOM host supervisor stop')
    if not record['cid'].startswith('175cb6c3e7e6') or record['container']['Id']!=record['cid']:
        raise ValueError('Wrong original Tele container')
    if labels.get('hybrid.arm')!='tele_raw' or labels.get('hybrid.freeze')!=freeze_sha:
        raise ValueError('Wrong original Tele resource identity')
    error=record.get('supervisor_error','')
    for text in ('OSError: [Errno 36] File name too long','full_state.py','exclusive_bytes','temporary.open',page_id):
        if text not in error:raise ValueError('Host failure traceback does not match exact filename fault')
    if record.get('foreign_processes'):raise ValueError('Foreign GPU activity is not an admitted host failure')
