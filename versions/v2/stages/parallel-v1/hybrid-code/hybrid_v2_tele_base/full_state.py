"""Durable full-run state. Terminals are immutable; incomplete pages are never skipped."""
import hashlib
import json
import os
import uuid
from pathlib import Path
from .lock_checks import digest


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while b:=stream.read(8*1024*1024):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def exclusive_json(path,value):
    encoded=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode()
    exclusive_bytes(path,encoded)


def exclusive_bytes(path,encoded):
    path=Path(path);temporary=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.pending')
    with temporary.open('xb') as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    # Atomic no-replace publication. An interrupted temporary is never a terminal.
    os.link(temporary,path)
    temporary.unlink()


def progress(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.pending')
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False).encode());stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def page_key(index):return 'p%05d'%index


def validate_input_paths(rows,image_root):
    """Preserve each manifest filename and bytes, including mixed original JPG/PNG."""
    image_root=Path(image_root).resolve();names=set();ids=set()
    for row in rows:
        path=Path(row['image_path'])
        if row['page_id'] in ids or path.name in names:raise ValueError('Duplicate original page or image path')
        ids.add(row['page_id']);names.add(path.name)
        if path.parent!=image_root or path.is_symlink() or not path.is_file() or path.resolve().parent!=image_root:
            raise ValueError('Original image escapes frozen directory or is not a regular file')
        if sha(path)!=digest(row['input_sha256']):raise ValueError('Original image bytes changed')
    if {p.name for p in image_root.iterdir()}!=names:raise ValueError('Unexpected image directory contents')


def phase_limit(phase):
    if phase in ('load','load_tele','load_paddle'):return 630
    if phase=='page':return 930
    return 120


def validate_session_exit(session):
    """Only a clean completion or an explicitly classified pause permits continuation."""
    session=Path(session);record=read(session/'EXIT.json');state=record['container']['State']
    if state['OOMKilled'] or state['Running'] or state['Status']=='created':raise ValueError('Unclean session exit')
    if record.get('supervisor_error') or record['reason'] not in (None,'page_timeout'):
        raise ValueError('Unresolved supervisor failure')
    if record['reason']=='page_timeout':return
    result=read(session/'SESSION_RESULT.json')
    if result.get('blocked'):raise ValueError('Unknown worker failure requires diagnosis')
    if result.get('complete') and state['ExitCode']==0:return
    if result.get('pause_reason') in ('PageDeadline','PauseAfterFallback','owner_or_resource_stop') and state['ExitCode']==2:return
    raise ValueError('Worker exit and classification disagree')


def pending_pages(root,pages,freeze_sha,arm):
    """Recover only terminals whose prediction and original session receipt agree."""
    root=Path(root);pending=[]
    expected={p['key'] for p in pages}
    terminal_dir=root/'terminals'
    if terminal_dir.exists() and any(p.stem not in expected for p in terminal_dir.glob('*.json')):
        raise ValueError('Unrecognized terminal page')
    for page in pages:
        path=terminal_dir/(page['key']+'.json')
        if not path.exists():pending.append(page);continue
        terminal=read(path)
        for key,value in [('freeze_sha256',freeze_sha),('arm',arm),('page_id',page['page_id']),
                          ('input_sha256',page['input_sha256']),('key',page['key'])]:
            if terminal.get(key)!=value:raise ValueError('Terminal identity mismatch: '+key)
        prediction=root/'predictions'/(page['page_id']+'.md')
        if sha(prediction)!=terminal['prediction_sha256']:raise ValueError('Prediction changed')
        evidence=(root/terminal['session_receipt']).resolve()
        if not evidence.is_relative_to((root/'sessions').resolve()) or sha(evidence)!=terminal['session_receipt_sha256']:
            raise ValueError('Terminal receipt changed or escaped session root')
        if terminal['status'] not in ('success','truncated','failed'):raise ValueError('Invalid terminal state')
        if terminal['status']=='failed' and prediction.stat().st_size:raise ValueError('Failed page must have empty primary')
    return pending


def commit_page(root,session,page,freeze_sha,arm):
    root,session=Path(root),Path(session)
    source=session/'pages'/page['key']/'PAGE_RESULT.json'
    receipt=read(source)
    if any(receipt.get(k)!=page[k] for k in ('key','page_id','input_sha256')):
        raise ValueError('Worker page identity mismatch')
    if receipt['freeze_sha256']!=freeze_sha or receipt['arm']!=arm:raise ValueError('Worker runtime mismatch')
    if receipt['status'] not in ('success','truncated','failed'):raise ValueError('Nonterminal page cannot be committed')
    filename=receipt.get('prediction_file','prediction.md')
    if filename not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected primary filename')
    raw=source.parent/filename
    if sha(raw)!=receipt['prediction_sha256']:raise ValueError('Worker primary changed')
    if receipt['status']=='failed' and raw.stat().st_size:raise ValueError('Failure primary must be empty')
    destination=root/'predictions'/(page['page_id']+'.md')
    terminal_path=root/'terminals'/(page['key']+'.json')
    terminal={**receipt,'session_receipt':str(source.relative_to(root)), 'session_receipt_sha256':sha(source)}
    if terminal_path.exists():
        if read(terminal_path)!=terminal or sha(destination)!=sha(raw):raise ValueError('Conflicting terminal write')
        return False
    # An interrupted promotion may leave only the exact primary; adopt it after validation.
    if destination.exists():
        if sha(destination)!=sha(raw):raise ValueError('Conflicting orphan prediction')
    else:
        exclusive_bytes(destination,raw.read_bytes())
    exclusive_json(terminal_path,terminal)
    return True
