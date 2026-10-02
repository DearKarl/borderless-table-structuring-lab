"""Exact native PageDeadline receipt plus immutable reused-page evidence; no exit override."""
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha,validate_session_exit,pending_pages

EMPTY='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'


def deadline_outcome(exit_record,result,page,pins):
    state=exit_record['container']['State']
    if exit_record['cid']!=pins['cid'] or exit_record['container']['Id']!=pins['cid'] or exit_record['time']!=pins['exit_time']:
        raise ValueError('Different native deadline session')
    if exit_record['reason'] is not None or exit_record['foreign_processes']!=[] or exit_record['supervisor_error'] is not None:
        raise ValueError('Not a native worker deadline pause')
    if state['Running'] or state['OOMKilled'] or state['Status']!='exited' or state['ExitCode']!=2:raise ValueError('Not a cleanly classified exit2 pause')
    expected={'arm':'tele_raw','freeze_sha256':pins['freeze_sha256'],'complete':False,'pause_reason':'PageDeadline',
        'completed_this_session':682,'model_loads':{'tele':1,'paddle':0}}
    if result.get('blocked') or any(result.get(k)!=v for k,v in expected.items()):raise ValueError('Native session outcome differs')
    expected={'arm':'tele_raw','freeze_sha256':pins['freeze_sha256'],'key':'p01367','page_id':'scihub_jb%2Fmvh022.pdf_1',
        'input_sha256':pins['timeout_input_sha256'],'status':'failed','bytes':0,'prediction_sha256':EMPTY}
    if any(page.get(k)!=v for k,v in expected.items()):raise ValueError('Deadline page terminal was altered')
    if page['error']['type']!='PageDeadline' or page['error']['message']!='Native page deadline 900 seconds':raise ValueError('Different page failure')
    if not 900<=page['elapsed_seconds']<930:raise ValueError('Native deadline timing differs')


def verify_original(base,pins):
    evidence={}
    for relative,digest in pins['original_evidence'].items():
        path=(base/relative).resolve()
        if not path.is_relative_to(base.resolve()) or sha(path)!=digest:raise ValueError('Original native pause evidence changed')
        evidence[str(path)]=digest
    root=base/'full-v2/tele_raw';session=root/'sessions/session-0003';page=session/'pages/p01367/PAGE_RESULT.json'
    deadline_outcome(read(session/'EXIT.json'),read(session/'SESSION_RESULT.json'),read(page),pins)
    # Invoke the untouched frozen validator. This PageDeadline was already allowed by the original contract.
    validate_session_exit(session)
    for path in (session/'pages/p01367/prediction.md',root/'predictions/scihub_jb%2Fmvh022.pdf_1.md'):
        if path.stat().st_size!=0 or sha(path)!=EMPTY:raise ValueError('Failed timeout primary changed')
        evidence[str(path)]=sha(path)
    terminal=read(root/'terminals/p01367.json')
    if terminal['status']!='failed' or terminal['session_receipt_sha256']!=sha(page) or (root/terminal['session_receipt']).resolve()!=page.resolve():raise ValueError('Timeout terminal source changed')
    host=read(base/'resource-resume-v1/tele_raw/attempt-0001/HOST_RESULT.json')
    if host['exit_code']!=2 or host.get('original_failures_preserved') is not True:raise ValueError('Old native pause host outcome differs')
    return evidence


def verify_reused(root,pages,rows,identity):
    by_key={p['key']:p for p in pages};expected={p['key'] for p in pages[:1368]};evidence={}
    if len(rows)!=1368 or {r['key'] for r in rows}!=expected:raise ValueError('Expected exactly original1368 reused terminals')
    for item in rows:
        key=item['key'];page=by_key[key];source=(root/item['session_receipt']).resolve()
        if not source.is_relative_to((root/'sessions').resolve()) or source.parent.name!=key or source.name!='PAGE_RESULT.json' or source.parent.parent.parent.name not in ('session-0001','session-0002','session-0003'):
            raise ValueError('Reused native source differs')
        receipt=read(source);terminal_path=root/'terminals'/(key+'.json');terminal=read(terminal_path)
        if sha(source)!=item['receipt_sha256'] or terminal['session_receipt_sha256']!=item['receipt_sha256'] or (root/terminal['session_receipt']).resolve()!=source:
            raise ValueError('A completed page was replaced or reinferred')
        for k,value in {'key':key,'page_id':page['page_id'],'input_sha256':page['input_sha256'],'arm':'tele_raw','freeze_sha256':identity}.items():
            if receipt.get(k)!=value or terminal.get(k)!=value:raise ValueError('Reused page identity differs')
        name=receipt.get('prediction_file','prediction.md')
        if name not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected native primary path')
        native=source.parent/name;primary=root/'predictions'/(page['page_id']+'.md')
        digest=sha(native);size=native.stat().st_size
        if digest!=receipt['prediction_sha256'] or digest!=terminal['prediction_sha256'] or digest!=item['prediction_sha256'] or sha(primary)!=digest or size!=item['bytes']:
            raise ValueError('Reused original prediction changed')
        if receipt['status'] not in ('success','truncated','failed') or terminal['status']!=receipt['status'] or item['status']!=receipt['status']:
            raise ValueError('Reused terminal status changed')
        if receipt['status']=='failed' and size:raise ValueError('Failed primary is not empty')
        for path in (source,terminal_path,native,primary):evidence[str(path)]=sha(path)
    return evidence


def audit_completed(root,pages,identity):
    pending=pending_pages(root,pages,identity,'tele_raw')
    if len(pending)!=283 or {p['key'] for p in pending}!={p['key'] for p in pages[1368:]}:raise ValueError('Expected exact283 missing-page complement')
    rows=[];seen=set()
    for session in sorted((root/'sessions').iterdir()):
        for source in sorted((session/'pages').glob('*/PAGE_RESULT.json')):
            result=read(source);key=result['key']
            if key in seen:raise ValueError('Duplicate completed native result')
            seen.add(key);name=result.get('prediction_file','prediction.md')
            if name not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected native primary path')
            path=source.parent/name
            rows.append({'key':key,'session_receipt':str(source.relative_to(root)),'receipt_sha256':sha(source),
                'prediction_sha256':sha(path),'bytes':path.stat().st_size,'status':result['status']})
    evidence=verify_reused(root,pages,rows,identity)
    return {'arm':'tele_raw','freeze_sha256':identity,'completed_pages_reused':1368,'remaining_pages':283,
        'completed_pages':rows,'read_only':True,'timeout_page_retained_failed':'p01367','evidence':evidence}
