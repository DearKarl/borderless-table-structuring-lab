"""Independent score admission for the exact unknown-ownership Hybrid interruption."""
import importlib.util
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha
from resource_score_gate import check_idle


def guard_header(proof,expected):
    for key,value in expected.items():
        if proof.get(key)!=value:raise ValueError('Unknown-ownership admission differs: '+key)
    required={'ownership_classification':'unknown','observed_guard_reason':'foreign_gpu_process',
        'original_failed_exit_preserved':True,'original_failure_reclassified_as_success':False,
        'signals_sent':False,'completed_pages_reused':646,'remaining_pages':1005}
    if any(proof.get(k)!=v for k,v in required.items()):raise ValueError('Unknown ownership, original failure or coverage was relabeled')


def reuse_pointer(terminal,item,root,source):
    if terminal['session_receipt_sha256']!=item['receipt_sha256'] or (root/terminal['session_receipt']).resolve()!=source.resolve():
        raise ValueError('Reused page source was replaced')


def verify_guard_recovery(base,pair,pages,binding,continuation=None,allow_session4=False,second_continuation=None,allow_session5=False,native_deadline=None,allow_session6=False):
    configured=binding['hybrid_guard'];code=base/configured['code_directory'];evidence={}
    def bind(path):
        path=Path(path);evidence[str(path)]=sha(path);return read(path)
    if sha(code/'CODE_LOCK.json')!=configured['code_lock_sha256']:raise ValueError('Guard recovery code lock changed')
    for name,value in bind(code/'CODE_LOCK.json').items():
        path=(code/name).resolve()
        if not path.is_relative_to(code.resolve()) or sha(path)!=value:raise ValueError('Guard recovery source changed')
        evidence[str(path)]=value
    pins=read(code/'BINDINGS.json');identity=pair['arms']['hybrid_v2']['freeze_sha256'];gpu=pair['arms']['hybrid_v2']['gpu_uuid']
    if pins['arm']!='hybrid_v2' or pins['freeze_sha256']!=identity or pins['ownership_classification']!='unknown':raise ValueError('Guard recovery binding changed')
    for relative,digest in pins['original_evidence'].items():
        path=(base/relative).resolve()
        if not path.is_relative_to(base.resolve()) or sha(path)!=digest:raise ValueError('Original diagnostic evidence changed')
        evidence[str(path)]=digest
    directory=base/'hybrid-guard-recovery-v1/attempt-0001';root=base/'full-v2/hybrid_v2';old=root/'sessions/session-0002';new=root/'sessions/session-0003'
    proof=bind(directory/'GUARD_INTERRUPTION_ADMISSION.json');audit=bind(directory/'READ_ONLY_AUDIT.json');host=bind(directory/'HOST_START.json');exit_record=bind(old/'EXIT.json')
    guard_header(proof,{'arm':'hybrid_v2','freeze_sha256':identity,'specific_session':str(old),
        'original_exit_sha256':sha(old/'EXIT.json'),'original_cid':pins['cid'],
        'code_lock_sha256':configured['code_lock_sha256'],'read_only_audit_sha256':sha(directory/'READ_ONLY_AUDIT.json'),
        'original_monitor_sha256':pins['original_monitor_sha256'],'guarded_monitor_sha256':sha(code/'guarded_monitor.py'),
        'gpu_guard_sha256':sha(code/'gpu_ownership_guard.py'),'host_pid':host['pid']})
    if any(host.get(k)!=v for k,v in {'arm':'hybrid_v2','freeze_sha256':identity,'code_lock_sha256':configured['code_lock_sha256']}.items()):raise ValueError('Guard host identity differs')
    if str(code/'launcher.py') not in host['command'] or configured['code_lock_sha256'] not in host['command']:raise ValueError('Guard host command differs')
    if (directory/'HOST_ERROR.json').exists():raise ValueError('Guard recovery host failed')
    if (directory/'HOST_RESULT.json').exists() and bind(directory/'HOST_RESULT.json')['exit_code']!=0:
        if continuation is None:raise ValueError('Guard recovery ended without success')
        from external_resume_policy import continuation_receipts
        spec=importlib.util.spec_from_file_location('external_proof_guard',code/'gpu_ownership_guard.py');verified_guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(verified_guard)
        continuation_receipts(base,continuation,read,sha,verified_guard)
    prior=base/'resource-resume-code-v1'
    spec=importlib.util.spec_from_file_location('guard_specific_exit_policy',prior/'resource_resume_policy.py');policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    policy.validate_foreign_exit(exit_record,pins['event'],'hybrid_v2',identity,gpu)
    if exit_record['cid']!=pins['cid']:raise ValueError('Full interrupted CID differs')
    required={'arm':'hybrid_v2','freeze_sha256':identity,'gpu_uuid':gpu,'specific_foreign_session':str(old),
        'original_exit_sha256':sha(old/'EXIT.json'),'original_exit':exit_record,'original_host_pid':2290776,
        'completed_native_pages':646,'remaining_after_reconciliation':1005,'read_only':True,'original_failure_preserved':True,
        'ownership_classification':'unknown','historical_guard_label_is_not_ownership_proof':True}
    if any(audit.get(k)!=v for k,v in required.items()):raise ValueError('Guard audit identity or ownership differs')
    check_idle(proof['gpu_idle_before_resume'],gpu,2)
    rows=audit['completed_pages'];keys={r['key'] for r in rows};by_key={p['key']:p for p in pages}
    if len(rows)!=646 or len(keys)!=646:raise ValueError('Guard reused coverage differs')
    for item in rows:
        key=item['key'];page=by_key[key]
        if item['session'] not in ('session-0001','session-0002') or item['page_id']!=page['page_id']:raise ValueError('Guard original page identity differs')
        source=root/'sessions'/item['session']/'pages'/key/'PAGE_RESULT.json';result=bind(source)
        if sha(source)!=item['receipt_sha256']:raise ValueError('Guard reused receipt changed')
        name=result.get('prediction_file','prediction.md')
        if name not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected primary path')
        path=source.parent/name;digest=sha(path);size=path.stat().st_size;evidence[str(path)]=digest
        policy.completed_candidate(result,page,'hybrid_v2',identity,digest,size)
        if item['prediction_sha256']!=digest or item['bytes']!=size or item['status']!=result['status']:raise ValueError('Guard reused content changed')
        terminal=bind(root/'terminals'/(key+'.json'));reuse_pointer(terminal,item,root,source)
    for item in audit['interrupted_pages']:
        if item['session'] not in ('session-0001','session-0002'):raise ValueError('Unexpected interrupted session')
        folder=root/'sessions'/item['session']/'pages'/item['key']
        if (folder/'PAGE_RESULT.json').exists() or sha(folder/'STARTED.json')!=item['started_sha256']:raise ValueError('Guard interrupted original evidence changed')
        bind(folder/'STARTED.json')
    for item in audit['session_exits']:
        path=root/'sessions'/item['session']/'EXIT.json'
        if item['session'] not in ('session-0001','session-0002') or sha(path)!=item['sha256']:raise ValueError('Guard historical exit changed')
        bind(path)
    expected_sessions={'session-0001','session-0002','session-0003'}
    if allow_session4:
        if continuation is None:raise ValueError('Missing exact continuation proof')
        expected_sessions.add('session-0004')
    if allow_session5:
        if not allow_session4 or second_continuation is None:raise ValueError('Missing exact second continuation')
        from external2_resume_policy import continuation_receipts as second_receipts
        second_receipts(base,second_continuation,read,sha,verified_guard)
        expected_sessions.add('session-0005')
    if allow_session6:
        if native_deadline is None:raise ValueError('Missing exact Hybrid native deadline proof')
        from hybrid_deadline_evidence import verify_original
        verify_original(base,native_deadline)
        expected_sessions.add('session-0006')
    if {p.name for p in (root/'sessions').iterdir()}!=expected_sessions:raise ValueError('Unapproved extra Hybrid session')
    assigned=bind(new/'ASSIGNED.json');start=bind(new/'START.json')
    for record in (assigned,start):
        if any(record.get(k)!=v for k,v in {'arm':'hybrid_v2','freeze_sha256':identity,'input_manifest_sha256':pair['input_manifest_sha256']}.items()):raise ValueError('Guard new session binding differs')
    if start['cid']==pins['cid'] or 'CUDA_VISIBLE_DEVICES='+gpu not in start['command']:raise ValueError('Guard resumed container/GPU differs')
    if len(assigned['pages'])!=1005 or {p['key'] for p in assigned['pages']}!=set(by_key)-keys:raise ValueError('Guard resumed completed pages or skipped missing pages')
    for page in assigned['pages']:
        if any(page[k]!=by_key[page['key']][k] for k in ('page_id','input_sha256')):raise ValueError('Guard resumed input differs')
    check_idle(start['before'],gpu,2)
    # Evidence grows during inference; its full immutable set is sealed only after COMPLETE.
    spec=importlib.util.spec_from_file_location('validated_gpu_guard_for_score',code/'gpu_ownership_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    for path in sorted((new/'GPU_GUARD').glob('*.json')):
        record=bind(path)
        if record['cid']!=start['cid'] or record['gpu_uuid']!=gpu or record['gpu_process_signals_sent'] is not False:raise ValueError('GPU guard evidence identity differs')
        owned=guard.owned_pids(record['second_owned']);active=guard.active_pids(record['second_active'],gpu)
        for key,decision in record['decisions'].items():
            actual=guard.classify(int(key),record['before'].get(key,{'error':'new_second_snapshot_pid'}),record['after'][key],record['anchor_before'],record['anchor_after'],owned,active,start['cid'])
            if actual!=decision:raise ValueError('GPU ownership evidence decision differs')
    return {'admitted_session':str(old),'original_exit_sha256':sha(old/'EXIT.json'),'ownership_classification':'unknown',
        'completed_pages_reused':646,'remaining_pages_at_resume':1005,'evidence':evidence}
