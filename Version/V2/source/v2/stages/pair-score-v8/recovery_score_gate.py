"""Admit one pinned engineering failure and verify the independent live-container handoff."""
import importlib.util
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha


def admission_header(proof,expected):
    for key,value in expected.items():
        if proof.get(key)!=value:raise ValueError('Recovery admission differs: '+key)
    if not proof.get('original_exit_preserved') or proof.get('original_failure_reclassified_as_success') is not False:
        raise ValueError('Original failed exit must remain failed')


def takeover_identity(ready,signal_record,preserved,host_start,pins,freeze_sha,cid):
    old=ready['old_supervisor']
    if old['pid']!=pins['old_hybrid_supervisor_pid'] or old['start_ticks']!=pins['old_hybrid_start_ticks'] or old['command']!=pins['old_hybrid_command']:
        raise ValueError('Old Hybrid host identity differs')
    expected={'old_pid':old['pid'],'old_start_ticks':old['start_ticks'],'SIGSTOP_sent':True,
              'SIGKILL_sent_to_exact_host_only':True,'container_signalled':False,'docker_attach_signalled':False}
    if any(signal_record.get(k)!=v for k,v in expected.items()):raise ValueError('Unexpected takeover signals')
    before=ready['container_before']
    if before['Id']!=cid or preserved['cid']!=cid or not before['State']['Running'] or preserved['running'] is not True:
        raise ValueError('Original live Hybrid container identity differs')
    if before['Config']['Labels'].get('hybrid.freeze')!=freeze_sha or before['Config']['Labels'].get('hybrid.arm')!='hybrid_v2':
        raise ValueError('Original Hybrid container resource binding differs')
    if not (before['State']['Pid']==preserved['native_pid_before']==preserved['native_pid_after']==1738801):
        raise ValueError('Hybrid native PID was not preserved')
    if preserved['native_model_reinitialized'] is not False:raise ValueError('Unexpected Hybrid model reload')
    if host_start['pid']!=1764764 or host_start['action']!='takeover-hybrid' or host_start['freeze_sha256']!=freeze_sha:
        raise ValueError('Repaired Hybrid host receipt differs')


def verify_recovery(base,pair,pages,binding):
    """Read-only proof check. Does not change any EXIT or treat arbitrary exit137 as admissible."""
    evidence={};packages={}
    def bind(path):
        path=Path(path);evidence[str(path)]=sha(path);return read(path)
    for arm,record in binding['host_repairs'].items():
        code=base/record['code_directory'];lock_path=code/'CODE_LOCK.json'
        if sha(lock_path)!=record['code_lock_sha256']:raise ValueError('Repair code lock changed')
        lock=bind(lock_path)
        for name,value in lock.items():
            path=(code/name).resolve()
            if not path.is_relative_to(code) or sha(path)!=value:raise ValueError('Repair source changed')
        packages[arm]=code
    raw=base/'full-v2/tele_raw';session=raw/'sessions/session-0001';code=packages['tele_raw']
    # Deliberately v3 only; the failed Tele v2 binding is never an admission source.
    proof_path=base/'host-repair-v3/tele_raw/SESSION_0001_ADMISSION.json';proof=bind(proof_path)
    admission_header(proof,{'arm':'tele_raw','freeze_sha256':pair['arms']['tele_raw']['freeze_sha256'],
        'original_terminal_count':205,'reconciled_terminal_count':206,'pending_pages':1445,
        'recovery_entry_sha256':sha(code/'launcher.py'),'policy_sha256':sha(code/'host_repair_policy.py'),
        'interrupted_nonterminal_pages_preserved_and_eligible_to_continue':['p00206']})
    old_exit=bind(session/'EXIT.json')
    if sha(session/'EXIT.json')!=proof['original_exit_sha256'] or old_exit['cid']!=proof['original_cid']:
        raise ValueError('Original failed Tele exit changed')
    spec=importlib.util.spec_from_file_location('validated_tele_host_policy_v3',code/'host_repair_policy.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    by_key={p['key']:p for p in pages};page=by_key['p00205'];policy.specific_host_fault(old_exit,pair['arms']['tele_raw']['freeze_sha256'],page['page_id'])
    reused=proof['reused_completed_pages']
    actual={p.parent.name for p in (session/'pages').glob('*/PAGE_RESULT.json')}
    if len(reused)!=206 or {p['key'] for p in reused}!=actual or len(actual)!=206:raise ValueError('Recovered native coverage differs')
    for item in reused:
        original=session/'pages'/item['key']/'PAGE_RESULT.json';terminal=bind(raw/'terminals'/(item['key']+'.json'))
        if item['page_id']!=by_key[item['key']]['page_id'] or sha(original)!=item['receipt_sha256']:
            raise ValueError('Recovered original page identity changed')
        if terminal['session_receipt_sha256']!=item['receipt_sha256'] or (raw/terminal['session_receipt']).resolve()!=original.resolve():
            raise ValueError('Recovered completed page was replaced or reinferred')
    p205=bind(session/'pages/p00205/PAGE_RESULT.json');pins=read(code/'BINDINGS.json')
    if p205['status']!='success' or p205['bytes']!=1515 or p205['prediction_sha256']!=pins['p00205_prediction_sha256']:
        raise ValueError('Recovered p00205 success changed')
    for path in (session/'pages/p00205/prediction.md',session/'pages/p00205/native/p00205/p00205.md'):
        if sha(path)!=pins['p00205_prediction_sha256']:raise ValueError('Original p00205 bytes changed')
        evidence[str(path)]=sha(path)
    if (session/'pages/p00206/PAGE_RESULT.json').exists() or not (session/'pages/p00206/STARTED.json').exists():
        raise ValueError('Interrupted original p00206 evidence differs')
    bind(session/'pages/p00206/STARTED.json')
    terminal=bind(raw/'terminals/p00206.json')
    if (raw/terminal['session_receipt']).resolve().is_relative_to(session.resolve()):raise ValueError('Interrupted p00206 was falsely completed in old session')
    # The preventive Hybrid takeover used v2, whose irrelevant erroneous Tele hash was never used by that action.
    directory=base/'host-repair-v2/hybrid_v2/attempt-0001';hcode=packages['hybrid_v2'];hpins=read(hcode/'BINDINGS.json')
    ready=bind(directory/'TAKEOVER_READY.json');signals=bind(directory/'HOST_SIGNAL_RECEIPT.json')
    preserved=bind(directory/'CONTAINER_PRESERVED.json');host_start=bind(directory/'HOST_START.json')
    hstart=bind(base/'full-v2/hybrid_v2/sessions/session-0001/START.json')
    if host_start['code_lock_sha256']!=binding['host_repairs']['hybrid_v2']['code_lock_sha256']:
        raise ValueError('Hybrid host code receipt differs')
    takeover_identity(ready,signals,preserved,host_start,hpins,pair['arms']['hybrid_v2']['freeze_sha256'],hstart['cid'])
    return {'admitted_failed_session':str(session),'original_exit_sha256':proof['original_exit_sha256'],
        'original_failed_exit_preserved':True,'reused_completed_pages':206,'interrupted_original_pages':['p00206'],
        'hybrid_original_native_pid_preserved':1738801,'evidence':evidence}
