"""Additional native-deadline evidence for scoring; never bypass the original exit validator."""
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha
from resource_score_gate import check_idle
from hybrid_deadline_evidence import verify_original,verify_reused


def hybrid_deadline_header(proof,expected):
    required={'arm':'hybrid_v2','completed_pages_reused':1368,'remaining_pages':283,
        'timeout_page_retained_failed':'p01367','timeout_seconds':900,'original_validator_unchanged':True,
        'original_failed_page_preserved':True,'original_failure_reclassified_as_success':False,'signals_sent':False}
    if any(proof.get(k)!=v for k,v in {**required,**expected}.items()):raise ValueError('Native deadline continuation proof differs')


def missing_assignment(rows,pages):
    expected={p['key']:p for p in pages[1368:]}
    if len(rows)!=283 or len(expected)!=283 or {p['key'] for p in rows}!=set(expected):raise ValueError('Tele resume did not assign exact283 missing pages')
    for row in rows:
        if any(row[k]!=expected[row['key']][k] for k in ('page_id','input_sha256')):raise ValueError('Missing-page original input changed')


def verify_hybrid_deadline_recovery(base,pair,pages,binding):
    record=binding['hybrid_deadline'];code=base/record['code_directory'];evidence={}
    def bind(path):
        path=Path(path);evidence[str(path)]=sha(path);return read(path)
    if sha(code/'CODE_LOCK.json')!=record['code_lock_sha256']:raise ValueError('Deadline package lock changed')
    for name,value in bind(code/'CODE_LOCK.json').items():
        path=(code/name).resolve()
        if not path.is_relative_to(code.resolve()) or sha(path)!=value:raise ValueError('Deadline source changed')
        evidence[str(path)]=value
    if sha(Path(__file__).with_name('hybrid_deadline_evidence.py'))!=sha(code/'hybrid_deadline_evidence.py'):raise ValueError('Deadline evidence implementation differs')
    pins=read(code/'BINDINGS.json');identity=pair['arms']['hybrid_v2']['freeze_sha256']
    if pins['freeze_sha256']!=identity or pins['original_policy']['page_seconds']!=900 or pins['original_policy']['controlled_page_failure']!='empty_primary_and_pause':raise ValueError('Frozen native failure policy differs')
    evidence.update(verify_original(base,pins))
    directory=base/'hybrid-deadline-recovery-v1/attempt-0001';root=base/'full-v2/hybrid_v2'
    proof=bind(directory/'NATIVE_DEADLINE_CONTINUATION.json');audit=bind(directory/'READ_ONLY_AUDIT.json');host=bind(directory/'HOST_START.json')
    hybrid_deadline_header(proof,{'freeze_sha256':identity,'code_lock_sha256':record['code_lock_sha256'],
        'original_exit_sha256':sha(root/'sessions/session-0005/EXIT.json'),
        'read_only_audit_sha256':sha(directory/'READ_ONLY_AUDIT.json'),'host_pid':host['pid']})
    if any(host.get(k)!=v for k,v in {'arm':'hybrid_v2','freeze_sha256':identity,'code_lock_sha256':record['code_lock_sha256']}.items()):raise ValueError('Deadline host identity differs')
    if str(code/'launcher.py') not in host['command'] or record['code_lock_sha256'] not in host['command']:raise ValueError('Deadline host command differs')
    if (directory/'HOST_ERROR.json').exists():raise ValueError('Deadline recovery host failed')
    if (directory/'HOST_RESULT.json').exists() and bind(directory/'HOST_RESULT.json')['exit_code']!=0:raise ValueError('Deadline recovery did not finish cleanly')
    expected={'arm':'hybrid_v2','freeze_sha256':identity,'completed_pages_reused':1368,'remaining_pages':283,'read_only':True,'timeout_page_retained_failed':'p01367'}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('Deadline read-only audit differs')
    reused=verify_reused(root,pages,audit['completed_pages'],identity)
    if reused!=audit['evidence']:raise ValueError('Original reused-page evidence changed')
    evidence.update(reused);check_idle(proof['gpu_idle_before_resume'],pair['arms']['hybrid_v2']['gpu_uuid'],2)
    if {p.name for p in (root/'sessions').iterdir()}!={'session-0001','session-0002','session-0003','session-0004','session-0005','session-0006'}:raise ValueError('Unexpected additional Tele session')
    new=root/'sessions/session-0006';assigned=bind(new/'ASSIGNED.json');start=bind(new/'START.json')
    for value in (assigned,start):
        if any(value.get(k)!=v for k,v in {'arm':'hybrid_v2','freeze_sha256':identity,'input_manifest_sha256':pair['input_manifest_sha256']}.items()):raise ValueError('New Tele session binding differs')
    missing_assignment(assigned['pages'],pages)
    if (new/'pages/p01367').exists():raise ValueError('Timeout page was reinferred')
    if start['cid']==pins['cid'] or 'CUDA_VISIBLE_DEVICES='+pair['arms']['hybrid_v2']['gpu_uuid'] not in start['command']:raise ValueError('New Tele container/GPU differs')
    check_idle(start['before'],pair['arms']['hybrid_v2']['gpu_uuid'],2)
    import importlib.util
    spec=importlib.util.spec_from_file_location('hybrid_deadline_scoring_guard',code/'gpu_ownership_guard.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    for path in sorted((new/'GPU_GUARD').glob('*.json')):
        record=bind(path)
        if record['cid']!=start['cid'] or record['gpu_uuid']!=pair['arms']['hybrid_v2']['gpu_uuid'] or record['gpu_process_signals_sent'] is not False:raise ValueError('Hybrid deadline guard identity differs')
        owned=guard.owned_pids(record['second_owned']);active=guard.active_pids(record['second_active'],record['gpu_uuid'])
        for key,decision in record['decisions'].items():
            if guard.classify(int(key),record['before'].get(key,{'error':'new_second_snapshot_pid'}),record['after'][key],record['anchor_before'],record['anchor_after'],owned,active,start['cid'])!=decision:raise ValueError('Hybrid guard proof decision differs')
    return {'native_validator_unchanged':True,'timeout_page_retained_failed':'p01367','reused_pages':1368,'remaining_at_resume':283,'evidence':evidence}


def load_hybrid_deadline_pins(base,binding):
    configured=binding['hybrid_deadline'];code=base/configured['code_directory']
    if sha(code/'CODE_LOCK.json')!=configured['code_lock_sha256']:raise ValueError('Hybrid deadline package lock changed')
    for name,value in read(code/'CODE_LOCK.json').items():
        path=(code/name).resolve()
        if not path.is_relative_to(code.resolve()) or sha(path)!=value:raise ValueError('Hybrid deadline package bytes changed')
    if sha(Path(__file__).with_name('hybrid_deadline_evidence.py'))!=sha(code/'hybrid_deadline_evidence.py'):raise ValueError('Hybrid deadline evidence implementation differs')
    pins=read(code/'BINDINGS.json');verify_original(base,pins)
    return pins
