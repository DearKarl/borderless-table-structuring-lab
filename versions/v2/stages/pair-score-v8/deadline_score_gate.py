"""Additional native-deadline evidence for scoring; never bypass the original exit validator."""
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha
from resource_score_gate import check_idle
from tele_deadline_evidence import verify_original,verify_reused


def deadline_header(proof,expected):
    required={'arm':'tele_raw','completed_pages_reused':1368,'remaining_pages':283,
        'timeout_page_retained_failed':'p01367','timeout_seconds':900,'original_validator_unchanged':True,
        'original_failed_page_preserved':True,'original_failure_reclassified_as_success':False,'signals_sent':False}
    if any(proof.get(k)!=v for k,v in {**required,**expected}.items()):raise ValueError('Native deadline continuation proof differs')


def missing_assignment(rows,pages):
    expected={p['key']:p for p in pages[1368:]}
    if len(rows)!=283 or len(expected)!=283 or {p['key'] for p in rows}!=set(expected):raise ValueError('Tele resume did not assign exact283 missing pages')
    for row in rows:
        if any(row[k]!=expected[row['key']][k] for k in ('page_id','input_sha256')):raise ValueError('Missing-page original input changed')


def verify_deadline_recovery(base,pair,pages,binding):
    record=binding['tele_deadline'];code=base/record['code_directory'];evidence={}
    def bind(path):
        path=Path(path);evidence[str(path)]=sha(path);return read(path)
    if sha(code/'CODE_LOCK.json')!=record['code_lock_sha256']:raise ValueError('Deadline package lock changed')
    for name,value in bind(code/'CODE_LOCK.json').items():
        path=(code/name).resolve()
        if not path.is_relative_to(code.resolve()) or sha(path)!=value:raise ValueError('Deadline source changed')
        evidence[str(path)]=value
    if sha(Path(__file__).with_name('tele_deadline_evidence.py'))!=sha(code/'tele_deadline_evidence.py'):raise ValueError('Deadline evidence implementation differs')
    pins=read(code/'BINDINGS.json');identity=pair['arms']['tele_raw']['freeze_sha256']
    if pins['freeze_sha256']!=identity or pins['original_policy']['page_seconds']!=900 or pins['original_policy']['controlled_page_failure']!='empty_primary_and_pause':raise ValueError('Frozen native failure policy differs')
    evidence.update(verify_original(base,pins))
    directory=base/'tele-deadline-recovery-v1/attempt-0001';root=base/'full-v2/tele_raw'
    proof=bind(directory/'NATIVE_DEADLINE_CONTINUATION.json');audit=bind(directory/'READ_ONLY_AUDIT.json');host=bind(directory/'HOST_START.json')
    deadline_header(proof,{'freeze_sha256':identity,'code_lock_sha256':record['code_lock_sha256'],
        'original_exit_sha256':sha(root/'sessions/session-0003/EXIT.json'),
        'read_only_audit_sha256':sha(directory/'READ_ONLY_AUDIT.json'),'host_pid':host['pid']})
    if any(host.get(k)!=v for k,v in {'arm':'tele_raw','freeze_sha256':identity,'code_lock_sha256':record['code_lock_sha256']}.items()):raise ValueError('Deadline host identity differs')
    if str(code/'launcher.py') not in host['command'] or record['code_lock_sha256'] not in host['command']:raise ValueError('Deadline host command differs')
    if (directory/'HOST_ERROR.json').exists():raise ValueError('Deadline recovery host failed')
    if (directory/'HOST_RESULT.json').exists() and bind(directory/'HOST_RESULT.json')['exit_code']!=0:raise ValueError('Deadline recovery did not finish cleanly')
    expected={'arm':'tele_raw','freeze_sha256':identity,'completed_pages_reused':1368,'remaining_pages':283,'read_only':True,'timeout_page_retained_failed':'p01367'}
    if any(audit.get(k)!=v for k,v in expected.items()):raise ValueError('Deadline read-only audit differs')
    reused=verify_reused(root,pages,audit['completed_pages'],identity)
    if reused!=audit['evidence']:raise ValueError('Original reused-page evidence changed')
    evidence.update(reused);check_idle(proof['gpu_idle_before_resume'],pair['arms']['tele_raw']['gpu_uuid'],1)
    if {p.name for p in (root/'sessions').iterdir()}!={'session-0001','session-0002','session-0003','session-0004'}:raise ValueError('Unexpected additional Tele session')
    new=root/'sessions/session-0004';assigned=bind(new/'ASSIGNED.json');start=bind(new/'START.json')
    for value in (assigned,start):
        if any(value.get(k)!=v for k,v in {'arm':'tele_raw','freeze_sha256':identity,'input_manifest_sha256':pair['input_manifest_sha256']}.items()):raise ValueError('New Tele session binding differs')
    missing_assignment(assigned['pages'],pages)
    if (new/'pages/p01367').exists():raise ValueError('Timeout page was reinferred')
    if start['cid']==pins['cid'] or 'CUDA_VISIBLE_DEVICES='+pair['arms']['tele_raw']['gpu_uuid'] not in start['command']:raise ValueError('New Tele container/GPU differs')
    check_idle(start['before'],pair['arms']['tele_raw']['gpu_uuid'],1)
    return {'native_validator_unchanged':True,'timeout_page_retained_failed':'p01367','reused_pages':1368,'remaining_at_resume':283,'evidence':evidence}
