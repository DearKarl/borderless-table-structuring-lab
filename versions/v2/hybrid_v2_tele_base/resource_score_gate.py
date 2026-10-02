"""Read-only admission of the two exact, preserved foreign-GPU interruptions."""
import importlib.util
from pathlib import Path
from hybrid_v2_tele_base.full_state import read,sha


def header(proof,audit,expected):
    for key,value in expected.items():
        if proof.get(key)!=value:raise ValueError('Resource admission differs: '+key)
    if proof.get('original_failed_exit_preserved') is not True or proof.get('original_failure_reclassified_as_success') is not False or proof.get('signals_sent') is not False:
        raise ValueError('Resource failure was relabeled or processes signalled')
    if audit.get('read_only') is not True or audit.get('original_failure_preserved') is not True:
        raise ValueError('Audit preservation differs')
    rows=audit['completed_pages'];keys=[r['key'] for r in rows]
    if len(keys)!=len(set(keys)) or len(keys)!=proof['completed_pages_reused'] or audit['completed_native_pages']!=len(keys) or audit['remaining_after_reconciliation']!=proof['remaining_pages']:
        raise ValueError('Resource reuse coverage differs')
    if proof['completed_pages_reused']+proof['remaining_pages']!=1651:raise ValueError('Original page count differs')


def check_idle(snapshot,uuid,index):
    rows=[list(map(str.strip,line.split(','))) for line in snapshot['inventory'].splitlines()]
    matched=[r for r in rows if len(r)==4 and r[1]==uuid]
    if len(matched)!=1 or matched[0][0]!=str(index) or int(matched[0][2])>10 or int(matched[0][3])!=0 or uuid in snapshot['processes']:
        raise ValueError('Original GPU was not freshly idle')


def verify_resource_recovery(base,pair,pages,binding):
    record=binding['resource_resume'];code=base/record['code_directory'];evidence={}
    def bind(path):
        path=Path(path);evidence[str(path)]=sha(path);return read(path)
    if sha(code/'CODE_LOCK.json')!=record['code_lock_sha256']:raise ValueError('Resource resume lock changed')
    for name,value in bind(code/'CODE_LOCK.json').items():
        path=(code/name).resolve()
        if not path.is_relative_to(code.resolve()) or sha(path)!=value:raise ValueError('Resource resume source changed')
        evidence[str(path)]=value
    pins=read(code/'BINDINGS.json')
    if pins['arms']!=pair['arms'] or pins['resource_pair_sha256']!=binding['resource_pair_sha256'] or pins['input_manifest_sha256']!=pair['input_manifest_sha256']:
        raise ValueError('Resource pair changed')
    spec=importlib.util.spec_from_file_location('locked_resource_resume_policy',code/'resource_resume_policy.py')
    policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)
    by_key={p['key']:p for p in pages};admitted={}
    for arm,pin in pins['events'].items():
        expected=record['arms'][arm];identity=pair['arms'][arm]['freeze_sha256'];uuid=pair['arms'][arm]['gpu_uuid']
        root=base/'full-v2'/arm;session=root/'sessions'/pin['session'];directory=base/'resource-resume-v1'/arm/'attempt-0001'
        proof=bind(directory/'RESOURCE_INTERRUPTION_ADMISSION.json');audit=bind(directory/'READ_ONLY_AUDIT.json');host=bind(directory/'HOST_START.json')
        old=bind(session/'EXIT.json');exit_sha=sha(session/'EXIT.json')
        header(proof,audit,{'arm':arm,'freeze_sha256':identity,'specific_foreign_session':str(session),
            'original_exit_sha256':exit_sha,'read_only_audit_sha256':sha(directory/'READ_ONLY_AUDIT.json'),
            'completed_pages_reused':expected['reused'],'remaining_pages':1651-expected['reused'],
            'entry_sha256':sha(code/'launcher.py'),'code_lock_sha256':record['code_lock_sha256']})
        policy.validate_foreign_exit(old,pin,arm,identity,uuid)
        if audit['original_exit']!=old or audit['original_exit_sha256']!=exit_sha or audit['specific_foreign_session']!=str(session) or audit['arm']!=arm or audit['freeze_sha256']!=identity or audit['gpu_uuid']!=uuid:
            raise ValueError('Resource audit identity differs')
        original_host=bind(base/pin['host_receipt'])
        if audit['original_host_pid']!=pin['host_pid'] or original_host['pid']!=pin['host_pid'] or audit['original_host_receipt_sha256']!=sha(base/pin['host_receipt']):raise ValueError('Old host identity differs')
        state=audit['actual_stopped_container_state']
        if state['Running'] or state['OOMKilled'] or state['ExitCode']!=137 or state['Status']!='exited':raise ValueError('Old container was not stopped non-OOM')
        if any(host.get(k)!=v for k,v in {'pid':expected['host_pid'],'arm':arm,'freeze_sha256':identity,'code_lock_sha256':record['code_lock_sha256']}.items()):raise ValueError('New host identity differs')
        if (directory/'HOST_ERROR.json').exists():raise ValueError('Recovery host failed')
        check_idle(proof['gpu_idle_before_resume'],uuid,expected['gpu_index'])
        completed=set()
        for item in audit['completed_pages']:
            key=item['key'];page=by_key[key];completed.add(key)
            if item['session']>pin['session'] or item['page_id']!=page['page_id']:raise ValueError('Reuse source differs')
            source=root/'sessions'/item['session']/'pages'/key/'PAGE_RESULT.json';result=bind(source)
            if sha(source)!=item['receipt_sha256']:raise ValueError('Reused native receipt changed')
            name=result.get('prediction_file','prediction.md')
            if name not in ('prediction.md','external-empty.md'):raise ValueError('Unexpected primary path')
            primary=source.parent/name;digest=sha(primary);size=primary.stat().st_size;evidence[str(primary)]=digest
            policy.completed_candidate(result,page,arm,identity,digest,size)
            if any(item[k]!=result[k] for k in ('prediction_sha256','status')) or item['bytes']!=size:raise ValueError('Reused content differs')
            terminal=bind(root/'terminals'/(key+'.json'))
            if terminal['session_receipt_sha256']!=sha(source) or (root/terminal['session_receipt']).resolve()!=source.resolve():raise ValueError('Completed page was replaced')
        for item in audit['interrupted_pages']:
            folder=root/'sessions'/item['session']/'pages'/item['key']
            if item['session']>pin['session'] or (folder/'PAGE_RESULT.json').exists() or sha(folder/'STARTED.json')!=item['started_sha256']:raise ValueError('Interrupted original evidence changed')
            bind(folder/'STARTED.json')
        for item in audit['session_exits']:
            path=root/'sessions'/item['session']/'EXIT.json'
            if item['session']>pin['session'] or sha(path)!=item['sha256']:raise ValueError('Historical exit changed')
            bind(path)
        next_session=root/'sessions'/expected['session'];assigned=bind(next_session/'ASSIGNED.json');start=bind(next_session/'START.json')
        missing=set(by_key)-completed
        if len(assigned['pages'])!=len(missing) or {p['key'] for p in assigned['pages']}!=missing:raise ValueError('Resume did not assign exactly remaining pages')
        for page in assigned['pages']:
            if any(page[k]!=by_key[page['key']][k] for k in ('page_id','input_sha256')):raise ValueError('Resumed input changed')
        for value in (assigned,start):
            if any(value.get(k)!=v for k,v in {'arm':arm,'freeze_sha256':identity,'input_manifest_sha256':pair['input_manifest_sha256']}.items()):raise ValueError('New session binding differs')
        if start['cid']==old['cid'] or 'CUDA_VISIBLE_DEVICES='+uuid not in start['command']:raise ValueError('Resume resource changed')
        check_idle(start['before'],uuid,expected['gpu_index'])
        for later in sorted((root/'sessions').iterdir()):
            if later.name>pin['session']:
                rows=read(later/'ASSIGNED.json')['pages']
                if completed & {p['key'] for p in rows}:raise ValueError('Completed pages scheduled again')
        admitted[str(session)]=exit_sha
    return {'admitted_sessions':admitted,'original_failures_preserved':True,'evidence':evidence}
