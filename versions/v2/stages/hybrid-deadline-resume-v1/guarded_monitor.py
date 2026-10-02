def monitor(root,session,pages,freeze_sha,arm,cid,adopted):
    started=time.monotonic();phase=None;phase_started=started;reason=None;foreign=[];process=None;current={'phase':'load'};supervisor_error=None
    if not adopted:
        stdout=(session/'stdout.log').open('ab');stderr=(session/'stderr.log').open('ab')
        process=subprocess.Popen(['docker','start','--attach',cid],stdout=stdout,stderr=stderr)
    try:
        while True:
            state=json.loads(query(['docker','inspect',cid]))[0]
            collect(root,session,pages,freeze_sha,arm)
            if state['State']['Status'] not in ('created','running'):break
            try:current=read(session/'PROGRESS.json')
            except FileNotFoundError:current={'phase':'load'}
            identity=(current['phase'],current.get('key'))
            if identity!=phase:
                phase=identity;phase_started=time.monotonic()-max(0,time.time()-current.get('time',time.time()))
                with (root/'PROGRESS.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps({'session':session.name,'time':time.time(),**current},ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
                progress(root/'LATEST.json',{'session':session.name,'time':time.time(),**current})
            if state['State']['Running']:
                guard_reason,foreign=guard_scan(query,exclusive_json,session,cid,state,UUID)
                if guard_reason:reason=guard_reason;break
            limit=phase_limit(current['phase'])
            if time.monotonic()-phase_started>limit:reason='page_timeout' if current['phase']=='page' else 'load_or_shutdown_timeout';break
            if time.monotonic()-started>72*3600:reason='supervisor_wall_limit';break
            time.sleep(5)
    except BaseException:
        reason='supervisor_error';supervisor_error=traceback.format_exc()
        raise
    finally:
        state=json.loads(query(['docker','inspect',cid]))[0]
        if state['State']['Running']:subprocess.run(['docker','kill',cid],check=True,capture_output=True)
        if state['State']['Status']!='created':subprocess.run(['docker','wait',cid],check=True,capture_output=True)
        state=json.loads(query(['docker','inspect',cid]))[0]
        if process is not None:
            process.wait(timeout=30);stdout.close();stderr.close()
        exclusive_json(session/'EXIT.json',{'cid':cid,'container':state,'reason':reason,'foreign_processes':foreign,
                                          'last_progress':current,'adopted_existing_container':adopted,'time':time.time(),
                                          'supervisor_error':supervisor_error})
    if reason=='page_timeout' and not state['State']['OOMKilled']:
        page=next(p for p in pages if p['key']==current['key'])
        folder=session/'pages'/page['key'];started_record=read(folder/'STARTED.json')
        if any(started_record[k]!=page[k] for k in ('key','page_id','input_sha256')):raise RuntimeError('Timeout page identity mismatch')
        if not (folder/'PAGE_RESULT.json').exists():
            exclusive_bytes(folder/'external-empty.md',b'')
            exclusive_json(folder/'PAGE_RESULT.json',{
                **{k:page[k] for k in ('key','page_id','input_sha256')},'arm':arm,'freeze_sha256':freeze_sha,
                'status':'failed','prediction_file':'external-empty.md','prediction_sha256':sha(folder/'external-empty.md'),
                'bytes':0,'error':{'type':'SupervisorPageDeadline','message':'Owned container stopped after page deadline'},
                'external_exit_sha256':sha(session/'EXIT.json'),'native_outputs_preserved':True})
    collect(root,session,pages,freeze_sha,arm)
    return state,reason
