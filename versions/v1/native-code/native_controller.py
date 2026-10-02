"""Frozen full-native lifecycle functions; unified entry supplies its locked image root.
Input-root and filename-only changes are bound by SOURCE_COMPATIBILITY.json.
No standalone CLI; caller verifies system freeze before any model call.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,re,subprocess,sys,threading,time
from pathlib import Path
import xml.etree.ElementTree as ET
from core import runtime_key,pending_pages,primary_bytes,complete_lock
from native_stop_audit import audit_stops
from docker_mount import bind_mount
from page_paths import component_name
from watchdog_policy import POLICY, failure_action, verify_terminal_timeout
from native_schema import validate_schema,paddle_context
from image_only_audit import validate_zero,validate_empty
ROOT=Path('/srv/hybrid-research')
ENVS={'mineru':'mineru-v6','paddle':'paddle-v2','tele':'tele-v2'}
CAPS={'mineru':4,'paddle':2,'tele':2}
DEFAULT_WORKERS={'mineru':2,'paddle':1,'tele':1}
MODELS={'mineru':['MinerU2.5-Pro-2605-1.2B','MinerU-4_models_torch'],'paddle':['paddle','paddle_layout'],'tele':['tele','tele-source']}
ALLOWED=set(filter(None, os.environ.get('HYBRID_GPU_ALLOWLIST', '').split(',')))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def query(cmd):return subprocess.check_output(cmd,text=True).strip()
def read_stage(path):
 try:return read(path)
 except (json.JSONDecodeError,OSError):return None
def artifact(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def verify_terminal(row,pages,key,pred):
 pid=row['page_id'];assert pid in pages and row['runtime_key']==key and row['input_sha256']==pages[pid]['input_sha256']
 assert row['status'] in ('success','failed','truncated')
 path=pred/component_name(pid, '.md');assert path.is_file() and not path.is_symlink() and sha(path)==row['prediction_sha256'] and path.stat().st_size==row['primary_bytes']
 if row['status']!='success':assert path.stat().st_size==0
 assert row['native_artifacts']
 for a in row['native_artifacts']:
  p=Path(a['path']);assert p.resolve().is_relative_to((ROOT/'artifacts/official-full-20260921-v1/native').resolve()) and p.is_file() and not p.is_symlink() and p.stat().st_size==a['bytes'] and sha(p)==a['sha256']
 row['native_artifacts_verified']=True

def acquire_gpu():
 lease_root=ROOT/'receipts/REMOTE_NATIVE_SMOKE_v1'
 active=query(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'])
 inventory=query(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'])
 for line in inventory.splitlines():
  uuid,memory,util=[x.strip() for x in line.split(',')]
  if uuid not in ALLOWED or uuid in active or int(memory)>10 or int(util)!=0:continue
  handle=(lease_root/(uuid+'.lease')).open('a')
  try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:handle.close();continue
  if uuid in query(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']):handle.close();continue
  xml=ET.fromstring(query(['nvidia-smi','-q','-x']));g=next(g for g in xml.findall('gpu') if g.findtext('uuid')==uuid)
  assert g.findtext('mig_mode/current_mig')=='Disabled'
  minor=int(g.findtext('minor_number'));devices=[f'/dev/nvidia{minor}','/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools']
  assert all(Path(x).is_char_device() for x in devices) and os.minor(Path(devices[0]).stat().st_rdev)==minor
  return uuid,devices,handle
 return None

def collect(group,out,arm,key,pred,terminal,exit_record):
 """Only real receipts or immutable started events can become terminal."""
 rows=[];remaining=[];pause=exit_record['state']['OOMKilled'] or exit_record['timeout']=='load'
 common=[p for p in out.iterdir() if p.is_file()]
 for page in group:
  pid=page['page_id'];folder=out/component_name(pid);rp=folder/'receipt.json';started=out/'events'/component_name(pid, '.started.json')
  if not rp.exists() and not started.exists():remaining.append(page);continue
  assert started.exists() and read(started)['page_id']==pid
  receipt=read(rp) if rp.exists() else {'page_id':pid,'input_sha256':page['input_sha256'],'status':'failed','error':'owned_container_'+str(exit_record['timeout'] or 'exited_without_page_receipt')}
  assert receipt['page_id']==pid and receipt['input_sha256']==page['input_sha256']
  if not rp.exists():
   folder.mkdir(exist_ok=True);save(folder/'external-terminal.json',receipt)
   # Preserve active root capture even if process died before per-page finally.
   for name in ['generation-calls.json','effective-generation-configs.json']:
    source=out/name
    if source.exists():
     with (folder/('external-'+name)).open('xb') as f:f.write(source.read_bytes())
  truncated,valid,stops=audit_stops(folder,receipt);raw=folder/'prediction.md'
  returned=receipt['status']=='returned' and raw.is_file() and receipt.get('prediction_sha256')==sha(raw)
  schema=False;schema_proof=None;zero=None;empty=None;zero_error=None
  if returned:
   native_page={**page,'image_path':'/images/'+Path(page['image_path']).name}
   try:
    schema_proof=validate_schema(folder,arm,native_page,Path(page['image_path']));schema=True
    if arm=='mineru':assert sha(raw)==sha(folder/'markdown.md')
   except (AssertionError,KeyError,ValueError,TypeError,OSError) as error:
    schema=False;schema_proof={'valid':False,'error_type':type(error).__name__}
   if schema and arm=='paddle' and not valid:
    try:
     assert read(folder/'generation-calls.json')==[]
     context=paddle_context(folder,native_page,Path(page['image_path']),out/'ASSIGNED.json',sha(Path(__file__).with_name('paddle_full.py')),key)
     if raw.read_bytes()==b'':
      empty=validate_empty(folder,receipt,context)
      save(folder/'EMPTY_RETURN_AUDIT.json',empty)
     else:
      zero=validate_zero(folder,receipt,context);valid=True
    except (AssertionError,KeyError,ValueError,TypeError,OSError) as error:
     zero_error={'valid':False,'error_type':type(error).__name__}

  status='failed' if not returned or not valid or not schema else 'truncated' if truncated else 'success'
  target=pred/component_name(pid, '.md');data=primary_bytes(status,raw.read_bytes() if raw.exists() else b'')
  with target.open('xb') as f:f.write(data)
  paths=[p for p in folder.rglob('*') if p.is_file()]+common+[started]
  row={'page_id':pid,'input_sha256':page['input_sha256'],'runtime_key':key,'status':status,'error':receipt.get('error'),'primary_bytes':len(data),'prediction_sha256':sha(target),'native_artifacts':[artifact(p) for p in sorted(set(paths))],'native_artifacts_verified':True,'native_stop_audit_complete':valid,'schema_present':schema,'schema_proof':schema_proof,'zero_generation_proof':zero,'empty_return_proof':empty,'zero_generation_error':zero_error,'stops':stops,'actual_started':True}
  row['failure_action']=failure_action(row,verify_terminal_timeout(row))
  save(terminal/component_name(pid, '.json'),row);rows.append(row);pause|=bool(receipt.get('arm_paused')) or row['failure_action']=='stop'
 return rows,remaining,pause

def run_shard(group,index,context,gpu):
 arm=context['arm'];out=context['run']/'shards'/f'{index:06d}';out.mkdir(parents=True,exist_ok=False);(out/'events').mkdir()
 uuid,devices,handle=gpu;records=[];mounts=[]
 for row in group:
  source=Path(row['image_path']);assert source.resolve().is_relative_to(Path(context['input_root']).resolve()) and sha(source)==row['input_sha256']
  target='/images/'+source.name;records.append({**row,'image_path':target});mounts.append((source,target,True))
 assigned=out/'ASSIGNED.json';save(assigned,{'pages':records,'runtime_key':context['key']})
 code=context['code'];env=ROOT/'inference/environments'/ENVS[arm];pbs=ROOT/'inference/environments/cpython-3.12.14-20260901'
 mounts += [(code,'/code',True),(env,str(env),True),(pbs,str(pbs),True),(assigned,'/input/manifest.json',True),(out,'/output',False),(ROOT/'inference/driver-bundle-v1/attempt-v2','/driver',True)]
 mounts += [(ROOT/'inference/models'/n,'/task/models/'+n,True) for n in MODELS[arm]]
 if arm=='mineru':
  locks=out/'model-locks';locks.mkdir();mounts.append((locks,'/task/models/.locks',False))
 name=f"full-native-v1-{arm}-{context['run_id']}-{index:06d}"
 assert name not in query(['docker','ps','-a','--format','{{.Names}}']).splitlines()
 cmd=['docker','create','--name',name,'--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--cpus','8','--memory','48g','--pids-limit','512','--shm-size','8g','--tmpfs','/tmp:rw,size=2g']
 for dev in devices:cmd+=['--device',dev+':'+dev]
 for k,v in {'HOME':'/output','HF_HOME':'/output/hf-cache','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','LD_LIBRARY_PATH':'/driver','CUDA_VISIBLE_DEVICES':uuid}.items():cmd+=['--env',k+'='+v]
 for source,target,ro in mounts:cmd+=['--mount',bind_mount(source,target,ro)]
 worker=[str(env/'bin/python'),'-B','/code/'+('paddle_full.py' if arm=='paddle' else 'native_full.py'),'--worker','--root','/task','--manifest','/input/manifest.json','--output','/output']
 if arm!='paddle':worker+=['--arm',arm]
 cmd += [context['image'],*worker]
 assert cmd.count('--device')==4 and len(set(devices))==4 and 'CUDA_VISIBLE_DEVICES='+uuid in cmd
 assert all(ro or source==out or source.parent==out for source,target,ro in mounts)
 assert all(ro for source,target,ro in mounts if target.startswith('/task/models/') and target!='/task/models/.locks')
 assert all('/evaluation/' not in str(source) for source,target,ro in mounts)
 cid=query(cmd);save(out/'START.json',{'cid':cid,'command':cmd,'gpu_uuid':uuid,'runtime_key':context['key'],'load_timeout':600,'page_timeout':900})
 timeout=None;began=time.monotonic();phase_start=began;previous_phase=None;last_stage={'phase':'load'}
 try:
  subprocess.run(['docker','start',cid],check=True,capture_output=True)
  while True:
   state=json.loads(query(['docker','inspect',cid]))[0]
   if not state['State']['Running']:break
   sampled=read_stage(out/'stage.json')
   if sampled is not None:last_stage=sampled
   stage=last_stage
   phase=(stage.get('phase'),stage.get('page_id'))
   if phase!=previous_phase:phase_start=time.monotonic();previous_phase=phase
   loaded=(out/'loaded.json').exists();now=time.monotonic()
   if not loaded and now-began>600:timeout='load';break
   if loaded and phase[0]=='page' and now-phase_start>900:
    timeout='page'
    active=[p for p in group if p['page_id']==phase[1]];assert len(active)==1
    event=out/'events'/component_name(phase[1],'.started.json');assert read(event)['page_id']==phase[1]
    watchdog={'policy':POLICY,'cid':cid,'runtime_key':context['key'],'phase':'page',
              'page_id':phase[1],'input_sha256':active[0]['input_sha256'],
              'page_budget_seconds':900,'phase_elapsed_seconds':now-phase_start,
              'loaded':True,'running_before_stop':state['State']['Running'],
              'started_event_sha256':sha(event),'assigned_sha256':sha(assigned),
              'loaded_sha256':sha(out/'loaded.json')}
    save(out/'WATCHDOG.json',watchdog);break
   if loaded and phase[0]!='page' and now-phase_start>900:timeout='worker_idle';break
   time.sleep(2)
 finally:
  state=json.loads(query(['docker','inspect',cid]))[0]
  kill_succeeded=False
  if state['State']['Running']:
   subprocess.run(['docker','kill',cid],check=True,capture_output=True);kill_succeeded=True
  if state['State'].get('Status')!='created':subprocess.run(['docker','wait',cid],check=True,capture_output=True,timeout=60)
  state=json.loads(query(['docker','inspect',cid]))[0];logs=subprocess.run(['docker','logs',cid],capture_output=True)
  (out/'stdout.log').write_bytes(logs.stdout);(out/'stderr.log').write_bytes(logs.stderr)
  exit_record={'cid':cid,'state':state['State'],'image':state['Image'],'timeout':timeout,'gpu_uuid':uuid,'all_owned_container_processes_stopped':not state['State']['Running']}
  exit_record.update(watchdog_kill_succeeded=kill_succeeded,watchdog_runtime_key=context['key'],
                     watchdog_sha256=sha(out/'WATCHDOG.json') if (out/'WATCHDOG.json').exists() else None)
  save(out/'EXIT.json',exit_record)
  handle.close()
 rows,remaining,pause=collect(group,out,arm,context['key'],context['pred'],context['terminal'],exit_record)
 if not rows:pause=True # no real progress never loops
 if exit_record['timeout']=='page' and sum(verify_terminal_timeout(r) for r in rows)!=1:pause=True
 if exit_record['state']['ExitCode']!=0 and exit_record['timeout']!='page':pause=True
 return rows,remaining,pause
