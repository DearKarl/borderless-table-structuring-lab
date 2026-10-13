"""Explicit GPU backends and sealed host/device evidence; stdlib only until worker guard."""
import ctypes
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET
from .core import ContractError, Deadline, canonical, closed, digest, read, sha, write

def require(ok,message):
    if not ok:raise ContractError(message)

def keys(value,wanted):
    require(isinstance(value,dict) and set(value)==set(wanted),'GPU binding fields differ')

def small_json(path):
    p=Path(path);require(p.is_file() and not p.is_symlink() and p.stat().st_size<=1024**2,'GPU binding file type/size')
    def pairs(rows):
        result={}
        for k,v in rows:
            require(k not in result,'Duplicate GPU binding field');result[k]=v
        return result
    return json.loads(p.read_text(encoding='utf-8'),object_pairs_hook=pairs)

def linux_path(value):
    require(isinstance(value,str) and value.startswith('/') and not any(t in value for t in (',','\x00','\\')),'Explicit Linux path required')
    p=PurePosixPath(value)
    require(str(p)==value and not any(x in ('.','..','') for x in value.split('/')[1:]),'Noncanonical driver path')
    require(not any(x.lower() in ('stubs','compat') for x in p.parts),'Stub/compat driver prohibited')
    return p

def basename(value):
    require(isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,159}',value) is not None,'Invalid driver basename')
    return value

def descriptor(value):
    keys(value,('schema','driver_version','bundle_root','files','aliases','provenance'))
    require(type(value['schema']) is int and value['schema']==1,'Driver descriptor schema')
    require(re.fullmatch(r'[0-9]+(?:\.[0-9]+)+',value['driver_version']) is not None,'Driver version required')
    bundle=linux_path(value['bundle_root'])
    require(bundle!=PurePosixPath('/') and not any(bundle==PurePosixPath(v) or PurePosixPath(v) in bundle.parents for v in ('/dev','/proc','/sys','/var/run','/run')) and 'docker.sock' not in value['bundle_root'],'Unsafe driver bundle root')
    rows=value['files'];require(isinstance(rows,list) and 1<=len(rows)<=16,'Closed driver file list required')
    names=set();targets=set()
    for row in rows:
        keys(row,('soname','target_basename','host_source','sha256'))
        soname=basename(row['soname']);target=basename(row['target_basename']);linux_path(row['host_source']);digest(row['sha256'])
        require(soname not in names and target not in targets,'Duplicate driver identity');names.add(soname);targets.add(target)
    require('libcuda.so.1' in names,'Real libcuda required')
    aliases=value['aliases'];require(isinstance(aliases,dict) and len(aliases)+len(targets)<=32,'Driver alias cap')
    for name,target in aliases.items():
        basename(name);basename(target);require(name not in targets,'Alias replaces locked ELF')
    def endpoint(name):
        seen=set()
        while name in aliases:
            require(name not in seen,'Driver alias cycle');seen.add(name);name=aliases[name]
        require(name in targets,'Driver alias outside closed files');return name
    for name in aliases:endpoint(name)
    for row in rows:require(endpoint(row['soname'])==row['target_basename'],'Soname does not resolve to locked target')
    keys(value['provenance'],('driver_receipt_sha256','freeze_sha256'))
    for v in value['provenance'].values():digest(v)
    return value

def kind(runtime):
    backend=runtime.get('gpu_backend',{'kind':'nvidia'})
    require(isinstance(backend,dict),'GPU backend must be explicit object')
    k=backend.get('kind');require(k in ('nvidia','manual'),'Unknown GPU backend; no fallback')
    keys(backend,('kind',) if k=='nvidia' else ('kind','host_binding'))
    if k=='manual':
        require(runtime.get('schema')==2,'Manual backend requires schema2')
        keys(backend['host_binding'],('path','sha256'));digest(backend['host_binding']['sha256'])
    return k

def validate_backend(runtime,runtime_root):
    if kind(runtime)=='nvidia':return None
    ref=runtime['gpu_backend']['host_binding'];p=closed(runtime_root,ref['path'])
    value=descriptor(small_json(p));require(sha(p)==ref['sha256'],'Driver descriptor SHA differs')
    return value

def copy_backend(config,base,out,allow_absolute=False):
    if 'gpu_backend' not in config:return None
    r={'schema':2,'gpu_backend':config['gpu_backend']}
    if kind(r)=='nvidia':return {'kind':'nvidia'}
    ref=r['gpu_backend']['host_binding'];source=Path(ref['path'])
    source=source.resolve() if allow_absolute and source.is_absolute() else closed(base,ref['path'])
    descriptor(small_json(source));require(sha(source)==ref['sha256'],'Driver descriptor input SHA differs')
    target=Path(out)/'HOST_DRIVER_BINDING.json'
    with target.open('xb') as f:f.write(source.read_bytes())
    require(sha(target)==ref['sha256'],'Copied driver descriptor differs')
    return {'kind':'manual','host_binding':{'path':target.name,'sha256':sha(target)}}

def bounded_text(path,cap=1024**2):
    with Path(path).open('rb') as f:raw=f.read(cap+1)
    require(len(raw)<=cap,'GPU evidence size cap');return raw.decode('utf-8')

def kernel_version():
    raw=bounded_text('/proc/driver/nvidia/version',16384)
    m=re.search(r'NVRM version:.*Kernel Module(?: for [A-Za-z0-9_]+)?\s+([0-9]+(?:\.[0-9]+)+)',raw)
    require(m is not None,'Kernel driver version unknown');return m.group(1)

def elf_hash(path,deadline):
    p=Path(path);before=p.stat();require(stat.S_ISREG(before.st_mode) and before.st_size<=256*1024**2,'Driver must be bounded regular ELF')
    with p.open('rb') as f:require(f.read(4)==b'\x7fELF','Driver is not ELF')
    result=sha(p,deadline);after=p.stat()
    require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Driver changed during read')
    return result

def verify_host_driver(binding,deadline):
    descriptor(binding);deadline.remaining();bundle=Path(binding['bundle_root'])
    require(bundle.resolve()==bundle and bundle.is_dir(),'Driver bundle path changed')
    entries=[]
    for p in bundle.iterdir():
        deadline.remaining();entries.append(p.name);require(len(entries)<=32,'Driver bundle inventory cap')
    expected={r['target_basename'] for r in binding['files']}|set(binding['aliases'])
    require(set(entries)==expected,'Extra/missing driver bundle entry')
    for name,link in binding['aliases'].items():
        p=bundle/name
        require(p.is_symlink() and os.readlink(p)==link and p.resolve().parent==bundle,'Driver alias differs/escapes')
    for row in binding['files']:
        deadline.remaining();target=bundle/row['target_basename'];source=Path(row['host_source'])
        require(not target.is_symlink() and target.resolve().parent==bundle,'Driver target not locked regular file')
        require(source.resolve().parent==source.parent,'Driver source link escapes explicit directory')
        require((bundle/row['soname']).resolve()==target,'Driver soname identity differs')
        for p in (source,target):require(elf_hash(p,deadline)==row['sha256'],'Driver source/target SHA differs: '+row['soname'])
    version=kernel_version();require(version==binding['driver_version'],'Kernel/driver binding version differs')
    return {'driver_version':version,'files':binding['files'],'aliases':binding['aliases'],'bundle_root':binding['bundle_root']}

def bind_host_driver(spec,output):
    value=descriptor(small_json(spec));clock=Deadline(time.monotonic(),300,0)
    verify_host_driver(value,clock);write(output,value)
    return {'path':str(Path(output).resolve()),'sha256':sha(output),'driver_verified':True,'GPU_queries':0,'model_loads':0}

def roles(runtime,mode):
    if runtime.get('v32',{}).get('execution')=='crop':return {'ovis':runtime['gpu_uuids'][0]}
    result={'native':runtime['gpu_uuids'][0]}
    if mode=='on' and 'v32' in runtime:result['ovis']=runtime['gpu_uuids'][1]
    if mode=='on' and runtime.get('v31',{}).get('components'):result['paddle']=runtime['gpu_uuids'][1]
    require(len(set(result.values()))==len(result),'Role UUID collision')
    require(all(isinstance(u,str) and re.fullmatch(r'GPU-[A-Za-z0-9-]+',u) for u in result.values()),'Unsafe role UUID')
    return result

def parse_xml(raw,uuids,ceiling,version):
    require(len(raw.encode())<=1024**2 and '<!ENTITY' not in raw.upper(),'Unsafe/large GPU XML')
    # nvidia-smi emits this external DTD declaration. Never fetch or expand a DTD.
    raw=re.sub(r'<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v[0-9]+\.dtd">','',raw)
    require('<!DOCTYPE' not in raw.upper(),'Unsupported GPU XML DTD')
    root=ET.fromstring(raw);require(root.findtext('driver_version')==version,'XML driver version differs')
    wanted=set(uuids);found={};minors=set()
    def number(text,unit=''):
        require(isinstance(text,str),'Missing GPU number')
        m=re.fullmatch(r'\s*([0-9]+)'+(r'\s*'+re.escape(unit) if unit else '')+r'\s*',text)
        require(m is not None,'Unsupported GPU numeric field');return int(m.group(1))
    for gpu in root.findall('gpu'):
        uid=gpu.findtext('uuid')
        if uid not in wanted:continue
        require(uid not in found,'Duplicate requested GPU UUID')
        minor=number(gpu.findtext('minor_number'));require(minor not in minors,'Duplicate requested GPU minor');minors.add(minor)
        require(gpu.findtext('mig_mode/current_mig')=='Disabled','MIG must be Disabled')
        memory=number(gpu.findtext('fb_memory_usage/used'),'MiB');util=number(gpu.findtext('utilization/gpu_util'),'%')
        processes=gpu.find('processes');require(processes is not None and len(processes)==0 and not (processes.text or '').strip(),'GPU process evidence missing/busy')
        require(util==0 and memory<=ceiling,'GPU utilization/memory busy')
        found[uid]={'uuid':uid,'minor':minor,'memory_mib':memory,'utilization':util,'mig':'Disabled','processes':[],'driver_version':version}
    require(set(found)==wanted,'Requested GPU XML identity missing');return found

def query_xml(uuids,ceiling,version,deadline,runner=None):
    runner=runner or subprocess.run
    r=runner(['nvidia-smi','-q','-x'],check=True,text=True,capture_output=True,timeout=deadline.bound(10))
    return {'observed_monotonic':time.monotonic(),'gpus':parse_xml(r.stdout,uuids,ceiling,version),'xml_sha256':canonical(r.stdout)}

def uvm_major():
    matches=[];character=False
    for line in bounded_text('/proc/devices',65536).splitlines():
        if line=='Character devices:':character=True
        elif line=='Block devices:':character=False
        elif character:
            parts=line.split()
            if len(parts)==2 and parts[1]=='nvidia-uvm' and parts[0].isdigit():matches.append(int(parts[0]))
    require(len(matches)==1,'Unknown UVM char major');return matches[0]

def device_stats(minor):
    major=uvm_major();result=[]
    for path,ma,mi in [(f'/dev/nvidia{minor}',195,minor),('/dev/nvidiactl',195,255),('/dev/nvidia-uvm',major,0),('/dev/nvidia-uvm-tools',major,1)]:
        p=Path(path);s=p.lstat()
        require(stat.S_ISCHR(s.st_mode) and not p.is_symlink() and (os.major(s.st_rdev),os.minor(s.st_rdev))==(ma,mi),'GPU character device identity differs: '+path)
        result.append({'path':path,'major':ma,'minor':mi})
    return result

def validate_devices(devices,minor):
    require(type(minor) is int and 0<=minor<255 and isinstance(devices,list) and len(devices)==4,'Device binding count/minor')
    require(all(set(x)=={'path','major','minor'} and type(x['major']) is int and type(x['minor']) is int for x in devices),'Device stat fields')
    expected=[{'path':f'/dev/nvidia{minor}','major':195,'minor':minor},{'path':'/dev/nvidiactl','major':195,'minor':255},
              {'path':'/dev/nvidia-uvm','major':devices[2]['major'],'minor':0},{'path':'/dev/nvidia-uvm-tools','major':devices[2]['major'],'minor':1}]
    require(devices==expected and devices[2]['major']>0,'Device binding differs')

def resolve_manual(control,runtime_root,out,identities,deadline):
    r=control['runtime'];binding=validate_backend(r,runtime_root);require(binding is not None,'Manual binding missing')
    driver=verify_host_driver(binding,deadline);mapping=roles(r,control['mode'])
    xml=query_xml(list(mapping.values()),r['idle_memory_mib'],binding['driver_version'],deadline)
    record={'schema':1,'backend':'manual','run_id':control['run_id'],'host_binding_sha256':r['gpu_backend']['host_binding']['sha256'],
            'driver':driver,'initial_xml':xml,'roles':{}}
    for role,uid in mapping.items():
        minor=xml['gpus'][uid]['minor'];devices=device_stats(minor);validate_devices(devices,minor)
        record['roles'][role]={'run_id':control['run_id'],'role':role,'uuid':uid,'minor':minor,'devices':devices,
                               'image_identity':identities['roles'][role],'driver':driver}
    source=closed(runtime_root,r['gpu_backend']['host_binding']['path'])
    with (Path(out)/'HOST_DRIVER_BINDING.json').open('xb') as f:f.write(source.read_bytes())
    require(sha(Path(out)/'HOST_DRIVER_BINDING.json')==record['host_binding_sha256'],'Sealed descriptor changed')
    write(Path(out)/'GPU_BINDINGS.json',record)
    return record

def fresh_role(control,binding,deadline):
    r=control['runtime'];uid=binding['uuid']
    xml=query_xml([uid],r['idle_memory_mib'],binding['driver']['driver_version'],deadline)
    require(xml['gpus'][uid]['minor']==binding['minor'] and device_stats(binding['minor'])==binding['devices'],'Fresh GPU mapping changed')
    require(kernel_version()==binding['driver']['driver_version'],'Fresh kernel changed')
    return xml

def seal_fresh(out,control,binding,xml,cid,owner,root):
    value={'schema':1,'run_id':control['run_id'],'role':binding['role'],'cid':cid,'owner':owner,'uuid':binding['uuid'],
           'binding_sha256':canonical(binding),'gpu_bindings_sha256':sha(Path(root)/'GPU_BINDINGS.json'),'xml':xml,
           'verified_before_start_monotonic':time.monotonic()}
    require(0<=value['verified_before_start_monotonic']-xml['observed_monotonic']<=240,'Stale start GPU identity')
    write(Path(out)/'GPU_BEFORE_START.json',value);return value

def manual_environment(binding):
    return {'CUDA_VISIBLE_DEVICES':binding['uuid'],'CUDA_DEVICE_ORDER':'PCI_BUS_ID','NVIDIA_VISIBLE_DEVICES':'void',
            'NVIDIA_DRIVER_CAPABILITIES':'','LD_LIBRARY_PATH':'/driver'}

def launch_flags(runtime,uid,binding=None):
    if kind(runtime)=='nvidia':
        require(binding is None,'Unexpected manual binding');return ['--gpus','device='+uid]
    require(binding is not None and binding['uuid']==uid,'Missing/mismatched manual binding')
    validate_devices(binding['devices'],binding['minor']);args=['--runtime','runc']
    for d in binding['devices']:args+=['--device',d['path']+':'+d['path']+':rwm']
    for k,v in manual_environment(binding).items():
        if k!='CUDA_DEVICE_ORDER':args+=['--env',k+'='+v]
    return args

def driver_mount(binding):return (binding['driver']['bundle_root'],'/driver',False)

def verify_container(state,binding):
    require(binding is not None,'Manual container requires role binding');validate_devices(binding['devices'],binding['minor'])
    hc=state['HostConfig'];devices=[{'PathOnHost':d['path'],'PathInContainer':d['path'],'CgroupPermissions':'rwm'} for d in binding['devices']]
    require(hc.get('Runtime')=='runc' and hc.get('Devices')==devices and not hc.get('DeviceRequests') and not hc.get('DeviceCgroupRules'),'Actual manual devices/runtime differ')
    require(not hc.get('Privileged') and hc.get('PidMode')!='host' and hc.get('IpcMode')!='host' and not hc.get('Annotations'),'Extra manual isolation configuration')
    require(hc.get('CapDrop')==['ALL'] and 'no-new-privileges' in (hc.get('SecurityOpt') or []) and hc.get('PidsLimit')==4096,'Manual capability/pid restrictions differ')
    require(set(hc.get('Tmpfs') or {})=={'/tmp'} and 'size=1g' in hc['Tmpfs']['/tmp'],'Manual tmpfs differs')
    values={}
    for entry in state['Config'].get('Env') or []:
        k,sep,v=entry.partition('=');require(sep and k not in values,'Duplicate/malformed container environment')
        require(not any(t in k.upper() for t in ('MIG','IMEX','CDI','MPS')) and k!='LD_PRELOAD','Extra GPU/loader injection environment')
        require(not k.startswith('NVIDIA_') or k in ('NVIDIA_VISIBLE_DEVICES','NVIDIA_DRIVER_CAPABILITIES') or k.startswith('NVIDIA_REQUIRE_'),'Unknown NVIDIA environment')
        values[k]=v
    require(all(values.get(k)==v for k,v in manual_environment(binding).items()),'Effective manual environment differs')
    mounts=[m for m in state['Mounts'] if m.get('Destination')=='/driver' or m.get('Destination','').startswith('/driver/')]
    require(len(mounts)==1 and mounts[0].get('Type')=='bind' and mounts[0].get('Source')==binding['driver']['bundle_root'] and mounts[0].get('RW') is False,'Actual driver bind differs')

def actual_gpu_uuid(expected,lib=None):
    class Uuid(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
    lib=lib or ctypes.CDLL('libcuda.so.1')
    for name,args in [('cuInit',[ctypes.c_uint]),('cuDeviceGetCount',[ctypes.POINTER(ctypes.c_int)]),('cuDeviceGet',[ctypes.POINTER(ctypes.c_int),ctypes.c_int]),('cuDeviceGetUuid',[ctypes.POINTER(Uuid),ctypes.c_int])]:
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=ctypes.c_int
    require(lib.cuInit(0)==0,'CUDA driver initialization failed')
    count=ctypes.c_int();require(lib.cuDeviceGetCount(ctypes.byref(count))==0 and count.value==1,'Expected exactly one visible GPU')
    device=ctypes.c_int();require(lib.cuDeviceGet(ctypes.byref(device),0)==0,'CUDA device unavailable')
    value=Uuid();require(lib.cuDeviceGetUuid(ctypes.byref(value),device)==0,'CUDA UUID query failed')
    actual='GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)))
    require(actual.lower()==expected.lower(),'Actual CUDA UUID differs');return actual

def worker_devices(binding,deadline):
    expected={d['path'] for d in binding['devices']};observed=set();count=0;majors={195,binding['devices'][2]['major']}
    character=False
    for line in bounded_text('/proc/devices',65536).splitlines():
        if line=='Character devices:':character=True
        elif line=='Block devices:':character=False
        elif character:
            fields=line.split()
            if len(fields)==2 and fields[0].isdigit() and 'nvidia' in fields[1].lower():majors.add(int(fields[0]))
    for directory,dirs,files in os.walk('/dev',followlinks=False):
        require(len(Path(directory).parts)<=8,'Device inventory depth')
        for name in dirs+files:
            deadline.remaining();count+=1;require(count<=4096,'Device inventory cap');p=Path(directory)/name
            try:s=p.stat()
            except FileNotFoundError:
                if p.is_symlink():continue
                raise
            if stat.S_ISDIR(s.st_mode):continue
            related=('nvidia' in str(p).lower() or str(p).startswith(('/dev/dri/','/dev/vfio/')) or str(p)=='/dev/kfd'
                     or (stat.S_ISCHR(s.st_mode) and os.major(s.st_rdev) in majors))
            if related:
                require(str(p) in expected and not p.is_symlink(),'Unexpected compute/NVIDIA device');observed.add(str(p))
    require(observed==expected,'Missing allowed NVIDIA device');actual=device_stats(binding['minor'])
    require(actual==binding['devices'],'Worker device major/minor differs');return actual

def loaded_driver(binding,deadline):
    row=next(r for r in binding['driver']['files'] if r['soname']=='libcuda.so.1')
    expected='/driver/'+row['target_basename']
    lib=ctypes.CDLL('/driver/libcuda.so.1',mode=os.RTLD_NOW|os.RTLD_LOCAL)
    mapped=set()
    for line in bounded_text('/proc/self/maps').splitlines():
        parts=line.split(maxsplit=5)
        if len(parts)==6 and Path(parts[5]).name.startswith('libcuda.so'):
            require(not parts[5].endswith(' (deleted)'),'Deleted CUDA mapping');mapped.add(str(Path(parts[5]).resolve()))
    require(mapped=={expected} and elf_hash(expected,deadline)==row['sha256'],'Loaded CUDA driver identity differs')
    mounts=[]
    for line in bounded_text('/proc/self/mountinfo').splitlines():
        parts=line.split();point=parts[4]
        if point=='/driver' or point.startswith('/driver/'):mounts.append({'mountpoint':point,'read_only':'ro' in parts[5].split(',')})
    require(mounts==[{'mountpoint':'/driver','read_only':True}],'Worker driver mount not exclusively read-only')
    require(kernel_version()==binding['driver']['driver_version'],'Worker kernel version differs')
    return lib,{'realpath':expected,'sha256':row['sha256'],'mounts':mounts,'driver_version':binding['driver']['driver_version']}

def worker_guard(control,role,out=Path('/output')):
    if kind(control['runtime'])!='manual':
        require('gpu_binding_schema' not in control,'Unexpected manual marker');return None
    require(control.get('gpu_binding_schema')==1,'Missing worker manual marker')
    out=Path(out);all_bindings=small_json(out/'GPU_BINDINGS.json');host=descriptor(small_json(out/'HOST_DRIVER_BINDING.json'))
    validate_bindings(control,all_bindings,host,sha(out/'HOST_DRIVER_BINDING.json'))
    binding=all_bindings['roles'][role];deadline=Deadline(time.monotonic(),max(.001,control['absolute_stop_monotonic']-time.monotonic()),0)
    devices=worker_devices(binding,deadline);lib,driver=loaded_driver(binding,deadline);actual=actual_gpu_uuid(binding['uuid'],lib)
    record={'schema':1,'run_id':control['run_id'],'role':role,'expected_uuid':binding['uuid'],'actual_uuid':actual,
            'devices':devices,'driver':driver,'binding_sha256':canonical(binding),'host_binding_sha256':sha(out/'HOST_DRIVER_BINDING.json'),
            'gpu_bindings_sha256':sha(out/'GPU_BINDINGS.json'),'observed_monotonic':time.monotonic()}
    write(out/('experts/' if role in ('paddle','ovis') else '')/'GPU_BINDING_CHECK.json',record);return record

def validate_bindings(control,record,host,host_sha):
    descriptor(host);r=control['runtime'];require(kind(r)=='manual','Sealed manual kind differs')
    require(record.get('schema')==1 and record.get('backend')=='manual' and record.get('run_id')==control['run_id'],'GPU bindings schema/run')
    require(record.get('host_binding_sha256')==host_sha==r['gpu_backend']['host_binding']['sha256'],'Sealed host binding SHA differs')
    driver={'driver_version':host['driver_version'],'files':host['files'],'aliases':host['aliases'],'bundle_root':host['bundle_root']}
    require(record.get('driver')==driver,'Sealed driver proof differs');mapping=roles(r,control['mode'])
    require(set(record['roles'])==set(mapping),'Effective manual roles differ')
    xml=record['initial_xml'];require(set(xml['gpus'])==set(mapping.values()),'Initial XML role coverage differs');digest(xml['xml_sha256'])
    minors=set()
    for role,uid in mapping.items():
        row=record['roles'][role];require(row['run_id']==control['run_id'] and row['role']==role and row['uuid']==uid and row['driver']==driver,'GPU binding role/driver differs')
        validate_devices(row['devices'],row['minor']);require(row['minor'] not in minors,'Sealed duplicate minor');minors.add(row['minor'])
        validate_xml_summary(xml['gpus'][uid],row,r['idle_memory_mib'])

def validate_xml_summary(value,binding,ceiling):
    require(value.get('uuid')==binding['uuid'] and value.get('minor')==binding['minor'] and value.get('driver_version')==binding['driver']['driver_version'],'XML GPU binding differs')
    require(value.get('mig')=='Disabled' and value.get('processes')==[] and type(value.get('utilization')) is int and value['utilization']==0 and type(value.get('memory_mib')) is int and 0<=value['memory_mib']<=ceiling,'Sealed GPU idleness invalid')

def required_evidence(control):
    manual=kind(control['runtime'])=='manual'
    if not manual:
        require('gpu_binding_schema' not in control,'Manual marker on NVIDIA backend');return []
    return ['HOST_DRIVER_BINDING.json','GPU_BINDINGS.json']+[prefix+n for role in roles(control['runtime'],control['mode']) for prefix in [('experts/' if role in ('paddle','ovis') else '')] for n in ('GPU_BEFORE_START.json','GPU_BINDING_CHECK.json')]

def validate_evidence(root,control,run):
    root=Path(root);manual=kind(control['runtime'])=='manual'
    if not manual:
        require('gpu_binding_schema' not in control and 'gpu_binding_schema' not in run,'Unexpected manual marker');return
    require(type(control.get('gpu_binding_schema')) is int and control['gpu_binding_schema']==1 and type(run.get('gpu_binding_schema')) is int and run['gpu_binding_schema']==1,'Manual audit markers absent/differ')
    host=descriptor(small_json(root/'HOST_DRIVER_BINDING.json'));record=small_json(root/'GPU_BINDINGS.json');host_sha=sha(root/'HOST_DRIVER_BINDING.json')
    validate_bindings(control,record,host,host_sha);all_sha=sha(root/'GPU_BINDINGS.json');images=read(root/'IMAGE_IDENTITIES.json')['roles']
    require(run['run_id']==control['run_id'],'Manual run identity differs')
    for role,binding in record['roles'].items():
        folder=root/('experts' if role in ('paddle','ovis') else '');fresh=read(folder/'GPU_BEFORE_START.json');check=read(folder/'GPU_BINDING_CHECK.json')
        created=read(folder/'CONTAINER_CREATED.json');exit_state=read(folder/'CONTAINER_EXIT.json');owner=control['run_id']+('-'+role if role in ('paddle','ovis') else '')
        require(created['owner']==owner and created['cid']==exit_state['Id'] and exit_state['Config']['Labels'].get('hybrid-v3-owner')==owner,'Manual owner/CID differs')
        require(binding['image_identity']==images[role],'Manual image binding differs')
        from .image_identity import verify_container_image
        reference=control['runtime']['image'] if role=='native' else control['runtime']['environments'][role]['image']
        verify_container_image(exit_state,reference,images[role]);verify_container(exit_state,binding)
        # Recover requested mounts from the sealed creation record, never from current host files.
        args=created['arguments'];mounts=[]
        for i,token in enumerate(args):
            if token=='--mount':
                fields=args[i+1].split(',');parts=dict(x.split('=',1) for x in fields if '=' in x)
                require(parts.get('type')=='bind','Unexpected sealed mount type')
                mounts.append((parts['src'],parts['dst'],'readonly' not in fields))
        budget=control['budget'] if role=='native' else {**control['budget'],'cpus':8,'ram_gib':48 if 'formula' in control['runtime'].get('v31',{}).get('components',[]) else 16,'swap_gib':0}
        from .lifecycle import verify_isolation
        verify_isolation(exit_state,mounts,budget,reference,[binding['uuid']],images[role],backend_binding=binding)
        for proof in (fresh,check):
            require(proof.get('schema')==1 and proof.get('run_id')==control['run_id'] and proof.get('role')==role and proof.get('binding_sha256')==canonical(binding) and proof.get('gpu_bindings_sha256')==all_sha,'Manual per-role proof differs')
        require(fresh['cid']==created['cid'] and fresh['owner']==owner and fresh['uuid']==binding['uuid'],'Fresh CID/role differs')
        require(set(fresh['xml']['gpus'])=={binding['uuid']},'Fresh query must concern only its role');digest(fresh['xml']['xml_sha256'])
        validate_xml_summary(fresh['xml']['gpus'][binding['uuid']],binding,control['runtime']['idle_memory_mib'])
        a=fresh['xml']['observed_monotonic'];b=fresh['verified_before_start_monotonic'];observed=check['observed_monotonic']
        require(all(type(t) in (float,int) and math.isfinite(t) for t in (a,b,observed)) and control['host_started_monotonic']<=record['initial_xml']['observed_monotonic']<=a<=b<=observed<=control['absolute_stop_monotonic'] and b-a<=240,'Fresh/start/check clock invalid')
        row=next(r for r in host['files'] if r['soname']=='libcuda.so.1')
        expected_driver={'realpath':'/driver/'+row['target_basename'],'sha256':row['sha256'],'mounts':[{'mountpoint':'/driver','read_only':True}],'driver_version':host['driver_version']}
        require(check['expected_uuid']==binding['uuid'] and check['actual_uuid'].lower()==binding['uuid'].lower() and check['devices']==binding['devices'] and check['driver']==expected_driver and check['host_binding_sha256']==host_sha,'Worker actual GPU/driver differs')
