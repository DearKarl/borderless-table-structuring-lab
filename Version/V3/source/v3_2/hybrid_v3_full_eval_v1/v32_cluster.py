"""Explicit cluster-only Ovis environment, with exact read-only mount evidence."""
import os
import math
from pathlib import Path, PurePosixPath
from .core import ContractError, canonical, digest, read, sha

BASE='sha256:787dd9a15d41c442e78a5faa9cdc721f154cbf3d4cf8624edb41026c3052b75a'
PREPARED='4d6ffa7de9027e121e722463e2adaff382a1bfe74c2942d8d00284f3ea3f1a24'
PYTHON='34a1af5926cf8b0c5c361d6d7e87817bc39405f88c9a57b5d58d4488687b1add'

def require(ok,message):
    if not ok:raise ContractError(message)

def bound_budget(runtime,budget,start):
    if 'v32' not in runtime:return budget
    end=runtime['v32'].get('host_absolute_deadline')
    require(type(end) in (int,float) and math.isfinite(end),'Original cumulative host deadline required')
    total=min(budget['total_seconds'],end-start)
    require(total>budget['cleanup_seconds'],'No cumulative execution budget remains')
    return {**budget,'total_seconds':total}

def absolute(value,allow_comma=False):
    require(isinstance(value,str) and value.startswith('/') and not any(x in value for x in ('\\','\n','\r','\x00'))
            and (allow_comma or ',' not in value)
            and '..' not in value.split('/') and str(PurePosixPath(value))==value,'Invalid cluster path')
    return value

def descriptor(env):
    require(env.get('kind')=='cluster_ro_bind','Explicit cluster environment kind required')
    ref=env['host_environment_binding'];d=ref['descriptor']
    absolute(ref['path']);digest(ref['sha256'])
    require(canonical(d)==ref['descriptor_sha256'],'Cluster descriptor changed')
    require(d.get('schema')==1 and d.get('base_image')==BASE==env['image'],'Cluster base differs')
    root=absolute(d['runtime_root']);venv=root+'/venv'
    require(root.endswith('/v32-ovis-text/runtime-003'),'Only bound runtime003 supported')
    require(d['prepared']=={'path':root+'/PREPARED.json','sha256':PREPARED,'installed_files':82431},'Prepared identity differs')
    require(env['python']==venv+'/bin/python' and env['resolved_python']=='/usr/bin/python3.12'
            and env['site_packages']==venv+'/lib/python3.12/site-packages','Cluster interpreter paths differ')
    require(env['image_files'].get('/usr/bin/python3.12')==PYTHON,'Cluster Python identity differs')
    require(d['files']==env['image_files'] and d['links']==env.get('image_links',{}),'Mounted file lock differs')
    require(d['packages']==env['packages'],'Mounted package lock differs')
    dirs={venv,'/usr/lib/python3.12','/usr/lib/gcc/x86_64-linux-gnu/11','/usr/include'}
    targets=set();directory_targets=set()
    for m in d['mounts']:
        source=absolute(m['source']);target=absolute(m['destination'])
        require(target not in targets and m['read_only'] is True,'Duplicate/writable cluster mount');targets.add(target)
        if m['type']=='directory':
            require(source==target and target in dirs,'Broad/unapproved host directory mount');directory_targets.add(target)
        elif source==target=='/etc/python3.12/sitecustomize.py':
            require(m['type']=='file' and m.get('sha256')=='e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
                    and d['files'].get(target)==m['sha256'],'Only frozen empty sitecustomize file allowed')
        else:
            require(m['type']=='file' and target.startswith(('/usr/bin/','/usr/lib/x86_64-linux-gnu/')),'Unapproved cluster leaf')
            require(source.startswith(('/usr/bin/','/usr/lib/x86_64-linux-gnu/')),'Unapproved cluster leaf source')
            require('/' not in target.removeprefix('/usr/bin/') if target.startswith('/usr/bin/') else '/' not in target.removeprefix('/usr/lib/x86_64-linux-gnu/'),'Cluster leaf must be direct child')
            require(not Path(target).name.startswith(('libc.so.','libm.so.','ld-linux','libz.so.','libexpat.so.')),'Base runtime library must remain untouched')
            require(d['files'].get(target)==m['sha256'],'Cluster leaf hash missing');digest(m['sha256'])
    require(directory_targets==dirs and '/usr/bin/python3.12' in targets,'Incomplete cluster support mounts')
    for name,h in d['files'].items():
        covered=isinstance(name,str) and any(name.startswith(p+'/') for p in directory_targets)
        absolute(name,allow_comma=covered);digest(h)
        require(name in targets or any(name.startswith(p+'/') for p in dirs),'File outside mounted support')
    require(d['files'] and len(d['files'])<=200000,'Cluster file inventory bound')
    require(d.get('include_system_site_packages') is False,'Host site packages are not isolated')
    return d

def mounts(env,deadline=None):
    d=descriptor(env);ref=env['host_environment_binding']
    require(sha(ref['path'],deadline)==ref['sha256'] and read(ref['path'])==d,'Cluster binding source changed')
    require(sha(d['prepared']['path'],deadline)==PREPARED,'Original PREPARED changed')
    result=[]
    for m in d['mounts']:
        if deadline:deadline.remaining()
        p=Path(m['source'])
        require(str(p.resolve())==m['source'],'Cluster source resolution changed')
        require(p.is_dir() if m['type']=='directory' else p.is_file(),'Missing cluster source')
        result.append((p,m['destination'],False))
    result.append((Path(ref['path']),'/cluster-environment.json',False))
    return result

def environment(env):
    d=descriptor(env)
    return {'PATH':d['runtime_root']+'/venv/bin:/usr/bin:/bin','PYTHONNOUSERSITE':'1',
            'TRITON_CACHE_DIR':'/output/experts/.cache/triton','XDG_CACHE_HOME':'/output/experts/.cache',
            'VLLM_USE_FLASHINFER_SAMPLER':'0'}

def worker_boundary(env):
    d=descriptor(env);ref=env['host_environment_binding']
    require(sha('/cluster-environment.json')==ref['sha256'] and read('/cluster-environment.json')==d,'Worker cluster descriptor differs')
    effective={}
    for k,v in environment(env).items():
        actual=os.environ.get(k);require(actual==v,'Cluster effective environment differs: '+k);effective[k]=actual
    cfg=Path(d['runtime_root']+'/venv/pyvenv.cfg').read_text()
    entries=dict(line.split('=',1) for line in cfg.splitlines() if '=' in line)
    entries={k.strip():v.strip().lower() for k,v in entries.items()}
    require(entries.get('include-system-site-packages')=='false','Venv admits host site packages')
    return {'source':'mounted_host_environment','binding_sha256':ref['sha256'],'descriptor_sha256':canonical(d),
            'effective_environment':effective}


def normalize_mount_evidence(evidence):
    """Normalize order only; preserve every field and reject duplicate targets."""
    require(isinstance(evidence,dict) and isinstance(evidence.get('mounts'),list),'Invalid cluster mount evidence')
    rows=[];destinations=set()
    for row in evidence['mounts']:
        require(isinstance(row,dict) and set(row)=={'source','destination','read_only'},'Invalid cluster mount row fields')
        absolute(row['source']);absolute(row['destination'])
        require(row['read_only'] is True,'Cluster mount is not strictly read-only')
        require(row['destination'] not in destinations,'Duplicate cluster mount destination')
        destinations.add(row['destination']);rows.append(dict(row))
    return {**evidence,'mounts':sorted(rows,key=lambda row:row['destination'])}

def mount_evidence(state,env):
    d=descriptor(env);ref=env['host_environment_binding']
    expected=[{'source':m['source'],'destination':m['destination'],'read_only':True} for m in d['mounts']]
    expected.append({'source':ref['path'],'destination':'/cluster-environment.json','read_only':True})
    targets={m['destination'] for m in expected}
    actual=[{'source':m['Source'],'destination':m['Destination'],'read_only':not m['RW']} for m in state['Mounts'] if m['Destination'] in targets]
    require(sorted(actual,key=lambda x:x['destination'])==sorted(expected,key=lambda x:x['destination']),'Cluster actual mounts differ')
    wanted=environment(env);effective={};config=state.get('Config')
    require(isinstance(config,dict) and isinstance(config.get('Env'),list),'Cluster actual Env must be a list')
    for item in config['Env']:
        require(isinstance(item,str),'Cluster actual Env entry must be a string')
        key,separator,value=item.partition('=')
        if key in wanted:
            require(separator=='=' and key not in effective and value==wanted[key],'Cluster actual environment differs: '+key)
            effective[key]=value
    require(set(effective)==set(wanted),'Cluster actual environment key missing')
    return normalize_mount_evidence({'source':'mounted_host_environment','binding_sha256':ref['sha256'],'descriptor_sha256':canonical(d),
            'cid':state['Id'],'mounts':actual,'effective_environment':effective})

def asset_root(runtime,root,name):
    from .core import closed
    a=runtime['assets'][name]
    if a.get('kind')=='cluster_ro_bind':
        require(name=='ovis_model' and 'v32' in runtime,'External asset only allowed for Ovis')
        d=descriptor(runtime['environments']['ovis'])
        require(a['path']==d['runtime_root']+'/model','External Ovis model source differs')
        p=Path(a['path']);require(str(p.resolve())==a['path'],'External Ovis model path changed');return p
    return closed(root,a['path'])
