"""Synthetic production-call-chain tests; never execute CUDA, Docker or models."""
import contextlib
import copy
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch,MagicMock
from . import gpu_backend as g
from .core import ContractError,Deadline,canonical,read,sha,write
from .image_identity_checks import Checks as ImageChecks,LaunchFake,identity,A,REPO,B,result
from .lifecycle import OwnedDocker,verify_isolation
from .completion import normal_completion,lifecycle_lock
ROOT=None

def host_descriptor():
    return {'schema':1,'driver_version':'999.1','bundle_root':'/synthetic/driver',
        'files':[{'soname':'libcuda.so.1','target_basename':'libcuda.so.999.1','host_source':'/synthetic/lib/libcuda.so.999.1','sha256':'a'*64}],
        'aliases':{'libcuda.so.1':'libcuda.so.999.1','libcuda.so':'libcuda.so.1'},
        'provenance':{'driver_receipt_sha256':'b'*64,'freeze_sha256':'c'*64}}

def driver_proof(desc):return {k:desc[k] for k in ('driver_version','files','aliases','bundle_root')}
def devices(minor):return [{'path':f'/dev/nvidia{minor}','major':195,'minor':minor},{'path':'/dev/nvidiactl','major':195,'minor':255},{'path':'/dev/nvidia-uvm','major':510,'minor':0},{'path':'/dev/nvidia-uvm-tools','major':510,'minor':1}]
def xml_summary(uid,minor):return {'uuid':uid,'minor':minor,'memory_mib':0,'utilization':0,'mig':'Disabled','processes':[],'driver_version':'999.1'}
def xml_text():return '<nvidia_smi_log><driver_version>999.1</driver_version><gpu id="ordinal-zero"><uuid>GPU-b</uuid><minor_number>7</minor_number><mig_mode><current_mig>Disabled</current_mig></mig_mode><fb_memory_usage><used>0 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes/></gpu><gpu id="ordinal-one"><uuid>GPU-a</uuid><minor_number>3</minor_number><mig_mode><current_mig>Disabled</current_mig></mig_mode><fb_memory_usage><used>0 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes/></gpu></nvidia_smi_log>'

class ManualFake(LaunchFake):
    def __init__(self,fault='normal'):super().__init__();self.manual_fault=fault
    def __call__(self,args,timeout,**kw):
        if args[1]=='start' and self.manual_fault=='start':raise subprocess.TimeoutExpired(args,timeout)
        r=super().__call__(args,timeout,**kw)
        if args[1]=='create':
            obj=self.objects[r.stdout];hc=obj['HostConfig']
            hc.update(Runtime='runc',Devices=[],CapDrop=['ALL'],SecurityOpt=['no-new-privileges'],PidsLimit=4096,Tmpfs={'/tmp':'rw,size=1g'})
            obj['Config']['Env']=[args[i+1] for i,v in enumerate(args) if v=='--env']
            for i,v in enumerate(args):
                if v=='--device':
                    src,dst,perm=args[i+1].split(':');hc['Devices'].append({'PathOnHost':src,'PathInContainer':dst,'CgroupPermissions':perm})
            if self.manual_fault=='device':hc['Devices'][0]['CgroupPermissions']='r'
            if self.manual_fault=='create':raise subprocess.TimeoutExpired(args,timeout)
        if args[1]=='start' and self.manual_fault=='worker':self.objects[args[-1]]['State']['ExitCode']=2
        return r

class Checks(unittest.TestCase):
    manifest=ImageChecks.manifest
    run_fixture=ImageChecks.run_fixture
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix='manual-',dir=ROOT))
        import os
        if os.name=='nt' and not str(self.root).startswith('\\\\?\\'):self.root=Path('\\\\?\\'+str(self.root.resolve()))
    def clock(self):return Deadline(time.monotonic(),600,240)
    def binding(self):
        return {'run_id':'fixture','role':'native','uuid':'GPU-a','minor':3,'devices':devices(3),'image_identity':identity(),'driver':driver_proof(host_descriptor())}
    def runtime(self):
        write(self.root/'HOST_DRIVER_BINDING.json',host_descriptor())
        return {'schema':2,'gpu_backend':{'kind':'manual','host_binding':{'path':'HOST_DRIVER_BINDING.json','sha256':sha(self.root/'HOST_DRIVER_BINDING.json')}}}
    def test_explicit_backend_and_schema(self):
        self.assertEqual(g.launch_flags({},'GPU-a'),['--gpus','device=GPU-a'])
        self.assertEqual(g.launch_flags({'gpu_backend':{'kind':'nvidia'}},'GPU-a'),['--gpus','device=GPU-a'])
        r=self.runtime();self.assertEqual(g.validate_backend(r,self.root),host_descriptor())
        for bad in ({'schema':1,'gpu_backend':r['gpu_backend']},{'gpu_backend':{'kind':'auto'}},{'gpu_backend':{'kind':'nvidia','argv':[]}}):
            with self.assertRaises(ContractError):g.kind(bad)
        with self.assertRaises(ContractError):g.launch_flags(r,'GPU-a')
    def test_descriptor_closed_fields_alias_and_paths(self):
        g.descriptor(host_descriptor())
        for fault in ('extra','root','traversal','comma','stub','absolutealias','cycle','target','duplicate','hash'):
            d=host_descriptor()
            if fault=='extra':d['argv']=[]
            if fault=='root':d['bundle_root']='/dev'
            if fault=='traversal':d['bundle_root']='/synthetic/../driver'
            if fault=='comma':d['bundle_root']='/synthetic/a,b'
            if fault=='stub':d['files'][0]['host_source']='/stubs/libcuda.so'
            if fault=='absolutealias':d['aliases']['libcuda.so']='/other'
            if fault=='cycle':d['aliases']['libcuda.so.1']='libcuda.so'
            if fault=='target':d['aliases']['libcuda.so']='unknown'
            if fault=='duplicate':d['files'].append(copy.deepcopy(d['files'][0]))
            if fault=='hash':d['files'][0]['sha256']='0'
            with self.subTest(fault=fault),self.assertRaises(ContractError):g.descriptor(d)
    def test_contract_verify_false_still_requires_manual_binding(self):
        fixture=ImageChecks('test_schema2_full_contract_and_environment_accept_local');fixture.root=self.root
        r,b=fixture.runtime_fixture(2,A);write(self.root/'HOST_DRIVER_BINDING.json',host_descriptor())
        r['gpu_backend']={'kind':'manual','host_binding':{'path':'HOST_DRIVER_BINDING.json','sha256':sha(self.root/'HOST_DRIVER_BINDING.json')}}
        (self.root/'runtime.json').write_text(json.dumps(r))
        from .contracts import runtime
        self.assertEqual(runtime(self.root/'runtime.json','native',b,verify=False)['gpu_backend'],r['gpu_backend'])
        (self.root/'HOST_DRIVER_BINDING.json').unlink()
        with self.assertRaises(ContractError):runtime(self.root/'runtime.json','native',b,verify=False)
    def test_copy_descriptor_only_no_host_process(self):
        r=self.runtime();out=self.root/'prepared';out.mkdir();bound=self.root/'bound';bound.mkdir()
        with patch.object(g.subprocess,'run',side_effect=AssertionError('No subprocess')):
            backend=g.copy_backend(r,self.root,out);second=g.copy_backend({'gpu_backend':backend},out,bound)
            self.assertEqual(g.validate_backend({'schema':2,'gpu_backend':second},bound),host_descriptor())
        self.assertEqual([p.name for p in bound.iterdir()],['HOST_DRIVER_BINDING.json'])
    def test_prepare_bind_preflight_complete_synthetic_assets(self):
        from . import asset_binder as binder,asset_binding as assets,contracts,runner
        original_vendor=binder.VENDOR;vendor=self.root/'vendor';shutil.copytree(original_vendor,vendor)
        model=self.root/'model';model.mkdir();write(model/'tokenizer_config.json',{'eos_token':'<end>'});(model/'weights.fixture').write_bytes(b'Synthetic non-model')
        model_files={p.name:sha(p) for p in model.iterdir()};parent=read(vendor/'PARENT_ASSET_LOCK.json');parent['model_files']['tele_model']=model_files
        (vendor/'PARENT_ASSET_LOCK.json').write_text(json.dumps(parent));parent_sha=sha(vendor/'PARENT_ASSET_LOCK.json')
        environment=self.root/'env';environment.mkdir();write(environment/'identity.json',{'synthetic':True})
        evidence=self.root/'parameters';evidence.mkdir();ledger=[]
        for name in ('tele','fasttext','onnx'):
            record={'component':name,'stored_elements':1,'unique_trainable':1,'buffers':0,'model_files_sha256':canonical(model_files),'method':'exact_auxiliary_inventory','measurement_source_sha256':'0'*64}
            write(evidence/(name+'.json'),record)
            ledger.append({k:record[k] for k in ('component','stored_elements','unique_trainable','buffers','model_files_sha256')})
            ledger[-1].update(model_asset='tele_model',evidence={'asset':'parameter_evidence','path':name+'.json','sha256':sha(evidence/(name+'.json'))})
        write(self.root/'ledger.json',ledger)
        env={'image':A,'python':'/opt/python','resolved_python':'/opt/python','site_packages':'/opt/site','packages':parent['packages']['native'],
             'image_files':{'/opt/python':'0'*64,**{'/opt/site/'+n:h for n,h in parent['image_auxiliaries']['native'].items()}},
             'asset_roles':['native_code','tele_source','tele_model','environment','parameter_evidence']}
        write(self.root/'environment.json',{'native':env});r=self.runtime()
        b={'pages':1,'native_calls_per_page':1,'expert_calls_per_page':0,'model_loads':3,'load_seconds':60,'page_seconds':60,'request_seconds':60,'total_seconds':600,'cleanup_seconds':240,'page_audit_seconds':15,'cpus':2,'ram_gib':8,'swap_gib':0,'shm_gib':1,'gpu_allowlist':['GPU-a'],'customer_gpu_uuids':[]}
        cfg={'profile':'native','mode':'off','gpu_backend':r['gpu_backend'],'asset_roots':{'tele_model':str(model),'environment':str(environment),'parameter_evidence':str(evidence)},
             'environment_lock':'environment.json','parameter_ledger':'ledger.json','gpu_uuids':['GPU-a'],'lease_directory':'/synthetic/leases','idle_memory_mib':0,'validation_budget':b}
        write(self.root/'configuration.json',cfg);write(self.root/'budget.json',b);manifest=self.manifest(1)
        original_read=assets.read;original_sha=contracts.sha
        def asset_read(path):return parent if Path(path)==original_vendor/'PARENT_ASSET_LOCK.json' else original_read(path)
        def contract_sha(path,*a):return parent_sha if Path(path)==original_vendor/'PARENT_ASSET_LOCK.json' else original_sha(path,*a)
        # Only external vendor/model identity input is synthetic; all prepare/bind/runtime/preflight validators run.
        with patch.object(binder,'VENDOR',vendor),patch.object(assets,'read',side_effect=asset_read),patch.object(contracts,'sha',side_effect=contract_sha),patch.object(g.subprocess,'run',side_effect=AssertionError('No host command during offline binding')):
            binder.prepare(self.root/'configuration.json',self.root/'prepared')
            binder.bind(self.root/'prepared/bind-plan.json',self.root/'bound')
            plan=runner.preflight('native',manifest,self.root/'bound/runtime.json',self.root/'budget.json',mode='off')
        self.assertEqual(plan['runtime']['gpu_backend']['kind'],'manual')
        self.assertNotIn('HOST_DRIVER_BINDING',plan['runtime']['assets'])
        self.assertEqual((self.root/'bound/HOST_DRIVER_BINDING.json').read_bytes(),(self.root/'HOST_DRIVER_BINDING.json').read_bytes())
    def test_xml_uuid_not_ordinal_and_faults(self):
        raw=xml_text();self.assertEqual(g.parse_xml(raw,['GPU-a','GPU-b'],0,'999.1')['GPU-a']['minor'],3)
        self.assertEqual(g.parse_xml('<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v12.dtd">'+raw,['GPU-a'],0,'999.1')['GPU-a']['minor'],3)
        with self.assertRaises(ContractError):g.parse_xml('<!DOCTYPE x [<!ENTITY x SYSTEM "file:///x">]>'+raw,['GPU-a'],0,'999.1')
        for bad in [raw.replace('<minor_number>3</minor_number>',''),raw.replace('>3</minor_number>','>N/A</minor_number>'),raw.replace('>3</minor_number>','>7</minor_number>'),raw.replace('GPU-b','GPU-a'),raw.replace('Disabled','Enabled'),raw.replace('<processes/>','<processes><process_info/></processes>'),raw.replace('0 %','1 %'),raw.replace('0 MiB','1 MiB')]:
            with self.assertRaises(ContractError):g.parse_xml(bad,['GPU-a','GPU-b'],0,'999.1')
        # Other GPU busy must not block the later role's fresh query.
        busy=raw.replace('<uuid>GPU-b</uuid>','<uuid>GPU-b</uuid>').replace('<used>0 MiB</used>','<used>999 MiB</used>',1)
        self.assertEqual(set(g.parse_xml(busy,['GPU-a'],0,'999.1')),{'GPU-a'})
    def test_host_driver_closed_inventory_and_aliases(self):
        d=host_descriptor();bundle=Path(d['bundle_root']);target=bundle/'libcuda.so.999.1';source=Path(d['files'][0]['host_source'])
        aliases={bundle/n:v for n,v in d['aliases'].items()};entries=[target,*aliases]
        for fault in (None,'extra','alias','regular','hash','version'):
            def resolve(p):return target if p in aliases else p
            with patch.object(Path,'resolve',resolve),patch.object(Path,'is_dir',return_value=True),patch.object(Path,'iterdir',return_value=iter(entries+([bundle/'extra'] if fault=='extra' else []))),patch.object(Path,'is_symlink',lambda p:p in aliases and fault!='regular'),patch.object(g.os,'readlink',side_effect=lambda p:'../escape' if fault=='alias' else aliases[p]),patch.object(g,'elf_hash',return_value='f'*64 if fault=='hash' else 'a'*64),patch.object(g,'kernel_version',return_value='wrong' if fault=='version' else '999.1'):
                if fault:
                    with self.subTest(fault=fault),self.assertRaises(ContractError):g.verify_host_driver(d,self.clock())
                else:self.assertEqual(g.verify_host_driver(d,self.clock()),driver_proof(d))
    def test_worker_actual_device_inventory_rejects_hidden_alias(self):
        import stat,os
        binding=self.binding();nodes={d['path']:types.SimpleNamespace(st_mode=stat.S_IFCHR,st_rdev=(d['major'],d['minor'])) for d in binding['devices']}
        for extra in (False,True):
            current=dict(nodes)
            if extra:current['/dev/hidden']=types.SimpleNamespace(st_mode=stat.S_IFCHR,st_rdev=(508,1))
            with patch.object(g.os,'walk',return_value=[('/dev',[],[Path(n).name for n in current])]),patch.object(Path,'stat',lambda p:current[p.as_posix()]),patch.object(Path,'is_symlink',return_value=False),patch.object(g,'bounded_text',return_value='Character devices:\n508 nvidia-caps\n'),patch.object(g,'device_stats',return_value=devices(3)):
                # Production is Linux; use POSIX paths at this syscall boundary on Windows too.
                with patch.object(g,'Path',PurePosixPath),patch.object(g.os,'major',lambda d:d[0],create=True):
                    with patch.object(PurePosixPath,'stat',lambda p:current[str(p)],create=True),patch.object(PurePosixPath,'is_symlink',return_value=False,create=True):
                        if extra:
                            with self.assertRaises(ContractError):g.worker_devices(binding,self.clock())
                        else:self.assertEqual(g.worker_devices(binding,self.clock()),devices(3))
    def test_loaded_driver_mapping_and_ro_guard(self):
        binding=self.binding();target='/driver/libcuda.so.999.1'
        for fault in (None,'path','sha','rw','version'):
            mapping='7f-8f r-xp 000 0:0 0 '+('/elsewhere/libcuda.so.999.1' if fault=='path' else target)
            def text(p,*args):return mapping if p=='/proc/self/maps' else '1 0 8:1 /bundle /driver '+('rw' if fault=='rw' else 'ro')+' - ext4 /disk rw'
            with patch.object(g.ctypes,'CDLL',return_value=object()),patch.object(g.os,'RTLD_NOW',2,create=True),patch.object(g.os,'RTLD_LOCAL',0,create=True),patch.object(g,'bounded_text',side_effect=text),patch.object(g,'elf_hash',return_value='b'*64 if fault=='sha' else 'a'*64),patch.object(g,'kernel_version',return_value='bad' if fault=='version' else '999.1'),patch.object(Path,'resolve',lambda p:PurePosixPath(p.as_posix())):
                if fault:
                    with self.assertRaises(ContractError):g.loaded_driver(binding,self.clock())
                else:self.assertEqual(g.loaded_driver(binding,self.clock())[1]['sha256'],'a'*64)
    def test_cuda_uuid_uses_signatures_count_and_exact_uuid(self):
        import ctypes,uuid
        expected='GPU-00000000-0000-0000-0000-000000000003'
        def count(ptr):ptr._obj.value=1;return 0
        def dev(ptr,index):ptr._obj.value=7;return 0
        def uid(ptr,device):
            for i,b in enumerate(uuid.UUID(expected[4:]).bytes):ptr._obj.bytes[i]=b
            return 0
        lib=types.SimpleNamespace(cuInit=MagicMock(return_value=0),cuDeviceGetCount=MagicMock(side_effect=count),cuDeviceGet=MagicMock(side_effect=dev),cuDeviceGetUuid=MagicMock(side_effect=uid))
        self.assertEqual(g.actual_gpu_uuid(expected,lib),expected)
        self.assertEqual(lib.cuDeviceGet.restype,ctypes.c_int)
        with self.assertRaises(ContractError):g.actual_gpu_uuid(expected[:-1]+'4',lib)
        lib.cuDeviceGetCount.side_effect=lambda ptr:1
        with self.assertRaises(ContractError):g.actual_gpu_uuid(expected,lib)
    def test_device_stat_contract(self):
        g.validate_devices(devices(3),3)
        for change in ('extra','minor','uvm','path'):
            d=devices(3)
            if change=='extra':d.append(d[0])
            if change=='minor':d[0]['minor']=4
            if change=='uvm':d[3]['major']=511
            if change=='path':d[0]['path']='/dev/nvidia0'
            with self.assertRaises(ContractError):g.validate_devices(d,3)
    def state(self):
        binding=self.binding();b={'cpus':2,'ram_gib':8,'swap_gib':0,'shm_gib':1}
        state={'Image':A,'Id':'cid','Config':{'Image':A,'Labels':{'hybrid-v3-owner':'fixture'},'Env':[k+'='+v for k,v in g.manual_environment(binding).items()]},
               'HostConfig':{'Runtime':'runc','Devices':[{'PathOnHost':x['path'],'PathInContainer':x['path'],'CgroupPermissions':'rwm'} for x in devices(3)],'DeviceRequests':[],
                   'NetworkMode':'none','ReadonlyRootfs':True,'Memory':8*1024**3,'MemorySwap':8*1024**3,'NanoCpus':2_000_000_000,'ShmSize':1024**3,
                   'CapDrop':['ALL'],'SecurityOpt':['no-new-privileges'],'PidsLimit':4096,'Tmpfs':{'/tmp':'rw,size=1g'}},
               'Mounts':[{'Type':'bind','Source':'/synthetic/driver','Destination':'/driver','RW':False}],
               'State':{'Running':False,'Pid':0,'OOMKilled':False,'ExitCode':0}}
        return state,b
    def test_effective_manual_isolation_faults(self):
        s,b=self.state();g.verify_container(s,self.binding())
        for fault in ('runtime','device','permission','requests','env','duplicate','preload','mig','mount','write','rules'):
            bad=copy.deepcopy(s);hc=bad['HostConfig']
            if fault=='runtime':hc['Runtime']='nvidia'
            if fault=='device':hc['Devices'].append(hc['Devices'][0])
            if fault=='permission':hc['Devices'][0]['CgroupPermissions']='rw'
            if fault=='requests':hc['DeviceRequests']=[{'DeviceIDs':['GPU-a']}]
            if fault=='env':bad['Config']['Env'][0]='CUDA_VISIBLE_DEVICES=GPU-b'
            if fault=='duplicate':bad['Config']['Env'].append(bad['Config']['Env'][0])
            if fault=='preload':bad['Config']['Env'].append('LD_PRELOAD=/shadow.so')
            if fault=='mig':bad['Config']['Env'].append('NVIDIA_MIG_CONFIG_DEVICES=all')
            if fault=='mount':bad['Mounts'][0]['Source']='/other'
            if fault=='write':bad['Mounts'][0]['RW']=True
            if fault=='rules':hc['DeviceCgroupRules']=['c 195:* rwm']
            with self.subTest(fault=fault),self.assertRaises(ContractError):g.verify_container(bad,self.binding())
    def host(self,mode='on',fault='normal'):
        from . import runner,image_identity as ii
        fake=ManualFake(fault);r=self.runtime();leases=self.root/'leases';leases.mkdir()
        r.update(image=A,python='/opt/python',gpu_uuids=['GPU-a','GPU-b'] if mode=='on' else ['GPU-a'],idle_memory_mib=0,lease_directory=str(leases),assets={},v31={'components':['formula']},environments={'paddle':{'image':REPO,'python':'/opt/paddle'}})
        b={'total_seconds':600,'cleanup_seconds':240,'cpus':8,'ram_gib':16,'swap_gib':0,'shm_gib':1,'load_seconds':60}
        plan={'inputs':{'pages':[]},'budget':b,'runtime':r,'source':{},'bindings':{},'states':{'runtime_verified':False,'integration_verified':False}}
        write(self.root/'budget.json',b);calls=[]
        idle_calls=[0]
        def idle(ids,*args):
            idle_calls[0]+=1
            if fault=='release' and idle_calls[0]>1:raise ContractError('Synthetic unknown release')
            return {u:0 for u in ids}
        class Docker(OwnedDocker):
            def __init__(self,token,clock,out):super().__init__(token,clock,out,fake)
        def query(ids,ceiling,version,deadline):
            calls.append({'uuids':ids[:],'creates':sum(c['args'][1]=='create' for c in fake.calls)})
            return {'observed_monotonic':time.monotonic(),'gpus':g.parse_xml(xml_text(),ids,ceiling,version),'xml_sha256':'a'*64}
        original_resolve=Path.resolve;original_exists=Path.exists
        def resolve(p,*a,**kw):return PurePosixPath('/synthetic/driver') if p.as_posix()=='/synthetic/driver' else original_resolve(p,*a,**kw)
        def exists(p):return True if p.as_posix()=='/synthetic/driver' else original_exists(p)
        with patch.object(runner,'os',types.SimpleNamespace(name='posix')),patch.object(runner,'preflight',return_value=plan),patch.object(runner,'Leases',return_value=MagicMock()),patch.object(runner,'gpu_idle',side_effect=idle),patch.object(runner,'OwnedDocker',Docker),patch('hybrid_v3_full_eval_v1.v31_service.OwnedDocker',Docker),patch.object(ii.subprocess,'run',side_effect=fake),patch.object(runner,'source_check',return_value={}),patch.object(g,'verify_host_driver',side_effect=ContractError('Synthetic driver mismatch') if fault=='driver' else lambda binding,clock:driver_proof(binding)),patch.object(g,'query_xml',side_effect=query),patch.object(g,'device_stats',side_effect=devices),patch.object(g,'kernel_version',return_value='999.1'),patch.object(Path,'resolve',resolve),patch.object(Path,'exists',exists):
            with self.assertRaises(ContractError):runner.infer('v31-formula','unused',self.root/'runtime.json',self.root/'budget.json',self.root/'run',mode)
        return fake,calls,read(self.root/'run/HOST_EXIT.json'),read(self.root/'run/ADMISSION_RESULT.json')
    def test_host_all_roles_before_create_and_role_fresh(self):
        fake,calls,exit_state,admission=self.host()
        self.assertEqual(calls[0],{'uuids':['GPU-a','GPU-b'],'creates':0})
        self.assertEqual([c['uuids'] for c in calls[1:]],[['GPU-b'],['GPU-a']])
        self.assertIsNone(exit_state['failure']);self.assertTrue(admission['released']);self.assertEqual(fake.objects,{})
        self.assertEqual(sum(c['args'][1]=='start' for c in fake.calls),2)
    def test_host_off_native_only(self):
        fake,calls,_,_=self.host('off');self.assertTrue(all(c['uuids']==['GPU-a'] for c in calls));self.assertEqual(sum(c['args'][1]=='start' for c in fake.calls),1)
    def test_host_pass_native_only(self):
        fake,calls,_,_=self.host('pass-through');self.assertTrue(all(c['uuids']==['GPU-a'] for c in calls));self.assertEqual(sum(c['args'][1]=='start' for c in fake.calls),1)
    def test_host_partial_create_cleanup(self):
        fake,_,state,admission=self.host(fault='create');self.assertIsNotNone(state['failure']);self.assertEqual(fake.objects,{});self.assertTrue(admission['released'])
    def test_host_start_failure_cleanup(self):
        fake,_,state,admission=self.host(fault='start');self.assertIsNotNone(state['failure']);self.assertEqual(fake.objects,{});self.assertTrue(admission['released'])
    def test_host_bad_device_no_start_cleanup(self):
        fake,_,state,admission=self.host(fault='device');self.assertIsNotNone(state['failure']);self.assertFalse(any(c['args'][1]=='start' for c in fake.calls));self.assertEqual(fake.objects,{})
    def test_host_worker_exit_cleanup(self):
        fake,_,state,_=self.host(fault='worker');self.assertIsNotNone(state['failure']);self.assertEqual(fake.objects,{})
    def test_host_driver_failure_no_create(self):
        fake,_,state,_=self.host(fault='driver');self.assertIsNotNone(state['failure']);self.assertFalse(any(c['args'][1]=='create' for c in fake.calls))
    def test_host_release_unknown_retains_block(self):
        fake,_,state,admission=self.host(fault='release');self.assertIsNotNone(state['failure']);self.assertTrue(admission['blocked']);self.assertFalse(admission['released'])
    def test_fresh_age_rejected(self):
        write(self.root/'GPU_BINDINGS.json',{})
        with patch.object(g.time,'monotonic',return_value=300),self.assertRaises(ContractError):
            g.seal_fresh(self.root,{'run_id':'fixture'},self.binding(),{'observed_monotonic':1},'cid','fixture',self.root)
    def sealed(self):
        path=self.run_fixture(1);root=path.parent;control=read(root/'CONTROL.json');run=read(path);desc=host_descriptor()
        control.update(mode='off',gpu_binding_schema=1,image_identity_schema=1,absolute_stop_monotonic=100)
        control['budget'].update(cpus=2,ram_gib=8,swap_gib=0,shm_gib=1)
        write(root/'HOST_DRIVER_BINDING.json',desc);h=sha(root/'HOST_DRIVER_BINDING.json')
        # schema2 marker activates existing load proofs as well as manual evidence.
        control['runtime'].update(schema=2,image=A,gpu_backend={'kind':'manual','host_binding':{'path':'absent-original.json','sha256':h}})
        binding=self.binding();binding['uuid']='GPU-fixture'
        record={'schema':1,'backend':'manual','run_id':'fixture','host_binding_sha256':h,'driver':driver_proof(desc),'roles':{'native':binding},
                'initial_xml':{'observed_monotonic':11,'gpus':{'GPU-fixture':xml_summary('GPU-fixture',3)},'xml_sha256':'a'*64}}
        write(root/'GPU_BINDINGS.json',record);all_sha=sha(root/'GPU_BINDINGS.json')
        write(root/'GPU_BEFORE_START.json',{'schema':1,'run_id':'fixture','role':'native','cid':'cid','owner':'fixture','uuid':'GPU-fixture','binding_sha256':canonical(binding),'gpu_bindings_sha256':all_sha,'verified_before_start_monotonic':13,'xml':{'observed_monotonic':12,'gpus':{'GPU-fixture':xml_summary('GPU-fixture',3)},'xml_sha256':'a'*64}})
        write(root/'GPU_BINDING_CHECK.json',{'schema':1,'run_id':'fixture','role':'native','expected_uuid':'GPU-fixture','actual_uuid':'GPU-fixture','devices':devices(3),'driver':{'realpath':'/driver/libcuda.so.999.1','sha256':'a'*64,'mounts':[{'mountpoint':'/driver','read_only':True}],'driver_version':'999.1'},'host_binding_sha256':h,'binding_sha256':canonical(binding),'gpu_bindings_sha256':all_sha,'observed_monotonic':14})
        state,_=self.state();state['Config']['Env']=[k+'='+v for k,v in g.manual_environment(binding).items()]
        (root/'CONTAINER_EXIT.json').write_text(json.dumps(state));created=read(root/'CONTAINER_CREATED.json');created['arguments']=['--mount','type=bind,src=/synthetic/driver,dst=/driver,readonly'];(root/'CONTAINER_CREATED.json').write_text(json.dumps(created))
        write(root/'IMAGE_IDENTITIES.json',{'schema':1,'run_id':'fixture','roles':{'native':identity()}})
        for n in ('NATIVE_LOAD_BOUNDARY.json','LOADED_PARAMETERS.json','PROCESSOR.json','TELE_LOAD.json'):write(root/n,{'synthetic':True})
        (root/'CONTROL.json').write_text(json.dumps(control));run.update(gpu_binding_schema=1,image_identity_schema=1,lifecycle=lifecycle_lock(root));path.write_text(json.dumps(run))
        folder=root/'pages/p0'
        for name in ('PRIMARY_CONTENT.json','V31_FINAL_RAW.json','V31_TRANSACTIONS.json','REQUEST_BINDINGS.json','CONTENT.json'):
            (folder/name).write_text('[]',encoding='utf-8')
        return path,run,control
    def offline(self,root,run):
        original=Path.resolve
        def resolve(p,*a,**kw):return PurePosixPath('/synthetic/driver') if p.as_posix()=='/synthetic/driver' else original(p,*a,**kw)
        with patch.object(Path,'resolve',resolve),patch.object(g.subprocess,'run',side_effect=AssertionError('Offline must not query host')):
            return normal_completion(root,run)
    def test_completion_valid_and_relocatable(self):
        path,run,_=self.sealed();self.assertTrue(self.offline(path.parent,run))
        moved=self.root/'moved';shutil.move(str(path.parent),str(moved));self.assertTrue(self.offline(moved,run))
        from .collection import collect
        original=Path.resolve
        def resolve(p,*a,**kw):return PurePosixPath('/synthetic/driver') if p.as_posix()=='/synthetic/driver' else original(p,*a,**kw)
        with patch.object(Path,'resolve',resolve):self.assertEqual(collect(moved/'RUN_MANIFEST.json',self.root/'collection')['page_count'],1)
    def test_completion_resealed_tamper_matrix(self):
        path,run,control=self.sealed();root=path.parent;original={p.name:p.read_bytes() for p in root.iterdir() if p.is_file()}
        for fault in ('markers','run','role','cid','sha','env','write','alias','missing'):
            for n,b in original.items():(root/n).write_bytes(b)
            current=copy.deepcopy(run)
            if fault=='markers':
                c=copy.deepcopy(control);c.pop('gpu_binding_schema');(root/'CONTROL.json').write_text(json.dumps(c));current.pop('gpu_binding_schema')
            elif fault=='missing':(root/'GPU_BINDING_CHECK.json').unlink()
            elif fault=='alias':
                d=read(root/'HOST_DRIVER_BINDING.json');d['aliases']['libcuda.so']='../bad';(root/'HOST_DRIVER_BINDING.json').write_text(json.dumps(d))
            elif fault in ('env','write'):
                s=read(root/'CONTAINER_EXIT.json')
                if fault=='env':s['Config']['Env'][0]='CUDA_VISIBLE_DEVICES=GPU-other'
                else:s['Mounts'][0]['RW']=True
                (root/'CONTAINER_EXIT.json').write_text(json.dumps(s))
            else:
                n='GPU_BEFORE_START.json' if fault=='cid' else 'GPU_BINDING_CHECK.json';v=read(root/n)
                if fault=='sha':v['driver']['sha256']='f'*64
                else:v[{'run':'run_id','role':'role','cid':'cid'}[fault]]='wrong'
                (root/n).write_text(json.dumps(v))
            current['lifecycle']=lifecycle_lock(root)
            with self.subTest(fault=fault),self.assertRaises(ContractError):self.offline(root,current)
    def test_worker_guard_before_load_rejects_devices_driver_uuid(self):
        path,_,control=self.sealed();root=path.parent
        for fault in ('devices','driver','uuid'):
            with patch.object(g,'worker_devices',side_effect=ContractError('device') if fault=='devices' else None,return_value=devices(3)),patch.object(g,'loaded_driver',side_effect=ContractError('driver') if fault=='driver' else None,return_value=(object(),{})),patch.object(g,'actual_gpu_uuid',side_effect=ContractError('uuid')) as query:
                with self.assertRaises(ContractError):g.worker_guard(control,'native',root)
                self.assertEqual(query.call_count,1 if fault=='uuid' else 0)
    def test_worker_entrypoints_guard_before_model(self):
        import ast
        for filename,model in [('worker.py','Native'),('v31_worker.py','provider')]:
            text=(Path(__file__).parent/filename).read_text();tree=ast.parse(text)
            guard=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='worker_guard']
            if model=='Native':loads=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id==model]
            else:loads=[n.lineno for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='provider' for t in n.targets)]
            self.assertLess(min(guard),min(loads))

def main():
    import argparse
    global ROOT
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();ROOT=Path(a.output);ROOT.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (ROOT/'checks.log').write_text(stream.getvalue(),encoding='utf-8');report={'passed':r.wasSuccessful(),'tests':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'synthetic_cpu_only':True,'GPU':0,'models':0,'Docker':0}
    write(ROOT/'CPU_CHECKS.json',report);print(stream.getvalue());return 0 if r.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
