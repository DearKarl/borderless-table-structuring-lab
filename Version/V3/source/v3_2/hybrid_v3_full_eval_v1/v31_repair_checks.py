"""R1-R3 production-path CPU fault injection; no Docker, GPU or model execution."""
import copy
import io
import json
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
from . import v31_checks as original
from .core import ContractError, Deadline, read, write, sha
from .group_cleanup import finalize_group
from .bounded_cleanup import CleanupClock
from .gpu_admission import Admission, assert_unblocked, block_path, clear
from .lifecycle import OwnedDocker, Leases

class DockerFault:
    def __init__(self,now,mode='normal'):
        self.now,self.mode=now,mode;self.calls=[];self.objects={};self.counter=0
    def add(self,owner):
        cid='cid-'+owner
        self.objects[cid]={'Id':cid,'Name':'hybrid-v3-'+owner,
            'Config':{'Labels':{'hybrid-v3-owner':owner}},
            'State':{'Running':True,'Pid':123,'ExitCode':0,'OOMKilled':False}}
        return cid
    def __call__(self,args,timeout,**kwargs):
        self.now[0]+=timeout;self.calls.append({'args':args,'cap':timeout,'at':self.now[0]})
        if self.mode=='daemon':raise subprocess.TimeoutExpired(args,timeout)
        command=args[1];target=args[-1]
        obj=next((v for k,v in self.objects.items() if target in (k,v['Name'])),None)
        def result(code=0,out='',err=''):return types.SimpleNamespace(returncode=code,stdout=out,stderr=err)
        if command=='create':
            owner=args[args.index('--label')+1].split('=',1)[1];cid=self.add(owner)
            if self.mode=='partial':raise subprocess.TimeoutExpired(args,timeout)
            return result(out=cid)
        if command=='inspect':
            if obj is None:return result(1,out='[]',err='Error: No such object: '+target)
            if self.mode=='unknown':return result(1,err='daemon unavailable')
            value=copy.deepcopy(obj)
            if self.mode=='foreign':value['Config']['Labels']['hybrid-v3-owner']='foreign'
            return result(out=json.dumps([value]))
        if command=='kill':
            obj['State'].update(Running=False,Pid=0);return result()
        if command=='wait':return result(out='0')
        if command=='logs':
            if self.mode=='logs' and target.endswith('one'):raise subprocess.TimeoutExpired(args,timeout)
            return result(out='synthetic logs')
        if command=='rm':
            del self.objects[obj['Id']];return result(out=target)
        raise AssertionError(args)

class RepairChecks(original.Checks):
    def group(self,mode='normal',cached=True,partial=False,none=False):
        now=[1.];deadline=Deadline(0,600,240,clock=lambda:now[0]);fake=DockerFault(now,mode)
        leases=self.root/'leases';leases.mkdir()
        admission=Admission(leases,['GPU-a','GPU-b']);admission.reserve('run',['one','two'])
        resources=[]
        for name in ['one','two']:
            folder=self.root/name;folder.mkdir();d=OwnedDocker(name,deadline,folder,fake);d.admission=admission
            if not none:
                cid=fake.add(name);d.creation_attempted=True;d.cid=None if partial else cid
                if cached and not partial:d.last_verified=copy.deepcopy(fake.objects[cid])
            admission.observe(d);resources.append(d)
        def release(clock):
            for _ in range(2):now[0]+=clock.bound(5)
            return {'verified':True,'gpu_uuids':['GPU-a','GPU-b'],'memory_mib':{'GPU-a':0,'GPU-b':0},'observed_monotonic':now[0]}
        group=finalize_group(resources,release,CleanupClock(deadline))
        decision=admission.finish(group)
        write(self.root/'GROUP.json',group);write(self.root/'ADMISSION.json',decision);write(self.root/'CALLS.json',fake.calls)
        self.assertLessEqual(now[0]-1,240)
        return group,decision,fake,admission

    def test_r1_two_full_cap_containers_all_critical_attempts(self):
        group,decision,fake,_=self.group()
        self.assertTrue(group['complete']);self.assertTrue(decision['released'])
        self.assertEqual(group['finished_monotonic']-group['started_monotonic'],144)
        for cid in ['cid-one','cid-two']:
            self.assertTrue(any(c['args']==['docker','kill',cid] for c in fake.calls))
            self.assertTrue(any(c['args']==['docker','rm',cid] for c in fake.calls))
        assert_unblocked(self.root/'leases',['GPU-a','GPU-b'])

    def test_r1_log_failure_is_audit_failure_not_resource_leak(self):
        group,decision,fake,_=self.group('logs')
        self.assertFalse(group['complete']);self.assertTrue(group['resource_release_verified'])
        self.assertTrue(decision['released']);self.assertEqual(fake.objects,{})

    def test_r1_partial_create_resolves_exact_owner_and_removes_both(self):
        group,decision,fake,_=self.group(partial=True)
        self.assertTrue(group['complete']);self.assertTrue(decision['released'])
        self.assertEqual([r['cid'] for r in group['resources']],['cid-one','cid-two'])

    def test_r1_foreign_even_cached_identity_refuses_destructive_commands(self):
        group,decision,fake,_=self.group('foreign')
        self.assertFalse(group['complete']);self.assertTrue(decision['blocked'])
        self.assertFalse(any(c['args'][1] in ('kill','rm') for c in fake.calls))
        with self.assertRaises(ContractError):assert_unblocked(self.root/'leases',['GPU-a'])

    def test_r1_unknown_partial_identity_remains_durably_blocked(self):
        group,decision,fake,_=self.group('unknown',cached=False,partial=True)
        self.assertFalse(group['complete']);self.assertTrue(decision['blocked'])
        self.assertFalse(any(c['args'][1] in ('kill','rm') for c in fake.calls))
        record=read(block_path(self.root/'leases','GPU-a'))
        self.assertEqual(record['resources']['one']['container_name'],'hybrid-v3-one')

    def test_r1_daemon_lost_confirmed_resources_get_stop_and_remove_attempts(self):
        group,decision,fake,_=self.group('daemon')
        self.assertFalse(group['complete']);self.assertTrue(decision['blocked'])
        for cid in ['cid-one','cid-two']:
            self.assertTrue(any(c['args']==['docker','kill',cid] for c in fake.calls))
            self.assertTrue(any(c['args']==['docker','rm','--force',cid] for c in fake.calls))
        for key in ['GPU-a','GPU-b']:
            record=read(block_path(self.root/'leases',key))
            self.assertEqual(record['resources']['one']['cid'],'cid-one')
            self.assertEqual(record['gpu_uuids'],['GPU-a','GPU-b'])

    def test_r1_no_created_resource_does_not_leave_false_admission_block(self):
        group,decision,_,_=self.group(none=True)
        self.assertFalse(group['complete']);self.assertTrue(decision['released'])
        self.assertTrue(group['resource_release_verified'])

    def test_r1_real_create_timeout_journal_survives_and_cleanup_discovers_cid(self):
        now=[1.];deadline=Deadline(0,600,240,clock=lambda:now[0]);fake=DockerFault(now,'partial')
        leases=self.root/'leases';leases.mkdir();admission=Admission(leases,['GPU-a']);admission.reserve('run',['one'])
        d=OwnedDocker('one',deadline,self.root,fake);d.admission=admission
        with self.assertRaises(subprocess.TimeoutExpired):d.create([])
        self.assertTrue(read(block_path(leases,'GPU-a'))['resources']['one']['creation_attempted'])
        self.assertIsNone(d.cid)
        group=finalize_group([d],lambda c:{'verified':True},CleanupClock(deadline))
        self.assertTrue(group['resource_release_verified']);self.assertEqual(fake.objects,{})
        self.assertTrue(admission.finish(group)['released'])

    def test_r1_new_lease_refused_then_explicit_verified_clear(self):
        _,_,_,admission=self.group('daemon')
        locks=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        with patch.dict(sys.modules,{'fcntl':locks}):
            with self.assertRaises(ContractError):
                with Leases(admission.directory,admission.keys):pass
            def absent(args,**kw):
                if args[0]=='docker':return types.SimpleNamespace(returncode=1,stdout='[]',stderr='No such object: '+args[-1])
                return types.SimpleNamespace(returncode=0,stderr='',stdout='GPU-a, 0\nGPU-b, 0' if 'query-gpu' in args[1] else '')
            proof=clear(admission.directory,admission.keys,'run',0,self.root/'CLEAR.json',runner=absent)
            self.assertTrue(proof['admission_released'])
            with Leases(admission.directory,admission.keys):pass

    def test_r1_explicit_clear_requires_absence_and_full_gpu_set(self):
        _,_,_,admission=self.group('daemon')
        locks=types.SimpleNamespace(LOCK_EX=1,LOCK_NB=2,flock=lambda *a:None)
        with patch.dict(sys.modules,{'fcntl':locks}):
            with self.assertRaises(ContractError):clear(admission.directory,['GPU-a'],'run',0,self.root/'NO.json')
            with self.assertRaises(ContractError):
                clear(admission.directory,admission.keys,'run',0,self.root/'NO.json',
                      runner=lambda *a,**k:types.SimpleNamespace(returncode=1,stdout='',stderr='daemon unavailable'))
        with self.assertRaises(ContractError):assert_unblocked(admission.directory,admission.keys)

    def test_r1_gpu_release_failure_keeps_block_after_both_removed(self):
        now=[0.];fake=DockerFault(now);deadline=Deadline(0,600,240,clock=lambda:now[0])
        leases=self.root/'leases';leases.mkdir();admission=Admission(leases,['GPU-a']);admission.reserve('run',['one'])
        d=OwnedDocker('one',deadline,self.root,fake);d.admission=admission;d.create([])
        def busy(clock):raise ContractError('GPU memory is not idle')
        group=finalize_group([d],busy,CleanupClock(deadline))
        self.assertEqual(fake.objects,{});self.assertFalse(group['complete'])
        self.assertTrue(admission.finish(group)['blocked'])
        with self.assertRaises(ContractError):assert_unblocked(leases,['GPU-a'])

    def test_r1_group_and_admission_required_by_formal_completion(self):
        from .cpu_fixtures import lifecycle_fixture
        from .completion import normal_completion,lifecycle_lock
        manifest={'pages':[]};lifecycle_fixture(self.root,manifest)
        def replace(name,obj):(self.root/name).write_text(json.dumps(obj))
        control=read(self.root/'CONTROL.json');control['cleanup_schema']=2;replace('CONTROL.json',control)
        # Empty fixture coverage still needs a nonempty lifecycle ledger.
        (self.root/'LEDGER.jsonl').write_text(json.dumps({'event':'LOAD_START'})+'\n')
        cleanup=read(self.root/'CLEANUP.json');cleanup['owner']='fixture';replace('CLEANUP.json',cleanup)
        host=read(self.root/'HOST_EXIT.json');host['cleanup']=cleanup;replace('HOST_EXIT.json',host)
        release=read(self.root/'GPU_RELEASE.json')
        group={'complete':True,'resource_release_verified':True,'budget_exceeded':False,
               'release':release,'resources':[cleanup]}
        admission={'run_id':'fixture','released':True,'blocked':False,'gpu_release':release}
        write(self.root/'GROUP_CLEANUP.json',group);write(self.root/'ADMISSION_RESULT.json',admission)
        run={'run_id':'fixture','profile':'native','inputs':manifest,'cleanup':cleanup,'host_failure':None,
             'states':{'processing_complete':True,'runtime_verified':True,'integration_verified':True}}
        run['lifecycle']=lifecycle_lock(self.root);self.assertTrue(normal_completion(self.root,run))
        for field in ['blocked','released']:
            bad={**admission,field:field=='blocked'};replace('ADMISSION_RESULT.json',bad)
            run['lifecycle']=lifecycle_lock(self.root)
            with self.assertRaises(ContractError):normal_completion(self.root,run)
        replace('ADMISSION_RESULT.json',admission)
        (self.root/'GROUP_CLEANUP.json').unlink();run['lifecycle']=lifecycle_lock(self.root)
        with self.assertRaises(ContractError):normal_completion(self.root,run)

    def test_r2_formula_primary_retained_and_authoritative_selected(self):
        self.run_native([original.Block('equation',[.1,.1,.8,.3],content='{')],experts=original.Experts(formula='x'))
        self.assertEqual(read(self.root/'PRIMARY_CONTENT.json')[0]['raw_content'],'{')
        row=read(self.root/'CONTENT.json')[0]
        self.assertEqual(row['selected_content'],'x');self.assertEqual(row['source'],'paddle')
        self.assertTrue(row['expert_request_id']);self.assertTrue(row['native_request_id'])

    def test_r2_rejected_formula_records_attempt_and_rollback(self):
        self.run_native([original.Block('equation',[.1,.1,.8,.3],content='{')],experts=original.Experts(formula='{'))
        row=read(self.root/'CONTENT.json')[0]
        self.assertEqual(row['selected_content'],'{');self.assertEqual(row['source'],'native')
        self.assertTrue(row['decisions'][0]['expert_response']['request_id'])
        self.assertEqual(row['decisions'][0]['rollback'],'{')

    def test_r2_collection_rejects_selection_content_order_and_source_tamper(self):
        from .collection import terminals
        self.run_native([original.Block('equation',[.1,.1,.8,.3],content='{')],experts=original.Experts(formula='x'))
        # Build a real success-page collection boundary around Native.parse outputs.
        import shutil
        host=self.root/'host';folder=host/'pages'/'p';folder.parent.mkdir(parents=True)
        shutil.copytree(self.root,folder,ignore=shutil.ignore_patterns('host'))
        (folder/'prediction.md').write_text('Formatted Markdown is deliberately different')
        ident={'run_id':'run','page_id':'p','input_sha256':'a'*64}
        write(folder/'PAGE_START.json',ident)
        write(folder/'PAGE_RESULT.json',{**ident,'status':'success','start_sha256':sha(folder/'PAGE_START.json'),'prediction_sha256':sha(folder/'prediction.md')})
        write(host/'CONTROL.json',{'runtime':{'schema':2},'mode':'on'})
        run={'run_id':'run','inputs':{'pages':[self.page]},'claims':['p'],'all_slots_terminal':True,
             'states':{'processing_complete':False},'actual_page_starts':['p'],'actual_page_results':['p'],'unresolved':[]}
        write(host/'RUN_MANIFEST.json',run);terminals(host/'RUN_MANIFEST.json')
        source=read(folder/'CONTENT.json')
        for key,value in [('selected_content','{'),('source','native'),('slot_id','wrong'),('final_index',4)]:
            bad=copy.deepcopy(source);bad[0][key]=value
            (folder/'CONTENT.json').write_text(json.dumps(bad))
            with self.assertRaises(ContractError):terminals(host/'RUN_MANIFEST.json')

    def test_r3_offline_model_and_source_pins_remain_distinct(self):
        from .asset_binder import requirements,VENDOR
        result=requirements(self.root/'ENVIRONMENT_REQUIREMENTS.json')
        self.assertEqual(result['models']['tele']['repository'],'StarDoc-AI/TeleOCR')
        self.assertEqual(result['models']['tele']['revision'],'8706730e41382f4a8f2562616d47225fee8e2f7a')
        self.assertEqual(result['sources']['tele']['revision'],'9921cffe380efe4e2fa010258b3d0c3cb70bab2d')
        self.assertEqual(result['models']['tele']['files'],read(VENDOR/'PARENT_ASSET_LOCK.json')['model_files']['tele_model'])
        self.assertFalse(result['network']);self.assertEqual(result['model_loads'],0)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--tele-source',required=True);a=p.parse_args()
    original.ROOT=Path(a.output);original.ROOT.mkdir(parents=True,exist_ok=False)
    original.TELE=Path(a.tele_source)
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairChecks))
    (original.ROOT/'checks.log').write_text(stream.getvalue(),encoding='utf-8')
    report={'passed':result.wasSuccessful(),'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),
            'synthetic_cpu_only':True,'gpu':0,'model_loads':0,'scoring':0,'training':0}
    write(original.ROOT/'CPU_CHECKS.json',report);print(stream.getvalue());print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1
if __name__=='__main__':raise SystemExit(main())
