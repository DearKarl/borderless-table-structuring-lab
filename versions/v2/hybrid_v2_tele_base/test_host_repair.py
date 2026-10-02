import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).parent))
from host_repair_policy import exclusive_bytes,specific_host_fault,validate_binding_digests


class RepairTests(unittest.TestCase):
    def test_digest_binding_rejects_transcription_lengths_before_runtime_work(self):
        actual='43cf0febf2ea7056d80b6a7a4f186071898adaa8a6ba2fb64f25866729450136'
        validate_binding_digests({'p00205_prediction_sha256':actual,'arms':{'tele_raw':{'freeze_sha256':'a'*64}}})
        for bad in ('a'*63,'a'*65,'A'*64,'g'*64):
            with self.assertRaises(ValueError):validate_binding_digests({'nested':[{'prediction_sha256':bad}]})
    def test_original_commit_uses_repaired_host_writer_and_remains_idempotent(self):
        frozen=Path(__file__).parent.parent/'full-code-v2'
        if frozen.exists():sys.path.insert(0,str(frozen))
        from hybrid_v2_tele_base import full_state as state
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);session=root/'sessions/session-0001';native=session/'pages/p00205'
            for p in (native,root/'predictions',root/'terminals'):p.mkdir(parents=True)
            page={'key':'p00205','page_id':'汉'*84,'input_sha256':'c'*64}
            (native/'prediction.md').write_bytes(b'native unchanged')
            state.exclusive_json(native/'PAGE_RESULT.json',{**page,'status':'success','arm':'tele_raw','freeze_sha256':'a'*64,
                'prediction_sha256':state.sha(native/'prediction.md')})
            with patch.object(state,'exclusive_bytes',exclusive_bytes):
                self.assertTrue(state.commit_page(root,session,page,'a'*64,'tele_raw'))
                self.assertFalse(state.commit_page(root,session,page,'a'*64,'tele_raw'))
                self.assertEqual(state.pending_pages(root,[page],'a'*64,'tele_raw'),[])
            self.assertEqual((root/'predictions'/(page['page_id']+'.md')).read_bytes(),b'native unchanged')

    def test_255_byte_utf8_final_name_uses_short_atomic_temporary(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/('汉'*84+'.md');self.assertEqual(len(path.name.encode()),255)
            observed=[];original=Path.open
            def opened(target,*args,**kwargs):
                if args and args[0]=='xb':observed.append(target.name)
                return original(target,*args,**kwargs)
            with patch.object(Path,'open',opened):exclusive_bytes(path,b'original native content')
            self.assertEqual(path.read_bytes(),b'original native content')
            self.assertTrue(all(len(name.encode())<64 for name in observed))
            with self.assertRaises(FileExistsError):exclusive_bytes(path,b'must not replace')
            self.assertEqual(path.read_bytes(),b'original native content')

    def test_only_specific_non_oom_filename_fault_is_admitted(self):
        row={'cid':'175cb6c3e7e6exact','reason':'supervisor_error','foreign_processes':[],
             'supervisor_error':'full_state.py exclusive_bytes temporary.open OSError: [Errno 36] File name too long: original-page',
             'container':{'Id':'175cb6c3e7e6exact','State':{'Running':False,'OOMKilled':False,'ExitCode':137},
                          'Config':{'Labels':{'hybrid.arm':'tele_raw','hybrid.freeze':'a'*64}}}}
        specific_host_fault(row,'a'*64,'original-page')
        for key,value in [('OOMKilled',True),('Running',True),('ExitCode',1)]:
            changed=copy.deepcopy(row);changed['container']['State'][key]=value
            with self.assertRaises(ValueError):specific_host_fault(changed,'a'*64,'original-page')
        for key,value in [('supervisor_error','unknown failure'),('cid','other'),('reason','page_timeout')]:
            changed=copy.deepcopy(row);changed[key]=value
            with self.assertRaises(ValueError):specific_host_fault(changed,'a'*64,'original-page')

    def test_takeover_signals_only_verified_host_and_preserves_container_pid(self):
        path=Path(__file__).with_name('launcher.py')
        if not path.exists():path=Path(__file__).with_name('host_repair_launcher.py')
        spec=importlib.util.spec_from_file_location('repair_test_launcher',path);m=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):spec.loader.exec_module(m)
        from hybrid_v2_tele_base import full_state as state
        with tempfile.TemporaryDirectory() as folder:
            base=Path(folder);session=base/'full-v2/hybrid_v2/sessions/session-0001';session.mkdir(parents=True)
            attempt=base/'attempt';attempt.mkdir();cid='hybrid-exact';state.exclusive_json(session/'START.json',{'cid':cid})
            container={'Id':cid,'State':{'Running':True,'Pid':222},'Config':{'Labels':{'hybrid.arm':'hybrid_v2','hybrid.freeze':'b'*64}}}
            run=types.SimpleNamespace(query=Mock(return_value=json.dumps([container])))
            pins={'old_hybrid_supervisor_pid':111,'old_hybrid_start_ticks':'456','old_hybrid_command':['exact'],
                  'arms':{'hybrid_v2':{'freeze_sha256':'b'*64}},'hybrid_container_prefix':'hybrid'}
            signals=[]
            def send(fd,sig):self.assertEqual(fd,999);signals.append(sig)
            def proc(pid):return {'pid':pid,'state':'T','start_ticks':'456','command':['exact']}
            with patch.object(m,'BASE',base),patch.object(m,'process',side_effect=proc),\
                 patch.object(m.os,'pidfd_open',return_value=999,create=True),patch.object(m.os,'close'),\
                 patch.object(m.signal,'pidfd_send_signal',side_effect=send,create=True),\
                 patch.object(m.signal,'SIGSTOP',19,create=True),patch.object(m.signal,'SIGKILL',9,create=True),\
                 patch.object(m.signal,'SIGCONT',18,create=True),patch.object(m.select,'select',return_value=([999],[],[])):
                m.takeover(run,state,pins,attempt)
            self.assertEqual(signals,[19,9])
            result=state.read(attempt/'CONTAINER_PRESERVED.json')
            self.assertEqual(result['native_pid_before'],result['native_pid_after']);self.assertFalse(result['native_model_reinitialized'])


if __name__=='__main__':unittest.main()
