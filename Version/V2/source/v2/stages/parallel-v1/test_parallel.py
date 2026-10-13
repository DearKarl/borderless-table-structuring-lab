import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,Mock

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))
from parallel_contract import resource_only,serial_command,gpu_pair


class ParallelTests(unittest.TestCase):
    def test_only_resource_binding_may_change(self):
        parent={'runtime_manifest':{'gpu_uuid':'old','recipe':'native'},'native_runtime':{'dtype':'original'},
                'code_files':{'inference.py':'original','RUNTIME_MANIFEST.json':'a','hybrid_v2_tele_base/run_smoke.py':'b'}}
        child=copy.deepcopy(parent);child['runtime_manifest']['gpu_uuid']='new';child['resource_revision']={'parent':'old'}
        child['code_files']['RUNTIME_MANIFEST.json']='newhash'
        resource_only(parent,child,'old','new')
        for key in ('recipe','gpu_uuid'):
            broken=copy.deepcopy(child);broken['runtime_manifest'][key]='changed'
            with self.assertRaises(ValueError):resource_only(parent,broken,'old','new')
        broken=copy.deepcopy(child);broken['code_files']['inference.py']='changed'
        with self.assertRaises(ValueError):resource_only(parent,broken,'old','new')

    def test_serial_resume_and_hardware_mismatch_are_rejected(self):
        command=['python','entry','--arm','both'];serial_command(command,command)
        with self.assertRaises(ValueError):serial_command(command+['--resume'],command+['--resume'])
        rows=[['1','old','NVIDIA A100-SXM4-80GB','81920'],['2','new','NVIDIA A100-SXM4-80GB','81920']]
        gpu_pair(rows,'old','new');rows[1][2]='different'
        with self.assertRaises(ValueError):gpu_pair(rows,'old','new')

    def load_launcher(self):
        path=HERE/('launcher.py' if (HERE/'launcher.py').exists() else 'parallel_launcher.py')
        spec=importlib.util.spec_from_file_location('parallel_test_launcher',path);module=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'fcntl':types.SimpleNamespace()}):spec.loader.exec_module(module)
        return module

    def test_original_run_arm_guard_precedes_all_mutation_and_gpu_calls(self):
        module=self.load_launcher()
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);target=base/'full-v2/hybrid_v2';target.mkdir(parents=True)
            (target/'reservation').write_bytes(b'keep')
            with patch.object(module.run,'BASE',base),patch.object(module.run,'idle') as idle:
                with self.assertRaisesRegex(RuntimeError,'Existing arm requires explicit --resume'):
                    module.run.run_arm(None,None,None,None,None,'hybrid_v2',False)
                idle.assert_not_called()
            self.assertEqual([(p.name,p.read_bytes()) for p in target.iterdir()],[('reservation',b'keep')])

    def test_failed_reservation_always_resumes_supervisor_without_container_signal(self):
        module=self.load_launcher()
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);(base/'full-v2').mkdir();state={'stopped':False};sent=[]
            command=['python','entry','--arm','both'];pair={'original_supervisor_pid':123,'original_supervisor_command':command}
            def send(fd,sig):
                sent.append(sig)
                if sig==module.signal.SIGSTOP:
                    state['stopped']=True;(base/'full-v2/hybrid_v2').mkdir()
            def proc(pid):return {'pid':pid,'command':command,'state':'T' if state['stopped'] else 'R','start_ticks':'99'}
            with patch.object(module,'BASE',base),patch.object(module,'process',side_effect=proc),\
                 patch.object(module.os,'pidfd_open',return_value=999,create=True),patch.object(module.os,'close'),\
                 patch.object(module.signal,'pidfd_send_signal',side_effect=send,create=True),\
                 patch.object(module.signal,'SIGSTOP',19,create=True),patch.object(module.signal,'SIGCONT',18,create=True),\
                 patch.object(module.subprocess,'Popen') as watchdog,patch.object(module.run,'query') as docker:
                with self.assertRaisesRegex(RuntimeError,'Serial arm already exists'):module.reserve(pair)
                self.assertEqual(sent,[19,18]);watchdog.assert_called_once();docker.assert_not_called()
            receipt=json.loads((base/'parallel-switch-v1/SUPERVISOR_RELEASE.json').read_bytes())
            self.assertTrue(receipt['SIGCONT_sent']);self.assertFalse(receipt['container_stop_or_kill_sent'])


if __name__=='__main__':unittest.main()
