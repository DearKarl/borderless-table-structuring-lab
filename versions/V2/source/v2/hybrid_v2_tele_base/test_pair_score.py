import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent
sys.path.insert(0,str(HERE))


def load():
    spec=importlib.util.spec_from_file_location('pair_score_tested',HERE/'pair_score.py');module=importlib.util.module_from_spec(spec)
    fake=types.SimpleNamespace(flock=lambda *args:None,LOCK_EX=1,LOCK_NB=2)
    with patch.dict(sys.modules,{'fcntl':fake}):spec.loader.exec_module(module)
    return module


class PairTests(unittest.TestCase):
    def test_resource_completion_cannot_be_substituted_between_arms(self):
        m=load();pair={'arms':{'tele_raw':{'freeze_sha256':'a'*64},'hybrid_v2':{'freeze_sha256':'b'*64}},'input_manifest_sha256':'c'*64}
        raw={'arm':'tele_raw','freeze_sha256':'a'*64,'input_manifest_sha256':'c'*64,'pages':1651}
        m.completed_binding(raw,'tele_raw',pair)
        with self.assertRaises(ValueError):m.completed_binding(raw,'hybrid_v2',pair)
        with self.assertRaises(ValueError):m.completed_binding({**raw,'pages':1650},'tele_raw',pair)

    def test_two_distinct_resource_freezes_seal_native_truncation_and_detect_tampering(self):
        m=load()
        from hybrid_v2_tele_base.full_state import exclusive_json,exclusive_bytes,commit_page,sha,read
        with tempfile.TemporaryDirectory() as temporary,patch.object(m,'BASE',Path(temporary)):
            base=Path(temporary);pages=[{'key':'p00000','page_id':'fixture-only','input_sha256':'d'*64}]
            pair={'input_manifest_sha256':'c'*64,'arms':{}}
            for arm,identity in [('tele_raw','a'*64),('hybrid_v2','b'*64)]:
                root=base/'full-v2'/arm;session=root/'sessions/session-0001';folder=session/'pages/p00000'
                for path in (folder,root/'predictions',root/'terminals'):path.mkdir(parents=True)
                pair['arms'][arm]={'freeze_sha256':identity,'gpu_uuid':'test-'+arm,'output':str(root)}
                exclusive_bytes(folder/'prediction.md',b'native truncated fixture')
                exclusive_json(folder/'PAGE_RESULT.json',{**pages[0],'arm':arm,'freeze_sha256':identity,'status':'truncated',
                    'prediction_sha256':sha(folder/'prediction.md')})
                commit_page(root,session,pages[0],identity,arm)
                exclusive_json(session/'EXIT.json',{'reason':None,'container':{'State':{'OOMKilled':False,'Running':False,'Status':'exited','ExitCode':0}}})
                exclusive_json(session/'SESSION_RESULT.json',{'complete':True})
                exclusive_json(root/'COMPLETE.json',{'arm':arm,'freeze_sha256':identity,'input_manifest_sha256':'c'*64,'pages':1651})
            # Small state fixture isolates resource routing; full 1651 coverage is enforced by preflight.
            target=m.seal(pair,'e'*64,pages);lock=read(target);m.validate_dual(lock,pair,'e'*64,pages)
            self.assertNotEqual(lock['arms']['tele_raw']['freeze_sha256'],lock['arms']['hybrid_v2']['freeze_sha256'])
            self.assertEqual((base/'full-v2/hybrid_v2/predictions/fixture-only.md').read_bytes(),b'native truncated fixture')
            changed=copy.deepcopy(lock);changed['arms']['hybrid_v2']['freeze_sha256']='a'*64
            with self.assertRaises(ValueError):m.validate_dual(changed,pair,'e'*64,pages)
            (base/'full-v2/tele_raw/predictions/fixture-only.md').write_bytes(b'changed')
            with self.assertRaises(ValueError):m.validate_dual(lock,pair,'e'*64,pages)


if __name__=='__main__':unittest.main()
