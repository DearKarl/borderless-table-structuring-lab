import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).parent.parent/'full-code-v2'))
from tele_deadline_evidence import deadline_outcome
from hybrid_v2_tele_base.full_state import validate_session_exit
HERE=Path(__file__).parent

class DeadlineTests(unittest.TestCase):
    def fixture(self):
        return [json.loads((HERE/name).read_bytes()) for name in ('ORIGINAL_EXIT.json','ORIGINAL_SESSION_RESULT.json','ORIGINAL_PAGE_RESULT.json','BINDINGS.json')]
    def test_actual_native_deadline_passes_original_frozen_validator(self):
        exit_record,result,page,pins=self.fixture();before=copy.deepcopy((exit_record,result,page))
        deadline_outcome(exit_record,result,page,pins)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'EXIT.json').write_text(json.dumps(exit_record));(root/'SESSION_RESULT.json').write_text(json.dumps(result))
            validate_session_exit(root)
        self.assertEqual((exit_record,result,page),before);self.assertEqual(page['status'],'failed');self.assertEqual(page['bytes'],0)
    def test_no_generic_exit2_or_host_timeout_admission(self):
        exit_record,result,page,pins=self.fixture()
        for key,value in [('reason','page_timeout'),('foreign_processes',['GPU, 1']),('supervisor_error','unknown'),('cid','other')]:
            with self.subTest(key=key),self.assertRaises(ValueError):deadline_outcome({**exit_record,key:value},result,page,pins)
        with self.assertRaises(ValueError):deadline_outcome(exit_record,{**result,'blocked':True},page,pins)
        changed=copy.deepcopy(exit_record);changed['container']['State']['OOMKilled']=True
        with self.assertRaises(ValueError):deadline_outcome(changed,result,page,pins)
    def test_timeout_page_cannot_be_relabelled_or_extended(self):
        exit_record,result,page,pins=self.fixture()
        for key,value in [('status','success'),('bytes',1),('key','p01368'),('elapsed_seconds',1800),('prediction_sha256','a'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):deadline_outcome(exit_record,result,{**page,key:value},pins)

if __name__=='__main__':unittest.main()
