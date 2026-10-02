from pathlib import Path
import json
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).parent.parent/'parallel-code-v1/hybrid-code'))
sys.path.append(str(Path(__file__).parent.parent/'hybrid-external2-resume-v1'))
from external2_score_gate import external2_header
from external2_resume_policy import continuation_receipts
import gpu_ownership_guard as guard

class ExternalScoreTests(unittest.TestCase):
    def test_exact_external_admission_coverage(self):
        expected={'guard_record_sha256':'a'*64,'original_exit_sha256':'b'*64}
        proof={**expected,'ownership_classification':'external_confirmed','observed_guard_reason':'foreign_gpu_process',
            'original_failed_exit_preserved':True,'original_failure_reclassified_as_success':False,'signals_sent':False,
            'completed_pages_reused':805,'remaining_pages':846}
        external2_header(proof,expected)
        for key,value in [('ownership_classification','unknown'),('completed_pages_reused',646),('remaining_pages',1005),('original_failure_reclassified_as_success',True),('guard_record_sha256','c'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):external2_header({**proof,key:value},expected)
    def test_prior_failed_host_requires_exact_exit_guard_and_result_hashes(self):
        source=Path(__file__).parent.parent/'hybrid-external2-resume-v1'
        pins=json.loads((source/'BINDINGS.json').read_bytes())
        original=json.loads((source/'ORIGINAL_EXIT.json').read_bytes());record=json.loads((source/'EXTERNAL_GUARD.json').read_bytes())
        values={'EXIT.json':original,pins['guard_record_name']:record,'HOST_RESULT.json':{'exit_code':2,'original_failures_preserved':True}}
        hashes={'EXIT.json':pins['interrupted_exit_sha256'],pins['guard_record_name']:pins['guard_record_sha256'],'HOST_RESULT.json':pins['prior_host_result_sha256']}
        continuation_receipts(Path('/test'),pins,lambda p:values[p.name],lambda p:hashes[p.name],guard)
        for name in hashes:
            with self.subTest(name=name),self.assertRaises(ValueError):continuation_receipts(Path('/test'),pins,lambda p:values[p.name],lambda p:'f'*64 if p.name==name else hashes[p.name],guard)
        values['HOST_RESULT.json']={'exit_code':0,'original_failures_preserved':True}
        with self.assertRaises(ValueError):continuation_receipts(Path('/test'),pins,lambda p:values[p.name],lambda p:hashes[p.name],guard)

if __name__=='__main__':unittest.main()
