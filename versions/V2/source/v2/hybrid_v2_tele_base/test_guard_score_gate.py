from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).parent.parent/"parallel-code-v1/hybrid-code"))
from guard_score_gate import guard_header,reuse_pointer

class GuardScoreTests(unittest.TestCase):
    def test_unknown_origin_and_failed_exit_cannot_be_reclassified(self):
        expected={'code_lock_sha256':'a'*64,'original_exit_sha256':'b'*64}
        proof={**expected,'ownership_classification':'unknown','observed_guard_reason':'foreign_gpu_process',
            'original_failed_exit_preserved':True,'original_failure_reclassified_as_success':False,'signals_sent':False,
            'completed_pages_reused':646,'remaining_pages':1005}
        guard_header(proof,expected)
        for key,value in [('ownership_classification','external'),('ownership_classification','owned'),('original_failure_reclassified_as_success',True),('completed_pages_reused',644),('remaining_pages',1007),('signals_sent',True),('original_exit_sha256','c'*64)]:
            with self.subTest(key=key),self.assertRaises(ValueError):guard_header({**proof,key:value},expected)

    def test_same_content_in_reinferred_session_is_rejected(self):
        root=Path.cwd();source=root/'sessions/session-0002/pages/p00645/PAGE_RESULT.json'
        item={'receipt_sha256':'a'*64};terminal={'session_receipt_sha256':'a'*64,'session_receipt':'sessions/session-0002/pages/p00645/PAGE_RESULT.json'}
        reuse_pointer(terminal,item,root,source)
        with self.assertRaises(ValueError):reuse_pointer({**terminal,'session_receipt':'sessions/session-0003/pages/p00645/PAGE_RESULT.json'},item,root,source)
        with self.assertRaises(ValueError):reuse_pointer({**terminal,'session_receipt_sha256':'b'*64},item,root,source)

if __name__=='__main__':unittest.main()
