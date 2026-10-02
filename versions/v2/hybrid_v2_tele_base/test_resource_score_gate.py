from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).parent.parent/'parallel-code-v1/hybrid-code'))
from resource_score_gate import header,check_idle

class ResourceScoreTests(unittest.TestCase):
    def test_reuse_counts_and_failure_preservation(self):
        expected={'completed_pages_reused':2,'remaining_pages':1649,'arm':'tele_raw'}
        proof={**expected,'original_failed_exit_preserved':True,'original_failure_reclassified_as_success':False,'signals_sent':False}
        audit={'read_only':True,'original_failure_preserved':True,'completed_pages':[{'key':'p00000'},{'key':'p00001'}],'completed_native_pages':2,'remaining_after_reconciliation':1649}
        header(proof,audit,expected)
        for key,value in [('completed_pages_reused',3),('remaining_pages',1648),('original_failed_exit_preserved',False),('original_failure_reclassified_as_success',True),('signals_sent',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):header({**proof,key:value},audit,expected)
        with self.assertRaises(ValueError):header(proof,{**audit,'completed_pages':[{'key':'p00000'},{'key':'p00000'}]},expected)

    def test_idle_is_bound_to_original_gpu(self):
        snapshot={'inventory':'1, GPU-original, 0, 0\n2, GPU-other, 1000, 50','processes':'GPU-other, 33, customer, 1000'}
        check_idle(snapshot,'GPU-original',1)
        for value in [{'inventory':snapshot['inventory'],'processes':'GPU-original, 44, customer, 100'},
                      {'inventory':'1, GPU-original, 1000, 0','processes':''},
                      {'inventory':'2, GPU-original, 0, 0','processes':''}]:
            with self.assertRaises(ValueError):check_idle(value,'GPU-original',1)

if __name__=='__main__':unittest.main()
