from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent.parent/'full-code-v2'))
from deadline_score_gate import deadline_header,missing_assignment

class DeadlineScoreTests(unittest.TestCase):
    def test_failed_page_policy_and_counts_cannot_change(self):
        proof={'arm':'tele_raw','completed_pages_reused':1368,'remaining_pages':283,'timeout_page_retained_failed':'p01367',
            'timeout_seconds':900,'original_validator_unchanged':True,'original_failed_page_preserved':True,
            'original_failure_reclassified_as_success':False,'signals_sent':False,'code_lock_sha256':'a'*64}
        deadline_header(proof,{'code_lock_sha256':'a'*64})
        for key,value in [('remaining_pages',284),('timeout_seconds',1800),('original_validator_unchanged',False),
                          ('original_failure_reclassified_as_success',True),('timeout_page_retained_failed','p01368')]:
            with self.subTest(key=key),self.assertRaises(ValueError):deadline_header({**proof,key:value},{'code_lock_sha256':'a'*64})
    def test_assignment_rejects_timeout_page_duplicates_and_changed_inputs(self):
        pages=[{'key':'p%05d'%i,'page_id':str(i),'input_sha256':str(i)} for i in range(1651)]
        rows=pages[1368:];missing_assignment(rows,pages)
        for altered in ([pages[1367]]+rows[1:],rows[:-1]+[rows[0]],[{**rows[0],'input_sha256':'changed'}]+rows[1:]):
            with self.assertRaises(ValueError):missing_assignment(altered,pages)

if __name__=='__main__':unittest.main()
