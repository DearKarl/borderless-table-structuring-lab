import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
from resource_resume_policy import validate_foreign_exit,completed_candidate


class ResourceResumeTests(unittest.TestCase):
    def fixture(self):
        pin={'container_prefix':'known','exit_time':1790228534.206456,'foreign_pid':2218042}
        row={'reason':'foreign_gpu_process','supervisor_error':None,'time':pin['exit_time'],'cid':'known-exact',
             'foreign_processes':['GPU-original, 2218042'],'container':{'Id':'known-exact',
              'State':{'Running':False,'OOMKilled':False,'ExitCode':137,'Status':'exited'},
              'Config':{'Labels':{'hybrid.arm':'tele_raw','hybrid.freeze':'a'*64}}}}
        return pin,row

    def test_exact_record_is_admitted_without_changing_failure(self):
        pin,row=self.fixture();before=copy.deepcopy(row)
        validate_foreign_exit(row,pin,'tele_raw','a'*64,'GPU-original')
        self.assertEqual(row,before);self.assertEqual(row['reason'],'foreign_gpu_process')

    def test_oom_different_event_pid_or_identity_remain_blocked(self):
        pin,row=self.fixture()
        changes=[('reason','supervisor_error'),('supervisor_error','unknown'),('time',pin['exit_time']+1),
                 ('cid','other'),('foreign_processes',['GPU-original, 2218043'])]
        for key,value in changes:
            with self.subTest(key=key),self.assertRaises(ValueError):validate_foreign_exit({**row,key:value},pin,'tele_raw','a'*64,'GPU-original')
        for key,value in [('OOMKilled',True),('Running',True),('ExitCode',0)]:
            changed=copy.deepcopy(row);changed['container']['State'][key]=value
            with self.assertRaises(ValueError):validate_foreign_exit(changed,pin,'tele_raw','a'*64,'GPU-original')
        with self.assertRaises(ValueError):validate_foreign_exit(row,pin,'hybrid_v2','a'*64,'GPU-original')

    def test_completed_candidates_preserve_truncated_bytes_and_reject_corruption(self):
        page={'key':'p00686','page_id':'original','input_sha256':'c'*64}
        result={**page,'arm':'tele_raw','freeze_sha256':'a'*64,'status':'truncated','prediction_sha256':'b'*64,'bytes':7}
        completed_candidate(result,page,'tele_raw','a'*64,'b'*64,7)
        for changed in ({**result,'status':'running'},{**result,'status':'failed'},{**result,'page_id':'different'},
                        {**result,'prediction_sha256':'d'*64},{**result,'bytes':6}):
            with self.assertRaises(ValueError):completed_candidate(changed,page,'tele_raw','a'*64,'b'*64,7)


if __name__=='__main__':unittest.main()
