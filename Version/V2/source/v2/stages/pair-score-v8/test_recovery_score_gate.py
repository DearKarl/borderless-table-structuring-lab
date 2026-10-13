import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parent))
sys.path.insert(0,str(Path(__file__).parent.parent/'parallel-code-v1/hybrid-code'))
from recovery_score_gate import admission_header,takeover_identity


class RecoveryGateTests(unittest.TestCase):
    def test_wrong_tele_recovery_revision_or_reclassified_failure_is_rejected(self):
        expected={'recovery_entry_sha256':'a'*64,'policy_sha256':'b'*64,'reconciled_terminal_count':206}
        proof={**expected,'original_exit_preserved':True,'original_failure_reclassified_as_success':False}
        admission_header(proof,expected)
        for key,value in [('recovery_entry_sha256','c'*64),('policy_sha256','c'*64),('reconciled_terminal_count',205),
                          ('original_exit_preserved',False),('original_failure_reclassified_as_success',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):admission_header({**proof,key:value},expected)

    def test_container_or_native_pid_replacement_is_not_a_valid_hybrid_takeover(self):
        pins={'old_hybrid_supervisor_pid':1738626,'old_hybrid_start_ticks':'45622200','old_hybrid_command':['original']}
        ready={'old_supervisor':{'pid':1738626,'start_ticks':'45622200','command':['original']},
               'container_before':{'Id':'baee-exact','State':{'Running':True,'Pid':1738801},
                   'Config':{'Labels':{'hybrid.freeze':'d'*64,'hybrid.arm':'hybrid_v2'}}}}
        signals={'old_pid':1738626,'old_start_ticks':'45622200','SIGSTOP_sent':True,'SIGKILL_sent_to_exact_host_only':True,
                 'container_signalled':False,'docker_attach_signalled':False}
        preserved={'cid':'baee-exact','running':True,'native_pid_before':1738801,'native_pid_after':1738801,'native_model_reinitialized':False}
        host={'pid':1764764,'action':'takeover-hybrid','freeze_sha256':'d'*64}
        takeover_identity(ready,signals,preserved,host,pins,'d'*64,'baee-exact')
        for key,value in [('cid','different-container'),('native_pid_after',1738802),('native_model_reinitialized',True)]:
            with self.subTest(key=key),self.assertRaises(ValueError):takeover_identity(ready,signals,{**preserved,key:value},host,pins,'d'*64,'baee-exact')
        with self.assertRaises(ValueError):takeover_identity(ready,{**signals,'container_signalled':True},preserved,host,pins,'d'*64,'baee-exact')


if __name__=='__main__':unittest.main()
