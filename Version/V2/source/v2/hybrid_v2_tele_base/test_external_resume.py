import copy
import json
from pathlib import Path
import unittest
import gpu_ownership_guard as guard
from external_resume_policy import external_record
HERE=Path(__file__).parent

class ExternalPolicyTests(unittest.TestCase):
    def fixture(self):
        return json.loads((HERE/'EXTERNAL_GUARD.json').read_bytes()),json.loads((HERE/'ORIGINAL_EXIT.json').read_bytes()),json.loads((HERE/'BINDINGS.json').read_bytes())
    def test_original_external_event_is_verified_without_mutation(self):
        record,exit_record,pins=self.fixture();original=copy.deepcopy((record,exit_record))
        external_record(record,exit_record,pins,guard)
        self.assertEqual((record,exit_record),original);self.assertEqual(exit_record['container']['State']['ExitCode'],137)
    def test_changed_pid_membership_or_missing_proof_is_rejected(self):
        record,exit_record,pins=self.fixture()
        for key,value in [('start_ticks','1'),('ppid',1),('cgroup',record['anchor_before']['cgroup']),('pid_namespace',record['anchor_before']['pid_namespace'])]:
            changed=copy.deepcopy(record);changed['after']['2387855'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):external_record(changed,exit_record,pins,guard)
        changed=copy.deepcopy(record);changed['decisions']={'2387855':'unknown'}
        with self.assertRaises(ValueError):external_record(changed,exit_record,pins,guard)
    def test_other_failure_or_container_is_not_admitted(self):
        record,exit_record,pins=self.fixture()
        for key,value in [('cid','other'),('time',exit_record['time']+1),('supervisor_error','error'),('reason','gpu_ownership_unresolved')]:
            with self.subTest(key=key),self.assertRaises(ValueError):external_record(record,{**exit_record,key:value},pins,guard)
        changed=copy.deepcopy(exit_record);changed['container']['State']['OOMKilled']=True
        with self.assertRaises(ValueError):external_record(record,changed,pins,guard)

if __name__=='__main__':unittest.main()
