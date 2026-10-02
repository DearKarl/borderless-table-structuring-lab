import copy
import json
from pathlib import Path
import unittest
from gpu_ownership_guard import classify,scan

CID='a'*64

def identity(pid,group=None,ns='pid:[1]',start='10'):
    return {'pid':pid,'start_ticks':start,'ppid':1,'cgroup':'0::'+(group or '/docker/'+CID)+'\n','pid_namespace':ns,'mount_namespace':'mnt:[1]'}

class GuardTests(unittest.TestCase):
    def test_late_child_is_owned_only_with_stable_membership(self):
        root=identity(1);child=identity(2)
        self.assertEqual(classify(2,child,child,root,root,{1,2},{2},CID),'owned_confirmed')
        self.assertEqual(classify(2,child,child,root,root,{1},{2},CID),'unknown')
        self.assertEqual(classify(2,child,{**child,'start_ticks':'11'},root,root,{1,2},{2},CID),'unknown')
        self.assertEqual(classify(2,child,child,root,{**root,'start_ticks':'11'},{1,2},{2},CID),'unknown')

    def test_external_and_ambiguous_membership(self):
        root=identity(1);other=identity(2,'/other','pid:[2]')
        self.assertEqual(classify(2,other,other,root,root,{1},{2},CID),'external_confirmed')
        self.assertEqual(classify(2,other,other,root,root,{1,2},{2},CID),'unknown')
        self.assertEqual(classify(2,{'error':'PermissionError'},other,root,root,{1},{2},CID),'unknown')
        self.assertEqual(classify(2,other,other,root,root,{1},set(),CID),'gone_from_gpu')

    def run_scan(self,second_top,second_gpu,snapshots):
        replies=iter(['PID\n1','GPU-test, 2',second_top,second_gpu]);saved=[]
        class Folder:
            def __init__(self):pass
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            result=scan(lambda _:next(replies),lambda p,v:saved.append(v),Path(directory),CID,{'State':{'Pid':1}},'GPU-test',lambda p:copy.deepcopy(snapshots[p]))
        self.assertEqual(len(saved),1);self.assertFalse(saved[0]['gpu_process_signals_sent'])
        return result,saved[0]

    def test_scan_rechecks_and_preserves_late_child_evidence(self):
        result,evidence=self.run_scan('PID\n1\n2','GPU-test, 2',{1:identity(1),2:identity(2)})
        self.assertEqual(result,(None,[]));self.assertEqual(evidence['decisions'][2],'owned_confirmed')

    def test_scan_external_stops_without_signalling_pid(self):
        result,evidence=self.run_scan('PID\n1','GPU-test, 2',{1:identity(1),2:identity(2,'/other','pid:[2]')})
        self.assertEqual(result,('foreign_gpu_process',['GPU-test, 2']))

    def test_scan_unknown_stops_and_disappearance_is_not_claimed_owned(self):
        result,_=self.run_scan('PID\n1','GPU-test, 2',{1:identity(1),2:{'pid':2,'error':'PermissionError'}})
        self.assertEqual(result[0],'gpu_ownership_unresolved')
        result,evidence=self.run_scan('PID\n1','',{1:identity(1),2:{'pid':2,'error':'FileNotFoundError'}})
        self.assertEqual(result,(None,[]));self.assertEqual(evidence['decisions'][2],'gone_from_gpu')

    def test_new_second_snapshot_pid_is_not_silently_admitted(self):
        result,evidence=self.run_scan('PID\n1','GPU-test, 3',{1:identity(1),2:identity(2),3:identity(3)})
        self.assertEqual(result[0],'gpu_ownership_unresolved');self.assertEqual(evidence['decisions'][3],'unknown')

if __name__=='__main__':unittest.main()
