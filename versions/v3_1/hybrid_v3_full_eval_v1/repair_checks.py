"""R1-R4 CPU failure injection and regression. No Docker/model/GPU/scorer execution."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from . import cpu_checks as original
from .core import ContractError,Deadline,read,sha,write
from .collection import collect,verify_collection
from .completion import LIFECYCLE_FILES
from .lifecycle import OwnedDocker
from .official_entry import summarize
from .cpu_fixtures import official_fixture
from .request_binding import BindingGraph,bind_backend

class DockerFake:
    def __init__(self,now,mode="normal"):
        self.now,self.mode=now,mode
        self.commands=[];self.removed=False;self.running=True;self.inspect_calls=0
    def state(self):
        return {"Id":"cid","Config":{"Labels":{"hybrid-v3-owner":"owned"}},
                "State":{"Running":self.running,"Pid":42 if self.running else 0,
                         "ExitCode":0,"OOMKilled":False}}
    def __call__(self,args,**kwargs):
        self.commands.append((args,kwargs["timeout"]))
        if not 0<kwargs["timeout"]<=240:raise AssertionError("Unbounded cleanup command")
        self.now[0]+=0.1
        if args[1]=="inspect":
            self.inspect_calls+=1
            if self.mode=="slow_inspect" and self.inspect_calls==1:
                self.now[0]+=kwargs["timeout"]
                raise subprocess.TimeoutExpired(args,kwargs["timeout"])
            if self.removed:return subprocess.CompletedProcess(args,1,"","No such container")
            return subprocess.CompletedProcess(args,0,json.dumps([self.state()]),"")
        if args[1]=="kill":
            self.running=False
            if self.mode=="kill_race":return subprocess.CompletedProcess(args,1,"","container is not running")
        if args[1]=="logs" and self.mode=="logs_timeout":
            self.now[0]+=kwargs["timeout"]
            raise subprocess.TimeoutExpired(args,kwargs["timeout"])
        if args[1]=="rm":
            if self.mode=="remove_failure":return subprocess.CompletedProcess(args,1,"","daemon unavailable")
            self.removed=True
        return subprocess.CompletedProcess(args,0,"","")

class Image:
    mode="RGB";width=2;height=2
    def __init__(self,value):self.value=value
    def tobytes(self):return self.value.encode()*12
class Tensor:
    shape=(1,)
    def __init__(self,value):self.value=value
def tensor_binding(items):
    return {k:{"shape":[1],"dtype":"synthetic","sha256":hashlib.sha256(v.value.encode()).hexdigest()}
            for k,v in items.items()}
class Feature(dict):
    def to(self,*args,**kwargs):return self
class Processor:
    def __init__(self):self.calls=0
    def __call__(self,**kwargs):
        self.calls+=1
        self.feature=Feature({k:Tensor(kwargs["images"][0].value+k)
                              for k in ("input_ids","pixel_values","image_grid_thw")})
        return self.feature
class Backend:
    def __init__(self,graph):
        self.graph=graph;self.processor=Processor();self.audits=[]
    def _predict_one_batch(self,image_objs,chat_prompts,sampling_params,**kwargs):
        feature=self.processor(images=image_objs,text=chat_prompts).to("synthetic")
        self.graph.generating(feature)
        audit={"returned":True,"input_tensors":tensor_binding(feature),
               "effective_generation_config":{"max_length":128000},
               "termination":{"stop":"eos","token_ids":[2]}}
        self.graph.generated(audit);self.audits.append(audit)
        return ["decoded-"+image_objs[0].value]

class RepairChecks(original.Checks):
    def test_r1_single_and_last_system_failure_not_formal(self):
        base=self.root
        for count,index in ((1,0),(3,2)):
            folder=base/str(count);folder.mkdir();self.root=folder
            run=self.run_fixture(count,failed=(index,))
            data=read(run);self.assertTrue(data["all_slots_terminal"])
            with self.assertRaises(ContractError):collect(run,folder/"formal")
            diagnostic=collect(run,folder/"diagnostic",True)
            self.assertEqual(diagnostic["page_count"],count)
            with self.assertRaises(ContractError):verify_collection(run,folder/"diagnostic/COLLECTED_LOCK.json")
            data["states"]["processing_complete"]=True
            run.write_text(json.dumps(data))
            with self.assertRaises(ContractError):collect(run,folder/"forged-normal")

    def test_r1_success_missing_release_or_cleanup_rejected(self):
        base=self.root
        for index,name in enumerate(("CLEANUP.json","GPU_RELEASE.json","SESSION_RESULT.json")):
            folder=base/str(index);folder.mkdir();self.root=folder
            run=self.run_fixture(1)
            (run.parent/name).unlink()
            with self.assertRaises(ContractError):collect(run,folder/"formal")

    def test_r1_all_lifecycle_tampering_rejected(self):
        run=self.run_fixture(1)
        for name in LIFECYCLE_FILES:
            p=run.parent/name;old=p.read_bytes();p.write_bytes(old+b" ")
            with self.subTest(name=name),self.assertRaises(ContractError):collect(run,self.root/("bad-"+name))
            p.write_bytes(old)

    def test_r1_normal_lifecycle_positive_and_lock_coverage(self):
        run=self.run_fixture(2)
        result=collect(run,self.root/"collected")
        self.assertTrue(result["processing_complete"])
        lock=self.root/"collected/COLLECTED_LOCK.json";verify_collection(run,lock)
        data=read(lock);del data["run_evidence"]["HOST_EXIT.json"];lock.write_text(json.dumps(data))
        with self.assertRaises(ContractError):verify_collection(run,lock)

    def cleanup_case(self,mode="normal",expired=False,release=None):
        now=[1001. if expired else 100.]
        fake=DockerFake(now,mode)
        d=OwnedDocker("owned",Deadline(0,1000,240,clock=lambda:now[0]),self.root,fake)
        d.cid="cid";d.creation_attempted=True;d.last_verified=fake.state()
        result=d.cleanup(release)
        self.assertLessEqual(len(fake.commands),9)
        self.assertLessEqual(now[0]-(1001 if expired else 100),240)
        return result,fake

    def test_r2_expired_budget_still_stops_owned(self):
        result,fake=self.cleanup_case(expired=True)
        self.assertTrue(result["removed"]);self.assertTrue(result["budget_exceeded"])
        self.assertFalse(result["complete"])
        self.assertIn("kill",[a[1] for a,_ in fake.commands])

    def test_r2_slow_inspect_still_removes_confirmed_owned(self):
        result,fake=self.cleanup_case("slow_inspect")
        self.assertTrue(result["removed"]);self.assertFalse(result["complete"])

    def test_r2_logs_timeout_does_not_skip_remove(self):
        result,fake=self.cleanup_case("logs_timeout")
        self.assertTrue(result["removed"]);self.assertFalse(result["complete"])
        self.assertLess([a[1] for a,_ in fake.commands].index("logs"),[a[1] for a,_ in fake.commands].index("rm"))

    def test_r2_log_disk_error_does_not_skip_remove(self):
        with patch.object(Path,"write_text",side_effect=OSError("synthetic full disk")):
            result,fake=self.cleanup_case()
        self.assertTrue(result["removed"]);self.assertFalse(result["complete"])

    def test_r2_exited_kill_race_is_preserved(self):
        result,fake=self.cleanup_case("kill_race")
        self.assertTrue(result["removed"]);self.assertTrue(result["complete"])
        self.assertTrue(result["recovered_errors"][0]["recovered_kill_race"])
        self.assertEqual(result["final_state"]["State"]["Pid"],0)

    def test_r2_remove_failure_not_complete(self):
        result,fake=self.cleanup_case("remove_failure")
        self.assertFalse(result["complete"]);self.assertEqual(result["unresolved_owned_cid"],"cid")

    def test_r2_fresh_release_failure_not_complete(self):
        def failure(_):raise ContractError("GPU still busy")
        result,fake=self.cleanup_case(release=failure)
        self.assertTrue(result["removed"]);self.assertFalse(result["complete"])

    def test_r2_fresh_foreign_identity_overrides_cached_owner(self):
        now=[100.]
        known={"Id":"cid","Config":{"Labels":{"hybrid-v3-owner":"owned"}},
               "State":{"Running":True,"Pid":2,"ExitCode":0}}
        commands=[]
        def runner(args,**kwargs):
            commands.append(args)
            foreign=copy.deepcopy(known);foreign["Config"]["Labels"]["hybrid-v3-owner"]="foreign"
            return subprocess.CompletedProcess(args,0,json.dumps([foreign]),"")
        d=OwnedDocker("owned",Deadline(0,1000,240,clock=lambda:now[0]),self.root,runner)
        d.cid="cid";d.creation_attempted=True;d.last_verified=known
        result=d.cleanup()
        self.assertFalse(result["complete"])
        self.assertTrue(all(x[1]=="inspect" for x in commands))

    def test_r1_dropped_page_audit_from_collection_rejected(self):
        run=self.run_fixture(1);collect(run,self.root/"collected")
        lock=self.root/"collected/COLLECTED_LOCK.json";value=read(lock)
        del value["run_evidence"]["pages/p0/GENERATION.json"]
        lock.write_text(json.dumps(value))
        with self.assertRaises(ContractError):verify_collection(run,lock)

    def test_r3_required_file_missing(self):
        m=official_fixture(self.root)
        (self.root/"pred_quick_match_table_result.json").unlink()
        with self.assertRaises(FileNotFoundError):summarize(m,self.root,2)

    def test_r3_nonempty_missing_metric(self):
        m=official_fixture(self.root)
        del m["table"]["page"]["TEDS"]
        with self.assertRaises(ValueError):summarize(m,self.root,2)

    def test_r3_bad_metric_types(self):
        m=official_fixture(self.root)
        for value in (None,True,"0.5",{},float("inf"),"NaN"):
            bad=copy.deepcopy(m);bad["table"]["page"]["TEDS"]["ALL"]=value
            with self.subTest(value=str(value)),self.assertRaises(ValueError):summarize(bad,self.root,2)

    def test_r3_bad_result_schema(self):
        m=official_fixture(self.root)
        (self.root/"pred_quick_match_table_result.json").write_text("{}")
        with self.assertRaises(ValueError):summarize(m,self.root,2)

    def test_r3_defined_original_rounding(self):
        m=official_fixture(self.root)
        m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"]=.123456
        m["display_formula"]["page"]["CDM"]["ALL"]=.941234
        m["table"]["page"]["TEDS"]["ALL"]=.887654
        result=summarize(m,self.root,2)
        self.assertAlmostEqual(result["raw_overall"],((1-.123456)*100+94.1234+88.7654)/3)
        self.assertAlmostEqual(result["notebook_overall_after_component_rounding"],((1-.123)*100+94.123+88.765)/3)

    def test_r3_real_empty_and_zero_length_evidence(self):
        m=official_fixture(self.root,empty=("display_formula","table"))
        result=summarize(m,self.root,2);self.assertIsNone(result["raw_overall"])
        sample={"img_id":"p0.png","gt":"","pred":"","upper_len":0,"Edit_num":0,"metric":{}}
        (self.root/"pred_quick_match_text_block_result.json").write_text(json.dumps([sample]))
        m["text_block"]["all"]["Edit_dist"]["ALL_page_avg"]=float("nan")
        m["text_block"]["page"]={}
        report=read(self.root/"pred_quick_match_run_summary.json")
        report["page_denominators"]["text_block"]["Edit_dist"]["ALL"]=0
        (self.root/"pred_quick_match_run_summary.json").write_text(json.dumps(report))
        result=summarize(m,self.root,2)
        self.assertIn("text_block_Edit_dist",result["undefined_metrics"])

    def graph(self,page="p0"):
        return BindingGraph("run",{"page_id":page,"file_sha256":"a"*64},"b"*64,tensor_binding)

    def test_r4_sparse_indices_layout_and_new_slot(self):
        g=self.graph();backend=Backend(g);original_processor=backend.processor
        restore=bind_backend(backend,g)
        layout=Image("layout");g.register(layout,"layout")
        backend._predict_one_batch([layout],["layout prompt"],None)
        first,second=Image("first"),Image("second")
        bindings=g.prepare_content(([first,second],["a","b"],[None,None],[2,9]),[None]*10)
        # Dispatch in reverse order; identities still map to original slots.
        backend._predict_one_batch([second],["b"],None)
        backend._predict_one_batch([first],["a"],None)
        added=Image("added");g.register(added,"content","p0:added:proposal42")
        backend._predict_one_batch([added],["c"],None)
        records=g.validate_complete()
        self.assertEqual([a["slot_id"] for a in backend.audits],[None,"p0:native:9","p0:native:2","p0:added:proposal42"])
        self.assertNotEqual(backend.audits[0]["request_id"],bindings[0]["request_id"])
        self.assertEqual(original_processor.calls,4)
        self.assertEqual(len(records),4)
        restore();self.assertIs(backend.processor,original_processor)

    def test_r4_lengths_duplicates_unknown_and_missing(self):
        g=self.graph();a,b=Image("a"),Image("b")
        for returned in (([a],[],[None],[2]),([a,b],["a","b"],[None,None],[2,2]),
                         ([a],["a"],[None],[99])):
            with self.assertRaises(ContractError):g.prepare_content(returned,[None]*10)
        with self.assertRaises(ContractError):g.begin([a],["unknown"],None)
        g.prepare_content(([a],["a"],[None],[2]),[None]*10)
        with self.assertRaises(ContractError):g.validate_complete()

    def test_r4_cross_page_replay_and_duplicate_dispatch(self):
        g=self.graph();image=Image("a");g.register(image,"layout")
        backend=Backend(g);bind_backend(backend,g);backend._predict_one_batch([image],["x"],None)
        with self.assertRaises(ContractError):backend._predict_one_batch([image],["x"],None)
        other=self.graph("p1")
        with self.assertRaises(ContractError):other.begin([image],["x"],None)
        with self.assertRaises(ContractError):other.register(Image("b"),"content","p0:native:2")

    def test_r4_processor_tensor_swap_and_audit_replay(self):
        g=self.graph();image=Image("a");g.register(image,"layout");g.begin([image],["x"],None)
        feature=Feature({k:Tensor(k) for k in ("input_ids","pixel_values","image_grid_thw")})
        g.processor([image],feature).to()
        bad={**feature,"pixel_values":Tensor("pixel_values")}
        with self.assertRaises(ContractError):g.generating(bad)
        g.generating(feature)
        audit={"returned":True,"input_tensors":tensor_binding(feature),
            "effective_generation_config":{"max_length":128000},"termination":{"stop":"eos"},"page_id":"p9"}
        with self.assertRaises(ContractError):g.generated(audit)

def main():
    import argparse,io
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False);original.ROOT=root
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RepairChecks))
    (root/"checks.log").write_text(stream.getvalue())
    receipt={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
             "passed":result.wasSuccessful(),"gpu":0,"model_loads":0,"scoring":0,"training":0,
             "scope":"R1-R4 synthetic fault injection plus changed-framework regressions"}
    write(root/"CPU_CHECKS.json",receipt)
    print(stream.getvalue());print(json.dumps(receipt))
    return 0 if result.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
