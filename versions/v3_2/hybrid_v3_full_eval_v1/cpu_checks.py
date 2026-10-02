"""Targeted CPU contracts only; synthetic records are never runtime/quality proof."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from .core import ContractError, Deadline, DeadlineError, closed, read, sha, write, verify_tree
from .contracts import inputs, budget, runtime, response_identity, route_slots, PROFILES
from .collection import collect, verify_collection, terminals
from .evaluation import gt_subset
from .lifecycle import terminal_reason, OwnedDocker, mount_args, limits, verify_isolation
from .official_entry import summarize
from .cpu_fixtures import lifecycle_fixture, official_fixture

ROOT=None

class Checks(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix="t-",dir=ROOT))
        write(self.root/"CASE.json",{"test":self._testMethodName})

    def manifest(self,n=3):
        pages=[]
        for i in range(n):
            f=self.root/f"p{i}.png"; f.write_bytes(b"CPU identity fixture "+str(i).encode())
            pages.append({"page_id":f"p{i}","kind":"image","path":f.name,
                          "file_sha256":sha(f),"source_document_id":"unknown"})
        path=self.root/"INPUT.json";write(path,{"schema":1,"pages":pages});return path

    def run_fixture(self,n=3,failed=(),unrun=()):
        path=self.manifest(n);m=read(path);run=self.root/"run";run.mkdir()
        ids=[p["page_id"] for p in m["pages"]]
        for i,p in enumerate(m["pages"]):
            if i in unrun:continue
            folder=run/"pages"/p["page_id"];folder.mkdir(parents=True)
            start={"run_id":"fixture","page_id":p["page_id"],"input_sha256":p["file_sha256"]}
            write(folder/"PAGE_START.json",start)
            (folder/"prediction.md").write_bytes(b"" if i in failed else b"fixture text")
            write(folder/"PAGE_RESULT.json",{**start,"start_sha256":sha(folder/"PAGE_START.json"),
                "prediction_sha256":sha(folder/"prediction.md"),"status":"failed" if i in failed else "success"})
            if i not in failed:
                for name in ("INPUT","RENDER","GEOMETRY","CONTENT","GENERATION","REQUEST_BINDINGS","RESOURCE_AUDIT"):
                    write(folder/(name+".json"),{})
                write(folder/"AUDIT_STATUS.json",{"complete":True,"patches_restored":True,"error":None})
        lifecycle=lifecycle_fixture(run,m,failed,unrun)
        write(run/"RUN_MANIFEST.json",{"run_id":"fixture","profile":"native","inputs":m,
            "claims":ids,"all_slots_terminal":not unrun,"lifecycle":lifecycle,
            "host_failure":read(run/"HOST_EXIT.json")["failure"],
            "cleanup":read(run/"CLEANUP.json"),
            "actual_page_starts":[x for i,x in enumerate(ids) if i not in unrun],
            "actual_page_results":[x for i,x in enumerate(ids) if i not in unrun],
            "unresolved":[x for i,x in enumerate(ids) if i in unrun],
            "states":{"processing_complete":not unrun and not failed,
                      "runtime_verified":True,"integration_verified":not unrun and not failed}})
        return run/"RUN_MANIFEST.json"

    def test_arbitrary_n_order(self):
        for n in (1,7,19):
            folder=self.root/str(n);folder.mkdir()
            old=self.root;self.root=folder
            run=self.run_fixture(n)
            result=collect(run,folder/"collected")
            self.assertEqual(result["page_count"],n)
            self.assertEqual([x["page_id"] for x in result["predictions"]],[f"p{i}" for i in range(n)])
            self.root=old

    def test_failed_page_retains_denominator(self):
        run=self.run_fixture(4,failed=(1,3))
        result=collect(run,self.root/"collected",diagnostic=True)
        self.assertEqual(len(result["predictions"]),4)
        self.assertEqual(sum(x["status"]=="failed" for x in result["predictions"]),2)
        self.assertEqual((self.root/"collected/predictions/p1.md").read_bytes(),b"")

    def test_unrun_rejects_primary(self):
        run=self.run_fixture(3,unrun=(2,))
        with self.assertRaisesRegex(ContractError,"primary"):collect(run,self.root/"collected")
        result=collect(run,self.root/"diagnostic",True)
        self.assertFalse(result["processing_complete"])
        with self.assertRaises(ContractError):verify_collection(run,self.root/"diagnostic/COLLECTED_LOCK.json")
        verify_collection(run,self.root/"diagnostic/COLLECTED_LOCK.json",True)

    def test_false_processing_complete_rejected(self):
        run=self.run_fixture(2,unrun=(1,))
        data=read(run);data["states"]["processing_complete"]=True
        run.write_text(json.dumps(data))
        with self.assertRaises(ContractError):terminals(run)

    def test_duplicate_manifest_id(self):
        path=self.manifest();data=read(path);data["pages"][1]["page_id"]="P0"
        path.write_text(json.dumps(data))
        with self.assertRaises(ContractError):inputs(path)

    def test_closed_path(self):
        for name in ("../escape","/absolute","C:/escape","a/../b","a\\b"):
            with self.subTest(name=name),self.assertRaises(ContractError):closed(self.root,name)

    def test_input_tamper(self):
        path=self.manifest();(self.root/"p0.png").write_bytes(b"changed")
        with self.assertRaises(ContractError):inputs(path)

    def test_pdf_ordinal_contract(self):
        path=self.manifest(1);data=read(path);p=data["pages"][0]
        p.update(kind="pdf",page_count=3,page_ordinal=2,source_sha256=p["file_sha256"])
        path.write_text(json.dumps(data));inputs(path)
        p["page_ordinal"]=3;path.write_text(json.dumps(data))
        with self.assertRaises(ContractError):inputs(path)

    def test_all_expert_profiles_fail_explicitly(self):
        path=self.root/"runtime.json";write(path,{})
        for name in PROFILES:
            if name!="native":
                with self.subTest(profile=name),self.assertRaisesRegex(ContractError,"PROFILE_UNBOUND"):
                    runtime(path,name,{})

    def test_sparse_off_and_passthrough(self):
        slots=[{"slot_id":i,"category":c,"content":object()} for i,c in ((2,"text"),(9,"formula"),(31,"table"))]
        for mode in ("off","pass-through"):
            calls=[]
            def native(items):calls.append(items);return items
            result=route_slots(slots,mode,"formula",native)
            self.assertIs(result,slots);self.assertIs(calls[0],slots)
            self.assertEqual([x["slot_id"] for x in result],[2,9,31])

    def test_non_target_unchanged(self):
        slots=[{"slot_id":2,"category":"table","content":object()},
               {"slot_id":9,"category":"formula","content":"x"}]
        out=route_slots(slots,"on","formula",lambda x:x,lambda x:{**x,"content":"y"})
        self.assertIs(out[0],slots[0]);self.assertEqual(out[1]["slot_id"],9)

    def test_system_errors_propagate(self):
        class CudaOOM(RuntimeError):pass
        for error in (CudaOOM("oom"),ContractError("identity"),OSError("audit disk"),RuntimeError("unknown")):
            def expert(_):raise error
            with self.subTest(error=type(error)),self.assertRaises(type(error)):
                route_slots([{"slot_id":7,"category":"text"}],"on","text",lambda _:self.fail("fallback"),expert)

    def test_cross_page_tensor_response_rejected(self):
        request={k:k+"-bound" for k in ("run_id","page_id","slot_id","request_id","input_sha256",
                                      "crop_rgb_sha256","tensor_sha256","model_sha256","config_sha256")}
        response_identity(request,request.copy())
        for key in request:
            bad={**request,key:"other"}
            with self.subTest(key=key),self.assertRaises(ContractError):response_identity(request,bad)

    def test_absolute_deadline_no_reset(self):
        now=[100.]
        d=Deadline(90,30,5,clock=lambda:now[0])
        self.assertEqual(d.bound(600),15)
        now[0]=114;self.assertEqual(d.bound(900),1)
        now[0]=115
        with self.assertRaises(DeadlineError):d.remaining()
        self.assertEqual(d.remaining(cleanup=True),5)

    def test_exit_oom_priority(self):
        self.assertEqual(terminal_reason({"State":{"Running":False,"ExitCode":137,"OOMKilled":True}},1000,0,1),"oom")
        self.assertEqual(terminal_reason({"State":{"Running":False,"ExitCode":3}},1000,0,1),"system_exit")
        self.assertEqual(terminal_reason({"State":{"Running":True}},1000,0,1),"live_timeout")

    def test_owned_cleanup_refuses_foreign(self):
        commands=[]
        def runner(args,**kwargs):
            commands.append(args)
            return subprocess.CompletedProcess(args,0,json.dumps([{"Id":"cid","Config":{"Labels":{"hybrid-v3-owner":"foreign"}}}]),"")
        d=OwnedDocker("owned",Deadline(0,100,15,clock=lambda:1),self.root,runner)
        d.cid="cid"
        self.assertFalse(d.cleanup()["complete"])
        self.assertTrue(all(x[:2]==["docker","inspect"] for x in commands))

    def test_owned_cleanup_kills_only_owned(self):
        commands=[];running=[True];removed=[False]
        def runner(args,**kwargs):
            commands.append(args)
            if args[1]=="inspect":
                if removed[0]:return subprocess.CompletedProcess(args,1,"","No such container")
                obj={"Id":"cid","Config":{"Labels":{"hybrid-v3-owner":"owned"}},
                     "State":{"Running":running[0],"Pid":1 if running[0] else 0,"ExitCode":0,"OOMKilled":False}}
                return subprocess.CompletedProcess(args,0,json.dumps([obj]),"")
            if args[1]=="kill":running[0]=False
            if args[1]=="rm":removed[0]=True
            return subprocess.CompletedProcess(args,0,"","")
        d=OwnedDocker("owned",Deadline(0,100,15,clock=lambda:1),self.root,runner);d.cid="cid"
        result=d.cleanup();self.assertTrue(result["removed"])
        self.assertIn(["docker","kill","cid"],commands);self.assertIn(["docker","rm","cid"],commands)

    def test_source_lock_tamper_and_extra(self):
        p=self.root/"code.py";p.write_bytes(b"x")
        lock={"code.py":sha(p),"CASE.json":sha(self.root/"CASE.json")};verify_tree(self.root,lock)
        p.write_bytes(b"y")
        with self.assertRaises(ContractError):verify_tree(self.root,lock)
        p.write_bytes(b"x");(self.root/"extra").write_bytes(b"z")
        with self.assertRaises(ContractError):verify_tree(self.root,lock)

    def test_prediction_tamper(self):
        run=self.run_fixture();collect(run,self.root/"collected")
        lock=self.root/"collected/COLLECTED_LOCK.json";verify_collection(run,lock)
        (self.root/"collected/predictions/p0.md").write_bytes(b"changed")
        with self.assertRaises(ContractError):verify_collection(run,lock)

    def test_audit_tamper(self):
        run=self.run_fixture();collect(run,self.root/"collected")
        (run.parent/"pages/p0/GENERATION.json").write_text('{"changed":true}')
        with self.assertRaises(ContractError):verify_collection(run,self.root/"collected/COLLECTED_LOCK.json")

    def test_gt_exact_join_and_duplicate(self):
        rows=[{"page_info":{"image_path":f"/images/p{i}.png"}} for i in range(5)]
        self.assertEqual(gt_subset(rows,["p4","p1"]),[rows[4],rows[1]])
        with self.assertRaises(ContractError):gt_subset(rows,["absent"])
        with self.assertRaises(ContractError):gt_subset(rows+[rows[0]],["p1"])

    def test_missing_categories_do_not_invent_overall(self):
        metrics=official_fixture(self.root,empty=("text_block","display_formula","table","reading_order"))
        result=summarize(metrics,self.root,2)
        self.assertIsNone(result["raw_overall"]);self.assertFalse(result["overall_defined"])
        self.assertEqual(result["raw_metrics"],{})
        with self.assertRaises(ValueError):summarize(metrics,self.root,3)

    def test_only_output_writable_mount(self):
        with self.assertRaises(ContractError):mount_args([(self.root,"/model",True)],"/output")
        args=mount_args([(self.root,"/model",False)],"/output")
        self.assertIn("readonly",args[-1])

    def test_evaluator_resource_ceiling(self):
        b=dict(cpus=24,ram_gib=128,swap_gib=0,shm_gib=8,total_seconds=259200)
        limits(b,True)
        with self.assertRaises(ContractError):limits({**b,"swap_gib":1},True)
        with self.assertRaises(ContractError):limits({**b,"cpus":25},True)
        self.assertEqual(limits(b,True)[:2],["--pull","never"])

    def test_portable_manifest_after_move(self):
        import shutil
        original=self.root/"original";original.mkdir()
        self.root=original
        path=self.manifest(2)
        destination=original.parent/"relocated"
        shutil.copytree(original,destination)
        self.assertEqual(inputs(path),inputs(destination/"INPUT.json"))

    def test_effective_gpu_mount_isolation(self):
        from .image_identity import validate_identity
        image="sha256:"+"a"*64
        identity=validate_identity(image,{"reference":image,"kind":"local_id","image_id":image,"RepoDigests":[]})
        b=dict(cpus=8,ram_gib=16,swap_gib=0,shm_gib=1)
        mounts=[(self.root,"/output",True)]
        state={"Mounts":[{"Type":"bind","Source":str(self.root.resolve()),"Destination":"/output","RW":True}],
            "Image":image,"Config":{"Image":image},"HostConfig":{"NetworkMode":"none","ReadonlyRootfs":True,
            "Memory":16*1024**3,"MemorySwap":16*1024**3,"NanoCpus":8_000_000_000,
            "ShmSize":1024**3,"DeviceRequests":[{"DeviceIDs":["GPU-owned"]}]}}
        verify_isolation(state,mounts,b,image,["GPU-owned"],identity)
        with self.assertRaises(ContractError):verify_isolation(state,mounts,b,image,[],identity)
        state["HostConfig"]["NetworkMode"]="host"
        with self.assertRaises(ContractError):verify_isolation(state,mounts,b,image,["GPU-owned"],identity)

    def test_all_budget_fields_required_and_customer_excluded(self):
        b=dict(pages=3,native_calls_per_page=256,expert_calls_per_page=0,model_loads=5,
            load_seconds=600,page_seconds=900,request_seconds=120,total_seconds=1800,
            cleanup_seconds=240,page_audit_seconds=15,cpus=8,ram_gib=48,swap_gib=0,shm_gib=8,
            gpu_allowlist=["GPU-owned"],customer_gpu_uuids=[])
        budget(b,3)
        with self.assertRaises(ContractError):budget(b,4)
        with self.assertRaises(ContractError):budget({**b,"customer_gpu_uuids":["GPU-owned"]},3)
        with self.assertRaises(ContractError):budget({k:v for k,v in b.items() if k!="pages"},3)

    def test_failed_page_cannot_hide_nonempty_prediction(self):
        run=self.run_fixture(1,failed=(0,))
        folder=run.parent/"pages/p0"
        (folder/"prediction.md").write_bytes(b"should be empty")
        t=read(folder/"PAGE_RESULT.json");t["prediction_sha256"]=sha(folder/"prediction.md")
        (folder/"PAGE_RESULT.json").write_text(json.dumps(t))
        with self.assertRaises(ContractError):collect(run,self.root/"collected")

    def test_timedout_create_owned_recovery(self):
        commands=[];removed=[False]
        def runner(args,**kwargs):
            commands.append(args)
            if args[1]=="inspect":
                if removed[0]:return subprocess.CompletedProcess(args,1,"","No such container")
                return subprocess.CompletedProcess(args,0,json.dumps([{"Id":"cid",
                    "Config":{"Labels":{"hybrid-v3-owner":"owned"}},
                    "State":{"Running":False,"Pid":0,"ExitCode":0,"OOMKilled":False}}]),"")
            if args[1]=="rm":removed[0]=True
            return subprocess.CompletedProcess(args,0,"","")
        d=OwnedDocker("owned",Deadline(0,100,15,clock=lambda:1),self.root,runner)
        d.creation_attempted=True
        self.assertTrue(d.cleanup()["removed"])
        self.assertEqual(commands[0],["docker","inspect","hybrid-v3-owned"])

    def test_help_inspect_no_heavy_import(self):
        code=("import sys; from hybrid_v3_full_eval_v1.cli import main; "
              "main(['inspect-assets']); "
              "assert not any(n.split('.')[0] in {'torch','paddle','vllm','transformers'} for n in sys.modules)")
        proc=subprocess.run([sys.executable,"-B","-c",code],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)
        proc=subprocess.run([sys.executable,"-B","-m","hybrid_v3_full_eval_v1.cli","--help"],capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stderr)

def main():
    import argparse
    global ROOT
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    ROOT=Path(a.output).resolve();ROOT.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (ROOT/"checks.log").write_text(stream.getvalue(),encoding="utf-8")
    report={"tests":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),
            "passed":result.wasSuccessful(),"gpu_runs":0,"model_loads":0,"scoring_runs":0,
            "training_runs":0,"synthetic_contract_checks_only":True}
    write(ROOT/"CPU_CHECKS.json",report)
    print(stream.getvalue());print(json.dumps(report))
    return 0 if result.wasSuccessful() else 1

if __name__=="__main__":
    raise SystemExit(main())
