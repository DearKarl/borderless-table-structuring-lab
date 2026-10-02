"""Finite production-boundary CPU checks; all process calls are synthetic."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import types
import unittest
from unittest.mock import patch, MagicMock
from .core import ContractError, Deadline, canonical, read, sha, write
from . import image_identity as ii
from .lifecycle import verify_isolation, OwnedDocker
from .completion import normal_completion, lifecycle_lock
from .cpu_checks import Checks as OldChecks
from .v31_repair_checks import DockerFault

ROOT=None
A="sha256:"+"a"*64
B="sha256:"+"b"*64
REPO="registry.example:5000/team/image@sha256:"+"c"*64

def identity(ref=A, image_id=None):
    return {"reference":ref,"kind":ii.parse_reference(ref),"image_id":image_id or ref,
            "RepoDigests":[ref] if "@" in ref else []}

def result(obj, code=0):
    return types.SimpleNamespace(returncode=code,stdout=json.dumps(obj),stderr="synthetic")

class LaunchFake(DockerFault):
    def __init__(self, mode="normal"):
        super().__init__([0.])
        self.mode_image=mode
    def __call__(self,args,timeout,**kw):
        if args[:3]==["docker","image","inspect"]:
            self.calls.append({"args":args,"cap":timeout})
            ref=args[-1]
            if self.mode_image=="missing":raise subprocess.CalledProcessError(1,args)
            image_id=A if self.mode_image=="alias" else (B if "@" in ref else ref)
            return result([{"Id":image_id,"RepoDigests":[ref] if "@" in ref else [],"Os":"linux","Architecture":"amd64"}])
        if args[1]=="start":
            self.calls.append({"args":args,"cap":timeout})
            obj=self.objects[args[-1]];obj["State"].update(Running=False,Pid=0)
            if obj["Config"]["Labels"]["hybrid-v3-owner"].endswith("-paddle"):
                out=Path(next(x["Source"] for x in obj["Mounts"] if x["Destination"]=="/output"))
                c=read(out/"CONTROL.json")
                write(out/"experts/READY.json",{"run_id":c["run_id"],"gpu_uuid":"GPU-b",
                    "components":["formula"],"model_loads":1})
            return result(None)
        response=super().__call__(args,timeout=timeout,**kw)
        if args[1]=="create":
            obj=self.objects[response.stdout]
            ref=next(v for v in args if v.startswith("sha256:") or "@sha256:" in v)
            obj["Image"]=B if "@" in ref else ref
            if self.mode_image=="wrong-container":obj["Image"]="sha256:"+"d"*64
            obj["Config"]["Image"]=ref
            def value(flag):return args[args.index(flag)+1]
            mounts=[]
            for i,v in enumerate(args):
                if v=="--mount":
                    fields=args[i+1].split(",")
                    d=dict(s.split("=",1) for s in fields if "=" in s)
                    mounts.append({"Type":"bind","Source":d["src"],"Destination":d["dst"],"RW":"readonly" not in fields})
            obj["Mounts"]=mounts
            obj["HostConfig"]={"NetworkMode":"none","ReadonlyRootfs":True,
                "Memory":int(value("--memory")[:-1])*1024**3,
                "MemorySwap":int(value("--memory-swap")[:-1])*1024**3,
                "NanoCpus":int(float(value("--cpus"))*1e9),"ShmSize":int(value("--shm-size")[:-1])*1024**3,
                "DeviceRequests":[{"DeviceIDs":[value("--gpus").removeprefix("device=")]}] if "--gpus" in args else []}
        return response

class Checks(unittest.TestCase):
    manifest=OldChecks.manifest
    run_fixture=OldChecks.run_fixture
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix="case-",dir=ROOT))
        write(self.root/"CASE.json",{"test":self._testMethodName})
    def clock(self):return Deadline(time.monotonic(),600,240)
    def test_reference_local_and_repository(self):
        self.assertEqual(ii.parse_reference(A),"local_id")
        self.assertEqual(ii.parse_reference(REPO),"repo_digest")
        self.assertEqual(ii.parse_reference("ubuntu@sha256:"+"f"*64),"repo_digest")
    def test_reject_unsafe_references(self):
        for ref in ["a"*64,"sha256:abc","sha256:"+"A"*64,"sha512:"+"a"*64,A+" ",A+"\n",
                    " "+A,A+":extra","ubuntu","ubuntu:latest","--help","https://x/y@"+A,
                    "x@@"+A,"@"+A,"x:y@"+A,"x/../y@"+A,"x/y@"+A+"\n"]:
            with self.subTest(ref=ref),self.assertRaises(ContractError):ii.parse_reference(ref)
    def test_schema_one_local_rejected(self):
        with self.assertRaises(ContractError):ii.parse_reference(A,allow_local=False)
        self.assertEqual(ii.parse_reference(REPO,allow_local=False),"repo_digest")
    def test_local_empty_repo_digests(self):
        self.assertEqual(ii.resolve_image_identity(A,self.clock(),lambda *a,**k:result([{"Id":A,"RepoDigests":[]}]))["image_id"],A)
    def test_repository_manifest_is_not_config_id(self):
        r=ii.resolve_image_identity(REPO,self.clock(),lambda *a,**k:result([{"Id":A,"RepoDigests":[REPO]}]))
        self.assertEqual(r["image_id"],A)
    def test_local_id_mismatch(self):
        with self.assertRaises(ContractError):ii.resolve_image_identity(A,self.clock(),lambda *a,**k:result([{"Id":B}]))
    def test_repository_digest_missing_or_wrong(self):
        for values in [[],[REPO.replace("c"*64,"d"*64)],None,"string"]:
            with self.subTest(values=values),self.assertRaises(ContractError):
                ii.resolve_image_identity(REPO,self.clock(),lambda *a,**k:result([{"Id":A,"RepoDigests":values}]))
    def test_inspect_bad_shape_id_and_platform(self):
        for obj in [[],[{},{}],{},[None],[{}],[{"Id":"short"}],[{"Id":A,"Os":7}]]:
            with self.subTest(obj=obj),self.assertRaises(ContractError):
                ii.resolve_image_identity(A,self.clock(),lambda *a,**k:result(obj))
    def test_inspect_bad_json_and_nonzero(self):
        for p in [types.SimpleNamespace(returncode=0,stdout="broken"),result([{"Id":A}],1)]:
            with self.subTest(p=p),self.assertRaises(ContractError):
                ii.resolve_image_identity(A,self.clock(),lambda *a,**k:p)
    def test_inspect_timeout_and_deadline(self):
        calls=[]
        def run(args,**kw):
            calls.append((args,kw));raise subprocess.TimeoutExpired(args,kw["timeout"])
        with self.assertRaises(subprocess.TimeoutExpired):ii.resolve_image_identity(A,self.clock(),run)
        self.assertEqual(len(calls),1);self.assertLessEqual(calls[0][1]["timeout"],10)
        self.assertEqual(calls[0][0],["docker","image","inspect",A])
        now=[95.]
        clock=Deadline(0,100,0,clock=lambda:now[0])
        def late(args,**kw):
            self.assertEqual(kw["timeout"],5);now[0]=101;return result([{"Id":A}])
        with self.assertRaises(ContractError):ii.resolve_image_identity(A,clock,late)
    def isolation(self):
        b=dict(cpus=8,ram_gib=16,swap_gib=0,shm_gib=1)
        state={"Image":A,"Config":{"Image":A},"Mounts":[],
            "HostConfig":{"NetworkMode":"none","ReadonlyRootfs":True,"Memory":16*1024**3,
            "MemorySwap":16*1024**3,"NanoCpus":8_000_000_000,"ShmSize":1024**3,
            "DeviceRequests":[{"DeviceIDs":["GPU-a"]}]}}
        return state,b
    def test_container_actual_id_required(self):
        s,b=self.isolation()
        verify_isolation(s,[],b,A,["GPU-a"],identity())
        for value in [B,None]:
            bad=copy.deepcopy(s);bad["Image"]=value
            with self.assertRaises(ContractError):verify_isolation(bad,[],b,A,["GPU-a"],identity())
        del s["Image"]
        with self.assertRaises(ContractError):verify_isolation(s,[],b,A,["GPU-a"],identity())
    def test_identity_does_not_weaken_isolation(self):
        s,b=self.isolation()
        for key,value in [("NetworkMode","host"),("ReadonlyRootfs",False),("Memory",0),
                          ("DeviceRequests",[]),("Privileged",True)]:
            bad=copy.deepcopy(s);bad["HostConfig"][key]=value
            with self.subTest(key=key),self.assertRaises(ContractError):verify_isolation(bad,[],b,A,["GPU-a"],identity())
        with self.assertRaises(TypeError):verify_isolation(s,[],b,A,["GPU-a"])
        bad=copy.deepcopy(s);bad["Config"]["Image"]=B
        with self.assertRaises(ContractError):verify_isolation(bad,[],b,A,["GPU-a"],identity())
        bad=copy.deepcopy(s);bad["Mounts"]=[{"Type":"volume"}]
        with self.assertRaises(ContractError):verify_isolation(bad,[],b,A,["GPU-a"],identity())
    def test_cid_and_image_identity_separate(self):
        state={"Id":"cid","Image":A,"Config":{"Image":A,"Labels":{"hybrid-v3-owner":"owned"}}}
        docker=OwnedDocker("owned",self.clock(),self.root,lambda *a,**k:result([state]));docker.cid="cid"
        self.assertEqual(docker.inspect()["Image"],A)
        state["Id"]=A
        with self.assertRaises(ContractError):docker.inspect()
    def host(self,mode="on",fault="normal"):
        from . import runner
        fake=LaunchFake(fault)
        class Docker(OwnedDocker):
            def __init__(self,token,clock,out):super().__init__(token,clock,out,fake)
        leases=self.root/"leases";leases.mkdir()
        b={"total_seconds":600,"cleanup_seconds":240,"cpus":8,"ram_gib":16,"swap_gib":0,"shm_gib":1,"load_seconds":60}
        runtime={"image":A,"python":"/opt/python","gpu_uuids":["GPU-a","GPU-b"] if mode=="on" else ["GPU-a"],
                 "idle_memory_mib":0,"lease_directory":str(leases),"assets":{},"v31":{"components":["formula"]},
                 "environments":{"paddle":{"image":REPO,"python":"/opt/paddle"}}}
        plan={"inputs":{"pages":[]},"budget":b,"runtime":runtime,"source":{},"bindings":{},
              "states":{"runtime_verified":False,"integration_verified":False}}
        write(self.root/"budget.json",b)
        with patch.object(runner,"os",types.SimpleNamespace(name="posix")),patch.object(runner,"preflight",return_value=plan), \
             patch.object(runner,"Leases",return_value=MagicMock()),patch.object(runner,"gpu_idle",side_effect=lambda ids,*args:{u:0 for u in ids}), \
             patch.object(runner,"OwnedDocker",Docker),patch("hybrid_v3_full_eval_v1.v31_service.OwnedDocker",Docker), \
             patch.object(ii.subprocess,"run",side_effect=fake),patch.object(runner,"source_check",return_value={}):
            with self.assertRaises(ContractError):runner.infer("v31-formula","unused","unused",self.root/"budget.json",self.root/"run",mode)
        write(self.root/"CALLS.json",fake.calls)
        return fake.calls,read(self.root/"run/HOST_EXIT.json")
    def test_host_resolves_all_before_create_once(self):
        calls,host=self.host()
        images=[x for x in calls if x["args"][:3]==["docker","image","inspect"]]
        self.assertEqual([x["args"][-1] for x in images],[A,REPO])
        first=next(i for i,x in enumerate(calls) if x["args"][1]=="create")
        self.assertTrue(all(calls.index(x)<first for x in images))
        self.assertEqual(sum(x["args"][1]=="start" for x in calls),2)
        self.assertIsNone(host["failure"])
        self.assertTrue(all("never" in x["args"] for x in calls if x["args"][1]=="create"))
    def test_host_alias_same_id_no_create(self):
        calls,host=self.host(fault="alias")
        self.assertFalse(any(x["args"][1] in ("create","start","pull") for x in calls))
        self.assertIn("same image",host["failure"]["message"])
    def test_host_missing_image_no_start(self):
        calls,host=self.host(fault="missing")
        self.assertFalse(any(x["args"][1] in ("create","start","pull") for x in calls))
        self.assertIsNotNone(host["failure"])
    def test_host_wrong_container_image_prevents_start(self):
        calls,host=self.host(fault="wrong-container")
        self.assertFalse(any(x["args"][1]=="start" for x in calls))
        self.assertIn("actual image",host["failure"]["message"])
    def test_host_off_native_only(self):
        calls,_=self.host(mode="off")
        self.assertEqual([x["args"][-1] for x in calls if x["args"][:3]==["docker","image","inspect"]],[A])
        self.assertEqual(sum(x["args"][1]=="start" for x in calls),1)
    def test_host_pass_native_only(self):
        calls,_=self.host(mode="pass-through")
        self.assertEqual([x["args"][-1] for x in calls if x["args"][:3]==["docker","image","inspect"]],[A])
        self.assertEqual(sum(x["args"][1]=="start" for x in calls),1)
    def sealed(self,expert=False):
        path=self.run_fixture(1);root=path.parent;run=read(path);control=read(root/"CONTROL.json")
        control.update(image_identity_schema=1,mode="on" if expert else "off")
        control["runtime"]["image"]=A
        if expert:control["runtime"].update(v31={"components":["formula"]},environments={"paddle":{"image":REPO}})
        (root/"CONTROL.json").write_text(json.dumps(control))
        state=read(root/"CONTAINER_EXIT.json");state["Image"]=A;state["Config"]["Image"]=A
        (root/"CONTAINER_EXIT.json").write_text(json.dumps(state))
        roles={"native":identity()}
        if expert:
            roles["paddle"]=identity(REPO,B)
            out=root/"experts";out.mkdir()
            es=copy.deepcopy(state);es["Id"]="expert-cid";es["Image"]=B;es["Config"].update(Image=REPO,Labels={"hybrid-v3-owner":"fixture-paddle"})
            write(out/"CONTAINER_CREATED.json",{"owner":"fixture-paddle","cid":"expert-cid"});write(out/"CONTAINER_EXIT.json",es)
            write(out/"READY.json",{"run_id":"fixture","model_loads":0})
            write(out/"SESSION_RESULT.json",{"run_id":"fixture","normal_exit":True})
            write(out/"FINAL_AUDIT.json",{"close_errors":[],"resource_audit":{"unknown_resources":[]}})
            write(out/"CLEANUP.json",read(root/"CLEANUP.json"));(out/"LEDGER.jsonl").write_text("")
        write(root/"IMAGE_IDENTITIES.json",{"schema":1,"run_id":"fixture","roles":roles})
        run.update(image_identity_schema=1,lifecycle=lifecycle_lock(root));path.write_text(json.dumps(run))
        return path,run
    def test_completion_native_and_dual(self):
        path,run=self.sealed(True);self.assertTrue(normal_completion(path.parent,run))
        from .collection import collect
        self.assertEqual(collect(path,self.root/"collection")["page_count"],1)
    def test_completion_missing_and_modified_hash(self):
        path,run=self.sealed();e=path.parent/"IMAGE_IDENTITIES.json";original=e.read_bytes()
        e.unlink()
        with self.assertRaises(ContractError):normal_completion(path.parent,run)
        e.write_bytes(original+b" ")
        with self.assertRaises(ContractError):normal_completion(path.parent,run)
    def test_completion_resealed_wrong_role_reference_id(self):
        from .collection import collect
        path,run=self.sealed();root=path.parent;original=read(root/"IMAGE_IDENTITIES.json")
        for case in ("run","role","reference","id","exit","marker"):
            data=copy.deepcopy(original)
            if case=="run":data["run_id"]="wrong"
            if case=="role":data["roles"]["wrong"]=data["roles"].pop("native")
            if case=="reference":data["roles"]["native"]["reference"]=B
            if case=="id":data["roles"]["native"]["image_id"]=B
            if case=="exit":
                state=read(root/"CONTAINER_EXIT.json");state["Image"]=B
                (root/"CONTAINER_EXIT.json").write_text(json.dumps(state))
            if case=="marker":run.pop("image_identity_schema")
            (root/"IMAGE_IDENTITIES.json").write_text(json.dumps(data))
            run["lifecycle"]=lifecycle_lock(root);path.write_text(json.dumps(run))
            with self.subTest(case=case),self.assertRaises(ContractError):collect(path,self.root/("bad-"+case))
            state=read(root/"CONTAINER_EXIT.json");state["Image"]=A
            (root/"CONTAINER_EXIT.json").write_text(json.dumps(state))
    def test_official_repository_pin_identity(self):
        from .evaluation import PINS,evaluate
        value=ii.resolve_identities("fixture",{"official":PINS["image"]},self.clock(),
            lambda *a,**k:result([{"Id":A,"RepoDigests":[PINS["image"]]}]))
        state={"Image":A,"Config":{"Image":PINS["image"]}}
        ii.verify_container_image(state,PINS["image"],value["roles"]["official"])
        config={**PINS,"image":REPO,"budget":{"cpus":1,"ram_gib":1,"swap_gib":0,"shm_gib":1,"total_seconds":600,"cleanup_seconds":240}}
        write(self.root/"eval.json",config)
        with patch.object(ii.subprocess,"run",side_effect=AssertionError("No process allowed")):
            with self.assertRaisesRegex(ContractError,"Official evaluator identity"):
                evaluate("unused","unused","unused",self.root/"eval.json",self.root/"output")
    def test_cpu_cli_prepare_bind_no_process(self):
        from .asset_binder import prepare,bind,requirements
        from .cli import main
        with patch.object(ii.subprocess,"run",side_effect=AssertionError("CPU command started process")),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["inspect-assets"]),0)
            requirements(self.root/"requirements.json")
            write(self.root/"bad-plan.json",{"schema":1,"parent_asset_lock_sha256":"0"*64})
            with self.assertRaises(ContractError):bind(self.root/"bad-plan.json",self.root/"bound")
            write(self.root/"bad-config.json",{"profile":"native","asset_roots":{}})
            with self.assertRaises(ContractError):prepare(self.root/"bad-config.json",self.root/"prepared")

    def runtime_fixture(self, schema, image):
        from .contracts import NATIVE_FREEZE
        from .asset_binder import VENDOR
        from .asset_binding import special_tokens
        from .v31_rules import PROFILE_SHA
        parent=read(VENDOR/"PARENT_ASSET_LOCK.json")
        assets={}
        for role in ("native_code","tele_source","tele_model","environment","parameter_evidence"):
            folder=self.root/role;folder.mkdir()
            files=copy.deepcopy(parent["model_files"].get(role,{"fixture":"0"*64}))
            assets[role]={"role":role,"path":role,"files":files}
        parent_name="PARENT_ASSET_LOCK.json" if schema==2 else "SYSTEM_FREEZE.json"
        write(self.root/"native_code"/parent_name,parent)
        code={k:v for k,v in parent["code_files"].items() if not k.endswith("paddle_formula_worker.py")}
        code[parent_name]=sha(VENDOR/"PARENT_ASSET_LOCK.json") if schema==2 else NATIVE_FREEZE
        assets["native_code"]["files"]=code
        write(self.root/"tele_model/tokenizer_config.json",{"eos_token":"<end>"})
        token=special_tokens(self.root/"tele_model")
        env={"image":image,"python":"/opt/python","resolved_python":"/opt/python","site_packages":"/opt/site",
             "packages":parent["packages"]["native"],"image_files":{"/opt/python":"0"*64,
             **{"/opt/site/"+k:v for k,v in parent["image_auxiliaries"]["native"].items()}},
             "asset_roles":list(assets)}
        rows=[]
        for component in ("tele","fasttext","onnx"):
            row={"component":component,"model_asset":"tele_model",
                 "model_files_sha256":canonical(assets["tele_model"]["files"]),
                 "stored_elements":1,"unique_trainable":1,"buffers":0}
            record={**row,"method":"exact_auxiliary_inventory","measurement_source_sha256":"0"*64}
            file=self.root/"parameter_evidence"/(component+".json");write(file,record)
            row["evidence"]={"asset":"parameter_evidence","path":file.name,"sha256":sha(file)}
            rows.append(row)
        runtime={"schema":schema,"image":image,"python":"/opt/python","native_parent_freeze_sha256":NATIVE_FREEZE,
            "gpu_uuids":["GPU-a"],"lease_directory":"/leases","idle_memory_mib":0,"assets":assets,
            "native":{"tele_packages":parent["packages"]["native"],"capacity":{"passed":True,"conservative_upper_bound":3},
                      "expected_load_metadata":read(VENDOR/"tele_metadata.json"),
                      "identity_files":[{"asset":"tele_source","path":"TeleOCR/vlm_utils/TeleOCR_client.py",
                                        "sha256":assets["tele_source"]["files"]["TeleOCR/vlm_utils/TeleOCR_client.py"]}]},
            "environments":{"native":env},"parameter_ledger":rows,
            "v31":{"components":[],"profile_sha256":PROFILE_SHA,"special_tokens":{"tele":token["tokens"]},
                   "tokenizer_lock":{"tele":token}}}
        if schema==1:del assets["parameter_evidence"]
        write(self.root/"runtime.json",runtime)
        return runtime,{"gpu_allowlist":["GPU-a"],"cpus":8,"ram_gib":16,"swap_gib":0,"native_calls_per_page":1}
    def test_schema2_full_contract_and_environment_accept_local(self):
        from .contracts import runtime
        from .asset_binding import validate_v31
        value,b=self.runtime_fixture(2,A)
        # Synthetic model file hashes are not asserted as actual models; use the documented verify=False boundary.
        with patch.object(ii.subprocess,"run",side_effect=AssertionError("File-only validation")):
            self.assertEqual(runtime(self.root/"runtime.json","native",b,verify=False)["image"],A)
            value["environments"]["native"]["image"]="tag:latest"
            with self.assertRaises(ContractError):validate_v31(value,self.root,"native",b,"on",False)
    def test_schema2_full_contract_repository_acceptance(self):
        from .contracts import runtime
        value,b=self.runtime_fixture(2,REPO)
        self.assertEqual(runtime(self.root/"runtime.json","native",b,verify=False)["image"],REPO)
    def test_schema1_full_contract_repository_only(self):
        from .contracts import runtime
        value,b=self.runtime_fixture(1,REPO)
        self.assertEqual(runtime(self.root/"runtime.json","native",b,verify=False)["image"],REPO)
        value["image"]=A;(self.root/"runtime.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ContractError,"Schema 1"):runtime(self.root/"runtime.json","native",b,verify=False)
    def test_official_evaluate_production_identity_call_chain(self):
        from . import evaluation,runner
        pin=evaluation.PINS
        source=self.root/"source";source.mkdir()
        (self.root/"predictions").mkdir()
        for name in ("config","entry","run","predictions.json"):
            (self.root/name).write_text("synthetic")
        write(self.root/"gt.json",[{"page_info":{"image_path":"p0.png"}}])
        write(self.root/"source-receipt.json",{"source_revision":pin["revision"],"files":[]})
        budget={"cpus":1,"ram_gib":1,"swap_gib":0,"shm_gib":1,"total_seconds":600,"cleanup_seconds":240}
        cfg={**pin,"budget":budget,"source":"source","config":"config","original_entry":"entry",
             "source_receipt":"source-receipt.json","source_files":{},"gt_sha256":sha(self.root/"gt.json"),
             "lease_directory":str(self.root),"python":"/python"}
        write(self.root/"evaluator.json",cfg)
        class Fake(LaunchFake):
            def __call__(fake,args,timeout,**kw):
                response=super().__call__(args,timeout,**kw)
                if args[1]=="start":
                    obj=fake.objects[args[-1]]
                    work=Path(next(x["Source"] for x in obj["Mounts"] if x["Destination"]=="/work"))
                    names={"pred_quick_match_metric_result.json","pred_quick_match_run_summary.json"}|{
                        "pred_quick_match_"+x+"_result.json" for x in ("text_block","display_formula","table","reading_order")}
                    files={}
                    for name in names:
                        p=work/"result"/name;write(p,{});files[name]=sha(p)
                    write(work/"RESULT_VALIDATION.json",{"files":files})
                    write(work/"OFFICIAL_SCORE.json",{"validated_result_schema":True,
                        "validation_sha256":sha(work/"RESULT_VALIDATION.json"),"official_match_debug":{"page_count":1}})
                return response
        fake=Fake()
        class Docker(OwnedDocker):
            def __init__(self,token,clock,out):super().__init__(token,clock,out,fake)
        expected={str(self.root/"source-receipt.json"):pin["source_receipt_sha256"],
                  str(self.root/"config"):pin["config_sha256"],str(self.root/"entry"):pin["entry_sha256"]}
        def fixture_sha(p,*a):return expected.get(str(p)) or sha(p,*a)
        run={"run_id":"fixture","profile":"native","inputs":{"pages":[{"page_id":"p0"}]}}
        collection={"predictions":[],"unresolved":[]}
        with patch.object(evaluation,"os",types.SimpleNamespace(name="posix")), \
             patch.object(evaluation,"verify_collection",return_value=(run,collection)), \
             patch.object(evaluation,"verify_tree"),patch.object(evaluation,"sha",side_effect=fixture_sha), \
             patch.object(evaluation,"Leases",return_value=MagicMock()),patch.object(evaluation,"OwnedDocker",Docker), \
             patch.object(ii.subprocess,"run",side_effect=fake),patch.object(runner,"source_check",return_value={}):
            answer=evaluation.evaluate(self.root/"run",self.root/"predictions.json",self.root/"gt.json",
                                       self.root/"evaluator.json",self.root/"evaluation")
        self.assertTrue(answer["evaluation_complete"])
        self.assertEqual(answer["image_identities_sha256"],sha(self.root/"evaluation/IMAGE_IDENTITIES.json"))
        self.assertEqual([x["args"][-1] for x in fake.calls if x["args"][:3]==["docker","image","inspect"]],[pin["image"]])
        self.assertEqual(read(self.root/"evaluation/CONTAINER_EXIT.json")["Image"],B)
        write(self.root/"CALLS.json",fake.calls)

def main():
    global ROOT
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--output",required=True);a=p.parse_args()
    ROOT=Path(a.output);ROOT.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (ROOT/"checks.log").write_text(stream.getvalue(),encoding="utf-8")
    report={"tests":r.testsRun,"failures":len(r.failures),"errors":len(r.errors),"passed":r.wasSuccessful(),
            "synthetic_cpu_only":True,"Docker":0,"GPU":0,"model_loads":0,"scoring":0}
    write(ROOT/"CPU_CHECKS.json",report);print(stream.getvalue());print(json.dumps(report))
    return 0 if r.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
