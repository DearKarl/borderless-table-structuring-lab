"""Production V31 CPU fixtures. No model, Docker, GPU, or official scorer execution."""
import ast
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
from .core import ContractError, Deadline, read, sha, write, canonical
from .contracts import TELE_CLIENT
from .v31_rules import structure, formula_failure, i0, proposal
from .v31_transactions import V31Page
from .v31_protocol import validate, IDENTITY
from .request_binding import BindingGraph
from .repair_checks import Tensor, Feature, tensor_binding, DockerFake
from .bounded_cleanup import CleanupClock
from .lifecycle import OwnedDocker
ROOT=None;TELE=None

EOS={"stop":"eos","token_ids":[7,2],"eos_ids":[2]}
class Block(dict):
    def __init__(self,type,bbox,angle=0,content=None):
        super().__init__(type=type,bbox=bbox,angle=angle,content=content)
    def __getattr__(self,k):
        if k not in self:raise AttributeError(k)
        return self[k]
    def __setattr__(self,k,v):self[k]=v

class Experts:
    def __init__(self,formula="x",candidates=(),error=None):
        self.text,self.candidates,self.error=formula,list(candidates),error
        self.calls=[]
    def formula(self,crop,binding,render):
        self.calls.append(("formula",binding["slot_id"]))
        if self.error:raise self.error
        return {"status":"ok","raw_text":self.text,"termination":EOS,"request_id":"synthetic-expert/"+binding['request_id']}
    def layout(self,image,page,render):
        self.calls.append(("layout",page["page_id"]))
        if self.error:raise self.error
        return {"candidates":self.candidates,"request_id":"synthetic-layout/"+page['page_id']}

def method():
    source=TELE/"TeleOCR/vlm_utils/TeleOCR_client.py"
    if sha(source)!=TELE_CLIENT:raise ContractError("CPU fixture requires original frozen stepping source")
    parsed=ast.parse(source.read_text(encoding="utf-8"))
    node=next(n for n in ast.walk(parsed) if isinstance(n,ast.FunctionDef) and n.name=="stepping_two_step_extract")
    namespace={}
    module=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(source),"exec"),namespace)
    return namespace["stepping_two_step_extract"]

class Checks(unittest.TestCase):
    def setUp(self):
        self.root=Path(tempfile.mkdtemp(prefix="t-",dir=ROOT))
        write(self.root/"CASE.json",{"test":self._testMethodName})
        self.page={"page_id":"p","file_sha256":"a"*64,"kind":"image"}
    def run_native(self,blocks,profile="v31-formula",mode="on",experts=None,fresh_output="fresh"):
        """Run Native.parse and the original frozen stepping method, using finite CPU boundary doubles."""
        from PIL import Image
        import numpy
        from .native import Native
        native=Native.__new__(Native)
        native.profile,native.mode,native.experts=profile,mode,experts
        native.runtime={"schema":2,"v31":{"special_tokens":{"tele":[],"formula":[]}}}
        native.output=self.root;native.budget={"page_audit_seconds":15}
        native.ledger=lambda *a,**k:None;native.run_id="run"
        native.model_identity="b"*64;native.snapshot=copy.deepcopy
        native.tensor_binding=tensor_binding;native.expert_calls=0
        pageimage=Image.new("RGB",(100,100),"white")
        state={"post":0,"prepared":0,"dispatch":0,"fresh_crops":[]}
        class Processor:
            def __call__(self,images,text):
                value=hashlib.sha256(images[0].tobytes()).hexdigest()
                return Feature({k:Tensor(value+k) for k in ("input_ids","pixel_values","image_grid_thw")})
        class Backend:
            batch_size=1
            def __init__(self):self.processor=Processor()
            def _predict_one_batch(self,image_objs,chat_prompts,sampling_params):
                feature=self.processor(images=image_objs,text=chat_prompts).to()
                g=native.binding;record=g.generating(feature)
                output=("layout" if record["kind"]=="layout" else fresh_output if record["kind"]=="fresh"
                        else blocks[int(record["slot_id"].rsplit(":",1)[1])].content)
                audit={"returned":True,"input_tensors":tensor_binding(feature),
                       "effective_generation_config":{"max_length":128000},"termination":copy.deepcopy(EOS)}
                g.generated(audit);native.rt["generation"].append(audit);state["dispatch"]+=1
                return [output]
            def batch_predict(self,images,prompts,params,priority=None):
                return [self._predict_one_batch([im],[pr],pa)[0] for im,pr,pa in zip(images,prompts,params)]
        class Helper:
            def prepare_for_layout(self,image):return image.copy()
            def prepare_for_extract(self,image,current,not_extract_list=None):
                state["prepared"]+=1
                indices=[i for i,b in enumerate(current) if b.type not in ("image","equation_block")]
                crops=[image.crop(tuple(int(v*100) for v in current[i].bbox)) for i in indices]
                return crops,[current[i].type for i in indices],[None]*len(indices),indices
            def batch_prepare_for_extract(self,executor,images,bs,not_extract_list=None):
                return [self.prepare_for_extract(im,bl,not_extract_list) for im,bl in zip(images,bs)]
            def post_process(self,bs):state["post"]+=1;return bs
            def batch_post_process(self,executor,bs):return [self.post_process(b) for b in bs]
        helper=Helper();backend=Backend()
        class Client:
            batching_mode="stepping";executor=None;incremental_priority=False
            stepping_two_step_extract=method()
            def __init__(self,*args,**kwargs):self.helper=helper;self.client=backend
            def batch_layout_detect(self,images,priority):
                prepared=[self.helper.prepare_for_layout(im) for im in images]
                self.client.batch_predict(prepared,["layout"],[None])
                return [copy.deepcopy(blocks)]
        class Resources:
            def result(self):return {"unknown_resources":[]}
        final={}
        def parse(folder,names,pdfbytes,ordinals,predictor):
            result=predictor.stepping_two_step_extract([pageimage])[0];final["blocks"]=result
            d=Path(folder)/names[0];d.mkdir(parents=True)
            (d/(names[0]+".md")).write_text(json.dumps(result),encoding="utf-8")
            write(d/(names[0]+"_middle.json"),result)
        native.rt={"TeleOCRClient":Client,"model":None,"processor":None,"generation":[],
                   "do_parse":parse,"read_fn":lambda p:b"synthetic PDF","resources":Resources()}
        import importlib.util
        spec=importlib.util.spec_from_file_location("fixture_otsl",TELE/"TeleOCR/vlm_utils/post_process/otsl2html.py")
        table=importlib.util.module_from_spec(spec);spec.loader.exec_module(table)
        mods={"TeleOCR.vlm_utils.structs":types.SimpleNamespace(ContentBlock=Block),
              "TeleOCR.vlm_utils.post_process.otsl2html":table}
        # Structure converter is not used by these text/equation fixtures; table gate tested separately.
        originals=(helper.prepare_for_extract,helper.post_process,backend.processor)
        with patch.dict(sys.modules,mods),patch("hybrid_v3_full_eval_v1.native.signal.setitimer",create=True),patch("hybrid_v3_full_eval_v1.native.signal.ITIMER_REAL",0,create=True):
            path,truncated=native.parse(self.page,"synthetic",self.root)
        self.assertEqual((helper.prepare_for_extract,helper.post_process,backend.processor),originals)
        self.assertEqual(state["post"],1)
        from .selection_audit import verify
        self.assertTrue(verify(self.root))
        return final["blocks"],state

    def test_original_stepping_off_and_pass_sparse_equal(self):
        blocks=[Block("text",[.1,.1,.8,.2],content="a"),Block("image",[.1,.3,.8,.4],content=None),
                Block("equation",[.1,.5,.8,.6],content="x")]
        a,sa=self.run_native(blocks,mode="off")
        old=self.root;self.root=old/"pass";self.root.mkdir()
        b,sb=self.run_native(blocks,mode="pass-through")
        self.assertEqual(a,b);self.assertEqual(sa["dispatch"],3);self.assertEqual(sb["dispatch"],3)
        req=read(self.root/"REQUEST_BINDINGS.json")
        self.assertEqual([r["slot_id"] for r in req],[None,"p:native:0","p:native:2"])

    def test_formula_actual_native_hook_replaces_only_bad_equation(self):
        blocks=[Block("text",[.1,.1,.8,.2],content="{text"),Block("equation",[.1,.3,.8,.4],content="\\frac{a}{b"),
                Block("equation",[.1,.5,.8,.6],content="x")]
        e=Experts(formula=" $$\\frac{a}{b}$$ ")
        out,state=self.run_native(blocks,experts=e)
        self.assertEqual(out[0],blocks[0]);self.assertEqual(out[2],blocks[2])
        self.assertEqual(out[1].content," \\frac{a}{b} ")
        self.assertEqual(len(e.calls),1);self.assertEqual(state["dispatch"],4)

    def test_formula_unknown_retains_native(self):
        b=Block("equation",[.1,.1,.8,.3],content="\\verb|{|")
        e=Experts()
        out,_=self.run_native([b],experts=e)
        self.assertEqual(out,[b]);self.assertEqual(e.calls,[])

    def test_formula_bad_expert_no_second_tele(self):
        b=Block("equation",[.1,.1,.8,.3],content="{")
        e=Experts(formula="{")
        out,state=self.run_native([b],experts=e)
        self.assertEqual(out,[b]);self.assertEqual(state["dispatch"],2)

    def test_formula_system_error_stops(self):
        b=Block("equation",[.1,.1,.8,.3],content="{")
        with self.assertRaisesRegex(ContractError,"identity"):
            self.run_native([b],experts=Experts(error=ContractError("identity")))

    def test_lf_new_equation_fresh_crop_before_formula(self):
        blocks=[Block("text",[.1,.05,.8,.15],content="top"),
                Block("text",[.1,.8,.8,.9],content="bottom")]
        e=Experts(formula="z",candidates=[{"index":4,"label":"display_formula","score":.99,"bbox":[10,30,80,50]}])
        out,state=self.run_native(blocks,"v31-both",experts=e,fresh_output="{")
        self.assertEqual([b.content for b in out],["top","z","bottom"])
        self.assertEqual(state["prepared"],2);self.assertEqual(state["dispatch"],4)
        req=read(self.root/"REQUEST_BINDINGS.json")
        fresh=next(r for r in req if r["kind"]=="fresh")
        self.assertIn(":new:",fresh["slot_id"])
        self.assertEqual([x[0] for x in e.calls],["layout","formula"])
        selected=read(self.root/'CONTENT.json')
        self.assertEqual([r['selected_content'] for r in selected],['top','z','bottom'])
        self.assertEqual(selected[1]['source'],'paddle')
        self.assertEqual(selected[1]['fresh_request_id'],fresh['request_id'])
        self.assertIsNone(selected[1]['native_request_id'])
        self.assertTrue(selected[1]['layout_request_id'])

    def test_new_region_rejection_exact_rollback(self):
        blocks=[Block("text",[.1,.05,.8,.15],content="top"),Block("text",[.1,.8,.8,.9],content="bottom")]
        e=Experts(candidates=[{"index":2,"label":"text","score":.99,"bbox":[10,30,80,50]}])
        out,state=self.run_native(blocks,"v31-layout",experts=e,fresh_output="")
        self.assertEqual(out,blocks);self.assertEqual(state["prepared"],2)
        events=read(self.root/"V31_TRANSACTIONS.json")
        self.assertTrue(any(x.get("reason")=="fresh_content_rejected" for x in events))
        self.assertEqual([r['slot_id'] for r in read(self.root/'CONTENT.json')],['p:native:0','p:native:1'])

    def test_geometry_noop_caption_rotated_area_and_anchors(self):
        n=[{"slot_id":"p:native:0","type":"table","angle":0,"bbox":[10,10,90,50]}]
        c={"index":0,"label":"table","score":.95,"bbox":[10,8,90,50]}
        self.assertTrue(proposal(c,n,[],100,100,"a"*64)["accepted"])
        for change,reason in [({"bbox":[10,10,90,50]},"noop"),({"bbox":[0,0,100,70]},"union_area_or_bounds")]:
            self.assertEqual(proposal({**c,**change},n,[],100,100,"a"*64)["reason"],reason)
        caption={"slot_id":"p:native:1","type":"table_caption","angle":0,"bbox":[10,5,90,9]}
        self.assertEqual(proposal(c,n+[caption],[],100,100,"a"*64)["reason"],"table_conflict")
        self.assertEqual(proposal(c,[{**n[0],"angle":90}],[],100,100,"a"*64)["reason"],"rotated_table")
        self.assertFalse(proposal({**c,"label":"text","bbox":[10,60,90,70]},n,[],100,100,"a"*64)["accepted"])

    def test_table_expansion_fresh_content_and_rejected_rollback(self):
        blocks=[Block("table",[.1,.1,.9,.5],content="<fcel>old<nl>")]
        candidates=[{"index":0,"label":"table","score":.99,"bbox":[10,8,90,50]}]
        out,state=self.run_native(blocks,"v31-layout",experts=Experts(candidates=candidates),fresh_output="<fcel>new<nl>")
        self.assertEqual(out[0].bbox,[.1,.08,.9,.5]);self.assertEqual(out[0].content,"<fcel>new<nl>")
        self.assertEqual(state["prepared"],2)
        row=read(self.root/'CONTENT.json')[0]
        self.assertEqual(row['source'],'tele_fresh');self.assertTrue(row['fresh_request_id'])
        self.assertEqual(row['primary_content'],'<fcel>old<nl>')
        self.root=self.root/"reject";self.root.mkdir()
        out,_=self.run_native(blocks,"v31-layout",experts=Experts(candidates=candidates),fresh_output="broken")
        self.assertEqual(out,blocks)
        row=read(self.root/'CONTENT.json')[0]
        self.assertEqual(row['source'],'native');self.assertIsNone(row['fresh_request_id'])
        self.assertTrue(any(e.get('reason')=='fresh_content_rejected' for e in row['decisions']))

    def test_same_anchor_additions_reading_order_not_score_order(self):
        blocks=[Block("text",[.1,.05,.8,.15],content="top"),Block("text",[.1,.8,.8,.9],content="bottom")]
        candidates=[{"index":0,"label":"text","score":.99,"bbox":[10,55,80,65]},
                    {"index":1,"label":"text","score":.95,"bbox":[10,25,80,35]}]
        out,state=self.run_native(blocks,"v31-layout",experts=Experts(candidates=candidates))
        self.assertEqual([b.bbox[1] for b in out],[.05,.25,.55,.8]);self.assertEqual(state["prepared"],3)

    def test_capacity_requires_bound_evidence_not_boolean(self):
        from .asset_binding import capacity
        with self.assertRaises(ContractError):capacity({"native":{"capacity":{"passed":True}}},{})
        rows=[];assets={};roots={}
        for name in ("tele","fasttext","onnx"):
            d=self.root/name;d.mkdir();f=d/"weights";f.write_bytes(name.encode())
            files={"weights":sha(f)};assets[name]={"files":files};roots[name]=d
            row={"component":name,"stored_elements":2,"unique_trainable":1,"buffers":0,
                 "model_asset":name,"model_files_sha256":canonical(files)}
            evidence={**row,"method":"exact_auxiliary_inventory","measurement_source_sha256":"a"*64}
            p=self.root/(name+".json");write(p,evidence)
            row["evidence"]={"asset":"proof","path":p.name,"sha256":sha(p)};rows.append(row)
        r={"parameter_ledger":rows,"assets":assets};roots["proof"]=self.root
        self.assertEqual(capacity(r,roots),3)
        rows[0]["unique_trainable"]=2
        with self.assertRaises(ContractError):capacity(r,roots)

    def test_metadata_rebinding_changes_only_explicit_paths(self):
        from .asset_binding import normalize_metadata
        p=read(Path(__file__).parent/"assets/vendor/tele_metadata.json")["PROCESSOR.json"]
        p["tokenizer_init_kwargs"]["name_or_path"]="/old/model"
        original=copy.deepcopy(p);bound,diff=normalize_metadata("PROCESSOR.json",p)
        self.assertEqual(p,original);self.assertEqual(bound["chat_template"],p["chat_template"])
        self.assertEqual(bound["image_processor_config"],p["image_processor_config"])
        self.assertEqual(len(diff),3)

    def test_lexical_boundaries_and_true_token_repetition(self):
        self.assertEqual(structure(r"\{x\}",[])["state"],"valid")
        self.assertEqual(structure(r"\begin{a}x\end{b}",[])["state"],"bad")
        self.assertEqual(structure("x% {",[])["state"],"unknown")
        self.assertEqual(structure(r"\verb|{|",[])["state"],"unknown")
        self.assertEqual(structure(r"x\%",[])["state"],"valid")
        self.assertEqual(structure("x<special>",["<special>"])["reason"],"special_token_leak")
        t={"stop":"eos","token_ids":list(range(16))*4+[99],"eos_ids":[99]}
        self.assertEqual(formula_failure("x",t,[])["state"],"bad")
        self.assertEqual(formula_failure("x",{"stop":"eos"},[])["state"],"unknown")
        self.assertEqual(i0(" \n$$ a＋b $$\t")," \n a＋b \t")
        self.assertIsNone(i0("$$a$$ $$b$$"))

    def test_formula_cap_and_resident_reuse(self):
        from PIL import Image
        e=Experts();p=V31Page(self.page,"v31-formula","on",e,{"tele":[],"formula":[]},lambda _:False)
        for i in range(34):
            b=Block("equation",[0,0,1,1],content="{")
            p.formula(b,Image.new("RGB",(2,2)),{"slot_id":str(i),"request_id":str(i),"termination":EOS},{})
        self.assertEqual(len(e.calls),32);self.assertEqual(p.events[-1]["reason"],"formula_budget_abstain")
        other=V31Page(self.page,"v31-formula","on",e,{"tele":[],"formula":[]},lambda _:False)
        self.assertIs(other.experts,e)

    def test_protocol_identity_and_actual_tensor_required(self):
        req={k:k for k in IDENTITY};req["engine"]="formula"
        tensors={k:{"sha256":"a"*64} for k in ("input_ids","pixel_values","image_grid_thw")}
        audit={**EOS,"input_tensors":tensors,"returned":True}
        response={**req,"status":"ok","raw_text":"x","generation":audit,
                  "processor_tensors":tensors,"tensor_sha256":canonical(tensors)}
        self.assertEqual(validate(req,copy.deepcopy(response))["termination"],EOS)
        for key in ("page_id","slot_id","png_sha256","model_sha256"):
            with self.assertRaises(ContractError):validate(req,{**response,key:"foreign"})
        with self.assertRaises(ContractError):validate(req,{**response,"processor_tensors":{}})

    def test_shared_cleanup_clock_partial_start(self):
        now=[1.]
        clock=Deadline(0,300,240,clock=lambda:now[0]);shared=CleanupClock(clock)
        for i in range(2):
            path=self.root/str(i);path.mkdir();fake=DockerFake(now)
            docker=OwnedDocker("owned",clock,path,fake);docker.creation_attempted=True
            if i==0:docker.cid="cid"
            result=docker.cleanup(clock=shared)
            self.assertTrue(result["removed"]);self.assertTrue(result["complete"])
        self.assertEqual(shared.started,1.)

    def test_actual_formula_provider_processor_observation(self):
        from PIL import Image
        from .v31_providers import Formula
        class Parameter:
            def numel(self):return 1
        class Processor:
            def preprocess(self,data,**kwargs):
                return Feature({k:Tensor(k) for k in ("input_ids","pixel_values","image_grid_thw")})
        class Model:
            def parameters(self):return [parameter]
            def buffers(self):return []
            def generate(self,inputs,**kwargs):
                legacy.audits.append({**EOS,"returned":True,"input_tensors":tensor_binding(inputs)})
                return "x"
        parameter=Parameter()
        class Predictor:
            def __init__(self):
                self.processor=Processor();self.infer=Model()
            def _switch_inputs_to_device(self,data):return data
            def predict(self,data,**kwargs):
                features=self.processor.preprocess(data,min_pixels=kwargs["min_pixels"],max_pixels=kwargs["max_pixels"])
                features=self._switch_inputs_to_device(features)
                self.infer.generate(features,max_new_tokens=kwargs["max_new_tokens"],use_cache=kwargs["use_cache"])
                yield {"result":"x"}
            def close(self):pass
        class Legacy:
            def __init__(self,*a):
                self.predictor=Predictor();self.audits=[];self.calls=0
            def close(self):pass
        legacy=Legacy()
        modules={
          "hybrid_v2_tele_base":types.ModuleType("hybrid_v2_tele_base"),
          "hybrid_v2_tele_base.paddle_formula_worker":types.SimpleNamespace(FormulaPredictor=lambda _:legacy),
          "hybrid_v2_tele_base.audit":types.SimpleNamespace(tensor_binding=tensor_binding)}
        runtime={"v31":{"formula":{"model_asset":"formula_model"}},
                 "parameter_ledger":[{"component":"formula","unique_trainable":1,"buffers":0}]}
        with patch.dict(sys.modules,modules):
            p=Formula(runtime)
            response=p.call({},Image.new("RGB",(2,2)),None)
            self.assertEqual(response["status"],"ok")
            self.assertEqual(response["processor_tensors"],response["generation"]["input_tensors"])
            self.assertEqual(legacy.calls,1)
            p.close()

    def test_missing_environment_preparation_fails_without_load(self):
        from .asset_binder import prepare
        from .contracts import PROFILES
        roots={}
        for name in ("tele_model","environment","parameter_evidence"):
            d=self.root/name;d.mkdir();(d/"placeholder").write_bytes(b"synthetic")
            roots[name]=str(d)
        env=self.root/"env.json";write(env,{})
        cfg=self.root/"cfg.json";write(cfg,{"profile":"native","asset_roots":roots,"environment_lock":str(env)})
        with self.assertRaisesRegex(ContractError,"image locks"):
            prepare(cfg,self.root/"prepared")
        self.assertFalse((self.root/"prepared/bind-plan.json").exists())

    def test_bind_parent_tamper_rejected(self):
        from .asset_binder import bind
        plan=self.root/"plan.json";write(plan,{"schema":1,"parent_asset_lock_sha256":"0"*64})
        with self.assertRaisesRegex(ContractError,"Parent asset lock"):
            bind(plan,self.root/"bound")
        self.assertFalse((self.root/"bound").exists())

    def test_host_partial_expert_start_still_cleans_all_owned(self):
        from . import runner
        from unittest.mock import MagicMock
        from .v31_repair_checks import DockerFault
        fake=DockerFault([0.])
        class Docker(OwnedDocker):
            def __init__(self,token,clock,out):super().__init__(token,clock,out,fake)
        class Service:
            def __init__(self,control,clock,out,mounts,image_identity):
                folder=Path(out)/'experts';folder.mkdir()
                self.docker=Docker(control['run_id']+'-paddle',clock,folder)
            def start(self):
                self.docker.create([])
                raise ContractError("partial-start fixture")
        budget={"total_seconds":600,"cleanup_seconds":240}
        leases=self.root/'leases';leases.mkdir()
        runtime={"gpu_uuids":["GPU-a","GPU-b"],"idle_memory_mib":0,"lease_directory":str(leases),
                 "assets":{},"v31":{"components":["formula"]},
                 "image":"sha256:"+"a"*64,"environments":{"paddle":{"image":"sha256:"+"b"*64}}}
        plan={"inputs":{"pages":[]},"budget":budget,"runtime":runtime,"source":{},"bindings":{},
              "states":{"runtime_verified":False,"integration_verified":False}}
        (self.root/"budget.json").write_text(json.dumps(budget))
        with patch.object(runner,"os",types.SimpleNamespace(name="posix")),patch.object(runner,"preflight",return_value=plan), \
             patch.object(runner,"Leases",return_value=MagicMock()),patch.object(runner,"gpu_idle",return_value={"GPU-a":0,"GPU-b":0}), \
             patch.object(runner,"resolve_identities",return_value={"roles":{"paddle":{}}}), \
             patch.object(runner,"OwnedDocker",Docker),patch("hybrid_v3_full_eval_v1.v31_service.ExpertService",Service):
            with self.assertRaisesRegex(ContractError,"Run stopped"):
                runner.infer("v31-formula","unused","unused",self.root/"budget.json",self.root/"run")
        group=read(self.root/'run/GROUP_CLEANUP.json')
        self.assertEqual(len(group['resources']),2)
        self.assertTrue(group['resources'][0]['owner'].endswith('-paddle'))
        self.assertTrue(group['resources'][0]['removed'])
        self.assertFalse(group['resources'][1]['created'])
        self.assertTrue(read(self.root/'run/ADMISSION_RESULT.json')['released'])
        self.assertEqual(read(self.root/"run/HOST_EXIT.json")["failure"]["message"],"partial-start fixture")

def main():
    import argparse
    global ROOT,TELE
    p=argparse.ArgumentParser();p.add_argument("--output",required=True)
    p.add_argument("--tele-source",required=True)
    a=p.parse_args();ROOT=Path(a.output);ROOT.mkdir(parents=True,exist_ok=False);TELE=Path(a.tele_source)
    stream=io.StringIO()
    r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (ROOT/"checks.log").write_text(stream.getvalue(),encoding="utf-8")
    report={"passed":r.wasSuccessful(),"tests":r.testsRun,"errors":len(r.errors),"failures":len(r.failures),
            "original_stepping_source_sha256":sha(TELE/"TeleOCR/vlm_utils/TeleOCR_client.py"),
            "synthetic_cpu_only":True,"gpu":0,"model_loads":0,"scoring":0}
    write(ROOT/"CPU_CHECKS.json",report);print(stream.getvalue());print(json.dumps(report))
    return 0 if r.wasSuccessful() else 1
if __name__=="__main__":raise SystemExit(main())
