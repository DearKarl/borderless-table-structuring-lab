"""Original Tele loader, read_fn->PDF->do_parse; observation only."""
import copy
import hashlib
import importlib
import sys
import signal
import time
from pathlib import Path
from .core import ContractError, sha, write, canonical, progress
from .request_binding import BindingGraph, bind_backend, rgb_binding, params_snapshot

class Native:
    def __init__(self, runtime, output, budget, ledger, run_id, profile="native", mode="off", experts=None):
        self.profile,self.mode,self.experts=profile,mode,experts
        self.runtime=runtime
        self.expert_calls=0
        self.adoptions={"layout":0,"formula":0,"content_changed":0}
        self.output, self.budget, self.ledger = Path(output), budget, ledger
        self.count = 0
        self.page = None
        self.run_id,self.binding=run_id,None
        if runtime.get("schema")==2:
            from .asset_binding import load_boundary
            write(self.output/"NATIVE_LOAD_BOUNDARY.json",load_boundary(runtime,"native"))
        sys.path.insert(0, "/assets/native_code")
        loader = importlib.import_module("hybrid_v2_tele_base.runtime_load")
        audit = importlib.import_module("hybrid_v2_tele_base.audit")
        self.snapshot = audit.audit_snapshot
        self.tensor_binding = audit.tensor_binding
        manifest = copy.deepcopy(runtime["native"])
        manifest.update(tele_source="/assets/tele_source", tele_model="/assets/tele_model",
                        gpu_uuid=runtime["gpu_uuids"][0])
        manifest["identity_files"] = [
            {"path":r["image_path"] if "image_path" in r else "/assets/"+r["asset"]+"/"+r["path"],"sha256":r["sha256"]}
            for r in manifest["identity_files"]]
        summary = {"model_loads":{"tele":0,"paddle":0}}
        if budget["model_loads"] < 1:
            raise ContractError("Model load budget exceeded")
        self.load_count=1+(len(runtime.get("v31",{}).get("components",[])) if experts is not None else 0)
        if self.load_count>budget["model_loads"]:raise ContractError("Combined model load budget exceeded")
        original_audit=loader.ResourceAudit
        owner=self
        class CountedResources(original_audit):
            def bind_tele_loaders(self):
                super().bind_tele_loaders()
                import fasttext
                import onnxruntime
                def wrap(function,kind):
                    def call(*args,**kwargs):
                        if runtime.get("schema")==2:
                            from .core import sha
                            actual_path=str(Path(args[0]).resolve())
                            known={str(Path(row["path"]).resolve()):row["sha256"] for row in manifest["identity_files"]}
                            if actual_path not in known or sha(actual_path)!=known[actual_path]:
                                raise ContractError("Auxiliary asset differs at actual loader boundary")
                        if owner.load_count>=budget["model_loads"]:
                            raise ContractError("Auxiliary model load budget exceeded")
                        owner.load_count+=1
                        ledger("LOAD_START",engine=kind,load_ordinal=owner.load_count)
                        try:
                            result=function(*args,**kwargs)
                            ledger("LOAD_RESULT",engine=kind,returned=True)
                            return result
                        except BaseException as exc:
                            ledger("LOAD_RESULT",engine=kind,returned=False,error_type=type(exc).__name__)
                            raise
                    return call
                fasttext.load_model=wrap(fasttext.load_model,"fasttext_auxiliary")
                onnxruntime.InferenceSession=wrap(onnxruntime.InferenceSession,"onnx_auxiliary")
        ledger("LOAD_START", engine="tele")
        loader.ResourceAudit=CountedResources
        try:
            self.rt = loader.load_runtime(manifest, self.output, summary)
        except BaseException as exc:
            ledger("LOAD_RESULT",engine="tele",returned=False,error_type=type(exc).__name__)
            raise
        finally:
            loader.ResourceAudit=original_audit
        for name, expected in manifest["expected_load_metadata"].items():
            from .core import read
            if read(self.output/name) != expected:
                raise ContractError("Native loaded metadata differs: "+name)
        ledger("LOAD_RESULT", engine="tele", loads=summary["model_loads"])
        if runtime.get("schema")==2:
            from .asset_binding import loaded_count
            write(self.output/"LOADED_PARAMETERS.json",loaded_count(runtime,"tele",self.rt["model"].parameters(),self.rt["model"].buffers()))
        self.model_identity = canonical(runtime["assets"]["tele_model"]["files"])
        original = self.rt["model"].generate
        def generate(*args, **kwargs):
            self.count += 1
            if self.count > self.budget["native_calls_per_page"]:
                raise ContractError("Native per-page call budget exceeded")
            if self.binding is None:
                raise ContractError("Generation has no active page binding")
            bound=self.binding.generating(kwargs)
            request=bound["request_id"]
            before = time.monotonic()
            remaining, interval = signal.getitimer(signal.ITIMER_REAL)
            if remaining <= 0:
                raise ContractError("Native call outside a bounded page")
            signal.setitimer(signal.ITIMER_REAL, min(remaining, self.budget["request_seconds"]))
            ledger("CALL_START", engine="tele", request_id=request, page_id=self.page["page_id"])
            before_audit=len(self.rt["generation"])
            try:
                result = original(*args, **kwargs)
                if len(self.rt["generation"])!=before_audit+1:
                    raise ContractError("Actual generation audit arity differs")
                self.binding.generated(self.rt["generation"][-1])
                ledger("CALL_RESULT", engine="tele", request_id=request,
                       page_id=self.page["page_id"], returned=True)
                return result
            except BaseException as exc:
                if len(self.rt["generation"])==before_audit+1:
                    self.rt["generation"][-1].update({k:bound[k] for k in
                        ("run_id","page_id","slot_id","request_id","input_sha256","model_sha256","kind")})
                ledger("CALL_RESULT", engine="tele", request_id=request,
                       page_id=self.page["page_id"], returned=False, error_type=type(exc).__name__)
                raise
            finally:
                # Never reset the enclosing page deadline when leaving a request.
                left = remaining - (time.monotonic() - before)
                signal.setitimer(signal.ITIMER_REAL, max(left, 0.000001), interval)
        self.rt["model"].generate = generate

    def parse(self, page, source, folder):
        rt = self.rt; folder = Path(folder)
        self.page, self.count = page, 0
        self.expert_calls=0
        self.adoptions={"layout":0,"formula":0,"content_changed":0}
        rt["generation"].clear()
        client = rt["TeleOCRClient"](backend="transformers", model=rt["model"],
                                     processor=rt["processor"], batch_size=1, max_concurrency=1)
        if client.batching_mode!="stepping" or client.client.batch_size!=1 or client.executor is not None:
            raise ContractError("Only serial original batch-one stepping can be audited")
        graph=BindingGraph(self.run_id,page,self.model_identity,self.tensor_binding)
        self.binding=graph
        restore_backend=bind_backend(client.client,graph)
        helper = client.helper
        prepare, post = helper.prepare_for_extract, helper.post_process
        prepare_layout=helper.prepare_for_layout
        geometry, renders, selected = [], [], []
        actual_crops={};actual_render=[];transaction=None
        if self.mode!="off" and self.runtime.get("schema")==2:
            from .v31_transactions import V31Page, original_table_validator
            table_module=importlib.import_module("TeleOCR.vlm_utils.post_process.otsl2html")
            transaction=V31Page(page,self.profile,self.mode,self.experts,
                self.runtime["v31"]["special_tokens"],original_table_validator(table_module))
        def rgb(image):
            value = image.convert("RGB")
            return {"width":value.width,"height":value.height,"mode":"RGB",
                    "rgb_sha256":hashlib.sha256(value.tobytes()).hexdigest()}
        def observe(image, blocks, not_extract_list=None):
            import numpy as np
            result = prepare(image, blocks, not_extract_list)
            bindings=graph.prepare_content(result,blocks)
            if actual_render:raise ContractError("Multiple rendered pages in one page transaction")
            actual_render.append(image)
            crops, prompts, params, indices = result
            render = rgb(image)
            renders.append(render)
            image.save(folder/"render.png")
            width,height=image.size
            if width*height>64000000:
                scale=(64000000/(width*height))**0.5
                width,height=max(1,round(width*scale)),max(1,round(height*scale))
            for crop, prompt, param, index, binding in zip(crops,prompts,params,indices,bindings):
                slot = binding["slot_id"]
                actual_crops[slot]=crop
                path = folder / f"crop-{index:05d}.png"
                crop.save(path)
                block = blocks[index]
                points=np.array(block.bbox,dtype=np.float32).reshape(-1,2)
                points[:,0]*=width
                points[:,1]*=height
                geometry.append({"slot_id":slot,"native_index":index,
                    "request_id":binding["request_id"],
                    "page_id":page["page_id"], "input_sha256":page["file_sha256"],
                    "render":render,"crop":rgb(crop),"crop_file_sha256":sha(path),
                    "bbox":self.snapshot(block.bbox),"angle":self.snapshot(block.angle),
                    "helper_scaled_size":[width,height],
                    "integer_points_from_frozen_helper_rule":points.astype(np.int32).tolist(),
                    "prompt":self.snapshot(prompt),"parameters":params_snapshot(param),
                    "transform":"original frozen Tele helper; no additional recrop/resize",
                    "model_sha256":self.model_identity})
            return result
        def observe_layout(image):
            prepared=prepare_layout(image)
            graph.register(prepared,"layout")
            prepared.save(folder/"layout-input.png")
            return prepared
        def observe_post(blocks):
            for index, block in enumerate(blocks):
                content = getattr(block, "content", None)
                slot=f"{page['page_id']}:native:{index}"
                binding=graph.slots.get(slot)
                if binding is not None and (binding["stage"]!="completed" or binding["raw_content"]!=content):
                    raise ContractError("Content assignment differs from bound completion")
                selected.append({"slot_id":slot,
                    "request_id":None if binding is None else binding["request_id"],
                    "generation":"not_extracted_by_native_helper" if binding is None else "bound",
                    "native_index":index,"raw_content":copy.deepcopy(content)})
            primary=copy.deepcopy(selected)
            if transaction is not None:
                def fresh(image,draft,slot,index):
                    result=prepare(image,[draft])
                    if len(result)!=4 or any(len(x)!=1 for x in result) or result[3]!=[0]:
                        raise ContractError("Original helper did not prepare exactly the proposed slot")
                    crops,prompts,params,_=result
                    binding=graph.register(crops[0],"fresh",slot,index)
                    file=folder/f"fresh-{index:05d}.png"
                    crops[0].save(file)
                    geometry.append({"slot_id":slot,"request_id":binding["request_id"],
                        "kind":"fresh","crop":rgb(crops[0]),"crop_file_sha256":sha(file),
                        "render":rgb(image),"bbox":self.snapshot(draft.bbox),"angle":draft.angle,
                        "prompt":prompts[0],"parameters":params_snapshot(params[0]),
                        "transform":"original frozen Tele helper; no additional recrop/resize"})
                    values=client.client.batch_predict(crops,prompts,params)
                    if len(values)!=1 or binding["stage"]!="completed" or values[0]!=binding["raw_content"]:
                        raise ContractError("Fresh native recognition binding differs")
                    draft.content=values[0]
                    return crops[0],binding
                from TeleOCR.vlm_utils.structs import ContentBlock
                blocks,final=transaction.apply(actual_render[0],blocks,graph.slots,actual_crops,fresh,ContentBlock)
                self.expert_calls=transaction.formula_calls+(1 if transaction.layout_on else 0)
                self.adoptions={"layout":sum(e.get("reason")=="committed" for e in transaction.events),
                    "formula":sum(e.get("actual_content_source")=="paddle" for e in transaction.events),
                    "content_changed":sum(e.get("actual_content_source")=="paddle" and e["primary"]!=e["selected"] for e in transaction.events)}
            else:
                final=[{"slot_id":f"{page['page_id']}:native:{i}","block":copy.deepcopy(dict(b))}
                       for i,b in enumerate(blocks)]
            from .selection_audit import build
            write(folder/"PRIMARY_CONTENT.json",primary)
            write(folder/"V31_FINAL_RAW.json",final)
            selected[:]=build(primary,final,transaction.events if transaction else [],list(graph.requests.values()))
            return post(blocks)
        helper.prepare_for_extract, helper.post_process = observe, observe_post
        helper.prepare_for_layout=observe_layout
        error = None
        try:
            if page["kind"] == "pdf":
                pdfium = importlib.import_module("pypdfium2")
                pdf = pdfium.PdfDocument(str(source))
                try:
                    if len(pdf) != page["page_count"]:
                        raise ContractError("PDF page count changed")
                finally:
                    pdf.close()
            rt["do_parse"](str(folder/"native"), [page["page_id"]],
                           [rt["read_fn"](str(source))],
                           [[page["page_ordinal"]] if page["kind"] == "pdf" else None],
                           predictor=client)
            graph.validate_complete()
        except BaseException as exc:
            error = {"type":type(exc).__name__, "message":str(exc)}
            raise
        finally:
            if error is not None:
                # Only audit/patch restoration is permitted after failure; no further generation.
                signal.setitimer(signal.ITIMER_REAL,self.budget["page_audit_seconds"])
                progress(self.output/"PROGRESS.json",{"phase":"page_audit",
                    "monotonic":time.monotonic(),"page_id":page["page_id"]})
            helper.prepare_for_extract, helper.post_process = prepare, post
            helper.prepare_for_layout=prepare_layout
            restore_backend()
            self.binding=None
            # Audit write failure propagates, even when another error occurred.
            write(folder/"INPUT.json",page)
            write(folder/"RENDER.json",renders)
            write(folder/"GEOMETRY.json",geometry)
            write(folder/"CONTENT.json",selected)
            write(folder/"V31_TRANSACTIONS.json",transaction.events if transaction is not None else [])
            write(folder/"REQUEST_BINDINGS.json",list(graph.requests.values()))
            generations = self.snapshot(rt["generation"])
            write(folder/"GENERATION.json",generations)
            write(folder/"AUDIT_STATUS.json",{"complete":True,"patches_restored":True,"error":error})
            resources = rt["resources"].result()
            write(folder/"RESOURCE_AUDIT.json",resources)
            if resources["unknown_resources"]:
                raise ContractError("Unknown model/auxiliary resources")
        md = folder/"native"/page["page_id"]/(page["page_id"]+".md")
        middle = folder/"native"/page["page_id"]/(page["page_id"]+"_middle.json")
        if not md.is_file() or not middle.is_file() or not renders:
            raise ContractError("Native full-page output/audit missing")
        return md, any(g.get("returned") and g["termination"]["stop"] != "eos"
                       for g in rt["generation"])
