"""Reversible single-page TeleOCR hook and passive native-call observation."""
from contextlib import contextmanager
import hashlib
import inspect
import json
import threading
import time
from .inputs import transform_native_image, image_fingerprint

_PAGE_LOCK = threading.Lock()

@contextmanager
def page_input_hook(vlm_analyze, source, action, *, deadline_seconds=600, clock=time.monotonic, on_prepared=None):
    """Patch one serial native page; a supervisor enforces the hard deadline."""
    if action not in source.available_actions(): raise ValueError("Unavailable action")
    if not _PAGE_LOCK.acquire(blocking=False): raise RuntimeError("Concurrent page")
    original=None;handles=[];started=None
    audit={"action":action,"loader_calls":0,"page_starts":1,"deadline_seconds":deadline_seconds,
           "hard_deadline_supervisor_required":True}
    try:
        started=clock()
        original=vlm_analyze.load_images_from_pdf
        def wrapped(*args,**kwargs):
            if audit["loader_calls"]: raise RuntimeError("Repeated page render")
            remaining=deadline_seconds-(clock()-started)
            if remaining<=0: raise TimeoutError("Whole-page deadline")
            bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults()
            if bound.arguments.get("start_page_id",0)!=0 or bound.arguments.get("end_page_id") not in (None,0):
                raise ValueError("Expected selected single-page PDF")
            import pypdfium2 as pdfium
            check=pdfium.PdfDocument(bound.arguments["pdf_bytes"])
            try:
                if len(check)!=1: raise ValueError("Expected exactly one page")
                audit["selected_pdf_sha256"]=hashlib.sha256(bound.arguments["pdf_bytes"]).hexdigest()
            finally: check.close()
            audit["loader_calls"]+=1
            if action in ("B","C") and source.source_type=="original_pdf": bound.arguments["dpi"]=300
            if "timeout" in bound.arguments:
                native=bound.arguments["timeout"]
                if native is None:
                    resolver=getattr(original,"__globals__",{}).get("get_load_images_timeout")
                    if resolver: native=resolver()
                    else:
                        import os
                        try: native=int(os.environ.get("TeleOCR_PDF_RENDER_TIMEOUT","300"))
                        except ValueError: native=300
                        if native<=0: native=300
                if native<=0: raise ValueError("Invalid render timeout")
                bound.arguments["timeout"]=min(remaining,native)
                audit.update(native_render_timeout=native,effective_render_timeout=bound.arguments["timeout"])
            before=clock()
            images,doc=original(*bound.args,**bound.kwargs);handles.append(doc)
            if len(images)!=1 or len(doc)!=1: raise ValueError("Unexpected multi-page result")
            page=doc[0]
            try: size=tuple(page.get_size())
            finally: page.close()
            result,transform=transform_native_image(images[0],action,source,size)
            audit.update(transform);audit["native_middle_page_size"]=list(map(int,size))
            audit["render_and_transform_seconds"]=clock()-before
            if clock()-started>deadline_seconds: raise TimeoutError("Whole-page deadline")
            if on_prepared is not None:
                before_features=clock()
                on_prepared(result["img_pil"],audit)
                audit["input_feature_seconds"]=clock()-before_features
                if clock()-started>deadline_seconds: raise TimeoutError("Whole-page deadline during input features")
            return [result],doc
        vlm_analyze.load_images_from_pdf=wrapped
        yield audit
        if audit["loader_calls"]!=1: raise RuntimeError("Expected one native pipeline")
        if clock()-started>deadline_seconds: raise TimeoutError("Whole-page deadline")
    except BaseException:
        for doc in handles: doc.close()
        raise
    finally:
        try:
            if original is not None: vlm_analyze.load_images_from_pdf=original
        finally:
            try:
                if started is not None: audit["whole_page_seconds"]=clock()-started
            finally: _PAGE_LOCK.release()

def _tolist(value):
    return value.tolist() if hasattr(value, "tolist") else value

def _tensor_meta(value, *, hash_tensor=False):
    info = {"shape": list(value.shape), "dtype": str(value.dtype), "object_id": id(value)}
    if hash_tensor:
        # Never enabled implicitly for GPU tensors.
        if str(getattr(value, "device", "cpu")) != "cpu":
            raise ValueError("Tensor hash requires explicit CPU materialization policy")
        info["sha256"] = hashlib.sha256(value.numpy().tobytes() if hasattr(value, "numpy")
                                        else value.tobytes()).hexdigest()
    return info

class NativeObserver:
    """Bind crop, prompt, original processor, actual .to result and one generation."""
    def __init__(self, merge_size=2, *, page_id="page"):
        import uuid
        self.merge_size=merge_size;self.page_id=page_id;self.session=uuid.uuid4().hex
        self.events=[];self.requests={};self.images={};self.active=None;self.used=False
        self.conversion_restores=[];self.dispatch=None
        self.visual_tokens={"layout":0,"recognition":0}

    def register(self,image,stage,index=None,prompt=None,block=None):
        if id(image) in self.images: raise RuntimeError("Duplicate prepared image")
        slot="layout" if stage=="layout" else f"native:{index}"
        rid=f"{self.session}/{self.page_id}/{slot}"
        if rid in self.requests: raise RuntimeError("Duplicate slot")
        rec=dict(request_id=rid,page_id=self.page_id,slot_id=slot,native_index=index,
                 stage=stage,state="prepared",crop=image_fingerprint(image),
                 raw_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None)
        if block is not None: rec.update(bbox=block.bbox,angle=block.angle)
        self.images[id(image)]=(image,rec);self.requests[rid]=rec

    def begin(self,images,prompts):
        if self.active is not None or len(images)!=1 or len(prompts)!=1: raise RuntimeError("Serial batch-one required")
        pair=self.images.get(id(images[0]))
        if pair is None or pair[0] is not images[0]: raise RuntimeError("Unknown/cross-page crop")
        rec=pair[1]
        if rec["state"]!="prepared" or rec["crop"]!=image_fingerprint(images[0]): raise RuntimeError("Replayed/changed crop")
        dispatch=self.dispatch["entries"].get(id(images[0])) if self.dispatch is not None else None
        if dispatch is None or dispatch["image"] is not images[0] or dispatch["chat_prompt"]!=prompts[0]:
            raise RuntimeError("Unbound chat prompt")
        rec.update(state="dispatch",prompt_sha256=hashlib.sha256(prompts[0].encode()).hexdigest())
        self.active=dict(image=images[0],prompts=list(prompts),record=rec,processor_calls=0,to_calls=0,generate_calls=0,tensors=None)

    def processor_input(self,images,text):
        active=self.active
        if active is None or active["processor_calls"] or len(images)!=1 or images[0] is not active["image"] or list(text)!=active["prompts"]:
            raise RuntimeError("Unbound/repeated processor or changed prompt/crop")
        active["processor_calls"]+=1

    def processor_result(self,result):
        active=self.active;original_to=result.to
        missing=object();previous=vars(result).get("to",missing);restored=False
        def restore():
            nonlocal restored
            if not restored:
                if previous is missing: delattr(result,"to")
                else: setattr(result,"to",previous)
                restored=True
        def convert(*args,**kwargs):
            if self.active is not active or active["to_calls"]: raise RuntimeError("Replayed conversion")
            active["to_calls"]+=1
            try: converted=original_to(*args,**kwargs)
            finally: restore()
            tensors={k:v for k,v in converted.items() if hasattr(v,"shape")}
            if not {"input_ids","pixel_values","image_grid_thw"}<=set(tensors): raise RuntimeError("Missing tensor fields")
            active["tensors"]=tensors;active["metadata"]={k:_tensor_meta(v) for k,v in tensors.items()}
            grids=_tolist(converted["image_grid_thw"]);total=0
            for t,h,w in grids:
                product=int(t)*int(h)*int(w)
                if product%self.merge_size**2: raise RuntimeError("Nonintegral grid")
                total+=product//self.merge_size**2
            rec=active["record"]
            rec.update(processor_tensors=active["metadata"],image_grid_thw=grids,visual_tokens=total,
                       processor_feature_object_id=id(result),converted_feature_object_id=id(converted))
            self.visual_tokens[rec["stage"]]+=total
            return converted
        # Patch the instance only: both processor output and .to output retain identity.
        setattr(result,"to",convert);self.conversion_restores.append(restore)
        return result

    @contextmanager
    def observe(self,helper,backend):
        if self.used: raise RuntimeError("Observer cannot be replayed across pages")
        self.used=True;restores=[]
        def replace(obj,name,value):
            restores.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
        try:
            old_layout=helper.prepare_for_layout;old_extract=helper.prepare_for_extract
            old_processor=backend.processor;old_generate=backend.model.generate;old_batch=backend._predict_one_batch
            old_predict=backend.batch_predict
            observer=self
            def layout(*args,**kwargs):
                out=old_layout(*args,**kwargs);self.register(out,"layout");return out
            def extract(*args,**kwargs):
                out=old_extract(*args,**kwargs);crops,prompts,params,indices=out
                blocks=inspect.signature(old_extract).bind(*args,**kwargs).arguments["blocks"]
                if len({len(v) for v in out})!=1 or len(set(indices))!=len(indices): raise RuntimeError("Invalid extraction slots")
                for crop,prompt,index in zip(crops,prompts,indices):
                    if type(index) is not int or not 0<=index<len(blocks): raise RuntimeError("Invalid native index")
                    self.register(crop,"recognition",index,prompt,blocks[index])
                return out
            class Processor:
                def __call__(self,*args,**kwargs):
                    observer.processor_input(kwargs["images"],kwargs["text"])
                    result=old_processor(*args,**kwargs)
                    return observer.processor_result(result)
                def apply_chat_template(self,*args,**kwargs):
                    dispatch=observer.dispatch
                    if dispatch is None or dispatch["template_index"]>=len(dispatch["template_groups"]): raise RuntimeError("Unbound/repeated chat template")
                    result=old_processor.apply_chat_template(*args,**kwargs)
                    for entry in dispatch["template_groups"][dispatch["template_index"]]: entry["chat_prompt"]=result
                    dispatch["template_index"]+=1
                    return result
                def __getattr__(self,name): return getattr(old_processor,name)
            def predict(images,prompts="",sampling_params=None,*args,**kwargs):
                if self.dispatch is not None: raise RuntimeError("Concurrent native dispatch")
                prompt_list=[prompts]*len(images) if isinstance(prompts,str) else list(prompts)
                if len(prompt_list)!=len(images) or len({id(im) for im in images})!=len(images): raise RuntimeError("Invalid dispatch slots")
                entries=[]
                for im,prompt in zip(images,prompt_list):
                    pair=self.images.get(id(im))
                    if pair is None or pair[0] is not im: raise RuntimeError("Unknown/cross-page crop")
                    rec=pair[1];digest=hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt,str) else None
                    if rec["state"]!="prepared": raise RuntimeError("Replayed crop")
                    if digest is None or (rec["raw_prompt_sha256"] is not None and digest!=rec["raw_prompt_sha256"]):
                        raise RuntimeError("Prepared prompt mismatch")
                    rec["raw_prompt_sha256"]=digest;entries.append(dict(image=im,chat_prompt=None))
                self.dispatch=dict(entries={id(e["image"]):e for e in entries},template_index=0,
                                   template_groups=[entries] if isinstance(prompts,str) else [[e] for e in entries])
                try: return old_predict(images,prompts,sampling_params,*args,**kwargs)
                finally: self.dispatch=None
            def batch(image_objs,chat_prompts,sampling_params,**kwargs):
                self.begin(image_objs,chat_prompts);active=self.active
                try:
                    result=old_batch(image_objs,chat_prompts,sampling_params,**kwargs)
                    if active["generate_calls"]!=1 or active["record"]["state"]!="generated": raise RuntimeError("Missing bound generation")
                    if not isinstance(result,list) or len(result)!=1 or not isinstance(result[0],str): raise RuntimeError("Invalid decode result")
                    active["record"]["state"]="completed";return result
                except BaseException as e:
                    active["record"].update(state="failed",error=repr(e));raise
                finally: self.active=None
            def generate(*args,**kwargs):
                active=self.active
                if active is None or active["generate_calls"] or active["tensors"] is None: raise RuntimeError("Unbound/repeated generate")
                if args or {k for k,v in kwargs.items() if hasattr(v,"shape")}!=set(active["tensors"]) or any(kwargs.get(k) is not v for k,v in active["tensors"].items()): raise RuntimeError("Post-to tensor object mismatch")
                if {k:_tensor_meta(kwargs[k]) for k in active["tensors"]}!=active["metadata"]: raise RuntimeError("Tensor metadata changed")
                active["generate_calls"]+=1;rec=active["record"]
                event=dict(kind="generate",request_id=rec["request_id"],slot_id=rec["slot_id"],page_id=self.page_id,
                    stage=rec["stage"],prompt_sha256=rec["prompt_sha256"],model_object_id=id(backend.model),
                    processor_tensors=active["metadata"],kwargs={k:_tensor_meta(v) if hasattr(v,"shape") else v for k,v in kwargs.items()},
                    returned=False)
                self.events.append(event);before=time.monotonic()
                try:
                    out=old_generate(*args,**kwargs);seq=getattr(out,"sequences",out)
                    n=kwargs["input_ids"].shape[-1];length=seq.shape[-1]
                    event.update(returned=True,output_shape=list(seq.shape),generated_length=length-n)
                    config=getattr(backend.model,"generation_config",None)
                    eos=kwargs.get("eos_token_id",getattr(config,"eos_token_id",None))
                    eos=set(eos if isinstance(eos,(list,tuple,set)) else [eos]) if eos is not None else set()
                    last=_tolist(seq[0,-1]);cap=kwargs.get("max_new_tokens")
                    limit=n+cap if cap is not None else kwargs.get("max_length")
                    event["termination"]=dict(last_token=last,eos_reached=last in eos,
                        length_limit_reached=limit is not None and length>=limit,
                        reason="eos" if last in eos else "length" if limit is not None and length>=limit else "unknown")
                    rec["state"]="generated";return out
                except BaseException as e: event["error"]=repr(e);raise
                finally: event["elapsed_seconds"]=time.monotonic()-before
            replace(helper,"prepare_for_layout",layout);replace(helper,"prepare_for_extract",extract)
            replace(backend,"processor",Processor());replace(backend,"_predict_one_batch",batch)
            replace(backend,"batch_predict",predict)
            replace(backend.model,"generate",generate)
            yield self
            if any(r["state"]!="completed" for r in self.requests.values()): raise RuntimeError("Unfinished prepared request")
        finally:
            try:
                for restore in reversed(self.conversion_restores): restore()
            finally:
                for obj,name,old in reversed(restores): setattr(obj,name,old)

class PhaseTimes:
    """Collect host timings without invoking any model or pipeline itself."""
    def __init__(self):
        self.seconds={name:0. for name in ("initialization","render","features","selection","parser","serialization")}

    @contextmanager
    def measure(self, name):
        if name not in self.seconds:raise ValueError("Unknown timing phase")
        before=time.monotonic()
        try:yield
        finally:self.seconds[name]+=time.monotonic()-before
