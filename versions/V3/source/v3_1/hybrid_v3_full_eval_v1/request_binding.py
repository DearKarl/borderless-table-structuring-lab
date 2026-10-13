"""Explicit actual-object request graph: prepare -> processor -> generate -> completion."""
import hashlib
from .core import ContractError, canonical

def rgb_binding(image):
    if image.mode!="RGB":
        raise ContractError("Frozen helper must supply actual RGB objects")
    return {"width":image.width,"height":image.height,"mode":"RGB",
            "rgb_sha256":hashlib.sha256(image.tobytes()).hexdigest()}

def params_snapshot(params):
    if params is None:return None
    names=("temperature","top_p","top_k","presence_penalty","frequency_penalty",
           "repetition_penalty","no_repeat_ngram_size","max_new_tokens")
    return {key:getattr(params,key) for key in names}

class BindingGraph:
    def __init__(self, run_id, page, model_sha256, tensor_binding):
        self.run_id,self.page,self.model_sha256=run_id,page,model_sha256
        self.tensor_binding=tensor_binding
        self.objects={};self.requests={};self.slots={};self.current=None
        self.layout_count=0

    def register(self,image,kind,slot_id=None,attempt=None):
        if id(image) in self.objects:
            raise ContractError("Duplicate prepared image object")
        if kind=="layout":
            if slot_id is not None:raise ContractError("Layout request has no content slot")
            suffix=str(self.layout_count);self.layout_count+=1
        elif kind in ("content","fresh"):
            if slot_id is None or not slot_id.startswith(self.page["page_id"]+":"):
                raise ContractError("Cross-page or missing stable slot")
            if kind=="content" and slot_id in self.slots:raise ContractError("Duplicate slot binding")
            suffix=slot_id
            if kind=="fresh":
                if type(attempt) is not int or attempt<0:raise ContractError("Fresh attempt index required")
                suffix+="/"+str(attempt)
        else:raise ContractError("Unknown request kind")
        request_id=self.run_id+"/"+self.page["page_id"]+"/"+kind+"/"+suffix
        if request_id in self.requests:raise ContractError("Duplicate actual request")
        record={"run_id":self.run_id,"page_id":self.page["page_id"],"kind":kind,
                "slot_id":slot_id,"request_id":request_id,"input_sha256":self.page["file_sha256"],
                "crop":rgb_binding(image),"model_sha256":self.model_sha256,"stage":"prepared"}
        self.objects[id(image)]=(image,record)
        self.requests[request_id]=record
        if kind=="content":self.slots[slot_id]=record
        return record

    def prepare_content(self, returned, blocks):
        if not isinstance(returned,(tuple,list)) or len(returned)!=4:
            raise ContractError("Native prepare arity")
        crops,prompts,params,indices=returned
        if any(not isinstance(x,list) for x in returned):
            raise ContractError("Native prepare must return four lists")
        if len({len(x) for x in returned})!=1:
            raise ContractError("Native prepare lengths differ")
        if len(set(indices))!=len(indices) or any(type(i) is not int or not 0<=i<len(blocks) for i in indices):
            raise ContractError("Duplicate/unknown native slot index")
        records=[]
        for crop,index in zip(crops,indices):
            records.append(self.register(crop,"content",self.page["page_id"]+":native:"+str(index)))
        return records

    def begin(self,images,chat_prompts,sampling_params):
        if self.current is not None or len(images)!=1 or len(chat_prompts)!=1:
            raise ContractError("Only serial batch-one native dispatch is bound")
        pair=self.objects.get(id(images[0]))
        if pair is None or pair[0] is not images[0]:
            raise ContractError("Unknown/cross-page processor input")
        record=pair[1]
        if record["stage"]!="prepared" or record["crop"]!=rgb_binding(images[0]):
            raise ContractError("Duplicate dispatch or changed actual crop")
        record.update(stage="dispatch",chat_prompts=list(chat_prompts),
                      sampling_params=params_snapshot(sampling_params))
        self.current={"record":record,"image":images[0],"processor_calls":0,"generate_calls":0}

    def processor(self,images,feature):
        active=self.current
        if active is None or len(images)!=1 or images[0] is not active["image"]:
            raise ContractError("Processor received a different actual crop")
        if active["processor_calls"]:
            raise ContractError("Repeated processor call")
        active["processor_calls"]+=1
        active["record"]["processor_input_crop"]=rgb_binding(images[0])
        return ObservedFeatures(feature,self)

    def transformed(self,features):
        active=self.current
        if active is None or active.get("tensors") is not None:
            raise ContractError("Unbound/repeated actual processor .to")
        tensors={k:v for k,v in features.items() if hasattr(v,"shape")}
        if not {"input_ids","pixel_values","image_grid_thw"}<=set(tensors):
            raise ContractError("Actual processor tensor/grid fields missing")
        active["tensors"]=tensors
        active["record"]["processor_tensors"]=self.tensor_binding(tensors)
        return features

    def generating(self,kwargs):
        active=self.current
        if active is None or active["generate_calls"] or not active.get("tensors"):
            raise ContractError("Generate outside a uniquely bound actual processor call")
        if any(kwargs.get(k) is not v for k,v in active["tensors"].items()):
            raise ContractError("Generate tensor object differs from actual processor")
        actual=self.tensor_binding({k:kwargs[k] for k in active["tensors"]})
        if actual!=active["record"]["processor_tensors"]:
            raise ContractError("Processor tensor changed before generate")
        active["generate_calls"]+=1
        return active["record"]

    def generated(self,audit):
        active=self.current
        if active is None or active["generate_calls"]!=1 or audit.get("returned") is not True or active["record"]["stage"]!="dispatch":
            raise ContractError("Missing/failed native generation")
        if audit.get("input_tensors")!=active["record"]["processor_tensors"]:
            raise ContractError("Actual generation tensor audit differs")
        config=audit.get("effective_generation_config")
        if not isinstance(config,dict) or not isinstance(audit.get("termination"),dict):
            raise ContractError("Actual generation config/termination absent")
        record=active["record"]
        for key in ("run_id","page_id","slot_id","request_id","input_sha256","model_sha256"):
            if key in audit and audit[key]!=record[key]:
                raise ContractError("Cross-page/request replay in actual generation audit")
        record.update(stage="generated",config_sha256=canonical(config),
                      tensor_sha256=canonical(audit["input_tensors"]),termination=audit["termination"])
        audit.update({k:record[k] for k in ("run_id","page_id","kind","slot_id","request_id",
                       "input_sha256","model_sha256","config_sha256","tensor_sha256")})
        audit["crop_rgb_sha256"]=record["crop"]["rgb_sha256"]
        audit["processor_tensors"]=record["processor_tensors"]

    def complete(self,outputs):
        active=self.current
        if active is None or active["record"]["stage"]!="generated":
            raise ContractError("Completion has no bound generation")
        if not isinstance(outputs,list) or len(outputs)!=1 or not isinstance(outputs[0],str):
            raise ContractError("Native completion arity/type differs")
        active["record"].update(stage="completed",raw_content=outputs[0])
        self.current=None

    def validate_complete(self):
        if self.current is not None or self.layout_count!=1:
            raise ContractError("Unfinished request or missing/duplicate layout")
        if any(r["stage"]!="completed" for r in self.requests.values()):
            raise ContractError("Prepared slot never completed")
        return list(self.requests.values())

class ObservedFeatures:
    def __init__(self,feature,graph):self.feature,self.graph=feature,graph
    def to(self,*args,**kwargs):
        # Calls the original conversion exactly once and returns its original object.
        result=self.feature.to(*args,**kwargs)
        return self.graph.transformed(result)

class ProcessorProxy:
    def __init__(self,processor,graph):self.original,self.graph=processor,graph
    def __getattr__(self,name):return getattr(self.original,name)
    def __call__(self,*args,**kwargs):
        result=self.original(*args,**kwargs)
        return self.graph.processor(kwargs["images"],result)

def bind_backend(backend,graph):
    original_batch,original_processor=backend._predict_one_batch,backend.processor
    backend.processor=ProcessorProxy(original_processor,graph)
    def batch(image_objs,chat_prompts,sampling_params,**kwargs):
        graph.begin(image_objs,chat_prompts,sampling_params)
        outputs=original_batch(image_objs,chat_prompts,sampling_params,**kwargs)
        graph.complete(outputs)
        return outputs
    backend._predict_one_batch=batch
    def restore():
        backend._predict_one_batch,backend.processor=original_batch,original_processor
    return restore
