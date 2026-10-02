"""Lossless, serial, fail-closed IPC between owned native and Paddle containers."""
import hashlib
import io
import time
from pathlib import Path
from .core import ContractError, DeadlineError, canonical, read, sha, write
from .request_binding import rgb_binding
from .v31_rules import PROFILE_SHA, PROFILE

IDENTITY=("run_id","page_id","slot_id","request_id","input_sha256","render",
          "crop","png_sha256","model_sha256","config_sha256","source_sha256","profile_sha256","engine")

def formula_config():
    return {"task":"equation","query":"Formula Recognition:","min_pixels":112896,
            "max_pixels":1003520,"max_new_tokens":4096,"use_cache":True,"skip_special_tokens":True}

def validate(request,response):
    if any(response.get(k)!=request[k] for k in IDENTITY):
        raise ContractError("Expert response identity mismatch")
    if response.get("status")=="fatal":
        raise ContractError("Expert system error: "+str(response.get("error")))
    if request["engine"]=="formula":
        if response.get("status") not in ("ok","empty","length","other","decode_error"):
            raise ContractError("Unknown expert completion status")
        audit=response.get("generation",{})
        tensors=response.get("processor_tensors")
        if not tensors or tensors!=audit.get("input_tensors") or canonical(tensors)!=response.get("tensor_sha256"):
            raise ContractError("Actual expert processor/generate tensor binding missing")
        if not {"input_ids","pixel_values","image_grid_thw"}<=set(tensors):
            raise ContractError("Actual Paddle tensor/grid fields missing")
        if audit.get("returned") is not True:raise ContractError("Expert generation failed")
        ids=audit.get("token_ids");eos=audit.get("eos_ids")
        if (not isinstance(ids,list) or any(type(x) is not int for x in ids) or len(ids)>4096
            or not isinstance(eos,list) or not eos or any(type(x) is not int for x in eos)):
            raise ContractError("Invalid actual expert token evidence")
        stop="eos" if any(t in eos for t in ids) else "length" if len(ids)==4096 else "other"
        if stop!=audit.get("stop"):raise ContractError("Expert stop evidence differs")
        response["termination"]={k:audit[k] for k in ("token_ids","eos_ids","stop")}
        if response["status"]=="ok" and (stop!="eos" or not isinstance(response.get("raw_text"),str) or not response["raw_text"].strip()):
            raise ContractError("Expert accepted invalid completion")
    else:
        if response.get("status")!="ok" or not isinstance(response.get("candidates"),list):
            raise ContractError("Layout response failed")
        if not response.get("trace") or response.get("runner_calls")!=1:
            raise ContractError("Missing actual layout preprocess/runner trace")
        from .v31_coordinates import verify
        passed,reason=verify(response,request["render"]["width"],request["render"]["height"])
        if not passed:raise ContractError("Layout coordinate response mismatch: "+reason)
    return response

class ResidentClient:
    def __init__(self, control, output, ledger, clock=time.monotonic):
        self.control,self.output,self.ledger,self.clock=control,Path(output),ledger,clock
        self.root=self.output/"experts"/"ipc";self.sequence=0;self.counts={}
        self.seen=set()
    def call(self,engine,image,page,slot_id,render):
        self.sequence+=1
        key=(page["page_id"],engine);count=self.counts.get(key,0)
        cap=1 if engine=="layout" else min(32,self.control["budget"]["expert_calls_per_page"])
        if count>=cap:raise ContractError("Expert request budget exceeded")
        self.counts[key]=count+1
        config=self.control["runtime"]["v31"][engine]
        identifier=self.control["run_id"]+"/"+page["page_id"]+"/"+engine+"/"+str(self.sequence)
        if identifier in self.seen:raise ContractError("Duplicate expert dispatch")
        self.seen.add(identifier)
        name=f"{self.sequence:08d}"
        path=self.root/(name+".png")
        stream=io.BytesIO();image.save(stream,format="PNG")
        raw=stream.getvalue()
        with path.open("xb") as f:f.write(raw)
        request={"run_id":self.control["run_id"],"page_id":page["page_id"],"slot_id":slot_id,
                 "request_id":identifier,"input_sha256":page["file_sha256"],"render":render,
                 "crop":rgb_binding(image),"png_sha256":hashlib.sha256(raw).hexdigest(),
                 "model_sha256":config["model_sha256"],"config_sha256":config["config_sha256"],
                 "source_sha256":config["source_sha256"],"profile_sha256":PROFILE_SHA,
                 "engine":engine,"png_name":name+".png"}
        self.ledger("CALL_START",engine=engine,request_id=identifier,page_id=page["page_id"])
        # Atomic publication: worker cannot observe a partially written request.
        temp=self.root/(name+".pending");write(temp,request);temp.rename(self.root/(name+".request.json"))
        end=min(self.control["absolute_stop_monotonic"],self.clock()+(60 if engine=="layout" else 180))
        import signal
        left,_=signal.getitimer(signal.ITIMER_REAL)
        if left<=0:raise ContractError("Expert dispatch outside page deadline")
        end=min(end,self.clock()+left)
        answer=self.root/(name+".response.json")
        while not answer.exists():
            if (self.output/"experts"/"ERROR.json").exists():
                raise ContractError("Resident expert process failed")
            if self.clock()>=end:raise DeadlineError("Expert/page/host deadline exhausted")
            time.sleep(min(.05,end-self.clock()))
        response=validate(request,read(answer))
        self.ledger("CALL_RESULT",engine=engine,request_id=identifier,page_id=page["page_id"],returned=True)
        return response
    def formula(self,image,binding,render):
        page={"page_id":binding["page_id"],"file_sha256":binding["input_sha256"]}
        if rgb_binding(image)!=binding["crop"]:raise ContractError("Expert crop differs from native prepared crop")
        return self.call("formula",image,page,binding["slot_id"],render)
    def layout(self,image,page,render):
        if rgb_binding(image)!=render:raise ContractError("Layout input differs from actual Tele render")
        return self.call("layout",image,page,None,render)
