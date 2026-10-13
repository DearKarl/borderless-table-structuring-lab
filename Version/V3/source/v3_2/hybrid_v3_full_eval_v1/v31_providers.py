"""Actual Paddle providers; imported only inside the owned Paddle container."""
import base64
import importlib
import json
import sys
from pathlib import Path
from .core import ContractError, canonical, sha
from .request_binding import rgb_binding
from .asset_binding import loaded_count
from .v31_protocol import formula_config

class Formula:
    def __init__(self,runtime):
        sys.path.insert(0,"/assets/native_code")
        from hybrid_v2_tele_base.paddle_formula_worker import FormulaPredictor
        from hybrid_v2_tele_base.audit import tensor_binding
        self.original=FormulaPredictor(Path("/assets")/runtime["v31"]["formula"]["model_asset"])
        self.tensor_binding=tensor_binding
        predictor=self.original.predictor
        self.parameters=loaded_count(runtime,"formula",predictor.infer.parameters(),predictor.infer.buffers())
        self.current=None
        preprocess=predictor.processor.preprocess
        switch=predictor._switch_inputs_to_device
        generate=predictor.infer.generate
        def observed_preprocess(data,*args,**kwargs):
            if self.current is None or self.current.get("preprocessed"):
                raise ContractError("Repeated/unbound Paddle processor")
            if len(data)!=1 or data[0]["image"] is not self.current["image"]:
                raise ContractError("Paddle processor did not receive transported crop object")
            if data[0]["query"]!="Formula Recognition:":
                raise ContractError("Paddle prompt changed")
            self.current["preprocessed"]=True
            return preprocess(data,*args,**kwargs)
        def observed_switch(data):
            if self.current is None or not self.current.get("preprocessed") or self.current.get("features") is not None:
                raise ContractError("Unbound/repeated Paddle device transfer")
            actual=switch(data)
            self.current["features"]=actual
            self.current["tensors"]=tensor_binding(actual)
            return actual
        def observed_generate(inputs,*args,**kwargs):
            if self.current is None or inputs is not self.current.get("features") or self.current.get("generated"):
                raise ContractError("Paddle generation input differs from actual processor")
            if tensor_binding(inputs)!=self.current["tensors"]:raise ContractError("Paddle input mutated")
            self.current["generated"]=True
            return generate(inputs,*args,**kwargs)
        predictor.processor.preprocess=observed_preprocess
        predictor._switch_inputs_to_device=observed_switch
        predictor.infer.generate=observed_generate
        self.restore=lambda:(setattr(predictor.processor,"preprocess",preprocess),
                             setattr(predictor,"_switch_inputs_to_device",switch),
                             setattr(predictor.infer,"generate",generate))
    def call(self,request,image,png):
        # Invoke original predictor directly to preserve the same observed RGB object.
        self.current={"image":image}
        first=len(self.original.audits);self.original.calls+=1
        try:
            returned=list(self.original.predictor.predict([{"image":image,"query":"Formula Recognition:"}],
                min_pixels=112896,max_pixels=1003520,max_new_tokens=4096,use_cache=True,skip_special_tokens=True))
            audits=self.original.audits[first:]
            if len(returned)!=1 or len(audits)!=1 or not self.current.get("generated"):
                raise ContractError("Paddle result/audit arity differs")
            audit=audits[0]
            if not audit.get("returned") or audit["input_tensors"]!=self.current["tensors"]:
                raise ContractError("Paddle actual generation audit mismatch")
            raw=returned[0].get("result")
            if not isinstance(raw,str):raise ContractError("Paddle result not text")
            status=audit["stop"] if audit["stop"]!="eos" else "ok" if raw.strip() else "empty"
            return {"status":status,"raw_text":raw,"generation":audit,
                    "processor_tensors":self.current["tensors"],
                    "tensor_sha256":canonical(self.current["tensors"])}
        finally:self.current=None
    def close(self):
        self.restore();self.original.close()

class Layout:
    def __init__(self,runtime):
        import numpy as np
        from paddlex.inference.models import create_predictor
        from paddlex.inference.models.runners.paddle_static.config import PaddlePredictorOption
        self.params=runtime["v31"]["layout"]["parameters"]
        construct={k:v for k,v in self.params.items() if k not in ("layout_shape_mode","filter_overlap_boxes")}
        self.predictor=create_predictor("PP-DocLayoutV3",
            model_dir=str(Path("/assets")/runtime["v31"]["layout"]["model_asset"]),
            engine="paddle_static",device="gpu:0",batch_size=1,use_hpip=False,
            pp_option=PaddlePredictorOption(run_mode="paddle"),**construct)
        if type(self.predictor).__name__!="LayoutAnalysisRunnerPredictor":
            raise ContractError("Frozen standalone layout predictor differs")
        self.trace=[];self.calls=0;owner=self
        def snapshot(v):
            if isinstance(v,np.ndarray):
                r={"shape":list(v.shape),"dtype":str(v.dtype),
                   "sha256":__import__("hashlib").sha256(v.tobytes()).hexdigest()}
                if v.size<=16:r["values"]=v.tolist()
                return r
            if isinstance(v,dict):
                return {k:snapshot(x) for k,x in v.items() if k in ("ori_img","img","img_size","ori_img_size","scale_factors")}
            if isinstance(v,(tuple,list)):return [snapshot(x) for x in v]
            if isinstance(v,(str,int,float,bool)) or v is None:return v
            return {"type":type(v).__name__}
        class Capture:
            def __init__(self,op,label):self.op,self.label=op,label
            def __getattr__(self,key):return getattr(self.op,key)
            def __call__(self,*args,**kwargs):
                if self.label=="actual_runner_inputs_outputs":
                    if owner.calls:raise ContractError("Multiple layout backend executions")
                    owner.calls+=1
                before=snapshot(args);value=self.op(*args,**kwargs)
                owner.trace.append({"stage":self.label,"before":before,"after":snapshot(value)})
                return value
        self.predictor.pre_ops=[Capture(op,type(op).__name__) for op in self.predictor.pre_ops]
        self.predictor.runner=Capture(self.predictor.runner,"actual_runner_inputs_outputs")
        self.predictor.post_op=Capture(self.predictor.post_op,"actual_postprocess_inputs")
    def call(self,request,image,png):
        self.trace.clear();self.calls=0
        # Original Paddle image loader reads the identical lossless RGB transport file.
        outputs=list(self.predictor([str(png)],**self.params))
        if len(outputs)!=1 or self.calls!=1:raise ContractError("Layout output/backend arity differs")
        result=outputs[0]
        import numpy as np
        original=result["input_img"]
        if original.shape!=(image.height,image.width,3):
            raise ContractError("Layout loader dimensions differ")
        # Frozen reader is OpenCV BGR; verify actual channels, never assume equivalence.
        if not np.array_equal(original[:,:,::-1],np.asarray(image)):
            raise ContractError("Layout BGR loader pixels differ from Tele rendered RGB")
        raw=json.loads(json.dumps(result["boxes"],default=lambda x:x.tolist() if hasattr(x,"tolist") else float(x)))
        candidates=[{"index":i,"label":b["label"],"score":b["score"],
                     "bbox":b["coordinate"],"angle":0} for i,b in enumerate(raw)]
        response={"status":"ok","candidates":candidates,"raw_boxes":raw,"trace":list(self.trace),
                "input_image_shape":list(original.shape),"runner_calls":self.calls,
                "loader_color":"BGR","transport_color":"RGB"}
        from .v31_coordinates import verify
        passed,reason=verify(response,image.width,image.height)
        if not passed:raise ContractError("Actual layout coordinate evidence: "+reason)
        response["coordinate_evidence"]=reason
        return response
    def close(self):self.predictor.close()
