"""Evaluation-only local Q/P. Never imported by deployment input adapters."""
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import tempfile
import unicodedata
import hashlib
import threading
from contextlib import contextmanager

class InvalidAnnotation(ValueError): pass
class InvalidMeasurement(RuntimeError): pass
class PredictionSyntaxError(ValueError): pass

def prediction_from_native_blocks(blocks, *, page_id, input_sha256, category_map):
    """Convert final native blocks using a separately frozen type map, never GT.

    Map each native type to text/table/formula/ignore. Unknown types fail closed;
    table content remains atomic, so its internal formulas are not double counted.
    Native postprocessing must already have converted tables to HTML.
    """
    out=dict(page_id=page_id,input_sha256=input_sha256,
             frame="canonical_upright_original_page",text=[],table=[],formula=[])
    for index,block in enumerate(blocks):
        category=category_map.get(block["type"])
        if category not in ("text","table","formula","ignore"):
            raise InvalidMeasurement("Unmapped final native block type")
        content=block.get("content") or ""
        if category=="text":out["text"].append(content)
        elif category!="ignore":
            polygon=block.get("bbox")
            out[category].append(dict(id=index,bbox=normalized_box(polygon),
                                     original_polygon=polygon,content=content))
    return out

def canonical_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text.replace("\r\n","\n").replace("\r","\n"))).strip()

def text_quality(reference, prediction):
    a,b=canonical_text(reference),canonical_text(prediction)
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return 1-min(1,previous[-1]/max(len(a),len(b),1))

def normalized_box(coords, page_size=None):
    """Accept xyxy or flat polygon in the already upright displayed frame."""
    try:
        vals=[float(v) for v in coords]
        if len(vals)<4 or len(vals)%2 or not all(math.isfinite(v) for v in vals):
            return None
        if page_size is not None:
            if len(page_size)!=2 or min(page_size)<=0: return None
            vals=[v/page_size[i%2] for i,v in enumerate(vals)]
        if any(v < -1e-6 or v > 1+1e-6 for v in vals): return None
        vals=[min(1,max(0,v)) for v in vals]
        box=tuple(vals) if len(vals)==4 else (min(vals[::2]), min(vals[1::2]), max(vals[::2]), max(vals[1::2]))
        return box if box[2]>box[0] and box[3]>box[1] else None
    except (TypeError, ValueError, OverflowError): return None

def pdf_polygon_to_upright(points, cropbox, rotation):
    """Map bottom-left PDF coordinates to normalized displayed page coordinates."""
    x0,y0,x1,y1=cropbox;w,h=x1-x0,y1-y0
    if min(w,h)<=0 or rotation not in (0,90,180,270):raise ValueError("Invalid PDF frame")
    mapped=[]
    for x,y in points:
        x,y=x-x0,h-(y-y0)
        if rotation==90:x,y=h-y,x;dims=(h,w)
        elif rotation==180:x,y=w-x,h-y;dims=(w,h)
        elif rotation==270:x,y=y,w-x;dims=(h,w)
        else:dims=(w,h)
        mapped.extend((x/dims[0],y/dims[1]))
    return mapped

def iou(a,b):
    if a is None or b is None:return 0.
    area=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return area/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)

def geometric_assignment(reference, prediction):
    """geom_iou_assignment_v1: cardinality first, summed quantized IoU second."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    refs=sorted(reference,key=lambda x:str(x["id"]))
    preds=sorted(prediction,key=lambda x:str(x["id"]))
    if len({str(x["id"]) for x in refs})!=len(refs) or len({str(x["id"]) for x in preds})!=len(preds):
        raise ValueError("Instance IDs must be unique within category")
    n,m=len(refs),len(preds)
    if not n or not m:return [],[x["id"] for x in refs],[x["id"] for x in preds]
    k=min(n,m);weights=np.zeros((n+m,n+m),dtype=np.int64)
    weights[:n,:m]=-1
    overlaps={}
    for i,a in enumerate(refs):
        for j,b in enumerate(preds):
            overlap=iou(normalized_box(a.get("bbox")),normalized_box(b.get("bbox")))
            if overlap>=.5:
                weights[i,j]=k*1000000+1+round(overlap*1000000)
                overlaps[i,j]=overlap
    row,col=linear_sum_assignment(weights,maximize=True)
    pairs=[(refs[i],preds[j],overlaps[i,j]) for i,j in zip(row,col) if (i,j) in overlaps]
    ri={a["id"] for a,b,o in pairs};pi={b["id"] for a,b,o in pairs}
    return pairs,[x["id"] for x in refs if x["id"] not in ri],[x["id"] for x in preds if x["id"] not in pi]

def canonical_table(value):
    """Use the original TEDS HTML parser semantics, retaining th/thead and content."""
    from lxml import html, etree
    root=html.fromstring(value,parser=html.HTMLParser(remove_comments=True,encoding="utf-8"))
    tables=([root] if root.tag=="table" else [])+root.xpath(".//table")
    if len(tables)!=1 or not tables[0].xpath(".//tr"):
        raise ValueError("One usable table required")
    for cell in tables[0].xpath(".//td"):
        for attr in ("rowspan","colspan"): int(cell.attrib.get(attr,"1"))
    return "<html><body>"+etree.tostring(tables[0],encoding="unicode",method="html",with_tail=False)+"</body></html>"

def checked_teds(teds):
    if getattr(teds,"structure_only",None) is not False:raise ValueError("Full TEDS required")
    def score(gt,pred):
        try: g=canonical_table(gt)
        except Exception as e:raise InvalidAnnotation("Invalid reference HTML") from e
        try: p=canonical_table(pred)
        except Exception: return 0.
        try: value=teds.evaluate(p,g)
        except Exception as e:raise InvalidMeasurement("TEDS execution failed") from e
        return valid_score(value)
    return score

def valid_score(value):
    if not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise InvalidMeasurement("Metric value outside [0,1]")
    return float(value)

_CDM_LOCK = threading.Lock()

class StrictCDM:
    """Observe real renderer globals and reject hidden tool/tokenizer failures."""
    def __init__(self,kernel,temporary_root,prediction_syntax_validator=None,*,identity=None):
        self.kernel=kernel;self.root=Path(temporary_root);self.syntax_validator=prediction_syntax_validator
        self.identity=dict(identity or {})
        required={"metric_source_sha256","metric_runtime_config_sha256"}
        if set(self.identity)!=required or any(not isinstance(v,str) or not re.fullmatch(r"[0-9a-f]{64}",v) for v in self.identity.values()):
            raise ValueError("Frozen source and runtime/config hashes required")
        self.last_audit=[]

    @contextmanager
    def observe_renderer(self,role):
        renderer=self.kernel.latex2bbox_color
        namespace=getattr(renderer,"__globals__",None)
        if namespace is None or not callable(namespace.get("run_cmd")) or not callable(namespace.get("_tokenize_latex_with_timeout")):
            raise InvalidMeasurement("Renderer observation boundaries unavailable")
        original_cmd=namespace["run_cmd"];original_token=namespace["_tokenize_latex_with_timeout"]
        audit={"role":role,"commands":[],"tokenization":[],**self.identity}
        self.last_audit.append(audit)
        def command(*args,**kwargs):
            rec={"command_sha256":hashlib.sha256(str(args[0] if args else kwargs).encode()).hexdigest()}
            audit["commands"].append(rec)
            try:
                result=original_cmd(*args,**kwargs);rec.update(returncode=result,success=type(result) is int and result==0)
                if not rec["success"]: raise InvalidMeasurement(role+" renderer command failed")
                return result
            except BaseException as e:rec.update(success=False,error=repr(e));raise
        def tokenize(*args,**kwargs):
            rec={};audit["tokenization"].append(rec)
            try:
                result=original_token(*args,**kwargs)
                rec["success"]=isinstance(result,tuple) and len(result)==2 and bool(result[0] and result[1])
                rec["fallback_detected"]=not rec["success"]
                if not rec["success"]: raise InvalidMeasurement(role+" tokenizer fallback")
                return result
            except BaseException as e:rec.update(success=False,error=repr(e),fallback_detected=True);raise
        namespace["run_cmd"]=command;namespace["_tokenize_latex_with_timeout"]=tokenize
        try:
            yield audit
        finally:
            namespace["run_cmd"]=original_cmd;namespace["_tokenize_latex_with_timeout"]=original_token

    def render(self,latex,role,base):
        box=base/role;tmp=base/("temp-"+role)
        for d in (box/"bbox",box/"vis",tmp):d.mkdir(parents=True,exist_ok=True)
        with self.observe_renderer(role) as audit:
            audit.update(content_sha256=hashlib.sha256(latex.encode()).hexdigest(),returned=False)
            try:
                self.kernel.latex2bbox_color((latex,"sample_0",str(box),str(tmp),self.kernel._get_total_color_list()))
                audit["returned"]=True
            except BaseException as e: audit["error"]=repr(e);raise
        if not audit["commands"] or not audit["tokenization"] or any(not x.get("success") for key in ("commands","tokenization") for x in audit[key]):
            raise InvalidMeasurement(role+" command/tokenization failed or was unobserved")
        path=box/"bbox/sample_0.jsonl"
        if not path.exists():raise InvalidMeasurement(role+" missing bbox artifacts")
        tokens=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
        if not tokens or not any(x.get("bbox") for x in tokens):raise InvalidMeasurement(role+" missing visible tokens")
        from PIL import Image
        with Image.open(box/"vis/sample_0_base.png") as image:image.verify()
        return str(box)

    def validate_reference_render(self,content):
        if not _CDM_LOCK.acquire(blocking=False):raise InvalidMeasurement("Concurrent CDM observation")
        self.last_audit=[]
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.root,prefix="strict-cdm-gt-") as work:
                self.render(content,"gt",Path(work))
            return dict(status="success",content_sha256=hashlib.sha256(content.encode()).hexdigest(),**self.identity)
        except InvalidMeasurement:raise
        except Exception as e:raise InvalidMeasurement("Reference render failed") from e
        finally:_CDM_LOCK.release()

    def __call__(self,gt,pred):
        if not _CDM_LOCK.acquire(blocking=False):raise InvalidMeasurement("Concurrent CDM observation")
        self.last_audit=[]
        try:
            self.root.mkdir(parents=True,exist_ok=True)
            with tempfile.TemporaryDirectory(dir=self.root,prefix="strict-cdm-") as work:
                base=Path(work);gt_dir=self.render(gt,"gt",base)
                if self.syntax_validator:
                    try:self.syntax_validator(pred)
                    except PredictionSyntaxError:return 0.
                pred_dir=self.render(pred,"pred",base)
                result=self.kernel.process_single_image(("sample_0",gt_dir,pred_dir,str(base/"matches"),5,2,30,500),save_vis=False)
                if result is None:raise InvalidMeasurement("CDM kernel returned no result")
                metrics=result[1]
                if metrics.get("gt_tokens",0)<=0 or metrics.get("pred_tokens",0)<=0:raise InvalidMeasurement("CDM missing token evidence")
                return valid_score(metrics["F1_score"])
        except InvalidMeasurement:raise
        except Exception as e:raise InvalidMeasurement("CDM execution/render failed") from e
        finally:_CDM_LOCK.release()

def validate_reference(ref, metric_identity=None):
    required=("page_id","input_sha256","source_family","split","annotation_sha256",
              "provenance","license","reviewer","independently_reviewed")
    if any(k not in ref for k in required):raise InvalidAnnotation("Incomplete annotation provenance")
    if ref.get("frame")!="canonical_upright_original_page" or not ref["independently_reviewed"]:
        raise InvalidAnnotation("Unchecked annotation or frame")
    for kind in ("text","table","formula"):
        category=ref[kind];state=category["state"];items=category["items"]
        if state not in ("present","verified_absent","unlabeled"):raise InvalidAnnotation("Unknown state")
        if state=="unlabeled":continue
        if (state=="present")!=bool(items):raise InvalidAnnotation("Category state/content mismatch")
        if kind=="text" and (any(not isinstance(x,str) for x in items) or (state=="present" and not canonical_text(" ".join(items)))):
            raise InvalidAnnotation("Present text needs recognized nonempty strings")
        if kind!="text":
            if len({str(x["id"]) for x in items})!=len(items):raise InvalidAnnotation("Duplicate GT ID")
            for item in items:
                if normalized_box(item.get("bbox")) is None:raise InvalidAnnotation("Invalid GT geometry")
                if not isinstance(item.get("content"),str) or not item["content"].strip():
                    raise InvalidAnnotation("Missing reference content")
                if kind=="table":
                    try:canonical_table(item["content"])
                    except Exception as e:raise InvalidAnnotation("Invalid reference HTML") from e
                if kind=="formula":
                    receipt=item.get("render_validation",{})
                    if not metric_identity or receipt.get("status")!="success" or receipt.get("content_sha256")!=hashlib.sha256(item["content"].encode()).hexdigest() or any(receipt.get(k)!=v for k,v in metric_identity.items()):
                        raise InvalidAnnotation("Reference formula render evidence does not match content/metric/runtime")

def score_page(ref,pred,*,table_metric,formula_metric,status="success"):
    base={"page_id":ref.get("page_id"),"source_family":ref.get("source_family"),
          "status":status,"Q":None,"components":None,"matching":{}}
    try:validate_reference(ref,getattr(formula_metric,"identity",None))
    except Exception as e:return dict(base,status="invalid_annotation",error=str(e))
    if any(ref[k]["state"]=="unlabeled" for k in ("text","table","formula")):
        return dict(base,status="target_ineligible")
    if status in ("not_started","deadline_incomplete"):return base
    if status in ("timeout","empty_nonempty_page","fatal_malformed"):
        return dict(base,status="terminal_model_failure",Q=0.,components=dict(text=0.,table=0.,formula=0.))
    if status not in ("success","partial"):return dict(base,status="invalid_measurement",error="Unknown execution status")
    try:
        if pred.get("page_id")!=ref["page_id"] or pred.get("input_sha256")!=ref["input_sha256"]:
            raise InvalidMeasurement("Prediction identity mismatch")
        if pred.get("frame")!=ref["frame"]:raise InvalidMeasurement("Prediction frame mismatch")
        has_content=bool(canonical_text(" ".join(pred["text"]))) or any(isinstance(x.get("content"),str) and canonical_text(x["content"]) for kind in ("table","formula") for x in pred[kind])
        if any(ref[k]["items"] for k in ("text","table","formula")) and not has_content:
            return dict(base,status="terminal_model_failure",Q=0.,components=dict(text=0.,table=0.,formula=0.))
        # Each side supplies its own reading order; no reference-based filtering.
        components={"text":text_quality(" ".join(ref["text"]["items"])," ".join(pred["text"]))}
        for kind,metric in (("table",table_metric),("formula",formula_metric)):
            gt=ref[kind]["items"];pr=pred[kind]
            pairs,unmatched_gt,unmatched_pred=geometric_assignment(gt,pr)
            scored=[]
            for a,b,overlap in pairs:
                value=0. if kind=="formula" and not canonical_text(b["content"]) else valid_score(metric(a["content"],b["content"]))
                scored.append({"reference_id":a["id"],"prediction_id":b["id"],"iou":overlap,"quality":value})
            components[kind]=sum(x["quality"] for x in scored)/max(len(gt),len(pr)) if gt or pr else 1.
            base["matching"][kind]={"pairs":scored,"unmatched_reference":unmatched_gt,"unmatched_prediction":unmatched_pred,
                "reference_count":len(gt),"prediction_count":len(pr)}
        return dict(base,Q=sum(components.values())/3,components=components)
    except Exception as e:return dict(base,status="invalid_measurement",error=str(e))

def source_equal_summary(results, expected_page_ids):
    if len(results)!=len(expected_page_ids) or {r["page_id"] for r in results}!=set(expected_page_ids):
        raise InvalidMeasurement("Incomplete/duplicate page denominator")
    if any(r["Q"] is None for r in results):raise InvalidMeasurement("Bank contains invalid/incomplete targets")
    families=defaultdict(list)
    for r in results:families[r["source_family"]].append(r)
    if not families:raise InvalidMeasurement("Empty bank")
    def average(key):
        return 100*sum(sum(r["Q"] if key=="Q" else r["components"][key] for r in pages)/len(pages)
                       for pages in families.values())/len(families)
    return {"name":"local_P","P":average("Q"),"components":{k:average(k) for k in ("text","table","formula")},
            "families":len(families),"pages":len(results)}
