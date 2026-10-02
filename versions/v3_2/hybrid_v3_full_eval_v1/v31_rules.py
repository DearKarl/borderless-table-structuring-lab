"""Frozen V31-CONSERVATIVE-r1 decisions; no semantic quality estimates."""
import math
import re
from .core import ContractError, canonical

PROFILE = {
    "name": "V31-CONSERVATIVE-r1", "admission_score": .90,
    "table_native_coverage": .80, "table_union_area_ratio": 1.25,
    "table_expansion_pixels": 2, "max_table_expansions": 2,
    "max_additions": 2, "max_candidate_overlap": .01, "anchor_horizontal_overlap": .5,
    "max_formula_requests": 32, "repetition_period": 16, "repetition_count": 4,
    "add_labels": {"text":"text", "display_formula":"equation", "table":"table"},
    "formula_type": "equation", "protected_commands":
        ["verb","verbatim","lstinline","lstlisting","mintinline","minted","catcode",
         "csname","endcsname","def","edef","gdef","xdef","newcommand","renewcommand"],
    "formula_query": "Formula Recognition:", "min_pixels":112896,
    "max_pixels":1003520, "max_new_tokens":4096, "quality_claim":False,
}
PROFILE_SHA = canonical(PROFILE)

def escaped(text, at):
    n=0
    while at>0 and text[at-1]=="\\":
        n+=1;at-=1
    return bool(n%2)

def structure(text, special_tokens):
    """Finite escape-aware scanner. Unknown contexts always abstain."""
    if not isinstance(text,str):
        raise ContractError("Formula completion must be a string")
    if any(c=="%" and not escaped(text,i) for i,c in enumerate(text)):
        return {"state":"unknown","reason":"comment"}
    commands=re.findall(r"(?<!\\)(?:\\\\)*\\([A-Za-z]+)",text)
    if any(c in PROFILE["protected_commands"] for c in commands) or chr(96)*3 in text:
        return {"state":"unknown","reason":"protected_lexical_context"}
    if any(re.search(r"\\begin\s*\{"+name+r"\}",text) for name in ("verbatim","lstlisting","minted")):
        return {"state":"unknown","reason":"protected_environment"}
    if not text.strip():return {"state":"bad","reason":"empty"}
    if any(token and token in text for token in special_tokens):
        return {"state":"bad","reason":"special_token_leak"}
    depth=0; env=[]; i=0
    while i<len(text):
        if escaped(text,i):i+=1;continue
        if text[i]=="\\":
            m=re.match(r"\\(begin|end)\s*\{([A-Za-z][A-Za-z0-9*]*)\}",text[i:])
            if m:
                kind,name=m.groups()
                if kind=="begin":env.append(name)
                elif not env or env.pop()!=name:
                    return {"state":"bad","reason":"environment_mismatch"}
                i+=m.end();continue
            if re.match(r"\\(?:begin|end)(?![A-Za-z])",text[i:]):
                return {"state":"unknown","reason":"unparsed_environment"}
        if text[i]=="{":depth+=1
        elif text[i]=="}":
            depth-=1
            if depth<0:return {"state":"bad","reason":"unmatched_brace"}
        i+=1
    if depth:return {"state":"bad","reason":"unmatched_brace"}
    if env:return {"state":"bad","reason":"unclosed_environment"}
    return {"state":"valid","reason":"limited_structure_pass"}

def repetition(termination):
    ids=termination.get("token_ids"); eos=termination.get("eos_ids")
    if not isinstance(ids,list) or any(type(x) is not int for x in ids):
        return None
    if not isinstance(eos,list) or any(type(x) is not int for x in eos):
        return None
    if ids and ids[-1] in eos:ids=ids[:-1]
    return any(ids[i:i+16]*4==ids[i:i+64] for i in range(max(0,len(ids)-63)))

def formula_failure(text, termination, special_tokens):
    s=structure(text,special_tokens)
    if s["state"]=="unknown":return s
    if s["state"]=="bad":return s
    if not isinstance(termination,dict) or termination.get("stop") not in ("eos","length","other"):
        raise ContractError("Missing or unknown real completion evidence")
    if termination["stop"]!="eos":return {"state":"bad","reason":"non_eos"}
    repeated=repetition(termination)
    if repeated is True:return {"state":"bad","reason":"16_tokens_repeated_4_times"}
    if repeated is None:return {"state":"unknown","reason":"real_token_ids_missing"}
    return s

def i0(raw):
    """Remove exactly one complete display shell; keep every other codepoint."""
    if structure(raw,[])["state"]=="unknown":return None
    found=[];i=0
    while i<len(raw):
        if not escaped(raw,i):
            if raw.startswith("$$",i):found.append((i,"$$"));i+=2;continue
            if raw[i]=="$":found.append((i,"$"))
            if raw[i:i+2] in ("\\[","\\]","\\(","\\)"):
                found.append((i,raw[i:i+2]));i+=2;continue
        i+=1
    if not found:return raw
    if len(found)!=2:return None
    (a,left),(b,right)=found
    if ((left,right) not in (("$$","$$"),("\\[","\\]")) or
        a!=len(raw)-len(raw.lstrip()) or b+len(right)!=len(raw.rstrip())):return None
    body=raw[a+len(left):b]
    return raw[:a]+body+raw[b+len(right):] if body.strip() else None

def rect(value):
    if not isinstance(value,(tuple,list)) or len(value)!=4:
        raise ContractError("Axis-aligned bbox must contain four coordinates")
    r=tuple(float(v) for v in value)
    if not all(math.isfinite(v) for v in r) or r[0]>=r[2] or r[1]>=r[3]:
        raise ContractError("Invalid bbox")
    return r

def area(r):return (r[2]-r[0])*(r[3]-r[1])
def intersect(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def union(a,b):return (min(a[0],b[0]),min(a[1],b[1]),max(a[2],b[2]),max(a[3],b[3]))
def horizontal(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))/min(a[2]-a[0],b[2]-b[0])
def inside(a,w,h):return 0<=a[0]<a[2]<=w and 0<=a[1]<a[3]<=h

def proposal(candidate, native, added, width, height, page_sha):
    """Native records use rendered-pixel bboxes in original reading order."""
    c=rect(candidate["bbox"]);index=candidate["index"]
    if type(index) is not int or index<0:raise ContractError("Invalid detector index")
    score=candidate["score"]
    if type(score) not in (int,float) or not math.isfinite(score):raise ContractError("Invalid score")
    def reject(reason):return {"accepted":False,"reason":reason,"detector_index":index}
    if candidate["label"] not in PROFILE["add_labels"]:return reject("unsupported_label")
    if score<.9:return reject("low_score")
    if candidate.get("angle",0)!=0 or not inside(c,width,height):return reject("rotated_or_outside")
    targets=[n for n in native if n["type"]=="table" and intersect(c,n["bbox"])/area(n["bbox"])>=.8]
    if candidate["label"]=="table" and targets:
        if len(targets)!=1:return reject("ambiguous_table")
        target=targets[0];a=target["bbox"];u=union(a,c)
        if target["angle"]!=0:return reject("rotated_table")
        if any(n["slot_id"]!=target["slot_id"] and (intersect(c,n["bbox"])>0 or intersect(u,n["bbox"])>0) for n in native+added):
            return reject("table_conflict")
        if not inside(u,width,height) or area(u)/area(a)>1.25:return reject("union_area_or_bounds")
        if max(a[0]-u[0],a[1]-u[1],u[2]-a[2],u[3]-a[3])<2:return reject("noop")
        return {"accepted":True,"kind":"expand","slot_id":target["slot_id"],"type":"table",
                "bbox":list(u),"detector_index":index}
    for n in native+added:
        overlap=intersect(c,n["bbox"])
        if overlap/area(c)>.01:return reject("candidate_overlap")
        if overlap>0 and (n["type"] in ("table","image","chart","char") or "caption" in n["type"]):
            return reject("protected_overlap")
    anchors=[]
    for prev,nxt in zip(native,native[1:]):
        a,b=prev["bbox"],nxt["bbox"]
        if (prev["angle"]==0 and nxt["angle"]==0 and a[3]<=c[1]<c[3]<=b[1]
            and horizontal(c,a)>=.5 and horizontal(c,b)>=.5):
            anchors.append([prev["slot_id"],nxt["slot_id"]])
    if len(anchors)!=1:return reject("nonunique_or_missing_anchors")
    return {"accepted":True,"kind":"add","slot_id":"new:"+canonical(
            [page_sha,index,PROFILE_SHA]),"type":PROFILE["add_labels"][candidate["label"]],
            "bbox":list(c),"anchors":anchors[0],"detector_index":index}

