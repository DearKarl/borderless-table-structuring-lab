"""Portable asset and capacity evidence checks at preflight and actual load boundaries."""
import copy
import importlib.metadata
import sys
from pathlib import Path
from .core import ContractError, canonical, closed, digest, read, sha, verify_tree

METADATA_PARENT={"PROCESSOR.json":"7c21e42f57f3ae5f425c7f8b7ddfbc6ed15374d7317316aba3c38e896087678e",
                 "TELE_LOAD.json":"710c2f46e0a118dc3b0d0b9e121eb94a904bd50d025c390d9ad3ec4fb9b42994"}
PATH_FIELDS={
 "PROCESSOR.json":{("tokenizer_init_kwargs","vocab_file"):"/assets/tele_model/vocab.json",
                   ("tokenizer_init_kwargs","merges_file"):"/assets/tele_model/merges.txt",
                   ("tokenizer_init_kwargs","name_or_path"):"/assets/tele_model"},
 "TELE_LOAD.json":{("config","_name_or_path"):"/assets/tele_model"},
}

def normalize_metadata(name,value):
    result=copy.deepcopy(value);changes=[]
    for keys,target in PATH_FIELDS[name].items():
        node=result
        for k in keys[:-1]:node=node[k]
        original=node[keys[-1]]
        if not isinstance(original,str) or not original.startswith("/"):
            raise ContractError("Unexpected original metadata path")
        node[keys[-1]]=target
        changes.append({"file":name,"field":list(keys),"before":original,"after":target})
    return result,changes

def special_tokens(model):
    path=Path(model)/"tokenizer_config.json";config=read(path);tokens=set()
    def add(value):
        if isinstance(value,str):tokens.add(value)
        elif isinstance(value,dict) and isinstance(value.get("content"),str):tokens.add(value["content"])
    for key,value in config.items():
        if key in ("bos_token","eos_token","pad_token","unk_token","mask_token"):add(value)
    for v in config.get("additional_special_tokens",[]):add(v)
    for v in config.get("added_tokens_decoder",{}).values():
        if isinstance(v,dict) and v.get("special") is True:add(v)
    if not tokens:raise ContractError("Frozen tokenizer has no explicit special tokens")
    return {"tokens":sorted(tokens),"tokenizer_config_sha256":sha(path)}

CHARDET_LOCK_SHA="ee37fb10b14d4b2d3e204796f4a607f46708205376ea719ac180fef55083348c"

def chardet_lock():
    path=Path(__file__).parent/"assets/chardet_capacity.json"
    if sha(path)!=CHARDET_LOCK_SHA:raise ContractError("Chardet classification lock changed")
    return read(path)

def chardet_deployment(runtime):
    """Schema-2 deployment identity, independent of routing and cached capacity."""
    if runtime.get("schema")!=2:return False,set()
    lock=chardet_lock();members=lock["members"]
    aux=runtime.get("assets",{}).get("auxiliary_models",{}).get("files",{})
    roles=set();required=bool(set(members)&set(aux))
    for role,env in runtime.get("environments",{}).items():
        prefix=env.get("site_packages","").rstrip("/")+"/"
        files=env.get("image_files",{})
        if role=="paddle" or any(prefix+n in files for n in members):
            required=True;roles.add(role)
            if any(files.get(prefix+n)!=h for n,h in members.items()):
                raise ContractError("Chardet image identity incomplete or changed")
            if "auxiliary_models" not in env.get("asset_roles",[]):
                raise ContractError("Chardet load boundary omits auxiliary_models")
    if required and any(aux.get(n)!=h for n,h in members.items()):
        raise ContractError("Chardet auxiliary identity incomplete or changed")
    return required,roles

def capacity(runtime, roots):
    ledger=runtime.get("parameter_ledger")
    required={"tele","fasttext","onnx"}|set(runtime.get("v31",{}).get("components",[]))
    has_chardet,_=chardet_deployment(runtime)
    if has_chardet:required.add("chardet")
    if not isinstance(ledger,list) or {r.get("component") for r in ledger}!=required or len(ledger)!=len(required):
        raise ContractError("Parameter ledger must cover every deployed model and auxiliary")
    total=0
    for row in ledger:
        for key in ("stored_elements","unique_trainable","buffers"):
            if type(row.get(key)) is not int or row[key]<0:raise ContractError("Exact parameter categories required")
        if row["unique_trainable"]>row["stored_elements"]:raise ContractError("Unique count exceeds stored count")
        evidence=row["evidence"];path=closed(roots[evidence["asset"]],evidence["path"])
        if sha(path)!=digest(evidence["sha256"]):raise ContractError("Parameter evidence changed")
        record=read(path)
        keys=("component","stored_elements","unique_trainable","buffers","model_files_sha256")
        if any(record.get(k)!=row.get(k) for k in keys):
            raise ContractError("Parameter ledger differs from bound measured evidence")
        if record.get("method") not in ("loaded_unique_parameters","static_graph_parameters","exact_auxiliary_inventory"):
            raise ContractError("Unsupported parameter evidence method")
        if not record.get("measurement_source_sha256"):raise ContractError("Parameter measurement source absent")
        digest(record["measurement_source_sha256"])
        if row["component"]=="chardet":
            lock=chardet_lock()
            if (row.get("model_asset")!="auxiliary_models" or evidence.get("asset")!="parameter_evidence"
                or record.get("method")!=lock["method"] or record.get("classification_lock_sha256")!=CHARDET_LOCK_SHA
                or record.get("relevant_members")!=lock["members"]
                or any(type(record.get(k)) is not int or record[k]!=v for k,v in lock["counts"].items())):
                raise ContractError("Chardet exact classification evidence differs")
        if row["model_files_sha256"]!=canonical(runtime["assets"][row["model_asset"]]["files"]):
            raise ContractError("Parameter evidence not bound to deployed model bytes")
        total+=row["unique_trainable"]
    if total>4_000_000_000:raise ContractError("Combined unique trainable parameter cap exceeded")
    return total

def loaded_count(runtime,component,parameters,buffers=()):
    seen=set();unique=0;buffer_count=0
    for p in parameters:
        # Torch/Paddle parameters are shared by object; data_ptr where supported catches shared storage.
        identity=(str(p.device),p.data_ptr()) if hasattr(p,"data_ptr") and hasattr(p,"device") else id(p)
        if identity not in seen:unique+=int(p.numel());seen.add(identity)
    for b in buffers:buffer_count+=int(b.numel())
    row=next(r for r in runtime["parameter_ledger"] if r["component"]==component)
    if unique!=row["unique_trainable"] or buffer_count!=row["buffers"]:
        raise ContractError("Actual loaded parameter/buffer count differs: "+component)
    return {"unique_trainable":unique,"buffers":buffer_count}

def load_boundary(runtime,role):
    """Image-local Python/source dependencies and RO asset bytes, immediately before model load."""
    env=runtime["environments"][role]
    if str(Path(sys.executable).resolve())!=env["resolved_python"]:
        raise ContractError("Container Python resolves outside frozen image environment")
    for name,expected in env["image_files"].items():
        if not name.startswith("/") or sha(name)!=expected:
            raise ContractError("Image environment file mismatch: "+name)
    if {k:importlib.metadata.version(k) for k in env["packages"]}!=env["packages"]:
        raise ContractError("Image environment package mismatch")
    roots={name:Path("/assets")/name for name in runtime["assets"]}
    has_chardet,_=chardet_deployment(runtime)
    if has_chardet:
        verify_tree(roots["auxiliary_models"],runtime["assets"]["auxiliary_models"]["files"])
    for name in env["asset_roles"]:
        verify_tree(roots[name],runtime["assets"][name]["files"])
    capacity(runtime,roots)
    return {"environment_sha256":canonical(env),"assets":{n:canonical(runtime["assets"][n]["files"]) for n in env["asset_roles"]}}


def validate_v31(runtime,root,profile,budget,mode,verify=True):
    from .v31_rules import PROFILE_SHA
    from .v31_protocol import formula_config
    if mode not in ("off","pass-through","on"):raise ContractError("Unknown V31 routing mode")
    v=runtime.get("v31",{})
    from .asset_binder import ASSET_ROLES
    if not set(runtime["assets"])<=ASSET_ROLES:raise ContractError("Unknown inference asset role")
    if v.get("profile_sha256")!=PROFILE_SHA:raise ContractError("Frozen V31 profile changed")
    expected={"native":set(),"v31-layout":{"layout"},"v31-formula":{"formula"},"v31-both":{"layout","formula"}}[profile]
    components=v.get("components")
    if not isinstance(components,list) or set(components)!=expected or len(components)!=len(expected):
        raise ContractError("Declared profile/provider set differs")
    if runtime["native"]["expected_load_metadata"]!=read(Path(__file__).parent/"assets/vendor/tele_metadata.json"):
        raise ContractError("Frozen processor/model behavior metadata changed")
    environments=runtime.get("environments",{})
    if set(environments)!=({"native","paddle"} if components else {"native"}):
        raise ContractError("Independent native/Paddle image environment locks required")
    _,chardet_roles=chardet_deployment(runtime)
    for role,env in environments.items():
        from .image_identity import parse_reference
        parse_reference(env["image"], allow_local=runtime.get("schema")==2)
        if (not env["python"].startswith("/") or not env["resolved_python"].startswith("/") or not env["site_packages"].startswith("/")
            or env["resolved_python"] not in env["image_files"] or not env["packages"]):
            raise ContractError("Image-local executable and package file lock required")
        if any(env[key].startswith(("/assets/","/output/","/framework/","/inputs/")) for key in ("python","resolved_python","site_packages")):
            raise ContractError("Relocated mounted venv is not an image-local environment")
        for name,h in env["image_files"].items():
            if not name.startswith("/") or "/../" in name:raise ContractError("Image file path invalid")
            digest(h)
        if not env["asset_roles"] or not set(env["asset_roles"])<=set(runtime["assets"]):
            raise ContractError("Environment load asset set differs")
        needed={"native_code","tele_source","tele_model","environment","parameter_evidence"} if role=="native" else {"native_code","environment","parameter_evidence"}|{v[c]["model_asset"] for c in components}
        if role in chardet_roles:needed.add("auxiliary_models")
        if not needed<=set(env["asset_roles"]):raise ContractError("Load boundary omits a required model/source/evidence root")
    if runtime["python"]!=environments["native"]["python"] or runtime["image"]!=environments["native"]["image"]:
        raise ContractError("Native image/environment mismatch")
    if components and environments["native"]["image"]==environments["paddle"]["image"]:
        raise ContractError("Separate frozen environment images required")
    if budget["cpus"]>8 or budget["ram_gib"]+budget["swap_gib"]>48 or budget["native_calls_per_page"]>256:
        raise ContractError("V31 native resource envelope exceeded")
    gpu_count=2 if components and mode=="on" else 1
    if len(runtime["gpu_uuids"])!=gpu_count:raise ContractError("Wrong effective V31 GPU count")
    roots={k:closed(root,a["path"]) for k,a in runtime["assets"].items()}
    parent=read(Path(__file__).parent/"assets/vendor/PARENT_ASSET_LOCK.json")
    for role,files in parent["model_files"].items():
        if role not in runtime["assets"]:continue
        if any(runtime["assets"][role]["files"].get(n)!=h for n,h in files.items()):
            raise ContractError("Original frozen model/source asset differs")
    for role,env in environments.items():
        if any(env["packages"].get(n)!=v for n,v in parent["packages"][role].items()):
            raise ContractError("Original environment package versions differ")
        if any(env["image_files"].get(env["site_packages"]+"/"+n)!=h for n,h in parent["image_auxiliaries"][role].items()):
            raise ContractError("Original auxiliary resource missing from image lock")
    layout=read(Path(__file__).parent/"assets/vendor/layout_metadata.json")
    for component in components:
        item=v[component];asset=runtime["assets"][item["model_asset"]]
        if item["model_sha256"]!=canonical(asset["files"]):raise ContractError("Expert model identity differs")
        if item["source_sha256"]!=canonical(environments["paddle"]["image_files"]):
            raise ContractError("Expert source/environment binding differs")
        expected_config=formula_config() if component=="formula" else layout["parameters"]
        if component=="layout":
            if item.get("parameters")!=expected_config or item.get("revision")!=layout["model_revision"]:
                raise ContractError("Original layout parameters/revision changed")
            expected_files={f["name"]:f["sha256"] for f in layout["model_files"]}
            if asset["files"]!=expected_files:raise ContractError("Frozen layout model changed")
            for name,version in layout["expected_versions"].items():
                if environments["paddle"]["packages"].get(name)!=version:raise ContractError("Paddle version changed")
            for f in layout["code_files"]:
                name=environments["paddle"]["site_packages"]+"/"+f["relative_path"]
                if environments["paddle"]["image_files"].get(name)!=f["sha256"]:
                    raise ContractError("Original Paddle layout implementation changed")
        if item["config_sha256"]!=canonical(expected_config):raise ContractError("Expert generation/config changed")
    for component,asset in (("tele","tele_model"),("formula","formula_model")):
        if asset not in roots:continue
        actual=special_tokens(roots[asset])
        if actual!=v["tokenizer_lock"][component] or actual["tokens"]!=v["special_tokens"][component]:
            raise ContractError("Special token table differs from frozen tokenizer")
    capacity(runtime,roots)
    return True
