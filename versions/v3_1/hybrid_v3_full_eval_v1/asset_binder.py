"""Offline preparation and binding; no installs, model imports, Docker or network."""
import copy
import shutil
from pathlib import Path
from .core import ContractError, canonical, closed, digest, identifier, new_dir, read, sha, write, verify_tree
from .asset_binding import METADATA_PARENT, special_tokens, capacity
from .v31_rules import PROFILE, PROFILE_SHA
from .v31_protocol import formula_config
VENDOR=Path(__file__).parent/"assets/vendor"
ASSET_ROLES={"native_code","tele_source","tele_model","environment","parameter_evidence","auxiliary_models","layout_model","formula_model"}

def source_path(base,value):
    p=Path(value)
    return p.resolve() if p.is_absolute() else closed(base,value)

def external_tele_files(root, parent):
    """Validate only the pinned inference subset of an externally supplied checkout."""
    files = dict(parent["model_files"]["tele_source"])
    for name, expected in files.items():
        if sha(closed(root, name)) != expected:
            raise ContractError("External TeleOCR source differs from pinned revision")
    return files


def prepare(configuration_path,output):
    """Build an executable binding plan from explicit model roots and prepared environment locks."""
    config_path=Path(configuration_path).resolve();cfg=read(config_path);base=config_path.parent
    out=new_dir(output);parent=read(VENDOR/"PARENT_ASSET_LOCK.json")
    from .contracts import PROFILES
    profile=cfg["profile"]
    if profile=="v32-text" or profile not in PROFILES:raise ContractError("V31 profile required")
    components=list(PROFILES[profile])
    roots={k:source_path(base,v) for k,v in cfg["asset_roots"].items()}
    if not set(roots)<=ASSET_ROLES:raise ContractError("Unknown asset role; inference assets only")
    if "tele_source" not in roots:
        raise ContractError("Explicit external tele_source required; see ENVIRONMENT_PREPARATION.md")
    native=out/"native-source";native.mkdir()
    names=list(parent["code_files"])
    if "formula" not in components:names.remove("hybrid_v2_tele_base/paddle_formula_worker.py")
    for name in names:
        dest=closed(native,name);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(closed(VENDOR/"native_code",name),dest)
    shutil.copyfile(VENDOR/"PARENT_ASSET_LOCK.json",native/"PARENT_ASSET_LOCK.json")
    roots["native_code"]=native
    mandatory={"native_code","tele_source","tele_model","environment","parameter_evidence"}
    mandatory|=({"formula_model"} if "formula" in components else set())
    mandatory|=({"layout_model"} if "layout" in components else set())
    if not mandatory<=set(roots):raise ContractError("Missing explicit model/environment/evidence roots")
    assets={}
    for role,root in roots.items():
        if not root.is_dir():raise ContractError("Missing prepared asset directory: "+role)
        if role == "tele_source":
            # Copy only the pinned inference subset, never an external checkout's metadata.
            files = external_tele_files(root, parent)
        else:
            files={p.relative_to(root).as_posix():sha(closed(root,p.relative_to(root).as_posix()))
                   for p in sorted(root.rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix!=".pyc"}
        assets[role]={"path":str(root),"files":files,"subset":True}
    env_path=source_path(base,cfg["environment_lock"]);environments=read(env_path)
    if set(environments)!=({"native","paddle"} if components else {"native"}):
        raise ContractError("Prepared separate image locks required; see ENVIRONMENT_PREPARATION.md")
    layout=read(VENDOR/"layout_metadata.json");v={"components":components}
    for component in components:
        v[component]={"model_asset":component+"_model"}
        if component=="layout":v[component].update(parameters=layout["parameters"],revision=layout["model_revision"])
    plan={"schema":1,"parent_asset_lock_sha256":sha(VENDOR/"PARENT_ASSET_LOCK.json"),
          "assets":assets,"environments":environments,
          "parameter_ledger":read(source_path(base,cfg["parameter_ledger"])),"v31":v,
          "gpu_uuids":cfg["gpu_uuids"],"lease_directory":cfg["lease_directory"],
          "idle_memory_mib":cfg["idle_memory_mib"],"validation_budget":cfg["validation_budget"],
          "validation_profile":profile,"validation_mode":cfg.get("mode","on")}
    from .gpu_backend import copy_backend
    backend=copy_backend(cfg,base,out,allow_absolute=True)
    if backend is not None:plan["gpu_backend"]=backend
    write(out/"bind-plan.json",plan)
    result={"status":"plan_created_not_runtime_verified","configuration_sha256":sha(config_path),
            "environment_lock_sha256":sha(env_path),"plan_sha256":sha(out/"bind-plan.json"),
            "installation":False,"model_loads":0,
            "next_command":"python -B -m hybrid_v3_full_eval_v1.cli bind-assets --plan bind-plan.json --output NEW_DIRECTORY"}
    write(out/"PREPARATION_PLAN.json",result);return result

def bind(plan_path,output):
    plan_path=Path(plan_path).resolve();plan=read(plan_path);base=plan_path.parent
    if plan.get("schema")!=1:raise ContractError("Schema-1 offline preparation plan required")
    if plan.get("parent_asset_lock_sha256")!=sha(VENDOR/"PARENT_ASSET_LOCK.json"):
        raise ContractError("Parent asset lock changed")
    out=new_dir(output)
    write(out/"PREPARATION_INPUT.json",{"plan_sha256":sha(plan_path),"status":"started","installation":False})
    parent=read(VENDOR/"PARENT_ASSET_LOCK.json")
    runtime={"schema":2,"native_parent_freeze_sha256":parent["original_freeze_sha256"],"assets":{}}
    from .gpu_backend import copy_backend
    backend=copy_backend(plan,base,out)
    if backend is not None:runtime["gpu_backend"]=backend
    origins={}
    for role,item in plan["assets"].items():
        identifier(role)
        if role not in ASSET_ROLES:raise ContractError("Unknown asset role; inference assets only")
        root=source_path(base,item["path"])
        if item.get("subset") is True:
            for name,h in item["files"].items():
                if sha(closed(root,name))!=digest(h):raise ContractError("Frozen subset changed")
        else:verify_tree(root,item["files"])
        target=out/"assets"/role;target.mkdir(parents=True)
        for name,h in item["files"].items():
            source=closed(root,name);dest=closed(target,name);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
            if sha(dest)!=h:raise ContractError("Asset copy differs")
        runtime["assets"][role]={"role":role,"path":"assets/"+role,"files":item["files"]}
        origins[role]={"parent_asset_lock_sha256":canonical(item["files"]),
                       "bound_asset_lock_sha256":canonical(runtime["assets"][role]["files"])}
    runtime["environments"]=plan["environments"];identities=[]
    for role,files in parent["model_files"].items():
        if role not in runtime["assets"]:continue
        for name,h in files.items():
            if runtime["assets"][role]["files"].get(name)!=h:
                raise ContractError("Frozen model/source member missing or changed")
            identities.append({"asset":role,"path":name,"sha256":h})
    for role,files in parent["image_auxiliaries"].items():
        if role not in runtime["environments"]:continue
        env=runtime["environments"][role]
        for name,h in files.items():
            path=env["site_packages"]+"/"+name
            if env["image_files"].get(path)!=h:raise ContractError("Frozen auxiliary image member missing/changed")
            identities.append({"image_role":role,"image_path":path,"sha256":h})
    runtime["native"]={"tele_packages":parent["packages"]["native"],"identity_files":identities,
        "expected_load_metadata":parent["native_runtime"]}
    runtime.update({k:plan[k] for k in ("gpu_uuids","lease_directory","idle_memory_mib","parameter_ledger")})
    runtime["image"]=plan["environments"]["native"]["image"]
    runtime["python"]=plan["environments"]["native"]["python"]
    runtime["v31"]=copy.deepcopy(plan["v31"]);v=runtime["v31"]
    v["profile_sha256"]=PROFILE_SHA;v["special_tokens"]={};token_lock={}
    for component,asset in (("tele","tele_model"),("formula","formula_model")):
        if asset in runtime["assets"]:
            data=special_tokens(out/"assets"/asset);v["special_tokens"][component]=data["tokens"]
            token_lock[component]=data
    if "formula" not in token_lock:v["special_tokens"]["formula"]=[]
    v["tokenizer_lock"]=token_lock
    for component in v["components"]:
        entry=v[component]
        entry["model_sha256"]=canonical(runtime["assets"][entry["model_asset"]]["files"])
        entry["source_sha256"]=canonical(plan["environments"]["paddle"]["image_files"])
        entry["config_sha256"]=canonical(formula_config() if component=="formula" else entry["parameters"])
    total=capacity(runtime,{k:out/a["path"] for k,a in runtime["assets"].items()})
    runtime["native"]["capacity"]={"passed":True,"conservative_upper_bound":total}
    # Only four path fields were normalized when deriving the shipped parent metadata.
    # This is a path binding, not removal of model/processor mismatch fields.
    diff=parent["path_rebinding"]
    write(out/"PATH_REBINDING.json",diff)
    runtime["binding"]={"parent_asset_lock_sha256":sha(VENDOR/"PARENT_ASSET_LOCK.json"),
                       "preparation_plan_sha256":sha(plan_path),"asset_rebinding":origins,
                       "path_rebinding_sha256":sha(out/"PATH_REBINDING.json")}
    write(out/"PROFILE.json",PROFILE);write(out/"TOKENIZER_LOCK.json",token_lock);write(out/"runtime.json",runtime)
    from .contracts import runtime as validate_runtime,budget
    b=budget(plan["validation_budget"],1)
    validate_runtime(out/"runtime.json",plan["validation_profile"],b,mode=plan["validation_mode"])
    result={"passed":True,"runtime_sha256":sha(out/"runtime.json"),
            "parent_asset_lock_sha256":sha(VENDOR/"PARENT_ASSET_LOCK.json"),"profile_sha256":PROFILE_SHA,
            "unique_trainable_total":total,"runtime_verified":False,"model_loads":0,"installation":False}
    if backend is not None:result["gpu_backend"]=backend
    write(out/"ASSET_BINDING.json",result);return result


def requirements(output):
    """Emit exact known prerequisites and explicit unknowns without choosing an image or wheel."""
    parent=read(VENDOR/"PARENT_ASSET_LOCK.json");layout=read(VENDOR/"layout_metadata.json")
    result={"schema":1,"status":"preparation_required_not_runtime_verified",
        "native_parent_freeze_sha256":parent["original_freeze_sha256"],
        "parent_asset_lock_sha256":sha(VENDOR/"PARENT_ASSET_LOCK.json"),
        "environments":{},
        "sources":{"tele":{"repository":"https://github.com/caipeng328/TeleOCR",
                           "revision":"9921cffe380efe4e2fa010258b3d0c3cb70bab2d"}},
        "models":{"tele":{"repository":"StarDoc-AI/TeleOCR","revision":"8706730e41382f4a8f2562616d47225fee8e2f7a","files":parent["model_files"]["tele_model"]},
                  "formula":{"identity":"original frozen PaddleOCR-VL1.6","files":parent["model_files"]["formula_model"]},
                  "layout":{"revision":layout["model_revision"],"files":{r["name"]:r["sha256"] for r in layout["model_files"]}}},
        "missing_required_evidence":["Exact immutable native/Paddle local image IDs or repository digests and resolved Python paths",
            "Complete image installed-file/native-library locks and original wheel/resolver receipts",
            "PP-DocLayoutV3 exact parameter graph count and source-bound evidence",
            "Unique parameters/buffers and auxiliary model inventories tied to each model asset lock",
            "Bounded real Linux CUDA/PDF/provider validation after deployment validation"],
        "installation_executed":False,"model_loads":0,"network":False}
    for role,packages in parent["packages"].items():
        result["environments"][role]={"packages":packages,"image":None,"python":None,
            "image_file_lock":None,"site_relative_auxiliary_hashes":parent["image_auxiliaries"][role],
            "required_source_hashes":layout["code_files"] if role=="paddle" else [],
            "action":"Provision a new separate immutable image using exactly these versions; preserve original wheel hashes; do not upgrade an existing environment."}
    from .asset_binding import chardet_lock,CHARDET_LOCK_SHA
    result["chardet_capacity"]={"classification_lock_sha256":CHARDET_LOCK_SHA,**chardet_lock(),
        "asset_role":"auxiliary_models","required_when":"schema2 paddle environment, any locked chardet image member, or any chardet auxiliary member"}
    path=Path(output);write(path,result);return result
