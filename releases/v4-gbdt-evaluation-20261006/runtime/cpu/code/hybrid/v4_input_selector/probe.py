"""Read-only static probe. Does not import model, renderer or training packages."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import shutil
import struct
import sys

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def check_files(root, expected):
    rows=[]
    for name,digest in sorted(expected.items()):
        path=Path(root)/name
        if not path.is_file():rows.append(dict(name=name,expected=digest,actual=None,status="missing"));continue
        actual=sha256(path)
        rows.append(dict(name=name,expected=digest,actual=actual,status="match" if actual==digest else "mismatch"))
    return rows

def site_package_paths(prefix):
    root=Path(prefix).resolve()
    candidates=[root/"Lib/site-packages"]
    for lib in (root/"lib",root/"lib64"):
        if lib.is_dir():candidates.extend(lib.glob("python*/site-packages"))
    return sorted({p.resolve() for p in candidates if p.is_dir() and p.resolve().is_relative_to(root)})

def packages(prefix):
    paths=[str(p) for p in site_package_paths(prefix)]
    found={}
    for d in importlib.metadata.distributions(path=paths):
        name=d.metadata.get("Name")
        if name:found[name.lower().replace("_","-")]=d.version
    return found

def requested_static_parity(model_rows,source_status,aux):
    return all(r["status"]=="match" for r in model_rows) and source_status["status"]=="match" and all(x.get("match") is True for x in aux)

def function_signatures(path):
    tree=ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    return {node.name:ast.unparse(node.args) for node in ast.walk(tree)
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}

def safetensors_header(path):
    with Path(path).open("rb") as f:
        n=struct.unpack("<Q",f.read(8))[0]
        if n>16*1024*1024:raise ValueError("Unexpected safetensors header size")
        header=json.loads(f.read(n))
    tensors={k:v for k,v in header.items() if k!="__metadata__"}
    return {"header_bytes":n,"tensor_count":len(tensors),
            "stored_elements":sum(math.prod(v["shape"]) for v in tensors.values()),
            "unique_learned_parameters":None,
            "note":"Stored elements do not establish tied unique parameters or fresh loaded count."}

def main():
    parser=argparse.ArgumentParser()
    for arg in ("source-root","model-root","parent-lock","runtime-prefix","training-prefix"):
        parser.add_argument("--"+arg,required=True)
    parser.add_argument("--mode",choices=["static"],required=True)
    parser.add_argument("--reuse-source-evidence",help="Previously verified source receipt; reuse is not a new hash pass")
    parser.add_argument("--aux-manifest",help="Explicit JSON resource path/expected SHA map; never scan the machine")
    args=parser.parse_args()
    lock=json.loads(Path(args.parent_lock).read_text(encoding="utf-8-sig"))
    model_rows=check_files(args.model_root,lock["model_files"]["tele_model"])
    source_rows=None
    if args.reuse_source_evidence:
        source_status={"status":"reused_evidence_not_fresh_parity","evidence_sha256":sha256(args.reuse_source_evidence)}
    else:
        source_rows=check_files(args.source_root,lock["model_files"]["tele_source"])
        source_status={"status":"match" if all(r["status"]=="match" for r in source_rows) else "failed","files":source_rows}
    config=json.loads((Path(args.model_root)/"config.json").read_text())
    processor=json.loads((Path(args.model_root)/"preprocessor_config.json").read_text())
    runtime,training=packages(args.runtime_prefix),packages(args.training_prefix)
    source=Path(args.source_root)
    signatures={}
    for rel in ("TeleOCR/tools/pdf_image_tools_pdfium.py","TeleOCR/tools/pdf_reader.py",
                "TeleOCR/vlm_utils/vlm_client/transformers_client.py"):
        signatures[rel]=function_signatures(source/rel)
    gbdt_paths=[p/"sklearn/ensemble/_hist_gradient_boosting/gradient_boosting.py" for p in site_package_paths(args.training_prefix)]
    gbdt_paths=[p for p in gbdt_paths if p.is_file()]
    if len(gbdt_paths)>1:raise ValueError("Ambiguous training package location")
    gbdt_path=gbdt_paths[0] if gbdt_paths else None
    gbdt={"source_present":gbdt_path is not None,"source_sha256":sha256(gbdt_path) if gbdt_path else None}
    if gbdt_path:
        tree=ast.parse(gbdt_path.read_text(encoding="utf-8"))
        gbdt["constructors"]={n.name:next((ast.unparse(x.args) for x in n.body if isinstance(x,ast.FunctionDef) and x.name=="__init__"),None)
            for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ("BaseHistGradientBoosting","HistGradientBoostingRegressor")}
    aux=[]
    if args.aux_manifest:
        for item in json.loads(Path(args.aux_manifest).read_text()):
            path=Path(item["path"]);actual=sha256(path) if path.is_file() else None
            aux.append({**item,"actual_sha256":actual,"match":actual is not None and actual==item["expected_sha256"]})
    result={"mode":"static","model_imported":False,"runtime_ready":False,
        "executable":sys.executable,"platform":platform.platform(),
        "source":source_status,"model_files":model_rows,
        "model_header":safetensors_header(Path(args.model_root)/"model.safetensors"),
        "max_position_embeddings":config.get("max_position_embeddings"),
        "text_max_position_embeddings":config.get("text_config",{}).get("max_position_embeddings"),
        "processor":processor,"runtime_packages":runtime,"training_packages":training,
        "native_signatures":signatures,"gbdt":gbdt,"aux":aux,
        "renderer_comparison":{"historical":"4.30.0","current_metadata":runtime.get("pypdfium2"),
             "match":runtime.get("pypdfium2")=="4.30.0"},
        "tools_on_path":{name:shutil.which(name) for name in ("pdflatex","xelatex","pdftoppm","node")},
        "unverified":["Fresh GPU lease/hardware","Loaded class/dtype/keys/parameter count",
            "Actual native stage grids/kwargs","Current auxiliary resource binding" if not aux else "Auxiliary loaded capacity",
            "Full metric render environment","Fresh target runtime parity"],
        "static_parity":requested_static_parity(model_rows,source_status,aux)}
    result["requested_static_checks_failed"]=not result["static_parity"]
    print(json.dumps(result,indent=2))
    return 0 if result["static_parity"] else 2

if __name__=="__main__":
    raise SystemExit(main())
