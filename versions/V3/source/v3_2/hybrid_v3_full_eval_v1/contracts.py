"""Frozen ordered inputs, explicit resources, isolated asset bindings."""
from pathlib import Path
from .core import ContractError, closed, digest, identifier, positive, read, sha, verify_tree

PROFILES = {
    "native": (),
    "v31-layout": ("layout",),
    "v31-formula": ("formula",),
    "v31-both": ("layout", "formula"),
    "v32-text": ("text",),
}
NATIVE_FREEZE = "399839e9f8a713bdab8d700d92865f72cccabc96d530773e8dc7b9b98e51acaa"
TELE_CLIENT = "9e998fa7408bf3bba5933275b0d6675a1c4926681d7da61a907a3f0e9a98f8fa"

def inputs(path, verify=True, deadline=None):
    path = Path(path).resolve()
    m = read(path)
    if m.get("schema") != 1 or not isinstance(m.get("pages"), list) or not m["pages"]:
        raise ContractError("Nonempty schema-1 input manifest required")
    ids, locations = set(), set()
    for p in m["pages"]:
        key = identifier(p["page_id"])
        if key.casefold() in ids:
            raise ContractError("Duplicate/case-colliding page_id")
        ids.add(key.casefold())
        f = closed(path.parent, p["path"])
        digest(p["file_sha256"])
        if not isinstance(p["source_document_id"], str) or not p["source_document_id"]:
            raise ContractError("source_document_id required; use unknown if unavailable")
        if p["kind"] not in ("image", "pdf"):
            raise ContractError("Unsupported input kind")
        ordinal = p.get("page_ordinal")
        if p["kind"] == "pdf":
            if type(ordinal) is not int or type(p.get("page_count")) is not int or not 0 <= ordinal < p["page_count"]:
                raise ContractError("PDF needs zero-based ordinal and original page_count")
            if p.get("source_sha256") != p["file_sha256"]:
                raise ContractError("PDF source SHA differs")
        elif ordinal is not None:
            raise ContractError("Image has unexpected PDF ordinal")
        location = (str(f).casefold(), ordinal)
        if location in locations:
            raise ContractError("Duplicate input page")
        locations.add(location)
        if verify and sha(f, deadline) != p["file_sha256"]:
            raise ContractError("Input tamper: " + key)
    return m

def budget(b, n):
    required = ("pages", "native_calls_per_page", "expert_calls_per_page", "model_loads",
                "load_seconds", "page_seconds", "request_seconds", "total_seconds",
                "cleanup_seconds", "page_audit_seconds", "cpus", "ram_gib", "swap_gib", "shm_gib",
                "gpu_allowlist", "customer_gpu_uuids")
    if any(k not in b for k in required):
        raise ContractError("All budget fields must be explicit")
    for k in ("pages", "native_calls_per_page", "expert_calls_per_page", "model_loads"):
        if type(b[k]) is not int:
            raise ContractError("Count budget must be an integer")
        positive(b[k], k, zero=k == "expert_calls_per_page")
    if n > b["pages"]:
        raise ContractError("Page budget exceeded")
    for k, ceiling in (("load_seconds",600), ("page_seconds",900), ("request_seconds",900),
                       ("total_seconds",None), ("cleanup_seconds",240), ("page_audit_seconds",15),
                       ("cpus",None), ("ram_gib",None), ("swap_gib",None), ("shm_gib",None)):
        positive(b[k], k, ceiling, zero=k == "swap_gib")
    if b["total_seconds"] <= b["cleanup_seconds"]:
        raise ContractError("No inference time remaining")
    if b["cleanup_seconds"] != 240:
        raise ContractError("Reserve the complete 240-second host cleanup path")
    g = b["gpu_allowlist"]
    if not isinstance(g, list) or not 1 <= len(g) <= 2 or len(set(g)) != len(g):
        raise ContractError("One or two distinct GPUs required")
    if any(not isinstance(x,str) or not x.startswith("GPU-") for x in g):
        raise ContractError("GPU UUIDs required")
    if set(g) & set(b["customer_gpu_uuids"]):
        raise ContractError("Customer GPU prohibited")
    return b

def runtime(path, profile, b, verify=True, deadline=None, mode="on"):
    r = read(path); root = Path(path).resolve().parent
    from .gpu_backend import validate_backend
    validate_backend(r,root) # Always validate manual descriptor, including verify=False.
    if profile not in PROFILES:
        raise ContractError("Unknown profile")
    # This release intentionally has no accepted Hybrid binding.
    if PROFILES[profile] and r.get("schema")!=2:
        raise ContractError("PROFILE_UNBOUND: " + profile +
                            "; expert integration requires schema 2")
    if r.get("schema") not in (1,2) or r.get("native_parent_freeze_sha256") != NATIVE_FREEZE:
        raise ContractError("Native parent asset identity mismatch")
    image = r.get("image", "")
    from .image_identity import parse_reference
    parse_reference(image, allow_local=r.get("schema")==2)
    if not r.get("python", "").startswith("/"):
        raise ContractError("Container Python path must be explicit")
    assets = r.get("assets", {})
    if (r.get("schema")==1 and set(assets) != {"native_code", "tele_source", "tele_model", "environment"}) or not {"native_code","tele_source","tele_model","environment"}<=set(assets):
        raise ContractError("Explicit code/source/model/environment asset roots required")
    if r.get("gpu_uuids") != b["gpu_allowlist"] or (r.get("schema")==1 and len(r["gpu_uuids"]) != 1):
        raise ContractError("Native profile requires one exact owned GPU")
    for name, a in assets.items():
        identifier(name)
        if a.get("role") != name or not a.get("files"):
            raise ContractError("Asset role/lock missing")
        from .v32_cluster import asset_root
        asset = asset_root(r, root, name)
        if not asset.is_dir():
            raise ContractError("Asset must be a closed directory")
        if verify:
            verify_tree(asset, a["files"], deadline)
    native = r.get("native", {})
    required = {"tele_packages", "identity_files", "capacity", "expected_load_metadata"}
    if not required <= set(native):
        raise ContractError("Native resource/model identities are incomplete")
    if set(native["expected_load_metadata"]) != {"PROCESSOR.json", "TELE_LOAD.json"}:
        raise ContractError("Exact processor/load metadata binding required")
    if not isinstance(r.get("lease_directory"), str) or not r["lease_directory"].startswith("/"):
        raise ContractError("Shared absolute Linux lease_directory required")
    positive(r["idle_memory_mib"], "idle_memory_mib", zero=True)
    if native["tele_packages"].get("pypdfium2") != "4.30.0":
        raise ContractError("Original PDFium 4.30.0 required")
    if not native["capacity"].get("passed") or native["capacity"]["conservative_upper_bound"] > 4_000_000_000:
        raise ContractError("Unverified parameter budget")
    if assets["tele_source"]["files"].get("TeleOCR/vlm_utils/TeleOCR_client.py") != TELE_CLIENT:
        raise ContractError("Original Tele client differs")
    code = assets["native_code"]["files"]
    parent_name="PARENT_ASSET_LOCK.json" if r.get("schema")==2 else "SYSTEM_FREEZE.json"
    parent_hash=sha(Path(__file__).parent/"assets/vendor/PARENT_ASSET_LOCK.json") if r.get("schema")==2 else NATIVE_FREEZE
    if code.get(parent_name) != parent_hash:
        raise ContractError("Parent freeze must be present, unchanged")
    frozen = read(closed(root, assets["native_code"]["path"]) / parent_name)
    required_code={parent_name} | {
        "hybrid_v2_tele_base/"+name+".py" for name in
        ("__init__","runtime_load","smoke","region_protocol","audit",
         "tele_adapter","expert_process","runtime_resources")}
    if r.get("schema")==2 and "formula" in r.get("v31",{}).get("components",[]):
        required_code.add("hybrid_v2_tele_base/paddle_formula_worker.py")
    if set(code)!=required_code:
        raise ContractError("Native asset root must contain only minimal frozen imports and parent freeze")
    for name in required_code-{parent_name}:
        if code[name] != frozen["code_files"][name]:
            raise ContractError("Native code no longer matches parent freeze: " + name)
    if not native["identity_files"]:
        raise ContractError("Native identity files required")
    for row in native["identity_files"]:
        if r.get("schema")==2 and "image_path" in row:
            if r["environments"][row["image_role"]]["image_files"].get(row["image_path"])!=digest(row["sha256"]):
                raise ContractError("Auxiliary image identity row not locked")
            continue
        if row["asset"] not in assets or assets[row["asset"]]["files"].get(row["path"]) != digest(row["sha256"]):
            raise ContractError("Identity row is not bound to locked asset")
    if r.get("schema")==2:
        from .asset_binding import validate_v31
        if profile=="v32-text":
            from .v32_binding import validate
            validate(r,root,profile,b,mode,verify)
        else:validate_v31(r,root,profile,b,mode,verify)
    return r

def response_identity(request, response):
    for k in ("run_id", "page_id", "slot_id", "request_id", "input_sha256",
              "crop_rgb_sha256", "tensor_sha256", "model_sha256", "config_sha256"):
        if not request.get(k) or response.get(k) != request[k]:
            raise ContractError("Cross-page/crop/tensor identity: " + k)
    return response

def route_slots(slots, mode, target, native, expert=None):
    """Integration primitive: preserve sparse IDs and original object order."""
    if mode not in ("off", "pass-through", "on"):
        raise ContractError("Unknown routing mode")
    ids = [x["slot_id"] for x in slots]
    if len(set(ids)) != len(ids):
        raise ContractError("Duplicate native slot")
    if mode != "on":
        return native(slots)
    if expert is None:
        raise ContractError("Expert missing")
    # No broad exception fallback: CUDA, identity, audit and unknown errors propagate.
    out = []
    for s in slots:
        out.append(expert(s) if s["category"] == target else native([s])[0])
    return out
