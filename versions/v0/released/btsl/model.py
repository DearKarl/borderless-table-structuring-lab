"""Pinned upstream assets and native loader; never substitute another model."""
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import sys
import urllib.request

from hybrid.native_tables.verify_model import verify
from hybrid.native_tables.runtime import generation, native_ops
from .io import read, sha

MANIFEST = Path(__file__).resolve().parents[1] / "hybrid/native_tables/model_manifest.json"


def verify_model(folder):
    return verify(Path(folder), read(MANIFEST))


def download(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for entry in read(MANIFEST)["files"]:
        path = folder / entry["path"]
        if path.exists():
            if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError("Existing model asset differs; will not overwrite " + path.name)
            continue
        partial = path.with_suffix(path.suffix + ".partial")
        if partial.exists():
            raise ValueError("Preserved incomplete download; inspect/remove this file before retry: " + str(partial))
        with urllib.request.urlopen(entry["source_url"], timeout=120) as source, partial.open("xb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        if partial.stat().st_size != entry["bytes"] or sha(partial) != entry["sha256"]:
            raise ValueError("Downloaded model SHA differs: " + path.name)
        partial.rename(path)
    return {"verified_files": verify_model(folder), "model_dir": str(folder.resolve())}


def runtime():
    import platform
    packages = {p: importlib.metadata.version(p) for p in (
        "torch", "torchvision", "transformers", "Pillow", "numpy", "scipy", "safetensors", "accelerate", "huggingface-hub")}
    required = {"torch": "2.12.0", "torchvision": "0.27.0", "transformers": "4.57.6", "Pillow": "12.2.0"}
    if any(packages[p].split("+")[0] != v for p, v in required.items()):
        raise ValueError("Install the pinned inference extra; changing runtime requires a new recipe")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def predict(model_dir, image, task, device):
    """One model lifetime per call. Raw Markdown and Gold never enter this function."""
    runtime()
    verify_model(model_dir)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") not in (None, "0"):
        raise ValueError("Implicit CPU fallback is prohibited")
    import torch
    from transformers import AutoConfig, AutoProcessor
    if device == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS unavailable")
        torch.mps.set_per_process_memory_fraction(12 * 1024**3 / torch.mps.recommended_max_memory())
    elif device == "cuda":
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA BF16 unavailable")
    else:
        raise ValueError("Only explicit MPS or CUDA BF16; no hidden backend fallback")
    source = Path(model_dir) / "modeling_naviocr.py"
    spec = importlib.util.spec_from_file_location("_sha_bound_navidc_model", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    config = AutoConfig.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model, info = module.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_dir), config=config, local_files_only=True, use_safetensors=True,
        ignore_mismatched_sizes=False, output_loading_info=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    generation.check_loading_info(info)
    model = model.to(device).eval()
    if model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr():
        raise ValueError("Expected tied embedding/head alias missing")
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, use_fast=True)
    if (processor.image_processor.min_pixels, processor.image_processor.max_pixels) != (3136, 12845056):
        raise ValueError("Native image preparation changed")
    if model.generation_config.repetition_penalty != 1.05:
        raise ValueError("Frozen repetition penalty changed")
    audit = {}
    if device == "mps":
        from hybrid.native_tables.runtime import navidc_vision_query_chunk_sdpa_001 as vision
        from hybrid.native_tables.runtime import navidc_decoder_query_chunk_sdpa_002 as decoder
        audit = {"vision": vision.bind_model(model), "decoder": decoder.bind_model(model)}
    # CUDA uses native SDPA: explicitly a distinct runtime, not the scored MPS identity.
    result = native_ops.generate(torch, model, processor, image, task, generation)
    result["load"] = {"loading_info": info, "device": str(model.device), "dtype": str(model.dtype)}
    result["attention"] = audit
    return result
