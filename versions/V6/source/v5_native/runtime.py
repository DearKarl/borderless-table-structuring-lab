"""Pinned runtime assertions retained from V5; native input remains unchanged."""
import importlib.metadata
import os
import sys
from .io_utils import utc

def runtime_receipt(client, protocol):
    import TeleOCR
    import TeleOCR_vllm
    import TeleOCR.config as config
    import torch
    from TeleOCR.vlm_utils.TeleOCR_client import DEFAULT_PROMPTS, DEFAULT_SAMPLING_PARAMS
    cfg = client.client.vllm_llm.llm_engine.vllm_config
    keys = ("temperature", "top_p", "top_k", "presence_penalty", "frequency_penalty",
            "repetition_penalty", "no_repeat_ngram_size", "max_new_tokens")
    sampling = {name: {key: getattr(value, key) for key in keys} for name, value in DEFAULT_SAMPLING_PARAMS.items()}
    packages = {}
    for name in ("torch", "torchvision", "vllm", "transformers", "tokenizers", "TeleOCR", "Pillow", "numpy",
                 "pypdfium2", "pymupdf", "lxml", "fast-langdetect", "fasttext-predict", "magika", "onnxruntime",
                 "triton", "flash-attn", "xformers", "safetensors", "opencv-python-headless"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    actual = {"dtype": str(cfg.model_config.dtype), "seed": cfg.model_config.seed,
              "max_model_len": cfg.model_config.max_model_len, "enforce_eager": cfg.model_config.enforce_eager,
              "gpu_memory_utilization": cfg.cache_config.gpu_memory_utilization,
              "enable_prefix_caching": cfg.cache_config.enable_prefix_caching,
              "max_num_seqs": cfg.scheduler_config.max_num_seqs,
              "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens}
    expected = protocol["engine"]
    for key, value in actual.items():
        reference = "torch.bfloat16" if key == "dtype" else expected[key]
        if value != reference:
            raise RuntimeError(f"Effective engine setting differs: {key}={value!r}, expected {reference!r}")
    if config.LAYOUT_MODE != "Detection" or config.MAX_PIXELS != 64000000:
        raise RuntimeError("Native configuration differs")
    if tuple(client.helper.layout_image_size) != (1036, 1036) or client.batching_mode != "stepping":
        raise RuntimeError("Native helper settings differ")
    return {"observed_at": utc(), "python": sys.version, "packages": packages, "engine": actual,
            "prompts": DEFAULT_PROMPTS, "sampling": sampling, "system_prompt": client.client.system_prompt,
            "tele_module": TeleOCR.__file__, "plugin_module": TeleOCR_vllm.__file__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_count": torch.cuda.device_count(), "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_device_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "compilation_config": str(cfg.compilation_config), "attention_config": str(getattr(cfg, "attention_config", None)),
            "helper": {key: getattr(client.helper, key) for key in (
                "layout_image_size", "min_image_edge", "max_image_edge_ratio", "simple_post_process",
                "handle_equation_block", "abandon_list", "abandon_paratext")},
            "allow_truncated_content": client.client.allow_truncated_content,
            "max_tokens_policy": "Native max_new_tokens=None resolves to max_model_len; engine context limits apply."}

