"""Instance-only, full-key/value vision SDPA query chunking; no import-time Torch.

This is a resource algorithm, not a window-attention approximation or a claim
of backend/token equivalence. Only native full-attention blocks 7/15/23/31 are
bound. Their original QKV/rotary/projection forward bytecode is retained. Every
256-row query slice calls the pinned HF SDPA with the entire original K and V.
MPS chunks synchronize to bound queued temporary allocations; no cache purge,
retry, device/dtype conversion, model loading, or global/library patch occurs.
"""
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, MethodType


QUERY_CHUNK_ROWS = 256
FULL_ATTENTION_BLOCKS = (7, 15, 23, 31)
MODEL_MODULE = "_sha_bound_navidc_model"
VISION_CLASS = "Qwen2_5_VLVisionAttention"
MODEL_SOURCE_SHA256 = "7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922"
SDPA_SOURCE_SHA256 = "dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013"
MARKER = "_navidc_vision_query_chunk_sdpa_001"
POLICY = "NAVIDC_FULL_VISION_SDPA_QUERY256_ALL_KEYS_VALUES_V1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_reference():
    """Read existing runtime classes; never construct or load a model."""
    import transformers
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

    source = sys.modules.get(MODEL_MODULE)
    if transformers.__version__ != "4.57.6" or source is None:
        raise ValueError("PINNED_TRANSFORMERS_AND_LOADED_NATIVE_MODULE_REQUIRED")
    vision_type = getattr(source, VISION_CLASS, None)
    if (vision_type is None or vision_type.__module__ != MODEL_MODULE
            or vision_type.__name__ != VISION_CLASS
            or vision_type.forward.__module__ != MODEL_MODULE
            or vision_type.forward.__qualname__ != VISION_CLASS + ".forward"
            or _sha(vision_type.forward.__code__.co_filename) != MODEL_SOURCE_SHA256
            or _sha(source.__file__) != MODEL_SOURCE_SHA256
            or _sha(inspect.getsourcefile(sdpa_attention_forward)) != SDPA_SOURCE_SHA256):
        raise ValueError("PINNED_NATIVE_VISION_OR_HF_SDPA_SOURCE_CHANGED")
    if vision_type.forward.__globals__["ALL_ATTENTION_FUNCTIONS"]["sdpa"] is not sdpa_attention_forward:
        raise ValueError("NATIVE_SDPA_REGISTRY_CHANGED")
    return vision_type, Qwen2_5_VLVisionConfig, sdpa_attention_forward


def _validate_attention(module):
    if (module.training or module.config._attn_implementation != "sdpa"
            or module.attention_dropout != 0.0 or module.is_causal is not False
            or module.num_key_value_groups != 1 or module.num_heads != 16
            or module.head_dim != 80 or module.dim != 1280
            or module.scaling != 80 ** -0.5):
        raise ValueError("ONLY_PINNED_EVAL_NONCAUSAL_16_HEAD_VISION_SDPA_SUPPORTED")


def _new_audit(paths):
    return {
        "policy": POLICY,
        "query_chunk_rows": QUERY_CHUNK_ROWS,
        "instance_paths": list(paths),
        "full_attention_block_indexes": list(FULL_ATTENTION_BLOCKS),
        "model_source_sha256": MODEL_SOURCE_SHA256,
        "sdpa_source_sha256": SDPA_SOURCE_SHA256,
        "all_original_keys_values_per_chunk": True,
        "native_forward_bytecode_preserved": True,
        "window_and_decoder_attention_unchanged": True,
        "mps_synchronize_after_each_chunk": True,
        "call_count": 0,
        "chunk_count": 0,
        "completed_call_count": 0,
        "passthrough_call_count": 0,
        "max_query_rows": 0,
        "max_key_rows": 0,
        "max_query_chunk_rows": 0,
        "instance_stats": {path: {"call_count": 0, "chunk_count": 0,
                                  "last_query_shape": None, "last_key_shape": None,
                                  "last_dtype": None, "last_device": None}
                           for path in paths},
    }


def _make_chunked_sdpa(original, audit, path, expected_instance):
    def chunked(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, is_causal=None, **kwargs):
        import torch

        if module is not expected_instance:
            raise ValueError("INSTANCE_LOCAL_SDPA_CANNOT_BE_REUSED")
        _validate_attention(module)
        if (attention_mask is not None or dropout != 0.0 or is_causal is not False
                or kwargs.get("output_attentions", False)
                or kwargs.get("head_mask") is not None):
            raise ValueError("MASK_DROPOUT_CAUSAL_OR_ATTENTION_OUTPUT_UNSUPPORTED")
        if (any(t.ndim != 4 for t in (query, key, value))
                or query.shape != key.shape or key.shape != value.shape
                or query.shape[0] != 1 or query.shape[1] != 16
                or query.shape[3] != 80 or query.shape[2] < 1
                or any(t.device != query.device or t.dtype != query.dtype for t in (key, value))):
            raise ValueError("PINNED_BATCH1_EQUAL_HEAD_FULL_SELF_ATTENTION_SHAPES_REQUIRED")
        rows = query.shape[2]
        stats = audit["instance_stats"][path]
        audit["call_count"] += 1
        stats["call_count"] += 1
        stats.update(last_query_shape=list(query.shape), last_key_shape=list(key.shape),
                     last_dtype=str(query.dtype), last_device=str(query.device))
        audit["max_query_rows"] = max(audit["max_query_rows"], rows)
        audit["max_key_rows"] = max(audit["max_key_rows"], key.shape[2])

        def native(q):
            audit["chunk_count"] += 1
            stats["chunk_count"] += 1
            audit["max_query_chunk_rows"] = max(audit["max_query_chunk_rows"], q.shape[2])
            result = original(module, q, key, value, attention_mask, dropout=dropout,
                              scaling=scaling, is_causal=is_causal, **kwargs)
            if query.device.type == "mps":
                torch.mps.synchronize()
            return result

        if rows <= QUERY_CHUNK_ROWS:
            audit["passthrough_call_count"] += 1
            result = native(query)  # Same tensors, arguments and returned pair.
        else:
            outputs = []
            for start in range(0, rows, QUERY_CHUNK_ROWS):
                output, weights = native(query[:, :, start:start + QUERY_CHUNK_ROWS, :])
                if weights is not None:
                    raise ValueError("PINNED_SDPA_UNEXPECTED_ATTENTION_WEIGHTS")
                outputs.append(output)
            # HF SDPA returns [batch, query_rows, heads, head_dim], not QKV order.
            result = (torch.cat(outputs, dim=1), None)
        audit["completed_call_count"] += 1
        return result

    return chunked


def bind_model(model):
    """Bind once after native eval load; return live JSON-serializable receipt.

    Counters measure attempted native calls/chunks and completed wrapper calls,
    including an attempted failing allocation. They contain no output content.
    All validation precedes mutation. Caller owns the enclosing runtime freeze.
    """
    if hasattr(model, MARKER) or model.training:
        raise ValueError("EVAL_MODEL_AND_SINGLE_BIND_REQUIRED")
    vision_type, config_type, original_sdpa = _load_reference()
    modules = dict(model.named_modules())
    visual = modules.get("model.visual")
    all_paths = [f"model.visual.blocks.{i}.attn" for i in range(32)]
    matches = {path for path, instance in modules.items() if type(instance).__name__ == VISION_CLASS}
    if (visual is None or matches != set(all_paths)
            or type(visual).__name__ != "Qwen2_5_VisionTransformerPretrainedModel"
            or type(visual).__module__ != MODEL_MODULE
            or type(visual.config) is not config_type
            or list(visual.fullatt_block_indexes) != list(FULL_ATTENTION_BLOCKS)
            or list(visual.config.fullatt_block_indexes) != list(FULL_ATTENTION_BLOCKS)
            or visual.config.depth != 32 or len(visual.blocks) != 32
            or visual.config.num_heads != 16 or visual.config.hidden_size != 1280
            or visual.config._attn_implementation != "sdpa" or visual.training):
        raise ValueError("PINNED_NATIVE_VISION_ARCHITECTURE_REQUIRED")
    selected = [f"model.visual.blocks.{i}.attn" for i in FULL_ATTENTION_BLOCKS]
    for path in all_paths:
        instance = modules[path]
        if (type(instance) is not vision_type or type(instance.config) is not config_type
                or instance.config is not visual.config or hasattr(instance, MARKER)
                or "forward" in vars(instance) or not inspect.ismethod(instance.forward)
                or instance.forward.__func__ is not vision_type.forward):
            raise ValueError("EXACT_UNPATCHED_NATIVE_VISION_INSTANCE_REQUIRED")
        _validate_attention(instance)

    audit = _new_audit(selected)
    prepared = []
    for path in selected:
        instance = modules[path]
        function = instance.forward.__func__
        private_globals = function.__globals__.copy()
        private_globals["ALL_ATTENTION_FUNCTIONS"] = {
            "sdpa": _make_chunked_sdpa(original_sdpa, audit, path, instance)}
        clone = FunctionType(function.__code__, private_globals, function.__name__,
                             function.__defaults__, function.__closure__)
        clone.__kwdefaults__ = function.__kwdefaults__
        clone.__annotations__ = function.__annotations__.copy()
        clone.__dict__.update(function.__dict__)
        clone.__qualname__ = function.__qualname__
        clone.__module__ = function.__module__
        clone.__doc__ = function.__doc__
        prepared.append((instance, MethodType(clone, instance)))
    for instance, forward in prepared:
        instance.forward = forward
        setattr(instance, MARKER, audit)
    setattr(model, MARKER, audit)
    return audit
