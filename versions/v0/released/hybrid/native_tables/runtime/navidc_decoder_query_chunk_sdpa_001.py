"""Instance-local decoder Q256/full-KV resource implementation.

No import-time Torch or model construction. Short queries call the original HF
adapter verbatim. Long queries keep its original GQA decision, scaling and full
KV, using explicit absolute-row causal masks. This is not a token/bit-equivalence
claim. No model, precision, image, token, cache or allocator policy is changed.
"""
import hashlib
import inspect
from pathlib import Path
import sys
from types import FunctionType, MethodType


QUERY_CHUNK_ROWS = 256
MODEL_MODULE = "_sha_bound_navidc_model"
ATTENTION_CLASS = "Qwen2_5_VLAttention"
MODEL_SOURCE_SHA256 = "7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922"
SDPA_SOURCE_SHA256 = "dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013"
DECORATOR_SOURCE_SHA256 = "aec6dcedb6c73e6bde3d29a4a5ffe75d0ece217e884c548aea72711efbb46d8e"
MARKER = "_navidc_decoder_query_chunk_sdpa_001"
POLICY = "NAVIDC_DECODER_SDPA_QUERY256_FULL_KV_ABSOLUTE_CAUSAL_V1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_reference():
    import transformers
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig

    source = sys.modules.get(MODEL_MODULE)
    if transformers.__version__ != "4.57.6" or source is None:
        raise ValueError("PINNED_TRANSFORMERS_AND_LOADED_NATIVE_MODULE_REQUIRED")
    cls = getattr(source, ATTENTION_CLASS, None)
    if cls is None or cls.__module__ != MODEL_MODULE or cls.__name__ != ATTENTION_CLASS:
        raise ValueError("PINNED_NATIVE_ATTENTION_CLASS_REQUIRED")
    outer, inner = cls.forward, inspect.unwrap(cls.forward)
    if (outer is inner or getattr(outer, "__wrapped__", None) is not inner
            or inner.__qualname__ != ATTENTION_CLASS + ".forward"
            or inner.__module__ != MODEL_MODULE
            or _sha(source.__file__) != MODEL_SOURCE_SHA256
            or _sha(inner.__code__.co_filename) != MODEL_SOURCE_SHA256
            or _sha(outer.__code__.co_filename) != DECORATOR_SOURCE_SHA256
            or _sha(inspect.getsourcefile(sdpa_attention_forward)) != SDPA_SOURCE_SHA256):
        raise ValueError("PINNED_NATIVE_OR_DECORATOR_OR_SDPA_SOURCE_CHANGED")
    if inner.__globals__["ALL_ATTENTION_FUNCTIONS"]["sdpa"] is not sdpa_attention_forward:
        raise ValueError("NATIVE_SDPA_REGISTRY_CHANGED")
    return cls, Qwen2_5_VLTextConfig, sdpa_attention_forward


def _validate_attention(module):
    if (module.training or module.config._attn_implementation != "sdpa"
            or module.attention_dropout != 0.0 or module.is_causal is not True
            or module.num_key_value_groups != 2 or module.num_heads != 16
            or module.num_key_value_heads != 8 or module.head_dim != 128
            or module.hidden_size != 1024 or module.sliding_window is not None
            or module.scaling != 128 ** -0.5):
        raise ValueError("PINNED_EVAL_FULL_CAUSAL_DECODER_REQUIRED")


def _new_audit(paths):
    return {"policy": POLICY, "query_chunk_rows": QUERY_CHUNK_ROWS,
            "model_source_sha256": MODEL_SOURCE_SHA256,
            "sdpa_source_sha256": SDPA_SOURCE_SHA256,
            "decorator_source_sha256": DECORATOR_SOURCE_SHA256,
            "instance_paths": list(paths), "full_original_kv_per_chunk": True,
            "absolute_upper_left_causal_offsets": True,
            "original_gqa_decision_preserved": True,
            "native_forward_and_decorator_bytecode_preserved": True,
            "global_registry_and_vision_unchanged": True,
            "short_query_original_passthrough": True,
            "mps_synchronize_after_each_chunk": True,
            "bitwise_equivalence_claimed": False,
            "call_count": 0, "completed_call_count": 0, "chunk_count": 0,
            "passthrough_call_count": 0, "max_query_rows": 0, "max_key_rows": 0,
            "max_query_chunk_rows": 0,
            "instance_stats": {p: {"call_count": 0, "chunk_count": 0,
                "last_query_shape": None, "last_key_shape": None,
                "last_dtype": None, "last_device": None,
                "last_effective_causal": None, "last_enable_gqa": None} for p in paths}}


def _validate_inputs(module, query, key, value, mask, dropout, scaling, is_causal, kwargs):
    import torch

    _validate_attention(module)
    allowed = {"position_ids", "sliding_window", "output_attentions", "head_mask"}
    if (set(kwargs) - allowed or kwargs.get("sliding_window") is not None
            or kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None
            or dropout != 0.0 or scaling not in (None, module.scaling)
            or (is_causal is not None and type(is_causal) is not bool)):
        raise ValueError("UNSUPPORTED_ATTENTION_FEATURE_OR_SCALING")
    if (not all(isinstance(t, torch.Tensor) and t.ndim == 4 for t in (query, key, value))
            or query.shape[0] != 1 or query.shape[1] != 16 or query.shape[3] != 128
            or key.shape[0] != 1 or key.shape[1] != 8 or key.shape[3] != 128
            or key.shape != value.shape or query.shape[2] < 1 or key.shape[2] < 1
            or any(t.dtype != query.dtype or t.device != query.device for t in (key, value))
            or query.dtype not in (torch.float64, torch.float32, torch.bfloat16)
            or query.device.type not in ("cpu", "mps")
            or (query.device.type == "mps" and query.dtype != torch.bfloat16)):
        raise ValueError("PINNED_BATCH1_GQA_SHAPES_DEVICE_DTYPE_REQUIRED")
    if mask is not None:
        if (not isinstance(mask, torch.Tensor) or mask.ndim != 4
                or mask.shape[0] != 1 or mask.shape[1] not in (1, 16)
                or mask.shape[2] not in (1, query.shape[2]) or mask.shape[3] < key.shape[2]
                or mask.device != query.device
                or mask.dtype not in (torch.bool, torch.float32, query.dtype)):
            raise ValueError("EXACT_BROADCAST_4D_BOOL_OR_FLOAT_MASK_REQUIRED")


def _make_chunked_sdpa(original, audit, path, expected_instance):
    def chunked(module, query, key, value, attention_mask, dropout=0.0,
                scaling=None, is_causal=None, **kwargs):
        import torch
        from transformers.integrations.sdpa_attention import repeat_kv, use_gqa_in_sdpa

        if module is not expected_instance:
            raise ValueError("INSTANCE_LOCAL_SDPA_CANNOT_BE_REUSED")
        _validate_inputs(module, query, key, value, attention_mask, dropout, scaling, is_causal, kwargs)
        rows, keys = query.shape[2], key.shape[2]
        effective_causal = (rows > 1 and attention_mask is None and module.is_causal
                            if is_causal is None else is_causal)
        stats = audit["instance_stats"][path]
        audit["call_count"] += 1
        stats["call_count"] += 1
        stats.update(last_query_shape=list(query.shape), last_key_shape=list(key.shape),
                     last_dtype=str(query.dtype), last_device=str(query.device),
                     last_effective_causal=effective_causal)
        audit["max_query_rows"] = max(audit["max_query_rows"], rows)
        audit["max_key_rows"] = max(audit["max_key_rows"], keys)
        if rows <= QUERY_CHUNK_ROWS:
            audit["passthrough_call_count"] += 1
            result = original(module, query, key, value, attention_mask, dropout=dropout,
                              scaling=scaling, is_causal=is_causal, **kwargs)
            audit["completed_call_count"] += 1
            return result

        # Preserve the original adapter's choice BEFORE explicit causal chunk
        # masks are introduced. Recalling HF with those masks would change GQA.
        enable_gqa = use_gqa_in_sdpa(attention_mask, key)
        stats["last_enable_gqa"] = enable_gqa
        if not enable_gqa:
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        sdpa_kwargs = {"enable_gqa": True} if enable_gqa else {}
        mask = attention_mask[..., :keys] if attention_mask is not None else None
        key_positions = torch.arange(keys, device=query.device) if effective_causal else None
        outputs = []
        for start in range(0, rows, QUERY_CHUNK_ROWS):
            end = min(start + QUERY_CHUNK_ROWS, rows)
            chunk_mask = mask if mask is None or mask.shape[-2] == 1 else mask[..., start:end, :]
            if effective_causal:
                causal = key_positions[None, :] <= torch.arange(start, end, device=query.device)[:, None]
                causal = causal[None, None, :, :]
                if chunk_mask is None:
                    chunk_mask = causal
                elif chunk_mask.dtype == torch.bool:
                    chunk_mask = chunk_mask & causal
                else:
                    # Addition, not masked_fill, preserves native additive-mask
                    # behavior including infinities at causally masked entries.
                    bias = torch.zeros(causal.shape, dtype=chunk_mask.dtype, device=query.device)
                    bias.masked_fill_(~causal, float("-inf"))
                    chunk_mask = chunk_mask + bias
            audit["chunk_count"] += 1
            stats["chunk_count"] += 1
            audit["max_query_chunk_rows"] = max(audit["max_query_chunk_rows"], end - start)
            output = torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:end, :], key, value, attn_mask=chunk_mask,
                dropout_p=dropout, scale=scaling, is_causal=False, **sdpa_kwargs)
            outputs.append(output.transpose(1, 2).contiguous())
            if query.device.type == "mps":
                torch.mps.synchronize()
        result = (torch.cat(outputs, dim=1), None)
        audit["completed_call_count"] += 1
        return result

    return chunked


def _clone_function(function, globals_dict, closure):
    clone = FunctionType(function.__code__, globals_dict, function.__name__, function.__defaults__, closure)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__.copy()
    clone.__dict__.update(function.__dict__)
    clone.__qualname__, clone.__module__, clone.__doc__ = function.__qualname__, function.__module__, function.__doc__
    return clone


def _cell(value):
    return (lambda: value).__closure__[0]


def _clone_native_forward(outer, registry):
    inner = getattr(outer, "__wrapped__", None)
    if inner is None or hasattr(inner, "__wrapped__"):
        raise ValueError("EXACT_SINGLE_NATIVE_DEPRECATION_WRAPPER_REQUIRED")
    cells = dict(zip(outer.__code__.co_freevars, outer.__closure__ or ()))
    if "func" not in cells or cells["func"].cell_contents is not inner:
        raise ValueError("NATIVE_DEPRECATION_FUNC_CLOSURE_CHANGED")
    private_globals = inner.__globals__.copy()
    private_globals["ALL_ATTENTION_FUNCTIONS"] = registry
    cloned_inner = _clone_function(inner, private_globals, inner.__closure__)
    closure = tuple(_cell(cloned_inner) if name == "func" else cell
                    for name, cell in zip(outer.__code__.co_freevars, outer.__closure__))
    cloned_outer = _clone_function(outer, outer.__globals__, closure)
    cloned_outer.__wrapped__ = cloned_inner
    return cloned_outer


def bind_model(model):
    """Bind only 28 already-loaded native decoder attention instances, once."""
    if model.training or hasattr(model, MARKER):
        raise ValueError("EVAL_MODEL_AND_SINGLE_BIND_REQUIRED")
    cls, config_type, original = _load_reference()
    modules = dict(model.named_modules())
    language = modules.get("model.language_model")
    paths = [f"model.language_model.layers.{i}.self_attn" for i in range(28)]
    matches = {p for p, v in modules.items() if type(v).__name__ == ATTENTION_CLASS}
    if (language is None or matches != set(paths)
            or type(language).__name__ != "Qwen2_5_VLTextModel"
            or type(language).__module__ != MODEL_MODULE or type(language.config) is not config_type
            or language.training or len(language.layers) != 28 or language.has_sliding_layers
            or language.config.num_hidden_layers != 28
            or list(language.config.layer_types) != ["full_attention"] * 28):
        raise ValueError("PINNED_28_LAYER_NATIVE_TEXT_MODEL_REQUIRED")
    for index, path in enumerate(paths):
        instance = modules[path]
        if (type(instance) is not cls or type(instance.config) is not config_type
                or instance.config is not language.config or instance.layer_idx != index
                or hasattr(instance, MARKER) or "forward" in vars(instance)
                or not inspect.ismethod(instance.forward) or instance.forward.__func__ is not cls.forward):
            raise ValueError("EXACT_UNPATCHED_NATIVE_DECODER_INSTANCE_REQUIRED")
        _validate_attention(instance)
    audit = _new_audit(paths)
    prepared = [(modules[p], MethodType(_clone_native_forward(cls.forward,
        {"sdpa": _make_chunked_sdpa(original, audit, p, modules[p])}), modules[p])) for p in paths]
    for instance, forward in prepared:
        instance.forward = forward
        setattr(instance, MARKER, audit)
    setattr(model, MARKER, audit)
    return audit
