"""SHA-bound decoder001 plus Q=1 MPS unused-allocator-cache reclamation.

Only fixed pre/post boundaries around the original short-query SDPA acquire
synchronization and empty_cache. The old SDPA arguments, returned object, long
Q256/full-KV implementation and native forward/decorator are unchanged. There
is no retry, GC, allocation-cap change or import-time Torch/model construction.
Telemetry retains only the latest event and failure, never tensors or locals.
"""
from collections import deque
import hashlib
import importlib.util
from pathlib import Path
from types import FunctionType


PREDECESSOR_PATH = Path(__file__).with_name("navidc_decoder_query_chunk_sdpa_001.py")
PREDECESSOR_SHA256 = "4ba5c30400282a83b428b0f3f5da2c713240411a421893753659388cc28d495d"
MARKER = "_navidc_decoder_query_chunk_sdpa_002"
HYGIENE_POLICY = "MPS_QUERY1_NATIVE_SDPA_PRE_POST_SYNCHRONIZE_UNUSED_CACHE_V1"
MAX_TRACEBACK_FRAMES = 32
MAX_ERROR_MESSAGE_CHARS = 2048
MAX_FRAME_TEXT_CHARS = 1024
OPERATION_ORDER = ["pre_telemetry", "pre_synchronize", "pre_empty_cache",
                   "post_pre_telemetry", "native_sdpa", "post_synchronize",
                   "post_empty_cache", "post_telemetry", "completed"]


def _load_predecessor():
    if hashlib.sha256(PREDECESSOR_PATH.read_bytes()).hexdigest() != PREDECESSOR_SHA256:
        raise ValueError("DECODER_HYGIENE_PREDECESSOR_SOURCE_CHANGED")
    spec = importlib.util.spec_from_file_location("_private_decoder_hygiene_predecessor001", PREDECESSOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_predecessor()
QUERY_CHUNK_ROWS = _base.QUERY_CHUNK_ROWS
MODEL_MODULE = _base.MODEL_MODULE
ATTENTION_CLASS = _base.ATTENTION_CLASS
POLICY = _base.POLICY


def _new_hygiene():
    return {"policy": HYGIENE_POLICY, "predecessor_helper_sha256": PREDECESSOR_SHA256,
            "device": "mps", "query_rows": 1, "operation_order": list(OPERATION_ORDER),
            "max_traceback_frames": MAX_TRACEBACK_FRAMES,
            "max_error_message_chars": MAX_ERROR_MESSAGE_CHARS,
            "max_frame_text_chars": MAX_FRAME_TEXT_CHARS,
            "retains_tensors_or_locals": False, "retry": False,
            "call_count": 0, "completed_call_count": 0, "failed_call_count": 0,
            "synchronize_count": 0, "empty_cache_count": 0,
            "last_event": None, "last_failure": None}


def _new_audit(paths):
    audit = _base._new_audit(paths)
    audit["memory_hygiene"] = _new_hygiene()
    return audit


def _snapshot(mps):
    # Values are allocator telemetry only. No tensor/model data is retained.
    return {"current_allocated_bytes": int(mps.current_allocated_memory()),
            "driver_allocated_bytes": int(mps.driver_allocated_memory())}


def _failure(exc, stage):
    frames, frame_count = deque(maxlen=MAX_TRACEBACK_FRAMES), 0
    tb = exc.__traceback__
    while tb is not None:
        code = tb.tb_frame.f_code
        frames.append({"filename": code.co_filename[:MAX_FRAME_TEXT_CHARS],
                       "function": code.co_name[:MAX_FRAME_TEXT_CHARS], "line": tb.tb_lineno})
        frame_count += 1
        tb = tb.tb_next
    message = str(exc)
    return {"stage": stage, "exception_type": type(exc).__name__[:MAX_FRAME_TEXT_CHARS],
            "exception_module": type(exc).__module__[:MAX_FRAME_TEXT_CHARS],
            "message": message[:MAX_ERROR_MESSAGE_CHARS], "message_chars": len(message),
            "traceback_frames": list(frames), "traceback_frame_count": frame_count,
            "traceback_keeps_innermost_frames": True, "captures_locals": False}


def _make_native_boundary(original, audit, path):
    def native(module, query, key, value, attention_mask, dropout=0.0,
               scaling=None, is_causal=None, **kwargs):
        if query.device.type != "mps" or query.shape[2] != 1:
            return original(module, query, key, value, attention_mask, dropout=dropout,
                            scaling=scaling, is_causal=is_causal, **kwargs)
        import torch

        hygiene = audit["memory_hygiene"]
        hygiene["call_count"] += 1
        event = {"call_index": hygiene["call_count"], "instance_path": path,
                 "key_rows": int(key.shape[2]), "stage": "pre_telemetry",
                 "pre_boundary": None, "after_pre_reclaim": None, "after_native_reclaim": None}
        hygiene["last_event"] = event
        try:
            event["pre_boundary"] = _snapshot(torch.mps)
            event["stage"] = "pre_synchronize"
            torch.mps.synchronize()
            hygiene["synchronize_count"] += 1
            event["stage"] = "pre_empty_cache"
            torch.mps.empty_cache()
            hygiene["empty_cache_count"] += 1
            event["stage"] = "post_pre_telemetry"
            event["after_pre_reclaim"] = _snapshot(torch.mps)
            event["stage"] = "native_sdpa"
            result = original(module, query, key, value, attention_mask, dropout=dropout,
                              scaling=scaling, is_causal=is_causal, **kwargs)
            event["stage"] = "post_synchronize"
            torch.mps.synchronize()
            hygiene["synchronize_count"] += 1
            event["stage"] = "post_empty_cache"
            torch.mps.empty_cache()
            hygiene["empty_cache_count"] += 1
            event["stage"] = "post_telemetry"
            event["after_native_reclaim"] = _snapshot(torch.mps)
            event["stage"] = "completed"
            hygiene["completed_call_count"] += 1
            return result
        except BaseException as exc:
            hygiene["failed_call_count"] += 1
            hygiene["last_failure"] = _failure(exc, event["stage"])
            raise

    return native


def _make_chunked_sdpa(original, audit, path, expected_instance):
    # The exact SHA-bound predecessor owns validation and all Q>1 arithmetic.
    return _base._make_chunked_sdpa(_make_native_boundary(original, audit, path),
                                   audit, path, expected_instance)


def bind_model(model):
    """Use the predecessor's exact atomic 28-instance binder with private globals."""
    namespace = _base.bind_model.__globals__.copy()
    namespace.update(_new_audit=_new_audit, _make_chunked_sdpa=_make_chunked_sdpa, MARKER=MARKER)
    binder = FunctionType(_base.bind_model.__code__, namespace, _base.bind_model.__name__)
    return binder(model)


def validate_memory_hygiene(audit, completed):
    """Pure bounded metadata checks for the parent; never imports Torch."""
    value, initial = audit["memory_hygiene"], _new_hygiene()
    counters = {"call_count", "completed_call_count", "failed_call_count",
                "synchronize_count", "empty_cache_count"}
    mutable = counters | {"last_event", "last_failure"}
    if (set(value) != set(initial)
            or any(value[k] != initial[k] for k in set(initial) - mutable)
            or any(type(value[k]) is not int or value[k] < 0 for k in counters)):
        raise ValueError("MEMORY_HYGIENE_POLICY_OR_COUNTER_CHANGED")
    calls, done, failed = (value[k] for k in ("call_count", "completed_call_count", "failed_call_count"))
    if (calls != done + failed or failed > 1
            or not 2 * done <= value["empty_cache_count"] <= value["synchronize_count"] <= 2 * calls
            or calls > audit["passthrough_call_count"]
            or (completed and failed)):
        raise ValueError("MEMORY_HYGIENE_COUNTER_CLOSURE_FAILED")
    event, failure = value["last_event"], value["last_failure"]
    if calls == 0:
        if event is not None or failure is not None:
            raise ValueError("MEMORY_HYGIENE_ORPHAN_EVENT")
        return
    event_keys = {"call_index", "instance_path", "key_rows", "stage", "pre_boundary",
                  "after_pre_reclaim", "after_native_reclaim"}
    if (not isinstance(event, dict) or set(event) != event_keys or event["call_index"] != calls
            or event["instance_path"] not in audit["instance_paths"]
            or type(event["key_rows"]) is not int or event["key_rows"] < 1
            or event["stage"] not in OPERATION_ORDER):
        raise ValueError("MEMORY_HYGIENE_EVENT_CHANGED")
    for field in ("pre_boundary", "after_pre_reclaim", "after_native_reclaim"):
        snapshot = event[field]
        if snapshot is not None and (not isinstance(snapshot, dict)
                or set(snapshot) != {"current_allocated_bytes", "driver_allocated_bytes"}
                or any(type(v) is not int or v < 0 for v in snapshot.values())):
            raise ValueError("MEMORY_HYGIENE_TELEMETRY_CHANGED")
    if not failed:
        if (failure is not None or event["stage"] != "completed"
                or any(event[k] is None for k in ("pre_boundary", "after_pre_reclaim", "after_native_reclaim"))):
            raise ValueError("MEMORY_HYGIENE_COMPLETION_CHANGED")
        return
    expected_failure_keys = {"stage", "exception_type", "exception_module", "message", "message_chars",
                             "traceback_frames", "traceback_frame_count", "traceback_keeps_innermost_frames",
                             "captures_locals"}
    if (not isinstance(failure, dict) or set(failure) != expected_failure_keys
            or failure["stage"] != event["stage"] or event["stage"] == "completed"
            or failure["captures_locals"] is not False
            or failure["traceback_keeps_innermost_frames"] is not True
            or type(failure["message_chars"]) is not int
            or not isinstance(failure["message"], str)
            or len(failure["message"]) != min(MAX_ERROR_MESSAGE_CHARS, failure["message_chars"])
            or type(failure["traceback_frame_count"]) is not int
            or failure["traceback_frame_count"] < 1
            or not isinstance(failure["traceback_frames"], list)
            or len(failure["traceback_frames"]) != min(MAX_TRACEBACK_FRAMES, failure["traceback_frame_count"])):
        raise ValueError("MEMORY_HYGIENE_FAILURE_CHANGED")
    for key in ("exception_type", "exception_module"):
        if not isinstance(failure[key], str) or len(failure[key]) > MAX_FRAME_TEXT_CHARS:
            raise ValueError("MEMORY_HYGIENE_EXCEPTION_TEXT_CHANGED")
    for frame in failure["traceback_frames"]:
        if (not isinstance(frame, dict) or set(frame) != {"filename", "function", "line"}
                or type(frame["line"]) is not int or frame["line"] < 1
                or any(not isinstance(frame[k], str) or len(frame[k]) > MAX_FRAME_TEXT_CHARS
                       for k in ("filename", "function"))):
            raise ValueError("MEMORY_HYGIENE_TRACEBACK_CHANGED")
