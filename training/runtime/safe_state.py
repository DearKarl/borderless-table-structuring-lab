from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch


class SafeStateError(RuntimeError):
    pass


MAGIC = b"EXPLICIT_SAFE_STATE_V1\n"
SCHEMA = "explicit-safe-state-2026.08.14.4"
MAX_HEADER_BYTES = 128 * 1024 * 1024
TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _torch_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    if tensor.layout != torch.strided or tensor.is_quantized:
        raise SafeStateError("only dense non-quantized tensors are supported")
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _encode(value: Any, blobs: list[dict[str, Any]], payloads: list[bytes]) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        dtype = str(tensor.dtype).removeprefix("torch.")
        if dtype not in TORCH_DTYPES:
            raise SafeStateError(f"unsupported torch dtype: {tensor.dtype}")
        payload = _torch_bytes(tensor)
        index = len(blobs)
        offset = sum(len(item) for item in payloads)
        blobs.append({
            "kind": "torch",
            "dtype": dtype,
            "shape": list(tensor.shape),
            "offset": offset,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        payloads.append(payload)
        return {"type": "blob", "index": index}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise SafeStateError("object or structured NumPy dtype is prohibited")
        payload = array.tobytes(order="C")
        index = len(blobs)
        offset = sum(len(item) for item in payloads)
        blobs.append({
            "kind": "numpy",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "offset": offset,
            "nbytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        payloads.append(payload)
        return {"type": "blob", "index": index}
    if isinstance(value, np.generic):
        return _encode(value.item(), blobs, payloads)
    if value is None:
        return {"type": "none"}
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SafeStateError("nonfinite scalar is prohibited")
        return {"type": "float", "value": value.hex()}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    if isinstance(value, bytes):
        return {"type": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, blobs, payloads) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, blobs, payloads) for item in value]}
    if isinstance(value, dict):
        items = [(_encode(key, blobs, payloads), _encode(item, blobs, payloads)) for key, item in value.items()]
        items.sort(key=lambda pair: _canonical(pair[0]))
        return {"type": "dict", "items": [[key, item] for key, item in items]}
    raise SafeStateError(f"unsupported state value: {type(value).__name__}")


def dumps_safe_state(value: Any) -> bytes:
    blobs: list[dict[str, Any]] = []
    payloads: list[bytes] = []
    root = _encode(value, blobs, payloads)
    payload = b"".join(payloads)
    header = {
        "schema_version": SCHEMA,
        "root": root,
        "blobs": blobs,
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    encoded = _canonical(header)
    if len(encoded) > MAX_HEADER_BYTES:
        raise SafeStateError("safe-state header exceeds frozen limit")
    return MAGIC + struct.pack(">Q", len(encoded)) + encoded + payload


def dump_safe_state(path: Path, value: Any) -> None:
    payload = dumps_safe_state(value)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def _decode(node: Any, blobs: list[dict[str, Any]], payload: bytes, references: list[int], depth: int = 0) -> Any:
    if depth > 512 or not isinstance(node, dict) or not isinstance(node.get("type"), str):
        raise SafeStateError("invalid safe-state node")
    kind = node["type"]
    if kind == "none" and set(node) == {"type"}:
        return None
    if kind == "bool" and set(node) == {"type", "value"} and isinstance(node["value"], bool):
        return node["value"]
    if kind == "int" and set(node) == {"type", "value"} and isinstance(node["value"], str):
        try:
            value = int(node["value"])
        except ValueError as exc:
            raise SafeStateError("invalid integer encoding") from exc
        if str(value) != node["value"]:
            raise SafeStateError("noncanonical integer encoding")
        return value
    if kind == "float" and set(node) == {"type", "value"} and isinstance(node["value"], str):
        try:
            value = float.fromhex(node["value"])
        except ValueError as exc:
            raise SafeStateError("invalid float encoding") from exc
        if not math.isfinite(value) or value.hex() != node["value"]:
            raise SafeStateError("noncanonical float encoding")
        return value
    if kind == "str" and set(node) == {"type", "value"} and isinstance(node["value"], str):
        return node["value"]
    if kind == "bytes" and set(node) == {"type", "value"} and isinstance(node["value"], str):
        try:
            return base64.b64decode(node["value"], validate=True)
        except ValueError as exc:
            raise SafeStateError("invalid bytes encoding") from exc
    if kind in {"tuple", "list"} and set(node) == {"type", "items"} and isinstance(node["items"], list):
        values = [_decode(item, blobs, payload, references, depth + 1) for item in node["items"]]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict" and set(node) == {"type", "items"} and isinstance(node["items"], list):
        result: dict[Any, Any] = {}
        encoded_keys: list[bytes] = []
        for pair in node["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise SafeStateError("invalid dictionary entry")
            encoded_keys.append(_canonical(pair[0]))
            key = _decode(pair[0], blobs, payload, references, depth + 1)
            if not isinstance(key, (type(None), bool, int, float, str, bytes, tuple)) or key in result:
                raise SafeStateError("unsafe or duplicate dictionary key")
            result[key] = _decode(pair[1], blobs, payload, references, depth + 1)
        if encoded_keys != sorted(encoded_keys):
            raise SafeStateError("dictionary key order is noncanonical")
        return result
    if kind == "blob" and set(node) == {"type", "index"} and isinstance(node["index"], int) and not isinstance(node["index"], bool):
        index = node["index"]
        if not 0 <= index < len(blobs):
            raise SafeStateError("blob index out of range")
        references.append(index)
        entry = blobs[index]
        required = {"kind", "dtype", "shape", "offset", "nbytes", "sha256"}
        if not isinstance(entry, dict) or set(entry) != required:
            raise SafeStateError("blob schema mismatch")
        offset, nbytes = entry["offset"], entry["nbytes"]
        shape = entry["shape"]
        if (not isinstance(offset, int) or isinstance(offset, bool) or not isinstance(nbytes, int)
                or isinstance(nbytes, bool) or offset < 0 or nbytes < 0 or not isinstance(shape, list)
                or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in shape)):
            raise SafeStateError("blob dimensions are invalid")
        raw = payload[offset:offset + nbytes]
        if len(raw) != nbytes or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise SafeStateError("blob identity mismatch")
        elements = math.prod(shape)
        if entry["kind"] == "torch" and entry["dtype"] in TORCH_DTYPES:
            dtype = TORCH_DTYPES[entry["dtype"]]
            expected = elements * torch.empty((), dtype=dtype).element_size()
            if expected != nbytes:
                raise SafeStateError("torch blob byte count mismatch")
            if elements == 0:
                return torch.empty(shape, dtype=dtype)
            octets = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
            return octets.view(dtype).reshape(shape)
        if entry["kind"] == "numpy" and isinstance(entry["dtype"], str):
            try:
                dtype = np.dtype(entry["dtype"])
            except TypeError as exc:
                raise SafeStateError("NumPy dtype is invalid") from exc
            if dtype.hasobject or dtype.fields is not None or elements * dtype.itemsize != nbytes:
                raise SafeStateError("NumPy blob byte count mismatch")
            return np.frombuffer(raw, dtype=dtype).copy().reshape(shape)
        raise SafeStateError("blob kind or dtype is unsupported")
    raise SafeStateError("safe-state node schema mismatch")


def loads_safe_state(document: bytes) -> Any:
    prefix = len(MAGIC) + 8
    if len(document) < prefix or document[:len(MAGIC)] != MAGIC:
        raise SafeStateError("safe-state magic mismatch")
    header_size = struct.unpack(">Q", document[len(MAGIC):prefix])[0]
    if header_size > MAX_HEADER_BYTES or prefix + header_size > len(document):
        raise SafeStateError("safe-state header length mismatch")
    header_bytes = document[prefix:prefix + header_size]
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeStateError("safe-state header parse failure") from exc
    required = {"schema_version", "root", "blobs", "payload_bytes", "payload_sha256"}
    if (not isinstance(header, dict) or set(header) != required or header["schema_version"] != SCHEMA
            or _canonical(header) != header_bytes or not isinstance(header["blobs"], list)):
        raise SafeStateError("safe-state header schema/canonicalization mismatch")
    payload = document[prefix + header_size:]
    if (not isinstance(header["payload_bytes"], int) or header["payload_bytes"] != len(payload)
            or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]):
        raise SafeStateError("safe-state payload identity mismatch")
    expected_offset = 0
    for entry in header["blobs"]:
        if not isinstance(entry, dict) or entry.get("offset") != expected_offset or not isinstance(entry.get("nbytes"), int):
            raise SafeStateError("safe-state blob layout is noncontiguous")
        expected_offset += entry["nbytes"]
    if expected_offset != len(payload):
        raise SafeStateError("safe-state blob layout length mismatch")
    references: list[int] = []
    value = _decode(header["root"], header["blobs"], payload, references)
    if sorted(references) != list(range(len(header["blobs"]))):
        raise SafeStateError("safe-state blob reference closure mismatch")
    return value


def load_safe_state(path: Path) -> Any:
    if path.is_symlink() or not path.is_file():
        raise SafeStateError("safe-state path is not a regular file")
    return loads_safe_state(path.read_bytes())
