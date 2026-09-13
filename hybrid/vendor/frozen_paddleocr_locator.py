#!/usr/bin/env python3
"""Run the frozen deployment PaddleOCR pipeline on a Gold-free manifest.

Every image and frozen model artifact is verified before inference. Successful
per-sample sidecars are written atomically and can be resumed safely. Gold
annotations are forbidden until all OCR outputs have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/frozen_paddleocr_v5_deployment_v1.json"
PARSER_VERSION = "paddleocr-v5-full-table-to-token/v1"
FORBIDDEN_KEYS = {
    "annotation",
    "annotation_path",
    "canonical_cells",
    "cells",
    "gold",
    "gold_cells",
    "labels",
    "ocr_tokens",
    "polygon",
    "residual_edit_labels",
    "structure_labels",
    "tabrecset_annotation",
    "text_candidates",
    "words_json",
    "xml_annotation",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object")
            yield value


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    temporary.replace(path)
    return count


def safe_id(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "_", value).strip("_")


def bounded_pending_records(
    records: list[tuple[dict[str, Any], Path]], maximum: int
) -> list[tuple[dict[str, Any], Path]]:
    """Bound one OCR process lifetime without changing candidate order."""
    maximum = int(maximum)
    return records if maximum <= 0 else records[:maximum]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artifact_sha256(root: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    if not names:
        raise ValueError(f"no artifact files configured for {root}")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe artifact file: {name}")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_gold_free(row: Mapping[str, Any]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if str(key).lower() in FORBIDDEN_KEYS:
                    raise ValueError(
                        f"{row.get('sample_id')}: forbidden OCR inference key {key}"
                    )
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(row)


def to_builtin(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_builtin(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(child) for child in value]
    if hasattr(value, "tolist"):
        return to_builtin(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def unwrap_result(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        value: Any = result
    else:
        serialized = getattr(result, "json", None)
        if callable(serialized):
            serialized = serialized()
        value = serialized
    value = to_builtin(value)
    if not isinstance(value, dict):
        raise ValueError("PaddleOCR result is not an object")
    if isinstance(value.get("res"), dict):
        value = value["res"]
    return value


def polygon_bbox(polygon: Sequence[Sequence[Any]]) -> list[float] | None:
    points: list[tuple[float, float]] = []
    for point in polygon:
        if not isinstance(point, Sequence) or len(point) < 2:
            return None
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        points.append((x, y))
    if len(points) < 4:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_ocr_tokens(
    result: Mapping[str, Any], width: int, height: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    texts = list(result.get("rec_texts") or [])
    scores = list(result.get("rec_scores") or [])
    polygons = list(result.get("rec_polys") or [])
    if not (len(texts) == len(scores) == len(polygons)):
        raise ValueError(
            "PaddleOCR result arrays are misaligned: "
            f"text={len(texts)} score={len(scores)} poly={len(polygons)}"
        )
    tokens: list[dict[str, Any]] = []
    dropped_empty = 0
    dropped_geometry = 0
    for source_index, (text_value, score_value, polygon_value) in enumerate(
        zip(texts, scores, polygons)
    ):
        text = str(text_value)
        if not text.strip():
            dropped_empty += 1
            continue
        try:
            score = float(score_value)
        except (TypeError, ValueError):
            dropped_geometry += 1
            continue
        if not math.isfinite(score) or not isinstance(polygon_value, Sequence):
            dropped_geometry += 1
            continue
        polygon = [list(point) for point in polygon_value]
        bbox = polygon_bbox(polygon)
        if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            dropped_geometry += 1
            continue
        clipped = [
            max(0.0, min(float(width), bbox[0])),
            max(0.0, min(float(height), bbox[1])),
            max(0.0, min(float(width), bbox[2])),
            max(0.0, min(float(height), bbox[3])),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            dropped_geometry += 1
            continue
        confidence = max(0.0, min(1.0, score))
        tokens.append(
            {
                "token_index": len(tokens),
                "source_detection_index": source_index,
                "text": text,
                "confidence": confidence,
                "bbox": clipped,
                "polygon": [
                    [float(point[0]), float(point[1])] for point in polygon
                ],
                "alternatives": [
                    {
                        "source": "PP-OCRv5_server_rec/full_table_crop",
                        "text": text,
                        "confidence": confidence,
                    }
                ],
            }
        )
    return tokens, {
        "detected": len(polygons),
        "retained": len(tokens),
        "dropped_empty_text": dropped_empty,
        "dropped_invalid_geometry": dropped_geometry,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, Any]:
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "paddleocr": package_version("paddleocr"),
        "paddlepaddle": package_version("paddlepaddle"),
        "paddlex": package_version("paddlex"),
        "numpy": package_version("numpy"),
        "opencv_contrib_python": package_version("opencv-contrib-python"),
        "pillow": package_version("Pillow"),
    }


def verify_runtime(config: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    expected = config["runtime"]
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if str(expected.get(key)) != str(actual.get(key))
    }
    if mismatches:
        raise RuntimeError(f"frozen OCR runtime mismatch: {mismatches}")


def verify_models(config: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for role, model in config["models"].items():
        directory = Path(model["directory"]).resolve()
        actual = artifact_sha256(directory, list(model["artifact_files"]))
        expected = str(model["artifact_sha256"])
        if actual != expected:
            raise RuntimeError(
                f"frozen OCR model hash mismatch for {role}: "
                f"expected={expected} actual={actual}"
            )
        hashes[str(role)] = actual
    return hashes


def apply_runtime_controls(config: Mapping[str, Any]) -> dict[str, str]:
    """Apply frozen process controls before Paddle imports native runtimes."""

    controls = config.get("runtime_controls", {}).get("environment", {})
    if not isinstance(controls, Mapping):
        raise ValueError("runtime_controls.environment must be an object")
    applied: dict[str, str] = {}
    for key, value in controls.items():
        name = str(key)
        text = str(value)
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise ValueError(f"unsafe runtime-control variable: {name!r}")
        os.environ[name] = text
        applied[name] = text
    return applied


def effective_inference_policy(
    config: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    fallback = config.get("large_image_fallback")
    width = int(row.get("image_width") or 0)
    height = int(row.get("image_height") or 0)
    pixels = width * height
    if not isinstance(fallback, Mapping):
        return {
            "policy_version": "paddleocr-v5-base/v1",
            "large_image_fallback_applied": False,
            "input_pixels": pixels,
            "predict_overrides": {},
        }
    tiers = fallback.get("tiers")
    if tiers is not None:
        if (
            fallback.get("trigger")
            != "tiered_image_pixels_strictly_greater_than"
            or not isinstance(tiers, list)
            or not tiers
        ):
            raise ValueError("invalid tiered large-image fallback policy")
        parsed: list[tuple[int, str, dict[str, Any]]] = []
        for tier in tiers:
            if not isinstance(tier, Mapping):
                raise ValueError("large-image fallback tier must be an object")
            threshold = int(tier.get("image_pixels_strictly_greater_than", 0))
            name = str(tier.get("name") or "")
            overrides = tier.get("predict_overrides", {})
            if threshold <= 0 or not name or not isinstance(overrides, Mapping):
                raise ValueError("invalid large-image fallback tier")
            parsed.append((threshold, name, dict(overrides)))
        thresholds = [threshold for threshold, _, _ in parsed]
        if thresholds != sorted(thresholds, reverse=True) or len(
            set(thresholds)
        ) != len(thresholds):
            raise ValueError(
                "large-image fallback tiers must have unique descending thresholds"
            )
        selected = next(
            (item for item in parsed if pixels > item[0]),
            None,
        )
        normal_maximum = min(thresholds)
        return {
            "policy_version": str(fallback.get("policy_version") or ""),
            "large_image_fallback_applied": selected is not None,
            "input_pixels": pixels,
            "normal_path_maximum_pixels": normal_maximum,
            "selected_tier": selected[1] if selected else "normal",
            "trigger_pixels_strictly_greater_than": (
                selected[0] if selected else None
            ),
            "predict_overrides": selected[2] if selected else {},
        }
    if fallback.get("trigger") != "image_pixels_strictly_greater_than":
        raise ValueError("unsupported large-image fallback trigger")
    threshold = int(fallback.get("maximum_normal_path_pixels", 0))
    overrides = fallback.get("predict_overrides", {})
    if threshold <= 0 or not isinstance(overrides, Mapping):
        raise ValueError("invalid large-image fallback policy")
    applied = pixels > threshold
    return {
        "policy_version": str(fallback.get("policy_version") or ""),
        "large_image_fallback_applied": applied,
        "input_pixels": pixels,
        "maximum_normal_path_pixels": threshold,
        "predict_overrides": dict(overrides) if applied else {},
    }


def compatible_normal_path_config_hashes(
    config: Mapping[str, Any],
) -> set[str]:
    extension = config.get("extends", {})
    if not isinstance(extension, Mapping):
        return set()
    hashes: set[str] = set()
    legacy = str(extension.get("inference_config_sha256") or "")
    if legacy:
        hashes.add(legacy)
    entries = extension.get("compatible_normal_path_configurations", [])
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, Mapping):
                digest = str(entry.get("inference_config_sha256") or "")
                if digest:
                    hashes.add(digest)
    return hashes


def existing_success(
    path: Path,
    image_hash: str,
    config_hash: str,
    runtime_hash: str,
    model_hashes: Mapping[str, str],
    config: Mapping[str, Any],
    row: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    policy = effective_inference_policy(config, row)
    sidecar_config_hash = value.get("inference_config_sha256")
    config_compatible = sidecar_config_hash == config_hash
    if not policy["large_image_fallback_applied"]:
        if (
            bool(
                config.get("large_image_fallback", {}).get(
                    "normal_path_is_predecessor_semantically_identical",
                    config.get("large_image_fallback", {}).get(
                        "normal_path_is_v1_semantically_identical", False
                    ),
                )
            )
            and sidecar_config_hash
            in compatible_normal_path_config_hashes(config)
        ):
            config_compatible = True
    if policy["large_image_fallback_applied"]:
        config_compatible = (
            sidecar_config_hash == config_hash
            and value.get("effective_inference") == policy
        )
    return (
        value.get("status") == "ok"
        and value.get("input_image_sha256") == image_hash
        and config_compatible
        and value.get("runtime_fingerprint_sha256") == runtime_hash
        and value.get("model_artifact_sha256") == dict(model_hashes)
        and value.get("parser_version") == PARSER_VERSION
    )


def create_pipeline(config: Mapping[str, Any]) -> Any:
    from paddleocr import PaddleOCR

    models = config["models"]
    pipeline = config["pipeline"]
    engine = config["engine"]
    return PaddleOCR(
        text_detection_model_name=models["text_detection"]["name"],
        text_detection_model_dir=models["text_detection"]["directory"],
        textline_orientation_model_name=models["textline_orientation"]["name"],
        textline_orientation_model_dir=models["textline_orientation"]["directory"],
        text_recognition_model_name=models["text_recognition"]["name"],
        text_recognition_model_dir=models["text_recognition"]["directory"],
        use_doc_orientation_classify=pipeline["use_doc_orientation_classify"],
        use_doc_unwarping=pipeline["use_doc_unwarping"],
        use_textline_orientation=pipeline["use_textline_orientation"],
        textline_orientation_batch_size=pipeline["textline_orientation_batch_size"],
        text_recognition_batch_size=pipeline["text_recognition_batch_size"],
        text_det_limit_side_len=pipeline["text_det_limit_side_len"],
        text_det_limit_type=pipeline["text_det_limit_type"],
        text_det_thresh=pipeline["text_det_thresh"],
        text_det_box_thresh=pipeline["text_det_box_thresh"],
        text_det_unclip_ratio=pipeline["text_det_unclip_ratio"],
        text_rec_score_thresh=pipeline["text_rec_score_thresh"],
        return_word_box=pipeline["return_word_box"],
        device=engine["device"],
        engine=engine["engine"],
        enable_hpi=engine["enable_hpi"],
        enable_mkldnn=engine["enable_mkldnn"],
        cpu_threads=engine["cpu_threads"],
        precision=engine["precision"],
        use_tensorrt=engine["use_tensorrt"],
        enable_cinn=engine["enable_cinn"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected_manifest_sha256", default="")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument(
        "--max_pending_samples",
        type=int,
        default=500,
        help="Process this many missing images before an in-place memory reset.",
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="verify contract, runtime, models, and manifest without loading PaddleOCR",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config).resolve()
    if not manifest.is_file() or not config_path.is_file():
        raise FileNotFoundError("OCR manifest or frozen config is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_FORMAL20K_OCR":
        raise RuntimeError("OCR configuration is not frozen")
    if config["parser"]["version"] != PARSER_VERSION:
        raise RuntimeError("OCR parser version disagrees with frozen config")

    runtime_controls = apply_runtime_controls(config)

    manifest_hash = file_sha256(manifest)
    if args.expected_manifest_sha256 and manifest_hash != args.expected_manifest_sha256:
        raise RuntimeError(
            "OCR manifest hash mismatch: "
            f"expected={args.expected_manifest_sha256} actual={manifest_hash}"
        )
    config_hash = file_sha256(config_path)
    actual_runtime = runtime_versions()
    verify_runtime(config, actual_runtime)
    runtime_hash = stable_json_hash(actual_runtime)
    model_hashes = verify_models(config)

    records = list(iter_jsonl(manifest))
    if args.max_samples > 0:
        records = records[: args.max_samples]
    sample_ids: set[str] = set()
    for row in records:
        validate_gold_free(row)
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"missing or duplicate sample_id: {sample_id!r}")
        sample_ids.add(sample_id)
        if not row.get("table_crop") or not row.get("input_image_sha256"):
            raise ValueError(f"{sample_id}: missing table_crop or image hash")

    sidecar_dir = output_dir / "sidecars"
    monitor_path = output_dir / "inference_monitor.jsonl"
    run_provenance = {
        "schema_version": "mpr-tsr-splitmerge/frozen-ocr-run-v1",
        "status": "VERIFIED_ONLY" if args.verify_only else "RUNNING",
        "manifest": str(manifest),
        "manifest_sha256": manifest_hash,
        "selected_records": len(records),
        "config": str(config_path),
        "inference_config_sha256": config_hash,
        "runtime": actual_runtime,
        "runtime_fingerprint_sha256": runtime_hash,
        "model_artifact_sha256": model_hashes,
        "parser_version": PARSER_VERSION,
        "runtime_controls": runtime_controls,
        "large_image_fallback": config.get("large_image_fallback"),
        "extends": config.get("extends"),
        "gold_visible_during_inference": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "run_provenance.json", run_provenance)
    if args.verify_only:
        print(json.dumps(run_provenance, ensure_ascii=False, indent=2))
        return

    pending: list[tuple[dict[str, Any], Path]] = []
    existing = 0
    for row in records:
        sidecar = sidecar_dir / f"{safe_id(str(row['sample_id']))}.json"
        if args.skip_existing and existing_success(
            sidecar,
            str(row["input_image_sha256"]),
            config_hash,
            runtime_hash,
            model_hashes,
            config,
            row,
        ):
            existing += 1
        else:
            pending.append((row, sidecar))

    pending_before_run = len(pending)
    pending = bounded_pending_records(pending, args.max_pending_samples)

    append_jsonl(
        monitor_path,
        {
            "event": "start",
            "time": time.time(),
            "manifest_sha256": manifest_hash,
            "selected": len(records),
            "existing_before_run": existing,
            "pending": pending_before_run,
            "scheduled_this_run": len(pending),
            "failures": 0,
            "input_hash_failures": 0,
        },
    )
    ocr = create_pipeline(config)
    processed = 0
    failures = 0
    input_hash_failures = 0
    total_tokens = 0
    started = time.time()
    for row, sidecar in pending:
        sample_started = time.time()
        sample_id = str(row["sample_id"])
        image_path = Path(row["table_crop"]).resolve()
        expected_image_hash = str(row["input_image_sha256"])
        status = "ok"
        error = None
        tokens: list[dict[str, Any]] = []
        counts = {
            "detected": 0,
            "retained": 0,
            "dropped_empty_text": 0,
            "dropped_invalid_geometry": 0,
        }
        raw_result_hash = None
        effective_policy = effective_inference_policy(config, row)
        try:
            actual_image_hash = file_sha256(image_path)
            if actual_image_hash != expected_image_hash:
                input_hash_failures += 1
                raise RuntimeError(
                    f"input image hash mismatch expected={expected_image_hash} "
                    f"actual={actual_image_hash}"
                )
            with Image.open(image_path) as image:
                width, height = image.size
            results = ocr.predict(
                str(image_path), **effective_policy["predict_overrides"]
            )
            if len(results) != 1:
                raise RuntimeError(f"PaddleOCR returned {len(results)} results")
            raw = unwrap_result(results[0])
            raw_result_hash = stable_json_hash(raw)
            tokens, counts = parse_ocr_tokens(raw, int(width), int(height))
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            failures += 1

        record = {
            "schema_version": "mpr-tsr-splitmerge/frozen-ocr-sidecar-v1",
            "sample_id": sample_id,
            "source_dataset": row.get("source_dataset"),
            "document_id": row.get("document_id"),
            "table_crop": str(image_path),
            "input_image_sha256": expected_image_hash,
            "status": status,
            "failure": error,
            "ocr_tokens": tokens,
            "token_counts": counts,
            "raw_result_sha256": raw_result_hash,
            "effective_inference": effective_policy,
            "inference_config_sha256": config_hash,
            "runtime_fingerprint_sha256": runtime_hash,
            "model_artifact_sha256": model_hashes,
            "parser_version": PARSER_VERSION,
            "elapsed_seconds": round(time.time() - sample_started, 6),
        }
        atomic_write_json(sidecar, record)
        processed += 1
        total_tokens += len(tokens)
        elapsed = max(1e-9, time.time() - started)
        remaining = len(pending) - processed
        progress = {
            "event": "progress",
            "time": time.time(),
            "processed": processed,
            "pending_total": len(pending),
            "existing_before_run": existing,
            "failures": failures,
            "input_hash_failures": input_hash_failures,
            "latest_sample": sample_id,
            "latest_status": status,
            "latest_seconds": record["elapsed_seconds"],
            "total_tokens": total_tokens,
            "samples_per_second_including_model_load": processed / elapsed,
            "eta_seconds": remaining / max(processed / elapsed, 1e-9),
        }
        append_jsonl(monitor_path, progress)
        print(json.dumps(progress, ensure_ascii=False), flush=True)

    remaining_after_run = pending_before_run - processed
    if (
        remaining_after_run > 0
        and failures == 0
        and input_hash_failures == 0
    ):
        checkpoint = {
            "event": "checkpoint_complete",
            "time": time.time(),
            "selected": len(records),
            "existing_before_run": existing,
            "processed": processed,
            "indexed": existing + processed,
            "failures": 0,
            "missing_sidecars": remaining_after_run,
            "input_hash_failures": 0,
            "remaining_after_run": remaining_after_run,
        }
        append_jsonl(monitor_path, checkpoint)
        run_provenance["status"] = "CHECKPOINT_COMPLETE"
        run_provenance["complete"] = checkpoint
        atomic_write_json(output_dir / "run_provenance.json", run_provenance)
        print(json.dumps(checkpoint, ensure_ascii=False, indent=2), flush=True)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    indexed: list[dict[str, Any]] = []
    missing_sidecars = 0
    for row in records:
        sidecar = sidecar_dir / f"{safe_id(str(row['sample_id']))}.json"
        if not sidecar.is_file():
            missing_sidecars += 1
            continue
        indexed.append(json.loads(sidecar.read_text(encoding="utf-8")))
    indexed.sort(key=lambda row: str(row["sample_id"]))
    index_path = output_dir / "ocr_sidecar_index.jsonl"
    write_jsonl(index_path, indexed)
    complete = {
        "event": "complete",
        "time": time.time(),
        "selected": len(records),
        "existing_before_run": existing,
        "processed": processed,
        "indexed": len(indexed),
        "failures": sum(row.get("status") != "ok" for row in indexed),
        "missing_sidecars": missing_sidecars,
        "input_hash_failures": input_hash_failures,
        "ocr_sidecar_index": str(index_path),
        "ocr_sidecar_index_sha256": file_sha256(index_path),
    }
    append_jsonl(monitor_path, complete)
    run_provenance["status"] = (
        "COMPLETE"
        if not complete["failures"] and not missing_sidecars
        else "INCOMPLETE_FAILURES_RETAINED"
    )
    run_provenance["complete"] = complete
    atomic_write_json(output_dir / "run_provenance.json", run_provenance)
    print(json.dumps(complete, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
