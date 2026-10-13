"""Bounded external measurements from native captures and frozen native regions."""
from __future__ import annotations

import argparse
import json
import signal
import time
import traceback
from pathlib import Path

from .features import extract, sample_scales, scaled
from .io_utils import (PageDeadline, atomic_json, commit_stage, digest, finalize_page,
                       process_identity, read_json, require_common_runtime, utc)
from .native import install_input_audit, native_page, pixel_identity, realized_view
from .regions import region_crop, runaway, typed_spans
from .runtime import runtime_receipt


def native_content(middle, kind):
    regions = [(obj, key) for obj, k, key in typed_spans(middle["pdf_info"][0]["para_blocks"]) if k == kind]
    if not regions:
        return "", {"matching_native_regions": 0, "selection": "no_native_target"}
    def area(item):
        x0, y0, x1, y1 = item[0]["bbox"]
        return max(0, x1 - x0) * max(0, y1 - y0)
    obj, key = max(regions, key=area)
    return str(obj.get(key) or ""), {"matching_native_regions": len(regions), "selection": "largest_native_target",
                                    "bbox": obj["bbox"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--worker-id", required=True)
    args = parser.parse_args()
    control = read_json(args.control)
    protocol = read_json(control["protocol_path"])
    if digest(control["protocol_path"]) != control["protocol_sha256"] or control["arm"] not in ("pilot", "compatibility", "regional"):
        raise RuntimeError("This worker only accepts frozen external preparation tracks")
    compatibility = control["arm"] == "compatibility"
    regional = control["arm"] == "regional"
    if control['arm'] == 'pilot':
        raise RuntimeError('Whole-crop supervision is superseded; use source-page native regions')
    output = Path(args.output)
    worker_dir = output / "workers" / args.worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    progress = {"worker_id": args.worker_id, "arm": control["arm"], "run_id": control["run_id"],
                "identity": process_identity(), "started_at": utc(), "phase": "model_loading",
                "current_page": None, "completed_in_process": 0,
                "physical_gpu_index": control["physical_gpu_index"], "gpu_uuid": control["gpu_uuid"]}
    def status(**kwargs):
        progress.update(kwargs, updated_at=utc(), heartbeat_monotonic=time.monotonic())
        atomic_json(worker_dir / "STATUS.json", progress)
    status()
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    import TeleOCR.config as config
    config.MAX_MODEL_LEN = protocol["engine"]["max_model_len"]
    config.GPU_MEMORY_UTILIZATION = protocol["engine"]["gpu_memory_utilization"]
    from TeleOCR.vlm_utils.TeleOCR_model import TeleOCRMODEL_SERVICE
    client = TeleOCRMODEL_SERVICE.get_model(protocol["backend"], control["model_path"], None, **protocol["engine"])
    client.client.use_tqdm = False
    runtime = runtime_receipt(client, protocol)
    if runtime["cuda_device_count"] != 1 or runtime["cuda_device_uuid"].removeprefix("GPU-") != control["gpu_uuid"].removeprefix("GPU-"):
        raise RuntimeError("Physical GPU assignment mismatch")
    require_common_runtime(runtime, control.get("expected_common_runtime", {}))
    atomic_json(worker_dir / "RUNTIME.json", runtime)
    engine = client.client.vllm_llm.llm_engine
    real_add = engine.add_request
    request_ids = []
    events = None
    stage = None
    page_start = None
    observed_tensors = {}
    recent_inputs = []
    def emit(row):
        if row.get("event") == "model_input":
            for feature in row["features"]:
                if feature["fields"]:
                    observed_tensors[feature["identifier"]] = feature["fields"]
                recent_inputs.append(observed_tensors.get(feature["identifier"]))
        if events is not None:
            events.write(json.dumps({"at": utc(), "stage": stage, "elapsed_seconds": time.monotonic() - page_start,
                                     **row}, ensure_ascii=False, default=str) + "\n")
            events.flush()
    def add_request(request_id, *pos, **kwargs):
        request_ids.append(request_id)
        return real_add(request_id, *pos, **kwargs)
    engine.add_request = add_request
    install_input_audit(client, emit)
    def cancel():
        if request_ids:
            engine.abort_request(list(request_ids))
        if engine.has_unfinished_requests():
            raise RuntimeError("Requests remain after cancellation")
    def alarm(*_):
        raise PageDeadline("Sample work deadline reached")
    signal.signal(signal.SIGALRM, alarm)
    status(phase="ready", model_ready_at=utc())
    failures = 0
    for sample in control["pages"]:
        identifier = sample["page_id"]
        if (output / "receipts" / (identifier + ".json")).exists():
            continue
        if time.time() >= control["absolute_deadline_unix"]:
            return 30
        folder = output / "pages" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "START.json").exists():
            raise RuntimeError("A started sample must be finalized, never replayed")
        source = Path(control["input_root"]) / sample["file"]
        if digest(source) != sample["input_sha256"]:
            raise RuntimeError("Frozen input changed")
        request_ids.clear()
        page_start = time.monotonic()
        record = {"arm": control["arm"], "worker_id": args.worker_id, "run_id": control["run_id"],
                  "started_at": utc(), "start_monotonic": page_start, "status": "success", "error": None,
                  "input_sha256": sample["input_sha256"], "completed_stages": []}
        atomic_json(folder / "START.json", record)
        events = open(folder / "events.jsonl", "a", encoding="utf-8")
        status(phase="inference", current_page=identifier, page_start_monotonic=page_start, stage="native")
        signal.setitimer(signal.ITIMER_REAL, min(590, max(.001, control["absolute_deadline_unix"] - time.time())))
        fatal = None
        try:
            stage = "native"
            if regional:
                if sample['unit_contract'] != 'source_page_native_predicted_region_v1':
                    raise RuntimeError('Regional labels require frozen native predicted crops')
                with Image.open(source) as saved:
                    image = saved.copy()
                if pixel_identity(image) != sample['native_pixel_identity']:
                    raise RuntimeError('Native crop pixel identity changed')
                baseline = sample['baseline']
                features = extract(image, baseline, sample['kind'])
                import numpy as np
                if not np.allclose(features, sample['features'], rtol=0, atol=1e-7):
                    raise RuntimeError('Native feature recomputation changed')
                selection = {'selection': 'reused_native_predicted_region', 'index': sample['native_region_index']}
                stage = 'reused_native_region'
                commit_stage(folder, stage, {'unit_contract': sample['unit_contract'], 'native_region': sample}, baseline)
            else:
                middle, markdown, image = native_page(client, source, folder, sample.get("source_page_index"))
                commit_stage(folder, stage, middle, markdown)
                baseline, selection = native_content(middle, sample["kind"])
                features = extract(image, baseline, sample["kind"])
            record["completed_stages"].append(stage)
            record["native_model_requests"] = len(request_ids)
            probe = {"region_id": identifier, "kind": sample["kind"], "split": sample["split"],
                     "scope": "external_source_page_native_region" if regional else "external_content_crop", "baseline": baseline, "native_selection": selection,
                     "features": features, "candidates": [], "baseline_request_count": len(request_ids)}
            atomic_json(folder / "PROBE.json", probe)
            if compatibility:
                page = middle["pdf_info"][0]
                predicted = []
                for index, (obj, kind, key) in enumerate(typed_spans(page["para_blocks"])):
                    crop = region_crop(image, page, obj, 1.0)
                    crop.save(folder / ("predicted-%03d.png" % index))
                    predicted.append({"index": index, "kind": kind, "bbox": obj["bbox"],
                                      "baseline": str(obj.get(key) or ""), "crop": pixel_identity(crop),
                                      "features": extract(crop, str(obj.get(key) or ""), kind)})
                atomic_json(folder / "COMPATIBILITY.json", {
                    "pair_id": sample["pair_id"], "context": sample["context"],
                    "source_page_index": sample.get("source_page_index"), "page_size": page["page_size"],
                    "native_image": pixel_identity(image), "predicted_regions": predicted,
                    "whole_capture_features": features, "baseline_requests": len(request_ids),
                    "gt_boxes_used_for_inference": False, "regional_rereads": 0})
            stage = "response_probe"
            status(stage=stage)
            seen = {}
            scales = sample.get('scales', sample_scales(sample["seed"])) if not compatibility else []
            if len(scales) > 8:
                raise RuntimeError('Offline regional view ceiling exceeded')
            for number, scale in enumerate(scales):
                candidate_start = time.monotonic()
                view = scaled(image, scale)
                preview_start = time.monotonic()
                realized = realized_view(client, view, sample["kind"])
                row = {"index": number, "scale": scale, "realized": realized, "request_count": 0,
                       "resize_seconds": preview_start-candidate_start, "preview_seconds": time.monotonic()-preview_start}
                if realized["identity"] in seen:
                    previous = seen[realized["identity"]]
                    row.update(alias_of=previous["index"], output=previous["output"], accepted=previous["accepted"],
                               rejection=previous["rejection"])
                else:
                    before = len(request_ids)
                    recent_inputs.clear()
                    ocr_start = time.monotonic()
                    value = client.batch_content_extract([view], [sample["kind"]])[0]
                    row['ocr_seconds'] = time.monotonic()-ocr_start
                    rejection = runaway(value, sample["kind"])
                    row.update(output=value, accepted=rejection is None, rejection=rejection,
                               request_count=len(request_ids) - before, alias_of=None)
                    if row["request_count"] != 1:
                        raise RuntimeError("One regional view must submit exactly one model request")
                    if len(recent_inputs) != 1 or not recent_inputs[0]:
                        raise RuntimeError("Actual request tensors were not observed")
                    for key in ("pixel_values", "image_grid_thw"):
                        if recent_inputs[0][key]["sha256"] != realized["tensors"][key]["sha256"]:
                            raise RuntimeError("Deduplication preview differs from actual inference tensors")
                    row["actual_tensor_identity_verified"] = True
                    seen[realized["identity"]] = row
                row["retained_output"] = row["output"] if row["accepted"] else baseline
                row["changed"] = row["accepted"] and row["output"] != baseline
                row['elapsed_seconds'] = time.monotonic()-candidate_start
                probe["candidates"].append(row)
                atomic_json(folder / "PROBE.json", probe)
                emit({"event": "candidate_complete", "index": number, "scale": scale,
                      "alias_of": row["alias_of"], "accepted": row["accepted"], "changed": row["changed"]})
            record["probe_requests"] = sum(r["request_count"] for r in probe["candidates"])
            failures = 0
        except PageDeadline as exc:
            record.update(status="timeout", error={"type": type(exc).__name__, "stage": stage, "message": str(exc)})
        except Exception as exc:
            record.update(status="failed", error={"type": type(exc).__name__, "stage": stage, "message": str(exc)[:2000]})
            emit({"event": "exception", "traceback": traceback.format_exc()})
            if type(exc).__name__ in ("EngineDeadError", "OutOfMemoryError", "AttributeError", "KeyError"):
                fatal = exc
            if 'tensors' in str(exc) or 'identity changed' in str(exc) or 'recomputation changed' in str(exc):
                fatal = exc
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                cancel()
            except Exception as exc:
                fatal = exc
            record.update(elapsed_seconds=time.monotonic() - page_start, request_count=len(request_ids))
            finalize_page(folder, output, identifier, record)
            events.close(); events = None
            status(current_page=None, page_start_monotonic=None, completed_in_process=progress["completed_in_process"] + 1)
        if record["status"] != "success":
            failures += 1
        if fatal is not None or failures >= 3:
            status(phase="systemic_failure", error=str(fatal))
            return 20
    status(phase="inference_complete", finished_at=utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
