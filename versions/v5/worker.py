"""One persistent, isolated vLLM worker serving a fixed arm and page shard."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

from .io_utils import (PageDeadline, atomic_json, commit_stage, digest, finalize_page,
                       process_identity, read_json, require_common_runtime, utc)


def native_page(client, path, folder, mode):
    from PIL import Image
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    if mode == "default_pdf":
        from TeleOCR.engine import do_parse
        from TeleOCR.tools.read_file import read_fn
        middle = do_parse(str(folder / "native"), ["page"], [read_fn(path)], [None], predictor=client)[0]
        markdown = (folder / "native" / "page" / "page.md").read_text(encoding="utf-8")
        return middle, markdown
    from TeleOCR.data_reader_writer import ImageDataWriter
    from TeleOCR.src.vlm_magic_model import MagicModel
    from TeleOCR.tools.cut_image import cut_image_and_table
    from TeleOCR.tools.enum_class import ContentType
    from TeleOCR.tools.hash_utils import bytes_md5
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    results = client.batch_two_step_extract(images=[image])
    if len(results) != 1:
        raise ValueError("Native output cardinality differs")
    model = MagicModel(results[0], image.width, image.height)
    writer = ImageDataWriter(str(folder / "images"))
    image_id = bytes_md5(image.tobytes())
    for span in model.get_all_spans():
        if span["type"] in (ContentType.IMAGE, ContentType.SEAL, ContentType.CHAR):
            cut_image_and_table(span, image, image_id, 0, writer, scale=1.0)
    page = {"para_blocks": model.get_page_blocks(), "discarded_blocks": [],
            "page_size": list(image.size), "page_idx": 0}
    middle = {"pdf_info": [page]}
    markdown = union_make([page], "images")
    writer.save_all_images()
    return middle, markdown


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--worker-id", required=True)
    args = ap.parse_args()
    control = read_json(args.control)
    protocol = read_json(control["protocol_path"])
    if digest(control["protocol_path"]) != control["protocol_sha256"]:
        raise RuntimeError("Protocol hash mismatch")
    output = Path(args.output)
    worker_dir = output / "workers" / args.worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    identity = process_identity()
    started = utc()
    progress = {"worker_id": args.worker_id, "arm": control["arm"], "run_id": control["run_id"],
                "identity": identity, "started_at": started, "phase": "model_loading", "current_page": None,
                "completed_in_process": 0, "physical_gpu_index": control["physical_gpu_index"], "gpu_uuid": control["gpu_uuid"]}

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
    from TeleOCR.src.vlm_middle_json_mkcontent import union_make
    from .regions import guard, reread
    client = TeleOCRMODEL_SERVICE.get_model(protocol["backend"], control["model_path"], None, **protocol["engine"])
    client.client.use_tqdm = False
    runtime = runtime_receipt(client, protocol)
    if runtime["cuda_device_count"] != 1 or runtime["cuda_device_uuid"].removeprefix("GPU-") != control["gpu_uuid"].removeprefix("GPU-"):
        raise RuntimeError("Visible CUDA device differs from the assigned physical GPU")
    require_common_runtime(runtime, control.get("expected_common_runtime", {}))
    atomic_json(worker_dir / "RUNTIME.json", runtime)
    engine = client.client.vllm_llm.llm_engine
    request_ids = []
    real_add = engine.add_request
    real_batch = client.client._predict_one_batch
    events = None
    stage = None
    page_start = None
    deadline_testing = False

    def emit(record):
        if events is None:
            return
        row = {"at": utc(), "elapsed_seconds": time.monotonic() - page_start, "stage": stage, **record}
        events.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        events.flush()

    def add_request(request_id, *pos, **kw):
        request_ids.append(request_id)
        result = real_add(request_id, *pos, **kw)
        if deadline_testing and len(request_ids) == 1:
            signal.setitimer(signal.ITIMER_REAL, control["test_deadline_seconds"])
        return result

    def batch(image_objs, chat_prompts, sampling_params):
        before = time.monotonic()
        emit({"event": "request_batch", "requests": len(image_objs),
              "inputs": [{"size": list(im.size), "rgb_sha256": hashlib.sha256(im.tobytes()).hexdigest(),
                          "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "sampling": str(sp)}
                         for im, prompt, sp in zip(image_objs, chat_prompts, sampling_params)]})
        try:
            result = real_batch(image_objs, chat_prompts, sampling_params)
            emit({"event": "request_batch_complete", "requests": len(result), "seconds": time.monotonic() - before})
            return result
        except BaseException as exc:
            emit({"event": "request_batch_failed", "exception": type(exc).__name__, "seconds": time.monotonic() - before})
            raise

    engine.add_request = add_request
    client.client._predict_one_batch = batch

    def cancel_pending():
        if request_ids:
            engine.abort_request(list(request_ids))
        if engine.has_unfinished_requests():
            raise RuntimeError("Pending inference remained after page cancellation")

    def alarm(*_):
        raise PageDeadline("Page work deadline reached")

    signal.signal(signal.SIGALRM, alarm)
    status(phase="ready", model_ready_at=utc())
    if control.get("test_deadline_seconds") is not None:
        # Smoke-only cancellation fixture; never enabled in a full control file.
        try:
            deadline_testing = True
            signal.setitimer(signal.ITIMER_REAL, 30)
            native_page(client, Path(control["input_root"]) / control["pages"][0]["file"], worker_dir / "deadline_fixture", "original_rgb")
            raise RuntimeError("Deadline fixture did not interrupt")
        except PageDeadline:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if not request_ids:
                raise RuntimeError("Deadline fixture failed to submit an inference request")
            cancel_pending()
            atomic_json(worker_dir / "DEADLINE_CHECK.json", {"interrupted": True, "request_count": len(request_ids),
                        "pending_after_abort": False, "at": utc()})
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            deadline_testing = False

    consecutive_native_failures = 0
    for page in control["pages"]:
        page_id = page["page_id"]
        if (output / "receipts" / f"{page_id}.json").is_file():
            continue
        if time.time() >= control["absolute_deadline_unix"]:
            status(phase="ceiling_reached")
            return 30
        folder = output / "pages" / page_id
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "START.json").exists():
            raise RuntimeError("An unfinished started page must be finalized by the supervisor, never rerun")
        path = Path(control["input_root"]) / page["file"]
        if digest(path) != page["input_sha256"]:
            raise RuntimeError("Input hash mismatch: " + page_id)
        page_start = time.monotonic()
        request_ids.clear()
        record = {"arm": control["arm"], "run_id": control["run_id"], "worker_id": args.worker_id,
                  "started_at": utc(), "start_monotonic": page_start, "input_sha256": page["input_sha256"],
                  "status": "success", "error": None, "completed_stages": []}
        atomic_json(folder / "START.json", record)
        events = open(folder / "events.jsonl", "a", encoding="utf-8")
        status(phase="inference", current_page=page_id, page_start_monotonic=page_start, stage="native")
        signal.setitimer(signal.ITIMER_REAL, min(protocol["limits"]["page_work_seconds"],
                                                max(0.001, control["absolute_deadline_unix"] - time.time())))
        fatal = None
        try:
            stage = "native"
            middle, markdown = native_page(client, path, folder, protocol["arms"][control["arm"]][0])
            if len(middle.get("pdf_info", [])) != 1:
                raise ValueError("Native middle JSON must retain exactly one page")
            commit_stage(folder, stage, middle, markdown)
            record["completed_stages"].append(stage)
            emit({"event": "stage_complete"})
            consecutive_native_failures = 0
            with Image.open(path) as opened:
                image = opened.convert("RGB")
            for stage in protocol["arms"][control["arm"]][1:]:
                status(stage=stage)
                if stage == "guard":
                    candidate, changed = guard(middle, image, client, emit, protocol["regions"]["guard_scales"])
                else:
                    kind, factor = ("table", 1.0) if stage == "table1x" else ("equation", 1.25)
                    candidate, changed = reread(middle, image, client, kind, factor, emit)
                next_markdown = union_make(candidate["pdf_info"], "images") if changed else markdown
                commit_stage(folder, stage, candidate, next_markdown)
                middle, markdown = candidate, next_markdown
                record["completed_stages"].append(stage)
                emit({"event": "stage_complete", "changed": changed})
        except PageDeadline as exc:
            record.update(status="timeout", error={"type": type(exc).__name__, "message": str(exc), "stage": stage})
        except Exception as exc:
            record.update(status="failed" if not record["completed_stages"] else "regional_fallback",
                          error={"type": type(exc).__name__, "message": str(exc)[:2000], "stage": stage})
            emit({"event": "exception", "traceback": traceback.format_exc()})
            if type(exc).__name__ in ("EngineDeadError", "OutOfMemoryError"):
                fatal = exc
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                cancel_pending()
            except Exception as exc:
                fatal = exc
            record["elapsed_seconds"] = time.monotonic() - page_start
            record["request_count"] = len(request_ids)
            finalize_page(folder, output, page_id, record)
            events.close()
            events = None
            status(completed_in_process=progress["completed_in_process"] + 1, current_page=None,
                   page_start_monotonic=None, last_progress_at=utc(), last_page_status=record["status"])
        if not record["completed_stages"]:
            consecutive_native_failures += 1
        if fatal is not None:
            status(phase="engine_failure", error=str(fatal))
            return 20
        if consecutive_native_failures >= 3:
            status(phase="systemic_failure", error="Three consecutive native page failures")
            return 21
    status(phase="inference_complete", finished_at=utc())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
