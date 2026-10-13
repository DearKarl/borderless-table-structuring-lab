"""Single coordinated image-to-native-table-to-Markdown pipeline.

Inputs are the original page image and a frozen upstream MinerU Markdown page.
Completed generations are cached; a process failure is never a Raw fallback.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

from hybrid.native_tables.assemble import assemble_record, SCHEMA
from hybrid.native_tables.assembly import assemble_page, digest
from hybrid.native_tables.formats import parse_layout, parse_otsl
from hybrid.native_tables.runtime.native_ops import crop_block, prepare_recognition_image
from .io import Lease, canonical, publish, read, sha, write_once
from .inputs import validate


def sources():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "btsl").glob("*.py"))
    native = root / "hybrid/native_tables"
    files += list(native.glob("*.py")) + list((native / "runtime").glob("*.py"))
    files += list((native / "vendor/omnidocbench").glob("*.py"))
    files += [native / "model_manifest.json", native / "architecture.json"]
    return {str(p.relative_to(root)): sha(p) for p in sorted(files)}


def verify_ready(folder, ready_name="READY.json"):
    folder = Path(folder)
    ready = read(folder / ready_name)
    for name, expected in ready["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or not (folder / relative).resolve().is_relative_to(folder.resolve()):
            raise ValueError("Unsafe receipt path")
        if sha(folder / relative) != expected:
            raise ValueError("Completed artifact changed: " + str(relative))
    return ready


class ProcessEngine:
    def __init__(self, folder, model_dir, device, run_binding, lock_fd, retry_failed=False):
        self.folder, self.model_dir, self.device = folder, model_dir, device
        self.run_binding, self.lock_fd, self.retry_failed = run_binding, lock_fd, retry_failed

    def generate(self, image, task):
        binding = {"rgb_sha256": digest(image.tobytes()), "dimensions": list(image.size),
                   "mode": image.mode, "task": task, "run_binding": self.run_binding,
                   "page_id": self.page_id, "call_index": self.call_index}
        self.call_index += 1
        key = digest(canonical(binding))
        cache = self.folder / "calls" / key
        cache.mkdir(parents=True, exist_ok=True)
        if (cache / "READY.json").exists():
            ready = verify_ready(cache)
            if ready["binding"] != binding:
                raise ValueError("Call binding mismatch")
            return read(cache / "generation.json")
        attempts = sorted(cache.glob("attempt-*"))
        # Recover a completed child if its parent exited before adopting the call.
        if attempts and (attempts[-1] / "READY.json").exists():
            attempt = attempts[-1]
            ready = verify_ready(attempt)
            if ready["binding"] != binding:
                raise ValueError("Child binding changed")
        else:
            if attempts and not self.retry_failed:
                raise RuntimeError("Incomplete call preserved. Inspect logs then explicitly use --resume --retry-failed: " + str(cache))
            attempt = cache / ("attempt-%04d" % (len(attempts) + 1))
            attempt.mkdir()
            image.save(attempt / "input.png", format="PNG")
            request = {"binding": binding, "model_dir": str(self.model_dir), "device": self.device,
                       "input_sha256": sha(attempt / "input.png")}
            publish(attempt / "request.json", request)
            env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
            command = [sys.executable, "-m", "btsl", "_call", "--request", str(attempt / "request.json")]
            with (attempt / "console.log").open("xb") as log:
                code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                       stdin=subprocess.DEVNULL, pass_fds=(self.lock_fd,))
            if code:
                raise RuntimeError("Model child failed (no Raw fill), exit=%d; %s" % (code, attempt / "console.log"))
            verify_ready(attempt)
        write_once(cache / "generation.json", (attempt / "generation.json").read_bytes())
        publish(cache / "READY.json", {"binding": binding, "attempt": attempt.name,
                "files": {"generation.json": sha(cache / "generation.json"),
                          str(attempt.relative_to(cache) / "READY.json"): sha(attempt / "READY.json")}})
        return read(cache / "generation.json")


def child(request_path):
    from PIL import Image
    from .model import predict
    request_path = Path(request_path).resolve()
    request, folder = read(request_path), request_path.parent
    if sha(folder / "input.png") != request["input_sha256"]:
        raise ValueError("Child image file changed")
    with Image.open(folder / "input.png") as original:
        image = original.convert("RGB")
    if digest(image.tobytes()) != request["binding"]["rgb_sha256"] or list(image.size) != request["binding"]["dimensions"]:
        raise ValueError("Prepared image binding changed")
    try:
        started = time.monotonic()
        result = predict(Path(request["model_dir"]), image, request["binding"]["task"], request["device"])
        result["elapsed_seconds"] = time.monotonic() - started
        publish(folder / "generation.json", result)
        publish(folder / "READY.json", {"binding": request["binding"], "files": {
            name: sha(folder / name) for name in ("request.json", "input.png", "generation.json")}})
    except BaseException as exc:
        publish(folder / "FAILURE.json", {"type": type(exc).__name__, "message": str(exc),
                "raw_fallback": False, "automatic_retry": False})
        raise
    finally:
        image.close()


def page(raw, image, image_sha, engine):
    from PIL import Image
    layout_image = image.resize((1036, 1036), Image.Resampling.BICUBIC)
    try:
        layout = engine.generate(layout_image, "layout")
    finally:
        layout_image.close()
    record = {"schema": SCHEMA, "status": "GENERATIONS_COMPLETE", "source_image_sha256": image_sha,
              "raw_markdown_sha256": digest(raw), "layout": layout, "tables": []}
    parsed = parse_layout(layout["decoded_text"], terminal_eos_present=layout["terminal_eos_present"], truncated=layout["truncated"])
    if not parsed["valid"]:
        final, receipt = assemble_record(raw, image_sha, record)
        return final, receipt, record
    if len(parsed["blocks"]) > 256:
        final, receipt = assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [], "TOO_MANY_LAYOUT_BLOCKS")
        return final, receipt, record
    failures = []
    for block in parsed["blocks"]:
        if block["type"] != "table":
            continue
        try:
            crop, geometry = crop_block(image, block)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        try:
            prepared, preparation = prepare_recognition_image(crop)
            try:
                generation = engine.generate(prepared, "table")
            finally:
                prepared.close()
        finally:
            crop.close()
        table = {**generation, "layout_index": block["index"], "crop": geometry, "preparation": preparation}
        record["tables"].append(table)
        parsed_table = parse_otsl(table["decoded_text"], terminal_eos_present=table["terminal_eos_present"], truncated=table["truncated"])
        if not parsed_table["valid"]:
            failures.append("INVALID_COMPLETED_OTSL: " + str(parsed_table["error"]))
    if failures:
        record["page_failures"] = failures
        final, receipt = assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [], "; ".join(failures))
        return final, receipt, record
    final, receipt = assemble_record(raw, image_sha, record)
    return final, receipt, record


def run(manifest, model_dir, output, device, resume=False, retry_failed=False):
    from PIL import Image
    from .model import verify_model, runtime, MANIFEST
    manifest, model_dir, output = (Path(p).resolve() for p in (manifest, model_dir, output))
    rows = validate(manifest)
    verify_model(model_dir)
    freeze = {"schema": "btsl-run/1", "manifest_sha256": sha(manifest), "pages": read(manifest)["pages"],
              "model_manifest_sha256": sha(MANIFEST), "sources": sources(), "device": device,
              "runtime": runtime(), "policy": "NAVIDC_NATIVE_TABLE_COLLECTION_WITH_MINERU_NONTABLE_V1",
              "gold_model_facing": 0, "score_claim": "UNSCORED_NEW_EXECUTION", "training_updates": 0}
    if output.exists() and not resume:
        raise ValueError("Existing run; use --resume to verify and reuse committed work")
    if resume and not (output / "FREEZE.json").is_file():
        raise ValueError("Resume requires an existing run freeze")
    output.mkdir(parents=True, exist_ok=True)
    with Lease(output / ".run.lock") as lock_fd:
        publish(output / "FREEZE.json", freeze)
        if (output / "READY.json").exists():
            return verify_ready(output)
        binding = sha(output / "FREEZE.json")
        engine = ProcessEngine(output, model_dir, device, binding, lock_fd, retry_failed)
        statuses = {}
        for i, row in enumerate(rows):
            folder = output / "page_receipts" / row["id"]
            folder.mkdir(parents=True, exist_ok=True)
            destination = output / "pages" / (row["id"] + ".md")
            if (folder / "READY.json").exists():
                ready = verify_ready(folder)
                if ready["run_binding"] != binding or sha(destination) != ready["prediction_sha256"]:
                    raise ValueError("Committed page binding changed")
                receipt = read(folder / "receipt.json")
            else:
                engine.page_id, engine.call_index = row["id"], 0
                raw = row["raw_path"].read_bytes()
                if digest(raw) != row["raw_sha256"] or sha(row["image_path"]) != row["image_sha256"]:
                    raise ValueError("Input changed after freeze")
                try:
                    with Image.open(row["image_path"]) as original:
                        image = original.convert("RGB")
                except (OSError, Image.UnidentifiedImageError) as exc:
                    final, receipt = assemble_page(raw, [], "PAGE_ABSTAIN_EXACT_RAW", [],
                                                   "SOURCE_IMAGE_DECODE_ERROR: " + str(exc))
                    record = {"status": "SOURCE_IMAGE_DECODE_ERROR", "source_image_sha256": row["image_sha256"]}
                else:
                    try:
                        final, receipt, record = page(raw, image, row["image_sha256"], engine)
                    finally:
                        image.close()
                publish(folder / "native.json", record)
                publish(folder / "receipt.json", receipt)
                write_once(destination, final)
                publish(folder / "READY.json", {"run_binding": binding, "prediction_sha256": sha(destination),
                        "files": {name: sha(folder / name) for name in ("native.json", "receipt.json")}})
            statuses[receipt["status"]] = statuses.get(receipt["status"], 0) + 1
            print("%d/%d %s %s" % (i + 1, len(rows), row["id"], receipt["status"]), flush=True)
        # No source or input mutation may be silently included in a completed run.
        if sources() != freeze["sources"]:
            raise ValueError("Implementation changed during execution")
        validate(manifest)
        files = {"FREEZE.json": binding}
        for base in (output / "pages", output / "page_receipts", output / "calls"):
            for p in base.rglob("*"):
                if p.is_file() and p.suffix in (".json", ".md", ".png"):
                    files[str(p.relative_to(output))] = sha(p)
        ready = {"schema": "btsl-complete-run/1", "pages": len(rows), "statuses": statuses,
                 "files": files, "scored": False, "gold_model_facing": 0}
        publish(output / "READY.json", ready)
        return ready
