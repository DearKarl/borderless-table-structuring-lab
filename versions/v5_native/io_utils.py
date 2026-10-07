"""Atomic receipts and content identities shared by worker and supervisor."""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with open(temp, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def atomic_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require_common_runtime(runtime, expected):
    for key, value in expected.items():
        actual_json = json.dumps(runtime[key], sort_keys=True, default=str)
        expected_json = json.dumps(value, sort_keys=True, default=str)
        if actual_json != expected_json:
            raise RuntimeError("Runtime changed after the smoke freeze: " + key)


def process_identity(pid=None):
    pid = os.getpid() if pid is None else pid
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "start_ticks": int(stat[19]), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


class PageDeadline(BaseException):
    """A deadline must escape ordinary per-region exception handling."""


def commit_stage(folder, stage, middle, markdown):
    folder = Path(folder)
    stage_dir = folder / "stages" / stage
    stage_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(stage_dir / "middle.json", middle)
    atomic_bytes(stage_dir / "prediction.md", markdown.encode())
    receipt = {"stage": stage, "completed_at": utc(), "monotonic": time.monotonic(),
               "middle_sha256": digest(stage_dir / "middle.json"),
               "markdown_sha256": digest(stage_dir / "prediction.md")}
    atomic_json(stage_dir / "COMPLETE.json", receipt)
    atomic_json(folder / "LAST_VALID.json", receipt)
    return receipt


def finalize_page(folder, output, page_id, record):
    folder, output = Path(folder), Path(output)
    last = folder / "LAST_VALID.json"
    data = b""
    if last.is_file():
        valid = read_json(last)
        stage_dir = folder / "stages" / valid["stage"]
        if digest(stage_dir / "prediction.md") != valid["markdown_sha256"] or digest(stage_dir / "middle.json") != valid["middle_sha256"]:
            raise RuntimeError("Atomic stage identity mismatch")
        data = (stage_dir / "prediction.md").read_bytes()
        record["retained_stage"] = valid["stage"]
    else:
        record["retained_stage"] = None
    atomic_bytes(output / "markdown" / f"{page_id}.md", data)
    record.update(page_id=page_id, prediction_bytes=len(data), prediction_sha256=hashlib.sha256(data).hexdigest(),
                  completed_at=utc(), empty_prediction=not bool(data))
    atomic_json(output / "receipts" / f"{page_id}.json", record)
    return record


def finalize_outputs(folder,output,identifier,record,versions):
    """Pure-stdlib recovery for shared workers, including host-side hard stops."""
    folder,output=Path(folder),Path(output)
    receipts={}
    for version in versions:
        receipts[version]=finalize_page(folder/'arms'/version,output/'arms'/version,identifier,
                                       {**record,'version':version,'shared_native':True})
    record.update(arm_receipts={k:{x:v[x] for x in ('prediction_sha256','empty_prediction','retained_stage')}
                               for k,v in receipts.items()},render_dpi=200,render_dpi_changed=False)
    return finalize_page(folder,output,identifier,record)
