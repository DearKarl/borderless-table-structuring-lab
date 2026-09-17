"""Isolated pinned Official-protocol evaluation; Gold never enters inference."""
from pathlib import Path
import shutil
import subprocess

from .io import publish, read, sha
from .pipeline import verify_ready

HISTORICAL_IMAGE = "sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617"
OFFICIAL_GOLD = "a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496"


def config(prediction, gold):
    return {"end2end_eval": {
        "metrics": {"table": {"metric": ["TEDS", "Edit_dist"], "teds_workers": 4}},
        "dataset": {"dataset_name": "end2end_dataset", "ground_truth": {"data_path": gold},
                    "prediction": {"data_path": prediction}, "match_method": "quick_match",
                    "match_workers": 4, "quick_match_truncated_timeout_sec": 300,
                    "match_timeout_sec": 420, "timeout_fallback_max_chunk_span": 10,
                    "timeout_fallback_order_penalty": 0.10}}}


def verify_source(source):
    manifest = read(source / "SOURCE_MANIFEST.json")
    if manifest["commit"] != "147cd5ac9472002f5751221d390bf00abdbc0d2f":
        raise ValueError("Wrong evaluator commit")
    for name, expected in manifest["files"].items():
        if sha(source / name) != expected:
            raise ValueError("Frozen evaluator source changed: " + name)
    return manifest


def evaluate(run, gold, output, docker_image, dataset="official", expected_pages=1651,
             baseline=None, execute=False, evaluator_source=None):
    run, gold, output = (Path(p).resolve() for p in (run, gold, output))
    ready = verify_ready(run)
    expected = {Path(n).name for n in ready["files"] if n.startswith("pages/") and n.endswith(".md")}
    present = {p.name for p in (run / "pages").iterdir()}
    if ready["pages"] != expected_pages or len(expected) != expected_pages or expected != present:
        raise ValueError("Prediction page completeness failed")
    gold_sha = sha(gold)  # Opaque integrity hash only. Isolated evaluator parses answers.
    if dataset == "official" and (expected_pages != 1651 or gold_sha != OFFICIAL_GOLD):
        raise ValueError("This is not the exact frozen Official input; explicitly select --dataset custom for another dataset")
    source = (Path(evaluator_source).resolve() if evaluator_source else
              Path(__file__).resolve().parents[1] / "evaluation/omnidocbench")
    manifest = verify_source(source)
    arms = {"candidate": run / "pages"}
    if baseline:
        arms["baseline"] = Path(baseline).resolve()
        if {p.name for p in arms["baseline"].iterdir()} != expected:
            raise ValueError("Baseline and candidate page sets differ")
    plan = {"schema": "btsl-evaluation-plan/1", "dataset": dataset, "pages": expected_pages,
            "gold_sha256": gold_sha, "inference_ready_sha256": sha(run / "READY.json"),
            "evaluator_source": manifest, "docker_image_requested": docker_image,
            "historical_docker_image": HISTORICAL_IMAGE, "public_leaderboard_submission": False,
            "configs": {a: config("/data_md/" + a, "/gt/OmniDocBench.json") for a in arms}}
    if not execute:
        return {"status": "PLAN_ONLY_NO_EVALUATOR_CALL", "plan": plan}
    if output.exists():
        raise ValueError("Evaluation directory exists; never overwrite or silently rerun scoring")
    image_info = subprocess.check_output(["docker", "image", "inspect", docker_image], text=True)
    import json
    image_info = json.loads(image_info)[0]
    plan["actual_image_id"] = image_info["Id"]
    plan["historical_runtime_identity"] = image_info["Id"] == HISTORICAL_IMAGE
    output.mkdir(parents=True)
    shutil.copytree(source, output / "frozen_evaluator")
    verify_source(output / "frozen_evaluator")
    # The parent /workspace mount is read-only. Its nested writable result
    # mountpoint must exist before Docker starts; Docker cannot create it there.
    (output / "frozen_evaluator/result").mkdir(exist_ok=True)
    (output / "results").mkdir()
    for arm, predictions in arms.items():
        shutil.copytree(predictions, output / "inputs" / arm)
        publish(output / "configs" / (arm + ".yaml"), plan["configs"][arm])
    plan["prediction_files"] = {str(p.relative_to(output)): sha(p) for p in sorted((output / "inputs").rglob("*.md"))}
    command = ["docker", "run", "--rm", "--network", "none", "--platform", "linux/amd64",
               "--entrypoint", "bash", "-e", "CUDA_VISIBLE_DEVICES=", "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-e", "PYTHONNOUSERSITE=1", "-e", "PYTHONPATH=", "-e", "HF_HUB_OFFLINE=1",
               "-e", "HF_DATASETS_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1"]
    mounts = [(output / "frozen_evaluator", "/workspace", "ro"),
              (output / "results", "/workspace/result", "rw"), (gold, "/gt/OmniDocBench.json", "ro"),
              (output / "inputs", "/data_md", "ro"), (output / "configs", "/run_configs", "ro")]
    for host, guest, mode in mounts:
        if ":" in str(host) or "\n" in str(host):
            raise ValueError("Unsupported Docker mount path")
        command += ["-v", f"{host}:{guest}:{mode}"]
    script = "set -euo pipefail; cd /workspace; " + "; ".join(
        f"python pdf_validation.py --config /run_configs/{a}.yaml > /workspace/result/{a}.console.log 2>&1" for a in arms)
    command += [image_info["Id"], "-lc", script]
    plan["command"] = command
    publish(output / "FREEZE.json", plan)
    with (output / "console.log").open("xb") as log:
        code = subprocess.call(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    if code:
        publish(output / "FAILURE.json", {"exit_code": code, "score_claim": False})
        raise RuntimeError("Evaluator failed; inspect preserved logs, do not treat as a result")
    verify_source(output / "frozen_evaluator")
    if sha(gold) != gold_sha or any(sha(output / n) != h for n, h in plan["prediction_files"].items()):
        raise ValueError("Evaluation inputs changed")
    aggregate = {}
    for arm in arms:
        matches = [p for p in (output / "results").rglob("*metric_result.json") if arm in p.name]
        if len(matches) != 1:
            raise ValueError("Ambiguous aggregate file: " + arm)
        result = read(matches[0])
        if result["match_debug"]["page_count"] != expected_pages:
            raise ValueError("Evaluator did not process every page")
        table = result["table"]["page"]
        aggregate[arm] = {"full_table_teds": table["TEDS"]["ALL"] * 100,
                          "structure_teds": table["TEDS_structure_only"]["ALL"] * 100,
                          "metric_source_sha256": sha(matches[0])}
    summary = {"dataset": dataset, "pages": expected_pages, "arms": aggregate,
               "public_leaderboard_submission": False, "historical_runtime_identity": plan["historical_runtime_identity"]}
    publish(output / "SUMMARY.json", summary)
    publish(output / "READY.json", {"files": {"FREEZE.json": sha(output / "FREEZE.json"),
            "SUMMARY.json": sha(output / "SUMMARY.json")}, "scored": True})
    return summary
