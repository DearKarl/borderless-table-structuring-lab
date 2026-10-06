"""Bounded host-only fitting and one-time held audit for NJU-Scale-GBDT v0.1.

Identity fields are supplied by the verified outcomes contract, never inferred
from historical receipts. Fit/cal and held files must be physically separate.
Use fit_calibrate once per new output directory; failures are never auto-retried.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import pickle
import random
import statistics
import subprocess
import sys
import time

from .schema import ACTION_CODES, FEATURES
from .scale_policy import (BINDING_KEYS, FEATURE_DEFINITION, MAX_JSON, MODEL_FILE,
                           PARAMETERS, SEED, ScalePolicy, canonical, fallback_action,
                           finite, is_hash, load_policy, read_bytes, read_json,
                           sha256, valid_vector, validate_bindings, verified_bundle)

CPU_SECONDS = 1800
TERMINAL = {"completed", "model_failure"}
STATUSES = TERMINAL | {"invalid_measurement", "not_run"}


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    """Exclusive creation and fsync; a partial failed delivery is not reusable."""
    data = canonical(value)
    with Path(path).open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256(data)


def limit_deadline(deadline):
    now = time.monotonic()
    if deadline is None:
        return now + CPU_SECONDS
    if not finite(deadline) or deadline <= now:
        raise TimeoutError("Fit/audit deadline already expired or invalid")
    return min(deadline, now + CPU_SECONDS)


def remaining(deadline):
    seconds = deadline - time.monotonic()
    if seconds <= 0:
        raise TimeoutError("Shared fit/calibration deadline exhausted")
    return seconds


def load_outcomes(value):
    raw = (read_bytes(value, MAX_JSON) if isinstance(value, (str, os.PathLike)) else
           json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    if len(raw) > MAX_JSON:
        raise ValueError("Outcomes exceed bounded input size")
    data = json.loads(raw)
    # Cost instrumentation faults disable deployment, not the real fit. Preserve
    # the original input hash, but do not serialize NaN/Inf into a saved report.
    pages = data.get("pages") if isinstance(data, dict) else None
    for page in pages if isinstance(pages, list) else []:
        if isinstance(page, dict):
            prep = page.get("selection_preparation_seconds")
            if prep is not None and (not finite(prep) or prep < 0):
                page["selection_preparation_seconds"] = None
                page["selection_preparation_invalid_reason"] = "nonfinite_nonnumeric_or_negative"
    return data, sha256(raw)


def validate_outcomes(data, *, held=False):
    if not isinstance(data, dict) or data.get("schema") != "nju_public_outcomes_v1":
        raise ValueError("Unsupported outcomes schema")
    if (not is_hash(data.get("roster_sha256"))
            or data.get("score_version") != "PublicTranscriptEdit_v1"
            or data.get("feature_definition_id") != FEATURE_DEFINITION
            or data.get("feature_names") != list(FEATURES)):
        raise ValueError("Unbound roster/score/feature contract")
    validate_bindings(data.get("bindings"))
    if data.get("bank_status") not in (None, "bank_complete", "bank_partial"):
        raise ValueError("Unknown declared bank status")
    pages = data.get("pages")
    if not isinstance(pages, list) or not pages or len(pages) > (8 if held else 40):
        raise ValueError("Invalid outcome page count")
    ids, groups, group_sizes, counts = set(), {}, Counter(), Counter()
    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Invalid page record")
        pid, group, split = (page.get(k) for k in ("page_id", "source_group", "split"))
        if not all(isinstance(v, str) and 0 < len(v) <= 512 for v in (pid, group)) or pid in ids:
            raise ValueError("Missing/duplicate page or source group identity")
        ids.add(pid)
        if split not in ({"held_out"} if held else {"fit", "calibration"}):
            raise ValueError("Held leakage or invalid split in outcomes")
        if groups.setdefault(group, split) != split:
            raise ValueError("Source group crosses splits")
        group_sizes[group] += 1
        if group_sizes[group] > 2:
            raise ValueError("More than two pages per source group")
        counts[split] += 1
        n = page.get("reference_symbols")
        if type(n) is not int or not 32 <= n <= 65536:
            raise ValueError("Invalid canonical reference length")
        available = page.get("available_actions")
        if (not isinstance(available, list) or len(available) != len(set(available))
                or "A" not in available or not set(available) <= set(ACTION_CODES)):
            raise ValueError("Invalid available actions; native A is required")
        actions = page.get("actions")
        if not isinstance(actions, dict) or set(actions) != set(available):
            raise ValueError("Outcome actions differ from availability")
        for action, row in actions.items():
            if not isinstance(row, dict) or row.get("status") not in STATUSES:
                raise ValueError("Unknown action status")
            quality, status = row.get("quality"), row["status"]
            if status in TERMINAL:
                if not finite(quality) or not 0 <= quality <= 1:
                    raise ValueError("Terminal quality must be finite in [0,1]")
                if status == "model_failure" and quality != 0:
                    raise ValueError("Terminal model failure must retain quality zero")
            elif quality is not None:
                raise ValueError("Invalid/unrun outcome quality must be null, never zero")
            seconds = row.get("seconds")
            if seconds is not None and (not finite(seconds) or seconds < 0):
                raise ValueError("Invalid measured seconds")
            vector = row.get("features")
            if vector is not None and not valid_vector(action, vector):
                raise ValueError("Invalid feature vector/order/action code")
        flags = {r["features"][FEATURES.index("is_original_pdf")]
                 for r in actions.values() if r.get("features") is not None}
        if len(flags) > 1 or ("C" in available and flags and flags != {1}):
            raise ValueError("Inconsistent original-source feature flags")
    if held and counts != {"held_out": 8}:
        raise ValueError("Exactly eight held pages required")
    if not held and (counts["fit"] > 32 or counts["calibration"] > 8):
        raise ValueError("Approved fit/cal split caps exceeded")
    if not held and data.get("bank_status") == "bank_complete" and (
            counts != {"fit": 32, "calibration": 8}
            or any(r["status"] == "not_run" for p in pages for r in p["actions"].values())):
        raise ValueError("Declared complete bank has missing fit/cal pages or starts")
    return pages


def paired_pages(pages):
    return [p for p in pages if all(r["status"] in TERMINAL for r in p["actions"].values())]


def usable_features(page):
    return all(r.get("features") is not None
               and r["features"][FEATURES.index("glyph_reliability")] == 1
               for r in page["actions"].values())


def group_mean(pages, values):
    grouped = defaultdict(list)
    for page, value in zip(pages, values):
        if value is None:
            return None
        grouped[page["source_group"]].append(value)
    if not grouped:
        return None
    return statistics.mean(statistics.mean(v) for v in grouped.values())


def policy_metrics(pages, decisions, *, include_selection_preparation=False):
    rows = [p["actions"][a] for p, a in zip(pages, decisions)]
    warm = [r.get("seconds") for r in rows]
    preparation = [p.get("selection_preparation_seconds") for p in pages]
    preparation = [v if finite(v) and v >= 0 else None for v in preparation]
    total = ([s + prep if s is not None and prep is not None else None
              for s, prep in zip(warm, preparation)] if include_selection_preparation else warm)
    total = [v if finite(v) and v >= 0 else None for v in total]
    return dict(pages=len(pages), groups=len({p["source_group"] for p in pages}),
                quality=group_mean(pages, [r["quality"] for r in rows]),
                seconds=group_mean(pages, total), warm_pipeline_seconds=group_mean(pages, warm),
                selection_preparation_included=include_selection_preparation,
                selection_preparation_seconds=group_mean(pages, preparation) if include_selection_preparation else None,
                unavailable_selection_preparation_pages=sum(v is None for v in preparation) if include_selection_preparation else 0,
                terminal_failures=sum(r["status"] == "model_failure" for r in rows),
                observed_visual_tokens=None,
                cost_semantics=("cached_action_seconds_plus_candidate_selection_preparation" if include_selection_preparation
                                else "measured_cached_action_seconds"),
                decisions=list(decisions))


def fixed_decisions(pages, action):
    return [fallback_action(action, p["available_actions"]) for p in pages]


def strongest_fixed(pages):
    if not pages:
        raise ValueError("No paired fit scores for F*")
    reports = {a: policy_metrics(pages, fixed_decisions(pages, a)) for a in ACTION_CODES}
    best_quality = max(v["quality"] for v in reports.values())
    ties = [a for a in ACTION_CODES if best_quality - reports[a]["quality"] <= 1e-12]
    fixed = min(ties, key=lambda a: (reports[a]["seconds"] if reports[a]["seconds"] is not None else math.inf, a))
    return fixed, reports


def training_table(pages, fixed):
    """Only eligible paired pages enter X; host metadata stays in separate rows."""
    eligible = [p for p in paired_pages(pages) if usable_features(p)]
    group_sizes = Counter(p["source_group"] for p in eligible)
    if len(eligible) < 8 or len(group_sizes) < 4:
        raise ValueError("Fit minimum is eight paired feature-valid pages from four groups")
    vectors, targets, weights, records = [], [], [], []
    for p in eligible:
        baseline = p["actions"][fallback_action(fixed, p["available_actions"])]
        epsilon = 2 / p["reference_symbols"]
        for action in sorted(p["actions"]):
            row = p["actions"][action]
            raw = row["quality"] - baseline["quality"]
            gain = (0.0 if row["status"] == baseline["status"] == "completed"
                    and abs(raw) <= epsilon else raw)
            weight = 1 / (len(group_sizes) * group_sizes[p["source_group"]] * len(p["actions"]))
            vectors.append(row["features"])
            targets.append(gain)
            weights.append(weight)
            records.append(dict(page_id=p["page_id"], source_group=p["source_group"],
                                action=action, raw_gain=raw, training_gain=gain,
                                epsilon=epsilon, weight=weight, status=row["status"]))
    return eligible, vectors, targets, weights, records


def feature_map(page):
    return {a: row.get("features") for a, row in page["actions"].items()}


def calibrate(estimator, config, fit, calibration):
    median_epsilon = statistics.median(2 / p["reference_symbols"] for p in fit)
    margins = sorted({0.0, median_epsilon, 2 * median_epsilon})
    pages = paired_pages(calibration)
    baseline = policy_metrics(pages, fixed_decisions(pages, config["fixed_action"]))
    tolerance = group_mean(pages, [2 / p["reference_symbols"] for p in pages])
    reports, accepted = [], []
    for margin in margins:
        candidate = ScalePolicy(estimator, dict(config, margin=margin, learned_enabled=True), "calibration")
        choices = [candidate.predict(feature_map(p))["action"] for p in pages]
        metric = policy_metrics(pages, choices, include_selection_preparation=True)
        improvement = None if not pages else metric["quality"] - baseline["quality"]
        added_failures = sum(p["actions"][a]["status"] == "model_failure"
                             and p["actions"][b]["status"] != "model_failure"
                             for p, a, b in zip(pages, choices, baseline["decisions"]))
        passes = (len(pages) == 8 and improvement > tolerance and added_failures == 0
                  and metric["seconds"] is not None and baseline["seconds"] is not None
                  and metric["seconds"] <= 1.10 * baseline["seconds"])
        report = dict(margin=margin, metrics=metric, improvement=improvement,
                      tolerance=tolerance, added_terminal_failures=added_failures, accepted=passes)
        reports.append(report)
        if passes:
            accepted.append(report)
    winner = min(accepted, key=lambda r: (-r["metrics"]["quality"], r["metrics"]["seconds"], -r["margin"])) if accepted else None
    return dict(margin=winner["margin"] if winner else 0.0, learned_enabled=winner is not None,
                baseline=baseline, candidates=reports, median_fit_epsilon=median_epsilon,
                reason="calibration_gate_passed" if winner else "conservative_fixed_default")


def controls(pages, policy, glyph_threshold):
    """Cached exploratory comparisons; random action counts match within availability."""
    pages = paired_pages(pages)
    predictions = [policy.predict(feature_map(p)) for p in pages]
    choices = [p["action"] for p in predictions]
    fixed = {a: policy_metrics(pages, fixed_decisions(pages, a)) for a in ACTION_CODES}
    glyph_index = FEATURES.index("glyph_height")
    simple, oracle = [], []
    for p in pages:
        vector = p["actions"]["A"].get("features")
        simple.append("B" if vector is not None and glyph_threshold is not None
                      and vector[glyph_index] <= glyph_threshold and "B" in p["actions"] else "A")
        oracle.append(min(p["actions"], key=lambda a: (-p["actions"][a]["quality"],
                          p["actions"][a]["seconds"] if p["actions"][a]["seconds"] is not None else math.inf, a)))
    strata = defaultdict(list)
    for i, p in enumerate(pages):
        strata[tuple(sorted(p["available_actions"]))].append(i)
    random_reports = []
    for seed in range(SEED, SEED + 5):
        rng, shuffled = random.Random(seed), list(choices)
        for indices in strata.values():
            values = [choices[i] for i in indices]
            rng.shuffle(values)
            for i, value in zip(indices, values):
                shuffled[i] = value
        random_reports.append(dict(seed=seed, metrics=policy_metrics(pages, shuffled)))
    return dict(fixed=fixed, fixed_star=policy_metrics(pages, fixed_decisions(pages, policy.config["fixed_action"])),
                simple=policy_metrics(pages, simple), glyph_threshold=glyph_threshold,
                learned=policy_metrics(pages, choices, include_selection_preparation=policy.config["learned_enabled"]),
                raw_predictions=predictions,
                random_controls=random_reports, random_matching="exact action counts within availability strata; not matched cost",
                oracle=policy_metrics(pages, oracle), oracle_kind="unconstrained GT-aware cached quality upper bound",
                excluded_unpaired_pages="reported by caller", quality_kind="noisy_public_reference_agreement_not_official_Overall")


def child_environment():
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = repo + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def fresh_process_check(root, policy, pages, deadline, original_input_path=None):
    maps = [feature_map(p) for p in pages[:4]]
    if maps:
        maps.extend([{a: None for a in maps[0]}, {a: v for a, v in maps[0].items() if a != "C"}])
    expected = [policy.predict(row) for row in maps]
    denied = [str(root / "host")]
    if original_input_path is not None:
        denied.append(original_input_path)
    command = [sys.executable, "-B", "-m", "hybrid.v4_input_selector.scale_policy", str(root / "policy"),
               "--expected-manifest-sha256", policy.identity]
    for path in denied:
        command.extend(["--deny-read", path])
    completed = subprocess.run(command, input=canonical(dict(action_features=maps)),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=child_environment(), timeout=remaining(deadline), check=False)
    if completed.returncode:
        raise RuntimeError("Fresh-process load/predict failed: " + completed.stderr.decode(errors="replace")[-4000:])
    observed = json.loads(completed.stdout)
    if observed.get("denied_paths_verified") != len(denied) or observed.get("policy_identity") != policy.identity:
        raise ValueError("Fresh-process identity/isolation guard not verified")
    actual = observed.get("predictions", [])
    if len(actual) != len(expected):
        raise ValueError("Fresh-process prediction count mismatch")
    for before, after in zip(expected, actual):
        if any(before[k] != after[k] for k in ("action", "reason", "policy_identity")) or set(before["predicted_gains"]) != set(after["predicted_gains"]):
            raise ValueError("Fresh-process decision mismatch")
        if any(not math.isclose(v, after["predicted_gains"][a], rel_tol=1e-12, abs_tol=1e-12)
               for a, v in before["predicted_gains"].items()):
            raise ValueError("Fresh-process raw prediction mismatch")
    return dict(status="artifact_load_passed", cases=len(maps), subprocess_returncode=0,
                decision_parity=True, prediction_tolerance=1e-12, host_store_read_denied=True,
                original_outcome_file_read_denied=original_input_path is not None,
                isolation="Python audit guard; host-only import denied; feature-only stdin; not an OS sandbox")


def _fit_impl(request, root, deadline):
    """Runs only in the bounded child; exactly one genuine estimator.fit call."""
    attempt = read_json(root / "ATTEMPT.json")
    if sha256(read_bytes(request, MAX_JSON)) != attempt.get("canonical_request_sha256"):
        raise ValueError("Fit child request differs from exclusive parent attempt")
    write_json(root / "FIT_STARTED.claim.json", dict(started_utc=utcnow(), retry_allowed=False))
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    started, cpu_started = time.monotonic(), time.process_time()
    data, outcome_hash = load_outcomes(request)
    pages = validate_outcomes(data)
    fit_all = [p for p in pages if p["split"] == "fit"]
    cal_all = [p for p in pages if p["split"] == "calibration"]
    fixed, fixed_reports = strongest_fixed(paired_pages(fit_all))
    fit, x, y, weights, records = training_table(fit_all, fixed)
    config = dict(schema="nju_scale_policy_v1", model_name="NJU-Scale-GBDT v0.1",
                  parameters=PARAMETERS, feature_names=list(FEATURES), feature_definition_id=FEATURE_DEFINITION,
                  sklearn_version=sklearn.__version__, fixed_action=fixed, margin=0.0, learned_enabled=False,
                  bindings=validate_bindings(data["bindings"]), roster_sha256=data["roster_sha256"],
                  outcome_sha256=outcome_hash, score_version=data["score_version"],
                  synthetic_fixture=data.get("synthetic_fixture") is True,
                  recipe_semantics={"A": "native 200-DPI PDF render / native raster",
                                    "B": "300-DPI PDF / 1.5x bicubic raster, native 3500 cap",
                                    "C": "genuine PDF B render then INTER_AREA to exact A dimensions; unavailable for raster"})
    remaining(deadline)
    estimator = HistGradientBoostingRegressor(**PARAMETERS)
    fit_started = time.monotonic()
    with threadpool_limits(limits=1):
        estimator.fit(x, y, sample_weight=weights)
        remaining(deadline)
        fit_seconds = time.monotonic() - fit_started
        calibration = calibrate(estimator, config, fit, cal_all)
        config.update(margin=calibration["margin"], learned_enabled=calibration["learned_enabled"])
        glyphs = sorted(p["actions"]["A"]["features"][FEATURES.index("glyph_height")] for p in fit)
        threshold = glyphs[math.floor((len(glyphs) - 1) * .25)]
        frozen_host = dict(page_ids=[p["page_id"] for p in pages], source_groups=sorted({p["source_group"] for p in pages}),
                           roster_sha256=data["roster_sha256"], bindings=validate_bindings(data["bindings"]),
                           glyph_threshold=threshold, outcome_sha256=outcome_hash)
        host_hash = write_json(root / "host" / "frozen.json", frozen_host)
        write_json(root / "host" / "targets.json", records)
        config_hash = write_json(root / "policy" / "config.json", config)
        model_bytes = pickle.dumps(estimator, protocol=pickle.HIGHEST_PROTOCOL)
        with (root / "policy" / MODEL_FILE).open("xb") as stream:
            stream.write(model_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        identity = write_json(root / "policy" / "manifest.json", dict(schema="nju_scale_bundle_v1", trusted_local_only=True,
                              files={MODEL_FILE: sha256(model_bytes), "config.json": config_hash}, host_state_sha256=host_hash))
        policy = load_policy(root, expected_manifest_sha256=identity)
        comparison = {"fit": controls(fit_all, policy, threshold), "calibration": controls(cal_all, policy, threshold)}
        write_json(root / "host" / "comparison.json", comparison)
        write_json(root / "host" / "calibration.json", calibration)
        load_check = fresh_process_check(root, policy, fit, deadline, attempt.get("original_input_path"))
    write_json(root / "host" / "LOAD_CHECK.json", load_check)
    remaining(deadline)
    receipt = dict(schema="nju_scale_fit_receipt_v1", status="fit_completed", actual_fit_calls=1,
                   synthetic_fixture=config["synthetic_fixture"], public_data_fit=not config["synthetic_fixture"],
                   fit_pages=len(fit), fit_groups=len({p["source_group"] for p in fit}), fit_rows=len(y),
                   submitted_fit_pages=len(fit_all), excluded_fit_pages=len(fit_all) - len(fit),
                   paired_score_fit_pages=len(paired_pages(fit_all)), calibration_pages=len(cal_all),
                   valid_calibration_pages=len(paired_pages(cal_all)),
                   training_data_status="full_fit_cal" if len(fit) == 32 and len(paired_pages(cal_all)) == 8 else "partial_fit_cal",
                   bank_status=data.get("bank_status", "not_verified_by_trainer"),
                   bank_status_authority="host declaration; held outcomes not inspected by trainer",
                   constant_targets=len(set(y)) == 1, all_zero_targets=all(v == 0 for v in y),
                   weight_sum=sum(weights), feature_target_weight_sha256=sha256(canonical(dict(X=x, y=y, weights=weights))),
                   outcome_sha256=outcome_hash, manifest_sha256=identity, model_sha256=sha256(model_bytes),
                   config_sha256=config_hash, parameters=PARAMETERS, sklearn_version=sklearn.__version__,
                   source_sha256={name: sha256(Path(__file__).with_name(name).read_bytes())
                                  for name in ("public_fit.py", "scale_policy.py")},
                   bindings=config["bindings"], fixed_action=fixed, fixed_fit_metrics=fixed_reports,
                   margin=config["margin"], learned_enabled=config["learned_enabled"],
                   benefit_status="benefit_observed" if config["learned_enabled"] else "inconclusive_or_no_benefit",
                   deployment_recommended="not_assessed", held_reference_consumed=False,
                   fit_seconds=fit_seconds, child_wall_seconds=time.monotonic() - started,
                   child_cpu_seconds=time.process_time() - cpu_started, completed_utc=utcnow(), load_check=load_check)
    write_json(root / "FIT_RECEIPT.json", receipt)
    return receipt


def fit_calibrate(outcomes, output_dir, *, deadline=None):
    started = time.monotonic()
    deadline = limit_deadline(deadline)
    data, original_hash = load_outcomes(outcomes)
    pages = validate_outcomes(data)
    if data.get("synthetic_fixture") is not True and data.get("smoke_passed") is not True:
        raise ValueError("Public-data fitting requires a functioning native smoke receipt binding")
    # Validate minima before creating any output or starting a fitting process.
    fit = [p for p in pages if p["split"] == "fit"]
    fixed, _ = strongest_fixed(paired_pages(fit))
    training_table(fit, fixed)
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "host").mkdir()
    (root / "policy").mkdir()
    request = root / "host" / "outcomes.json"
    request_hash = write_json(request, data)
    write_json(root / "ATTEMPT.json", dict(started_utc=utcnow(), original_input_sha256=original_hash,
               canonical_request_sha256=request_hash,
               original_input_path=str(Path(outcomes).resolve()) if isinstance(outcomes, (str, os.PathLike)) else None,
               wall_limit_seconds=remaining(deadline), synthetic_fixture=data.get("synthetic_fixture") is True))
    command = [sys.executable, "-B", "-m", "hybrid.v4_input_selector.public_fit", "_fit",
               str(request), str(root), "--seconds", str(remaining(deadline))]
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=remaining(deadline), env=child_environment(), check=False)
        (root / "host" / "child_stdout.txt").write_bytes(completed.stdout)
        (root / "host" / "child_stderr.txt").write_bytes(completed.stderr)
        if completed.returncode:
            raise RuntimeError("Fit child failed (" + str(completed.returncode) + "): " + completed.stderr.decode(errors="replace")[-4000:])
        remaining(deadline)
        receipt = read_json(root / "FIT_RECEIPT.json")
        write_json(root / "DELIVERY.json", dict(status="fit_delivery_completed",
                   manifest_sha256=receipt["manifest_sha256"], completed_utc=utcnow(),
                   consumed_budget_seconds=time.monotonic()-started,
                   budget_semantics="single-thread wall time, conservatively charged against shared 1800 CPU-second allowance"))
        return receipt
    except BaseException as exc:
        write_json(root / "FAILED.json", dict(status="fit_delivery_failed", error=type(exc).__name__,
                   message=str(exc), recorded_utc=utcnow(), retry_allowed=False))
        raise


def evaluate_held_once(outcomes, model_dir, output_dir, *, deadline=None):
    deadline = limit_deadline(deadline)
    if not isinstance(outcomes, (str, os.PathLike)):
        raise ValueError("Held input must be a sealed file path, not an already-read mapping")
    root = Path(model_dir).resolve()
    if root.name == "policy" and (root / "manifest.json").is_file():
        root = root.parent
    _, identity, config, _ = verified_bundle(root)
    fit_receipt = read_json(root / "FIT_RECEIPT.json")
    if fit_receipt.get("status") != "fit_completed" or fit_receipt.get("manifest_sha256") != identity:
        raise ValueError("A complete frozen fit/load delivery is required before held audit")
    delivery = read_json(root / "DELIVERY.json")
    consumed = delivery.get("consumed_budget_seconds")
    if (delivery.get("status") != "fit_delivery_completed" or delivery.get("manifest_sha256") != identity
            or not finite(consumed) or consumed < 0):
        raise ValueError("Missing frozen fit delivery/budget accounting")
    deadline = min(deadline, time.monotonic() + max(0, CPU_SECONDS-consumed))
    remaining(deadline)
    output = Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Held output must be separate from immutable model directory")
    if output.exists():
        raise FileExistsError("Held output must be new")
    # Permanent exclusive claim precedes all held path stat/open/read operations.
    write_json(root / "HELD_ONCE.claim.json", dict(schema="nju_held_once_claim_v1", manifest_sha256=identity,
               claimed_utc=utcnow(), output_dir=str(output), retry_allowed=False))
    output.mkdir(parents=True, exist_ok=False)
    try:
        remaining(deadline)
        frozen_raw = read_bytes(root / "host" / "frozen.json", MAX_JSON)
        manifest = read_json(root / "policy" / "manifest.json")
        if sha256(frozen_raw) != manifest.get("host_state_sha256"):
            raise ValueError("Frozen fit/cal separation metadata hash mismatch")
        frozen = json.loads(frozen_raw)
        data, held_hash = load_outcomes(outcomes)
        pages = validate_outcomes(data, held=True)
        if data["roster_sha256"] != config["roster_sha256"] or validate_bindings(data["bindings"]) != config["bindings"]:
            raise ValueError("Held roster/model/source/score identity mismatch")
        if {p["page_id"] for p in pages} & set(frozen["page_ids"]) or {p["source_group"] for p in pages} & set(frozen["source_groups"]):
            raise ValueError("Held page/source-group overlaps fit/cal")
        policy = load_policy(root, expected_manifest_sha256=identity)
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            report = controls(pages, policy, frozen["glyph_threshold"])
        remaining(deadline)
        verified_bundle(root, expected_manifest_sha256=identity)
        result = dict(schema="nju_held_audit_v1", status="held_audit_completed" if len(paired_pages(pages)) == 8 else "held_audit_inconclusive",
                      held_pages=8,
                      valid_paired_pages=len(paired_pages(pages)), held_outcome_sha256=held_hash,
                      manifest_sha256=identity, model_unchanged=True, model_refitted=False,
                      calibration_updated=False, comparison=report, completed_utc=utcnow(),
                      interpretation="one-time development comparison; not official OmniDocBench evaluation")
        write_json(output / "HELD_REPORT.json", result)
        return result
    except BaseException as exc:
        write_json(output / "FAILED.json", dict(status="held_audit_failed_claim_consumed", error=type(exc).__name__,
                   message=str(exc), retry_allowed=False, recorded_utc=utcnow()))
        raise
    finally:
        # Detection only; never overwrite or restore another writer's changes.
        verified_bundle(root, expected_manifest_sha256=identity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("fit", "held", "_fit"))
    parser.add_argument("outcomes")
    parser.add_argument("output")
    parser.add_argument("--model-dir")
    parser.add_argument("--seconds", type=float, default=CPU_SECONDS)
    args = parser.parse_args()
    if not finite(args.seconds) or not 0 < args.seconds <= CPU_SECONDS:
        parser.error("seconds must be in (0,1800]")
    deadline = time.monotonic() + args.seconds
    if args.operation == "fit":
        result = fit_calibrate(args.outcomes, args.output, deadline=deadline)
    elif args.operation == "held":
        if not args.model_dir:
            parser.error("held requires --model-dir")
        result = evaluate_held_once(args.outcomes, args.model_dir, args.output, deadline=deadline)
    else:
        result = _fit_impl(args.outcomes, Path(args.output), deadline)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
