"""Caller-owned load-once Tele workers. No GPU admission or remote activation.

Production run returns a base-compatible result plus a selection receipt. For
calibration the caller supplies sixteen preregistered arm items; each run makes
one native call, observing the same candidate boundary as deployment.
"""
import json
from pathlib import Path
import time
from hybrid.v4_input_selector.scale_policy import ScalePolicy
from hybrid.v4_input_selector.selector_input import (action_features, candidate_boundary,
    count_calls, observed_predict, select_and_run, verify_selected_render)
from hybrid.v4_input_selector.tele_worker import PageWorker
from hybrid.v4_input_selector.worker_contract import atomic_json, bound_file, file_sha
from .contracts import MARGINS, MODEL_SHA, ORIGINAL_MANIFEST_SHA
from .native_export import export_middle


class SelectedWorker(PageWorker):
    def __init__(self, native, *, policy):
        # Entry supplies the once-constructed, certificate-validated policy.
        if policy.config['learned_enabled'] is not True or policy.deployment_binding['model_sha256'] != MODEL_SHA:
            raise ValueError('Verified adaptive policy required')
        self.policy = policy
        if getattr(native, 'selector_rounding_evidence_sha256', None) != self.policy.deployment_binding['rounding_evidence_sha256']:
            raise ValueError('Rounding evidence differs from calibrated boundary')
        super().__init__(native)

    def run(self, item, input_root, output_root, deadline):
        if item.get('action') != 'policy_selected':
            raise ValueError('Only policy_selected items are accepted')
        receipt = select_and_run(self, item, input_root, output_root, self.policy, deadline=deadline)
        if receipt['result']['status']=='completed':export_middle(Path(output_root)/item['item_id'],item['item_id'])
        # Preserve the frozen supervisor's exact seven-key wire contract.
        # The additional receipt lives at the deterministic _selected path.
        return receipt['result']


class CalibrationWorker(PageWorker):
    """Unactivated collection entry. Input projection must already be hash frozen.

    Candidate preparation and three frozen-margin predictions are observed for
    each arm. The final gate charges that arm's real preparation + that margin's
    predict time + its actual recognition; it does not infer prep from renders.
    """
    def __init__(self, native, items, *, policy, rounding_evidence_sha256=None):
        if policy.config['learned_enabled'] is not False or policy.identity != ORIGINAL_MANIFEST_SHA:
            raise ValueError('Verified original calibration policy required')
        self.policy = policy
        if getattr(native, 'selector_rounding_evidence_sha256', None) != rounding_evidence_sha256:
            raise ValueError('Rounding evidence differs from calibration plan')
        if len(items) != 16 or len({i['item_id'] for i in items}) != 16:
            raise ValueError('Exactly sixteen frozen calibration items required')
        groups = {}
        for item in items:
            if item['source']['source_type'] != 'raster' or item['action'] not in ('A', 'B'):
                raise ValueError('Raster A/B only')
            groups.setdefault(item['page_id'], []).append(item['action'])
        if len(groups) != 8 or any(sorted(a) != ['A', 'B'] for a in groups.values()):
            raise ValueError('Eight frozen raster pairs required')
        for pid in groups:
            pair=[i for i in items if i['page_id']==pid]
            if any(pair[0][key]!=pair[1][key] for key in ('file','input_sha256','source')):
                raise ValueError('Calibration pair sources differ')
        self.items = {i['item_id']: i for i in items}
        self.claimed = set()
        super().__init__(native)

    def run(self, item, input_root, output_root, deadline):
        if item != self.items.get(item['item_id']) or item['item_id'] in self.claimed:
            raise ValueError('Unknown or repeated calibration start')
        self.claimed.add(item['item_id'])
        selection_started = time.monotonic()
        features, prep = action_features(self.native, bound_file(input_root, item['file']), item, deadline=deadline)
        predictions = []
        for margin in MARGINS:
            probe = ScalePolicy(self.policy.estimator, dict(self.policy.config, learned_enabled=True, margin=margin),
                                self.policy.identity)
            decision, timing = observed_predict(probe, features)
            predictions.append(dict(margin=margin, decision=decision, timing=timing))
        record = dict(schema='v4_calibration_predecision_v1', item=item, model_sha256=MODEL_SHA,
            original_manifest_sha256=self.policy.identity, preparation=prep, features=features,
            predictions=predictions, original_deadline=deadline, diagnostic_only=True,
            selected_for_recognition=item['action'])
        path = bound_file(output_root, '_calibration/' + item['item_id'] + '.json')
        if path.exists():
            raise FileExistsError('Calibration decision already exists')
        record_sha = atomic_json(path, record)
        selection_seconds = time.monotonic()-selection_started
        if time.monotonic() >= deadline:
            raise TimeoutError('Calibration preparation exhausted original item deadline')
        start = time.monotonic()
        with candidate_boundary(self.native):
            with count_calls(self.native.analyze, 'doc_analyze') as calls:
                result = PageWorker.run(self, item, input_root, output_root, deadline)
        seconds = time.monotonic()-start
        audit_path = bound_file(output_root, result['audit_file'])
        if file_sha(audit_path) != result['audit_sha256']:
            raise ValueError('Calibration audit changed')
        audit = json.loads(audit_path.read_bytes())
        if audit.get('render') is not None:
            verify_selected_render(prep['actions'][item['action']]['render'], audit['render'])
        if result['status'] == 'completed' and (calls['calls'] != 1 or audit.get('render') is None):
            raise ValueError('Invalid observed native call count')
        if calls['calls'] > 1:
            raise ValueError('Repeated native call')
        if result['status']=='completed':export_middle(Path(output_root)/item['item_id'],item['item_id'])
        seconds=time.monotonic()-start
        if time.monotonic()>=deadline:raise TimeoutError('Native export exhausted original item deadline')
        receipt = dict(schema='v4_calibration_result_v1', item_id=item['item_id'], result=result,
            predecision_file=path.relative_to(Path(output_root).resolve()).as_posix(), predecision_sha256=record_sha,
            recognition_seconds=seconds, native_page_pipeline_calls=calls['calls'],
            selection_seconds=selection_seconds,
            # Charge all preparation/persistence overhead; subtract only other
            # margins' observed prediction time from each counterfactual route.
            selection_nonprediction_seconds=selection_seconds-sum(p['timing']['seconds'] for p in predictions))
        receipt_sha = atomic_json(bound_file(output_root, '_calibration_results/' + item['item_id'] + '.json'), receipt)
        return result


def summarize(records):
    counts = dict(pages=0, learned_decisions=0, low_gain_fallbacks=0, feature_fallbacks=0,
                  fixed_policy_defaults=0, pages_with_candidate_aliases=0, estimator_predict_calls=0,
                  native_page_pipeline_calls=0)
    for row in records:
        counts['pages'] += 1
        reason = row['selection']['reason']
        key = {'learned_gain':'learned_decisions', 'below_margin':'low_gain_fallbacks',
               'fixed_policy_default':'fixed_policy_defaults'}.get(reason, 'feature_fallbacks')
        counts[key] += 1
        counts['pages_with_candidate_aliases'] += bool(row['preparation']['candidate_aliases'])
        counts['estimator_predict_calls'] += row['prediction']['estimator_predict_calls']
        counts['native_page_pipeline_calls'] += row['native_page_pipeline_calls']
    counts['observed_adaptive_decisions'] = counts['learned_decisions'] > 0
    return counts
