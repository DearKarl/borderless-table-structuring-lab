"""Post-evaluation descriptive analysis; never imported by inference workers.

Run only after all five evaluations finish. Ground-truth categories establish
the same eligible page set for paired component summaries, never patch choices.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from .io_utils import atomic_json, digest, read_json, utc


def page_name(img_id):
    img_id = str(img_id)
    return img_id if img_id.endswith(('.jpg', '.png')) else '_'.join(img_id.split('_')[:-1])


def page_metrics(samples, metric, gt_pages=()):
    values = defaultdict(list)
    for sample in samples:
        if metric not in sample.get('metric', {}):
            continue
        score = float(sample['metric'][metric])
        if not math.isfinite(score):
            raise ValueError('Non-finite sample metric')
        gt = sample.get('norm_gt') or sample.get('gt', '')
        pred = sample.get('norm_pred') or sample.get('pred', '')
        values[page_name(sample['img_id'])].append((score, max(len(gt), len(pred))))
    output = {}
    for page, rows in values.items():
        if metric == 'Edit_dist':
            weight = sum(row[1] for row in rows)
            output[page] = sum(score * n for score, n in rows) / weight if weight else 0.0
        else:
            output[page] = statistics.mean(score for score, _ in rows)
    if metric != 'Edit_dist':
        for page in gt_pages:
            output.setdefault(page, 0.0)
    return output


def paired(before, after, pages, higher_better):
    missing = sorted(set(pages) - (before.keys() & after.keys()))
    if missing:
        return {'status': 'incomplete_numeric_coverage', 'eligible_pages': len(pages),
                'missing_pages': len(missing), 'reason': 'No successful-only subset is substituted.'}
    raw = [after[page] - before[page] for page in sorted(pages)]
    oriented = raw if higher_better else [-value for value in raw]
    return {'status': 'complete', 'eligible_pages': len(pages), 'improved_pages': sum(x > 1e-12 for x in oriented),
            'regressed_pages': sum(x < -1e-12 for x in oriented), 'tied_pages': sum(abs(x) <= 1e-12 for x in oriented),
            'mean_raw_delta': statistics.mean(raw) if raw else None,
            'mean_quality_delta': statistics.mean(oriented) if oriented else None,
            'before_mean': statistics.mean(before[page] for page in pages) if pages else None,
            'after_mean': statistics.mean(after[page] for page in pages) if pages else None,
            'interpretation': 'Descriptive paired page comparison on the frozen eligible set; no significance claim.'}


def stage_summary(output):
    counts = defaultdict(Counter)
    pages_with_events = defaultdict(set)
    request_seconds = defaultdict(float)
    for file in (output / 'pages').glob('*/events.jsonl'):
        page_id = file.parent.name
        for line in file.read_text(encoding='utf-8').splitlines():
            event = json.loads(line)
            stage = event['stage']
            counts[stage][event['event']] += 1
            if event['event'] in ('region', 'guard_attempt'):
                pages_with_events[stage].add(page_id)
                counts[stage]['accepted_attempts'] += bool(event.get('accepted'))
                counts[stage]['changed_attempts'] += bool(event.get('changed'))
                counts[stage]['rejected_attempts'] += not bool(event.get('accepted'))
                counts[stage]['invalid_to_valid_attempts'] += bool(event.get('accepted') and event.get('before_reason'))
                if event['event'] == 'region':
                    counts[stage]['observed_completed_regions'] += 1
            if event['event'] == 'stage_coverage' and stage == 'guard':
                counts[stage]['guard_eligible_regions'] += event['eligible']
            if event['event'] == 'request_batch_complete':
                request_seconds[stage] += event['seconds']
                counts[stage]['completed_model_requests'] += event['requests']
    return {stage: {'counts': dict(values), 'pages_with_regional_attempts': len(pages_with_events[stage]),
                    'completed_request_seconds': request_seconds[stage],
                    'validity_note': 'Invalid-to-valid counts use syntax/runaway checks and do not establish accuracy gains.'}
            for stage, values in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round-root', required=True)
    ap.add_argument('--ground-truth', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    root, output = Path(args.round_root), Path(args.output)
    manifest = read_json(root / 'MANIFEST.json')
    terminal = read_json(root / 'EVALUATION_COMPLETE.json')
    if not terminal['success'] or len(terminal['arms']) != 5:
        raise RuntimeError('Five completed evaluations are required')
    binding = read_json(root / 'EVALUATION_BINDING.json')
    if digest(args.ground_truth) != binding['gt']['sha256']:
        raise RuntimeError('Ground-truth identity differs from the frozen evaluator')
    gt = read_json(args.ground_truth)
    gt_pages = {Path(page['page_info']['image_path']).name for page in gt}
    if len(gt_pages) != 1651:
        raise RuntimeError('Ground-truth page denominator differs')
    eligible = {'display_formula': set(), 'table': set()}
    for page in gt:
        name = Path(page['page_info']['image_path']).name
        for block in page.get('layout_dets', []):
            if block.get('ignore', False):
                continue
            if block.get('category_type') == 'equation_isolated':
                eligible['display_formula'].add(name)
            elif block.get('category_type') == 'table':
                eligible['table'].add(name)
    del gt
    inference = read_json(root / 'INFERENCE_COMPLETE.json')
    arms, numeric, native_hashes = {}, {}, {}
    definitions = {'text_edit': ('text_block', 'Edit_dist', gt_pages, False),
                   'cdm': ('display_formula', 'CDM', eligible['display_formula'], True),
                   'teds': ('table', 'TEDS', eligible['table'], True),
                   'structure_teds': ('table', 'TEDS_structure_only', eligible['table'], True),
                   'reading_order': ('reading_order', 'Edit_dist', gt_pages, False)}
    for arm, evaluation in terminal['arms'].items():
        arm_output = root / arm / 'output'
        records = [read_json(arm_output / 'receipts' / (row['page_id'] + '.json')) for row in manifest['pages']]
        if len(records) != 1651:
            raise RuntimeError('Missing inference receipts')
        official = root / arm / 'evaluation/work/result'
        numeric[arm] = {}
        for key, (element, metric, pages, _) in definitions.items():
            samples = read_json(official / f'pred_quick_match_{element}_result.json')
            numeric[arm][key] = page_metrics(samples, metric, pages)
        native_hashes[arm] = {}
        for page in manifest['pages']:
            file = arm_output / 'pages' / page['page_id'] / 'stages/native/COMPLETE.json'
            native_hashes[arm][page['page_id']] = read_json(file)['markdown_sha256'] if file.exists() else None
        resource = inference['arms'][arm]
        allocations = resource['allocations']
        allocation_start = min(row['start_unix'] for row in allocations)
        allocation_end = max(row['ended_unix'] for row in allocations)
        arms[arm] = {'metrics': evaluation['metrics'], 'denominator': 1651,
                     'raw_metric_result_sha256': evaluation['raw_result_sha256'],
                     'prediction_binding_sha256': digest(root / arm / 'evaluation/PREDICTION_BINDING.json'),
                     'status_counts': dict(Counter(row['status'] for row in records)),
                     'empty_predictions': sum(row['empty_prediction'] for row in records),
                     'retained_stage_counts': dict(Counter(str(row['retained_stage']) for row in records)),
                     'sum_page_seconds': sum(row['elapsed_seconds'] for row in records),
                     'median_page_seconds': statistics.median(row['elapsed_seconds'] for row in records),
                     'max_page_seconds': max(row['elapsed_seconds'] for row in records),
                     'allocated_gpu_seconds': resource['allocated_gpu_seconds'],
                     'inference_wall_seconds': allocation_end - allocation_start,
                     'concurrent_gpus': len(manifest['gpu_mapping'][arm]),
                     'peak_gpu_memory_mib': resource['peak_gpu_memory_mib'],
                     'resource_note': 'Allocation timing includes engine startup and up to one observation interval after exit; memory is the sampled device peak.',
                     'infrastructure_restarts': resource['infrastructure_restarts'],
                     'stage_coverage': stage_summary(arm_output)}
    comparisons = {}
    for baseline, candidate in [('V5.1', 'V5.2'), ('V5.2', 'V5.3'), ('V5.2', 'V5.4'), ('V5.2', 'V5.5'), ('V5.4', 'V5.5')]:
        key = candidate + '_minus_' + baseline
        comparisons[key] = {'overall_delta_points': arms[candidate]['metrics']['overall'] - arms[baseline]['metrics']['overall'],
                            'components': {name: paired(numeric[baseline][name], numeric[candidate][name], pages, higher)
                                           for name, (_, _, pages, higher) in definitions.items()}}
    native_agreement = {}
    for arm in ('V5.3', 'V5.4', 'V5.5'):
        common = [page for page in native_hashes['V5.2'] if native_hashes['V5.2'][page] is not None and native_hashes[arm][page] is not None]
        native_agreement[arm] = {'paired_valid_native_pages': len(common), 'all_pages': 1651,
                                 'byte_identical_native_pages': sum(native_hashes[arm][page] == native_hashes['V5.2'][page] for page in common),
                                 'missing_native_in_either_arm': 1651 - len(common)}
    request_audit = read_json(root / 'NATIVE_DIAGNOSTIC_FULL.json')
    if request_audit['run_id'] != manifest['run_id'] or request_audit['gt_used']:
        raise RuntimeError('Native request audit has inconsistent provenance')
    for arm, row in request_audit['comparisons'].items():
        if row['byte_identical_native_markdown'] != native_agreement[arm]['byte_identical_native_pages']:
            raise RuntimeError('Native agreement differs from the saved request audit')
    aggregate = {'schema_version': 1, 'run_id': manifest['run_id'], 'created_at': utc(), 'arms': arms,
                 'paired_comparisons': comparisons, 'native_agreement_with_V5_2': native_agreement,
                 'native_first_layout_request_audit': request_audit['comparisons'],
                 'preparation_allocated_gpu_seconds': manifest['preparation_allocated_gpu_seconds'],
                 'full_inference_allocated_gpu_seconds': inference['allocated_gpu_seconds'],
                 'all_recorded_allocated_gpu_seconds': manifest['preparation_allocated_gpu_seconds'] + inference['allocated_gpu_seconds'],
                 'evaluation_commit': binding['commit'], 'evaluation_config_sha256': binding['config']['sha256'],
                 'ground_truth_sha256': binding['gt']['sha256'], 'input_manifest_sha256': manifest['input_manifest_sha256'],
                 'shared_runtime': manifest['expected_common_runtime'],
                 'inference_image_sha256': manifest['image'], 'evaluation_image_sha256': binding['image'],
                 'frozen_source_hashes': {name.split('/frozen_code/', 1)[1]: value
                                          for name, value in manifest['frozen_files'].items()
                                          if '/frozen_code/' in name},
                 'limits': ['Local frozen protocol; no leaderboard submission or official rank claim.',
                            'Independent native predictions can differ numerically; native agreement is reported.',
                            'Validity repairs are distinct from paired metric improvements.',
                            'V5.5 versus V5.4 combines guard and table changes and cannot isolate guard causality.']}
    atomic_json(output / 'AGGREGATE.json', aggregate)
    atomic_json(output / 'PRIVATE_PAGE_METRICS.json', numeric)
    print(json.dumps({arm: row['metrics'] for arm, row in arms.items()}, indent=2))


if __name__ == '__main__':
    main()
