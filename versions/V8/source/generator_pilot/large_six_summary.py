"""Summarize frozen six-model results without selecting or rerunning models.

Inputs are hash-bound saved inference receipts and official CPU scoring reports.
Public output contains aggregates only; missing fields stay missing.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

FIELDS = ('overall', 'text_edit', 'formula_cdm', 'formula_edit', 'table_teds',
          'table_teds_structure', 'table_edit', 'reading_order_edit')
DIRECTIONS = {key: 'lower' if key.endswith('_edit') else 'higher' for key in FIELDS}
RECIPES = ('V8.1', 'V8.2', 'V8.3')
CONTROLS = ('base', 'legacy_simple_rule')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound(item):
    if digest(item['path']) != item['sha256']:
        raise ValueError('Input receipt bytes differ from manifest')
    return read(item['path'])


def public_identity(model):
    return f"{model['public_recipe']}/seed{model['seed']}"


def terminal_scores(report, expected, selection_sha256, fallback_config=None):
    if report['denominator'] != 1651 or report['status'] not in ('complete', 'failed'):
        raise ValueError('Scoring must have a full terminal disposition')
    if report['status'] == 'failed' and report['policies'] == {}:
        if (fallback_config is None or fallback_config['selection_sha256'] != selection_sha256
                or set(fallback_config['policies']) != set(expected)
                or len(fallback_config['full_input_freeze']['pages']) != 1651):
            raise ValueError('Hard scoring failure needs its frozen task configuration')
        return {policy: dict(status='failed', page_denominator=1651,
            missing_reasons={key:'CPU scoring controller terminated without policy metrics' for key in FIELDS}) for policy in expected}
    if report['selection_sha256'] != selection_sha256 or set(report['policies']) != set(expected):
        raise ValueError('Scoring policy or selection binding differs')
    return report['policies']


def numeric_tree(value):
    """Allow only aggregate dictionaries and numbers in category exports."""
    if isinstance(value, dict):
        if any('/' in key or '\\' in key or key.endswith(('.png', '.jpg')) for key in value):
            raise ValueError('Unexpected path-like category key')
        return {key: numeric_tree(item) for key, item in value.items()}
    if value is None or value == 'NaN':
        return None
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    raise ValueError('Non-numeric category payload requires explicit review')


def metric_record(value):
    if value.get('page_denominator') != 1651:
        raise ValueError('Full page denominator differs')
    if value['status'] not in ('complete', 'failed'):
        raise ValueError('Scoring is not terminal')
    values = value.get('values', {})
    result = dict(status=value['status'], page_denominator=1651,
        values={key: values.get(key) for key in FIELDS},
        missing_reasons={key: value.get('missing_reasons', {}).get(key, 'CPU scoring did not produce this field')
                         for key in FIELDS if values.get(key) is None},
        input_failures=value.get('input_failures'), empty_predictions=value.get('empty_predictions'),
        cpu_evaluator_elapsed_seconds=value.get('cpu_seconds'),
        category_breakdowns=numeric_tree(value.get('category_breakdowns', {})))
    for key, number in result['values'].items():
        if number is not None and (not isinstance(number, (int, float)) or not math.isfinite(number)
                                   or not 0 <= number <= (100 if key == 'overall' else 1)):
            raise ValueError('Metric outside declared range')
    return result


def contrast(left, right):
    values = {}
    for key in FIELDS:
        a, b = left['values'][key], right['values'][key]
        raw = a - b if a is not None and b is not None else None
        values[key] = dict(raw_left_minus_right=raw,
            improvement_signed_delta=None if raw is None else raw * (-1 if DIRECTIONS[key] == 'lower' else 1))
    return dict(metrics=values, inference='Descriptive matched-output contrast; no confidence interval or significance claim')


def inference_summary(rows, policy, baseline):
    if len(rows) != 1651 or len({r['page_id'] for r in rows}) != 1651:
        raise ValueError('Inference rows must retain all1651 pages')
    if set(baseline) != {r['page_id'] for r in rows}:
        raise ValueError('Candidate and base page membership differ')
    tables = [table for row in rows for table in row.get('tables', [])]
    return dict(page_denominator=1651, page_status_counts=dict(Counter(r['status'] for r in rows)),
        output_status_counts=dict(Counter(r['outputs'][policy]['status'] for r in rows)),
        empty_predictions=sum(r['outputs'][policy]['empty'] for r in rows),
        raw_markdown_changed_pages=sum(r['outputs'][policy]['sha256'] != baseline[r['page_id']]['outputs']['base']['sha256'] for r in rows),
        non_table_identity_verified_pages=sum(r.get('non_table_middle_exact') is True for r in rows),
        observed_table_actions=len(tables),
        exact_native_input_verified_actions=sum(t.get('native_input_exact') is True for t in tables),
        raw_table_content_changed_actions=sum(t.get('changed', t.get('legacy_changed', False)) is True for t in tables),
        legacy_rule_accepted_actions=sum(t.get('legacy_accepted') is True for t in tables),
        note='Output changes are not quality gains. Counts include explicit failures and partial action evidence; raw Markdown equality includes empty outputs.')


def cost_group(stage):
    for prefix, group in [('train-', 'formal_training'), ('dev-', 'DEV_generation'),
                          ('confirmation-', 'confirmation_generation'), ('benchmark-shared-', 'shared_benchmark'),
                          ('benchmark-model-', 'model_benchmark'), ('native-data-', 'native_input_preparation')]:
        if stage.startswith(prefix):
            return group
    if 'calibrat' in stage:
        return 'shared_gradient_calibration'
    return 'preparation_and_technical_checks'


def costs(ledger):
    stages = ledger['completed_stages']
    if len({row['stage'] for row in stages}) != len(stages):
        raise ValueError('Repeated stage would double-count GPU cost')
    grouped = defaultdict(lambda: dict(stages=0, failed_stages=0, allocated_gpu_seconds=0.0))
    for row in stages:
        if row['worker_exited'] is not True or row['lease_released'] is not True:
            raise ValueError('Cost ledger contains an allocation not verified released')
        if not math.isfinite(row['gpu_seconds']) or row['gpu_seconds'] < 0:
            raise ValueError('Invalid GPU allocation time')
        item = grouped[cost_group(row['stage'])]
        item['stages'] += 1
        item['failed_stages'] += row['status'] != 'passed'
        item['allocated_gpu_seconds'] += row['gpu_seconds']
    return dict(completed_allocations=len(stages), groups=dict(grouped),
        completed_allocated_gpu_seconds=sum(row['gpu_seconds'] for row in stages),
        note='Sum of terminal allocation seconds, including failed attempts. Shared preparation/calibration/controls are counted once. Concurrent allocation time is not campaign wall time; CPU scoring is reported separately.')


def summarize(manifest):
    selection = bound(manifest['selection'])
    if selection['pages_per_model'] != 1651 or selection['model_page_outputs'] != 9906:
        raise ValueError('Frozen scope differs')
    models = selection['models']
    expected = {f'{recipe}/seed{seed}' for recipe in RECIPES for seed in [0, 1]}
    if len(models) != 6 or {public_identity(model) for model in models} != expected:
        raise ValueError('Exactly six selected models required')
    if set(manifest['model_scores']) != expected or set(manifest['model_inference']) != expected:
        raise ValueError('All six terminal results required; do not omit failed models')
    shared = bound(manifest['shared_scores'])
    shared_config = bound(manifest['shared_scores']['scoring_config']) if 'scoring_config' in manifest['shared_scores'] else None
    shared_policies = terminal_scores(shared, CONTROLS, manifest['selection']['sha256'], shared_config)
    versions = {name: metric_record(shared_policies[name]) for name in CONTROLS}
    shared_rows = []
    for item in manifest['shared_inference']:
        value = bound(item)
        if value['status'] not in ('complete', 'failed') or value['selection_sha256'] != manifest['selection']['sha256']:
            raise ValueError('Shared inference not terminal or selection differs')
        shared_rows.extend(value['rows'])
    baseline = {row['page_id']: row for row in shared_rows}
    inference = {name: inference_summary(shared_rows, name, baseline) for name in CONTROLS}
    checkpoints = {}
    for model in models:
        name = public_identity(model)
        score = bound(manifest['model_scores'][name])
        score_config = bound(manifest['model_scores'][name]['scoring_config']) if 'scoring_config' in manifest['model_scores'][name] else None
        policies = terminal_scores(score, ['candidate'], manifest['selection']['sha256'], score_config)
        versions[name] = metric_record(policies['candidate'])
        raw = bound(manifest['model_inference'][name])
        if raw['status'] not in ('complete', 'failed') or raw['checkpoint_sha256'] != model['checkpoint_sha256']:
            raise ValueError('Selected inference checkpoint or state differs')
        if raw['selection_sha256'] != manifest['selection']['sha256'] or raw['model_run_id'] != model['run_id']:
            raise ValueError('Model inference identity differs')
        inference[name] = inference_summary(raw['rows'], 'candidate', baseline)
        checkpoints[name] = dict(epoch=model['epoch'], checkpoint_sha256=model['checkpoint_sha256'], wave=model['wave'])
    pairs = [(model, control) for model in sorted(expected) for control in CONTROLS]
    pairs += [(f'{a}/seed{seed}', f'{b}/seed{seed}') for a, b in [('V8.2', 'V8.1'), ('V8.3', 'V8.2')] for seed in [0, 1]]
    seeds = {}
    for recipe in RECIPES:
        seeds[recipe] = {}
        for key in FIELDS:
            values = [versions[f'{recipe}/seed{s}']['values'][key] for s in [0, 1]]
            seeds[recipe][key] = dict(individual_values=values,
                mean=statistics.mean(values) if None not in values else None,
                range=[min(values), max(values)] if None not in values else None)
    return dict(schema_version=1, status='all_evaluation_dispositions_summarized', distinct_pages=1651,
        model_evaluations=6, model_page_outputs=9906, control_policies=2, checkpoint_selection=checkpoints,
        selection_sha256=manifest['selection']['sha256'],
        overall_definition='100 * ((1 - text_edit) + formula_cdm + table_teds) / 3',
        units={key: '0-100' if key == 'overall' else '0-1' for key in FIELDS}, directions=DIRECTIONS,
        versions=versions, comparisons={a+' - '+b: contrast(versions[a], versions[b]) for a,b in pairs},
        seed_variability=seeds, inference=inference, costs=costs(bound(manifest['costs'])),
        limits=['Paired aggregate differences do not establish significance.',
                'Two optimization seeds give descriptive values and range, not a reliable confidence interval.',
                'Historical vLLM and public leaderboard scores are not matched Transformers controls.',
                'All1651 pages remain present; applicable official component denominators can differ.',
                'Official sample repair/harm and component-denominator supplements remain separate deliverables.'],
        benchmark_driven_reselection=False, external_leaderboard_submission=False,
        raw_GT_or_prediction_bodies_included=False)


def write_outputs(value, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / 'AGGREGATE.json').write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    with (output / 'METRICS.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['policy', 'status', 'page_denominator', *FIELDS])
        writer.writeheader()
        for name, row in value['versions'].items():
            writer.writerow(dict(policy=name, status=row['status'], page_denominator=1651, **row['values']))
    lines = ['# Six-model official benchmark metrics', '',
             'Descriptive matched evaluation. Overall uses 0–100; all other fields use 0–1. Missing values remain missing.', '',
             '| Policy | Status | '+' | '.join(FIELDS)+' |', '|---|---|'+ '|'.join(['---:']*len(FIELDS))+'|']
    for name, row in value['versions'].items():
        cells = ['missing' if row['values'][key] is None else f"{row['values'][key]:.6f}" for key in FIELDS]
        lines.append('| '+name+' | '+row['status']+' | '+' | '.join(cells)+' |')
    lines += ['', 'Every policy retains 1,651 pages. Category details, failures, selected checkpoint hashes and descriptive contrasts are in AGGREGATE.json.', '']
    (output / 'METRICS.md').write_text('\n'.join(lines), encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = summarize(read(args.manifest))
    write_outputs(value, Path(args.output))
    print(json.dumps(dict(model_evaluations=6, control_policies=2, distinct_pages=1651,
        completed_policies=sum(r['status']=='complete' for r in value['versions'].values()))))


if __name__ == '__main__':
    main()
