"""Read terminal official scoring outputs and export only paired aggregates.

No inference, score recomputation, checkpoint selection or bootstrap is performed.
Raw references and predictions are used locally for conservative correspondence
and never included in the returned aggregate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

COMPONENTS = {'table': {'TEDS': 1, 'TEDS_structure_only': 1, 'Edit_dist': -1},
              'display_formula': {'CDM': 1, 'Edit_dist': -1}}
RECIPES = ('V8.1', 'V8.2', 'V8.3')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def page_key(value):
    name = Path(str(value)).name
    return name[:-4] if name.lower().endswith(('.jpg', '.png')) else name


def sample_key(row):
    indices = row.get('gt_idx')
    page = page_key(row.get('image_name', row.get('img_id', '')))
    if not indices or not page:
        return None
    return page, json.dumps(indices, sort_keys=True), hashlib.sha256(str(row.get('gt', '')).encode()).hexdigest()


def outcome(before, after, direction):
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (before, after)):
        return 'unmatchable'
    difference = direction * (after - before)
    return 'improved' if difference > 1e-6 else 'worsened' if difference < -1e-6 else 'tied'


def sample_comparison(before, after, kind):
    maps = []
    for rows in (before, after):
        mapped = defaultdict(list)
        for row in rows:
            key = sample_key(row)
            if key is not None:
                mapped[key].append(row)
        maps.append(mapped)
    unknown = max(sum(sample_key(r) is None for r in rows) for rows in (before, after))
    summary = {}
    for metric, direction in COMPONENTS[kind].items():
        counts = Counter(improved=0, worsened=0, tied=0, unmatchable=unknown)
        deltas = []; repaired = harmed = 0
        for key in set(maps[0]) | set(maps[1]):
            left, right = maps[0].get(key, []), maps[1].get(key, [])
            if len(left) != 1 or len(right) != 1:
                counts['unmatchable'] += 1
                continue
            x = left[0].get('metric', {}).get(metric)
            y = right[0].get('metric', {}).get(metric)
            label = outcome(x, y, direction); counts[label] += 1
            if label != 'unmatchable':
                deltas.append(y - x)
                x_exact = abs(x - (1 if direction == 1 else 0)) <= 1e-6
                y_exact = abs(y - (1 if direction == 1 else 0)) <= 1e-6
                repaired += y_exact and not x_exact
                harmed += x_exact and not y_exact
        summary[metric] = dict(outcomes=dict(counts), comparison_unit_denominator=sum(counts.values()),
            baseline_scored_rows=len(before), candidate_scored_rows=len(after),
            mean_raw_candidate_minus_baseline_on_unique_pairs=sum(deltas)/len(deltas) if deltas else None,
            reached_metric_perfect_count=repaired, lost_metric_perfect_count=harmed,
            metric_perfect_note='Metric-perfect is not necessarily exact output-text equality',
            direction='higher' if direction == 1 else 'lower', tie_tolerance=1e-6)
    return summary


def page_comparison(before, after, pages, direction):
    if (set(before) | set(after)) - set(pages):
        raise ValueError('Official numeric page identities differ from frozen input membership')
    counts = Counter(improved=0, worsened=0, tied=0, unmatchable=0)
    for page in pages:
        counts[outcome(before.get(page), after.get(page), direction)] += 1
    return dict(outcomes=dict(counts), page_denominator=len(pages), tie_tolerance=1e-6,
        note='Absent component or missing score is unmatchable; no favorable value is imputed')


def policy_inputs(item, pages):
    score_path = Path(item['score_report'])
    if sha(score_path) != item['score_sha256']:
        raise ValueError('Official scoring receipt changed')
    report = read(score_path)
    if report['denominator'] != 1651 or report['status'] not in ('complete', 'failed'):
        raise ValueError('Full scoring report is not terminal')
    if report['status'] == 'failed' and report['policies'] == {}:
        return dict(status='failed', samples={}, pages={}, page_denominators={}, source_hashes={score_path.name:item['score_sha256']})
    metrics = report['policies'][item['policy']]
    if metrics['status'] not in ('complete', 'failed'):
        raise ValueError('Policy scoring is still running')
    folder = Path(item['result_directory'])
    if metrics['status'] == 'failed':
        return dict(status='failed', samples={}, pages={}, page_denominators={}, source_hashes={})
    raw = folder / 'pred_quick_match_metric_result.json'
    if sha(raw) != metrics['raw_result_sha256']:
        raise ValueError('Official raw metrics changed')
    summary_path = folder / 'pred_quick_match_run_summary.json'
    summary = read(summary_path)
    hashes = {raw.name: sha(raw), summary_path.name: sha(summary_path)}
    samples = {}; values = {}
    for kind in COMPONENTS:
        path = folder / f'pred_quick_match_{kind}_result.json'
        samples[kind] = read(path); hashes[path.name] = sha(path)
        if any(page_key(row.get('image_name', row.get('img_id', ''))) not in pages for row in samples[kind]):
            raise ValueError('Scored sample is outside the frozen benchmark')
        for metric in COMPONENTS[kind]:
            if metric == 'Edit_dist':
                continue
            groups = defaultdict(list)
            for row in samples[kind]:
                value = row.get('metric', {}).get(metric)
                if isinstance(value, (int, float)) and math.isfinite(value):
                    groups[page_key(row.get('image_name', row.get('img_id', '')))].append(value)
            values[kind+'_'+metric] = {key: sum(v)/len(v) for key,v in groups.items()}
    for kind in ('text_block', 'display_formula', 'table', 'reading_order'):
        path = folder / f'pred_quick_match_{kind}_per_page_edit.json'
        if path.exists():
            values[kind+'_edit'] = {page_key(key): value for key,value in read(path).items()}
            hashes[path.name] = sha(path)
        else:
            values[kind+'_edit'] = {}
    return dict(status='complete', samples=samples, pages=values,
        page_denominators=summary['page_denominators'], source_hashes=hashes)


def analyze(manifest):
    pages = {page_key(name) for name in manifest['original_page_ids']}
    if len(pages) != 1651 or len(manifest['original_page_ids']) != 1651:
        raise ValueError('Full1651-page membership is required')
    expected = {'base', 'legacy_simple_rule'} | {f'{recipe}/seed{seed}' for recipe in RECIPES for seed in (0,1)}
    if set(manifest['policies']) != expected:
        raise ValueError('Both controls and all six model dispositions required')
    inputs = {name: policy_inputs(item, pages) for name,item in manifest['policies'].items()}
    pairs = [(name, control) for name in sorted(expected - {'base','legacy_simple_rule'}) for control in ('base','legacy_simple_rule')]
    pairs += [(f'{a}/seed{seed}', f'{b}/seed{seed}') for a,b in [('V8.2','V8.1'),('V8.3','V8.2')] for seed in (0,1)]
    comparisons = {}
    metric_names = ['text_block_edit','display_formula_edit','table_edit','reading_order_edit',
                    'table_TEDS','table_TEDS_structure_only','display_formula_CDM']
    for candidate, baseline in pairs:
        a, b = inputs[candidate], inputs[baseline]
        comparisons[candidate+' - '+baseline] = dict(
            policy_statuses=dict(candidate=a['status'], baseline=b['status']),
            official_sample_outcomes={kind: sample_comparison(b['samples'].get(kind,[]), a['samples'].get(kind,[]), kind) for kind in COMPONENTS},
            page_outcomes={metric: page_comparison(b['pages'].get(metric,{}), a['pages'].get(metric,{}), pages,
                -1 if metric.endswith('_edit') else 1) for metric in metric_names})
    return dict(schema_version=1, page_denominator=1651, comparisons=comparisons,
        policy_component_page_denominators={name:item['page_denominators'] for name,item in inputs.items()},
        source_file_hashes={name:item['source_hashes'] for name,item in inputs.items()},
        correspondence='Same page, exact GT index group and GT-text hash; unique in both outputs. Ambiguous or unidentified samples remain unmatchable.',
        unidentified_sample_note='Unidentified sample count is the maximum across outputs, not an assertion that those rows correspond. Scored-row totals for both outputs are retained.',
        page_metric_note='Official per-page edit values; TEDS/CDM page values are diagnostic means of emitted sample metrics. No page-level Overall is constructed.',
        descriptive_only=True, raw_GT_or_prediction_bodies_included=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--manifest', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = analyze(read(args.manifest))
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(page_denominator=1651, comparisons=len(result['comparisons']), raw_bodies_returned=False)))


if __name__ == '__main__':
    main()
