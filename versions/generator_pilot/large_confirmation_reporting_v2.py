"""Qualify bootstrap wording and plot saved confirmation results without rescoring."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import textwrap

from .large_names import mapping, public_labels

RECIPES = ('V7.3.0', 'V7.3.1', 'V7.3.2')
MODELS = [f'{recipe}-seed{seed}' for recipe in RECIPES for seed in (0, 1)]
CONTROLS = ('base', 'legacy_simple_rule')
PUBLIC_FIELDS = ('cohort', 'denominator', 'nominee', 'aggregate', 'comparisons',
                 'former_promotion_gate_diagnostic', 'strata', 'seed_variability',
                 'nominee_reselected', 'full_benchmark_required', 'limits')
PRIMARY_NOTE = ('Nominal 97.5% marginal percentile-bootstrap interval; Bonferroni '
                'allocation targets 95% family coverage for the two registered '
                'primary contrasts. Bootstrap coverage is approximate, not guaranteed.')
SECONDARY_NOTE = ('Nominal 97.5% marginal percentile-bootstrap interval; descriptive '
                  'secondary contrast, without a family coverage guarantee.')


def qualify(value, nomination_sha256, nominee):
    if (value['status'] != 'complete' or value['cohort'] != 'confirmation'
            or value['denominator'] != 512 or value['nominee_reselected']
            or not value['full_benchmark_required']):
        raise ValueError('Complete unchanged confirmation analysis required')
    if value['nomination_sha256'] != nomination_sha256 or value['nominee'] != nominee or nominee not in MODELS:
        raise ValueError('Frozen nomination binding differs')
    if set(value['aggregate']) != set(MODELS + list(CONTROLS)):
        raise ValueError('All six selected models and both controls required')
    if set(value['seed_variability']) != set(RECIPES):
        raise ValueError('All three two-seed summaries required')
    for recipe in RECIPES:
        row = value['seed_variability'][recipe]
        points = [value['aggregate'][f'{recipe}-seed{seed}']['teds'] for seed in (0, 1)]
        if (row['values'] != points or row['range'] != [min(points), max(points)]
                or not math.isclose(row['mean'], sum(points) / 2, rel_tol=0, abs_tol=1e-15)):
            raise ValueError('Saved seed summary differs from its individual scores')
    pairs = [(model, control) for model in MODELS for control in CONTROLS]
    pairs += [(f'{left}-seed{seed}', f'{right}-seed{seed}')
              for left, right in [('V7.3.1', 'V7.3.0'), ('V7.3.2', 'V7.3.1')] for seed in (0, 1)]
    expected = {left + ' - ' + right for left, right in pairs}
    primary = {nominee + ' - ' + control for control in CONTROLS}
    if set(value['comparisons']) != expected:
        raise ValueError('Registered confirmation contrast membership differs')
    for name, row in value['comparisons'].items():
        if (row['primary'] != (name in primary) or row['source_denominator'] != 512
                or row['bootstrap_resamples'] != 10000 or row['bootstrap_seed'] != 0):
            raise ValueError('Frozen contrast interpretation or bootstrap settings differ')
        bounds = row['paired_percentile_97_5_ci']
        if len(bounds) != 2 or bounds[0] > bounds[1] or not all(math.isfinite(x) for x in [row['mean_delta'], *bounds]):
            raise ValueError('Invalid saved interval')
        counts = [row[key] for key in ('gain_count', 'harm_count', 'tie_count')]
        if any(type(x) is not int or x < 0 for x in counts) or sum(counts) != 512:
            raise ValueError('Paired outcome denominator differs')
    # public_labels recursively copies containers; the raw analysis is untouched.
    result = public_labels({key: value[key] for key in PUBLIC_FIELDS})
    for row in result['comparisons'].values():
        row['interval_interpretation'] = PRIMARY_NOTE if row['primary'] else SECONDARY_NOTE
    result['version_mapping'] = mapping()
    result['interval_reporting_qualification'] = dict(
        scope='Wording and visualization only; frozen numerical analysis retained',
        numerical_values_changed=False, source_denominator_changed=False,
        raw_analysis_preserved=True, bootstrap_coverage_is_approximate=True)
    return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--nomination', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source, nomination_path = Path(args.analysis), Path(args.nomination)
    value = qualify(json.loads(source.read_text(encoding='utf-8')), digest(nomination_path),
                    json.loads(nomination_path.read_text(encoding='utf-8'))['nominee_run_id'])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    aggregate = output / 'CONFIRMATION_AGGREGATE.json'
    aggregate.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    files = plot(value, output)
    manifest = dict(source_analysis_sha256=digest(source), nomination_sha256=digest(nomination_path),
                    qualified_aggregate_sha256=digest(aggregate), source_revision=Path(__file__).name,
                    source_revision_sha256=digest(Path(__file__)), files=files,
                    visual_inspection='pending', measured_values_only=True,
                    inference_or_scoring_repeated=False,
                    interval_reporting_qualification=value['interval_reporting_qualification'])
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figures=3, exports=len(files), numerical_values_changed=False, visual_inspection='pending')))


def plot(value, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 170})
    files = []
    def finish(fig, name, title, footer):
        fig.suptitle(title, fontsize=14)
        fig.text(.02, .015, textwrap.fill(footer, width=int(fig.get_figwidth() * 14)), fontsize=8, va='bottom')
        fig.tight_layout(rect=[0, .16, 1, .94])
        for suffix in ('png', 'svg'):
            path = output / f'{name}.{suffix}'
            fig.savefig(path, bbox_inches='tight')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n',
                                encoding='utf-8', newline='\n')
            files.append(dict(file=path.name, sha256=digest(path)))
        plt.close(fig)
    primary = [(name, row) for name, row in value['comparisons'].items() if row['primary']]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (_, row) in enumerate(primary):
        low, high = row['paired_percentile_97_5_ci']
        ax.hlines(i, low, high, color='#1769aa', linewidth=2)
        ax.scatter(row['mean_delta'], i, color='#1769aa', s=40, zorder=3)
        ax.plot([low, high], [i, i], '|', color='#1769aa', markersize=10)
    ax.axvline(0, color='#555', linewidth=1)
    ax.set(yticks=range(2), yticklabels=[name for name, _ in primary], ylim=(-.5, 1.5),
           xlabel='Paired mean full-TEDS difference (higher)')
    ax.grid(axis='x', alpha=.2)
    finish(fig, 'confirmation_primary', 'Prespecified nominee against both controls: 512 sources',
           '10,000 paired source resamples, seed 0. Nominal 97.5% marginal percentile intervals; '
           'Bonferroni allocation targets 95% family coverage for these two contrasts. Bootstrap coverage is approximate. '
           'Optimization-seed variation is separate; the DEV-selected nominee is unchanged.')
    names = public_labels(list(CONTROLS) + MODELS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].scatter(range(8), [value['aggregate'][name]['teds'] for name in names], color='#1769aa')
    axes[0].set(xticks=range(8), xticklabels=names, ylabel='Mean full TEDS (higher)', ylim=(0, 1))
    axes[0].tick_params(axis='x', rotation=60)
    for i, (recipe, color) in enumerate(zip(('V8.1', 'V8.2', 'V8.3'), ('#1769aa', '#d97706', '#16866b'))):
        row = value['seed_variability'][recipe]
        for seed, score in enumerate(row['values']):
            axes[1].scatter(i + (-.08 if seed == 0 else .08), score, color=color,
                            marker='o' if seed == 0 else '^', label=f'Seed {seed}' if i == 0 else None)
        axes[1].plot([i, i], row['range'], color=color)
        axes[1].scatter(i, row['mean'], marker='_', color=color, s=150)
    axes[1].set(xticks=range(3), xticklabels=['V8.1', 'V8.2', 'V8.3'], ylabel='Mean full TEDS (higher)', ylim=(0, 1))
    axes[1].legend(fontsize=8)
    finish(fig, 'confirmation_all_models', 'All six selected models and both controls',
           'All 512 sources retained. Full vertical axes. Right: dots and triangles are individual seeds, '
           'horizontal ticks are means, and vertical segments are ranges. Two-seed ranges are not confidence intervals. '
           'No confirmation-based reselection.')
    fig, ax = plt.subplots(figsize=(11, 5))
    left = [0, 0]
    for key, label, color in [('gain_count', 'TEDS improved', '#16866b'),
                              ('harm_count', 'TEDS worsened', '#bb3e3e'), ('tie_count', 'Tied', '#b7bdc5')]:
        counts = [row[key] for _, row in primary]
        ax.barh(range(2), counts, left=left, label=label, color=color)
        left = [a + b for a, b in zip(left, counts)]
    ax.set(yticks=range(2), yticklabels=[name for name, _ in primary], xlabel='Number of paired sources', xlim=(0, 512))
    ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncols=3, fontsize=8)
    finish(fig, 'confirmation_gains_harms', 'Paired source gains, harms and ties',
           'All 512 pairs, including missing inputs, remain in each comparison. Counts, exact-table repairs and harms, '
           'and unmatchable grids are preserved in CONFIRMATION_AGGREGATE.json.')
    return files


if __name__ == '__main__':
    main()
