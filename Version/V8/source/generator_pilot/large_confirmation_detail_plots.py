"""Plot saved confirmation seed and secondary-contrast detail without reanalysis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import textwrap


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aggregate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.aggregate)
    value = json.loads(source.read_text(encoding='utf-8'))
    assert value['cohort'] == 'confirmation' and value['denominator'] == 512
    assert not value['nominee_reselected']
    assert value['interval_reporting_qualification']['numerical_values_changed'] is False
    recipes = ('V8.1', 'V8.2', 'V8.3')
    names = [f'{left}/seed{seed} - {right}/seed{seed}'
             for left, right in (('V8.2', 'V8.1'), ('V8.3', 'V8.2')) for seed in (0, 1)]
    contrasts = [value['comparisons'][name] for name in names]
    assert all(not row['primary'] and row['source_denominator'] == 512
               and row['bootstrap_resamples'] == 10000 and row['bootstrap_seed'] == 0 for row in contrasts)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 170})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={'width_ratios': [1, 1.4]})
    seeds = []
    for i, (recipe, color) in enumerate(zip(recipes, ('#1769aa', '#d97706', '#16866b'))):
        row = value['seed_variability'][recipe]
        axes[0].plot([i, i], row['range'], color=color, linewidth=2)
        axes[0].scatter(i, row['mean'], marker='_', color=color, s=180, zorder=3)
        for seed, score in enumerate(row['values']):
            assert score == value['aggregate'][f'{recipe}/seed{seed}']['teds']
            axes[0].scatter(i + (-.08 if seed == 0 else .08), score, color=color,
                marker='o' if seed == 0 else '^', s=55, label=f'Seed {seed}' if i == 0 else None, zorder=4)
            seeds.append(dict(recipe=recipe, seed=seed, full_teds=score, mean=row['mean'],
                              range_min=row['range'][0], range_max=row['range'][1]))
    points = [row['full_teds'] for row in seeds]
    padding = max(.001, (max(points) - min(points)) * .2)
    axes[0].set(xticks=range(3), xticklabels=recipes, ylabel='Mean full TEDS (higher)',
                ylim=(min(points) - padding, max(points) + padding), title='Seed detail: expanded vertical axis')
    axes[0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(axis='y', alpha=.2)
    for i, row in enumerate(contrasts):
        low, high = row['paired_percentile_97_5_ci']
        color = '#d97706' if i < 2 else '#16866b'
        axes[1].hlines(i, 100 * low, 100 * high, color=color, linewidth=2)
        axes[1].plot([100 * low, 100 * high], [i, i], '|', color=color, markersize=9)
        axes[1].scatter(100 * row['mean_delta'], i, color=color, s=45, zorder=3)
    axes[1].axvline(0, color='#666', linewidth=1)
    axes[1].set(yticks=range(4), yticklabels=['GA - LoRA / seed 0', 'GA - LoRA / seed 1',
                 'GA + head - GA / seed 0', 'GA + head - GA / seed 1'], ylim=(3.6, -.6),
                xlabel='Paired full-TEDS difference (percentage points)', title='Secondary method contrasts: descriptive')
    axes[1].grid(axis='x', alpha=.2)
    fig.suptitle('Confirmation details: two seeds and method contrasts (512 sources)', fontsize=14)
    footer = ('Left: individual seeds, mean ticks and ranges on an expanded axis; ranges are not confidence intervals. '
              'Right: saved nominal 97.5% marginal percentile-bootstrap intervals, 10,000 paired source resamples, seed 0; '
              'secondary and descriptive, with no family coverage guarantee. No new analysis or model selection.')
    fig.text(.02, .015, textwrap.fill(footer, 165), fontsize=8, va='bottom')
    fig.tight_layout(rect=[0, .17, 1, .92], w_pad=3)
    files = []
    for suffix in ('png', 'svg'):
        path = output / f'confirmation_detail.{suffix}'
        fig.savefig(path, bbox_inches='tight')
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n',
                            encoding='utf-8', newline='\n')
        files.append(dict(file=path.name, sha256=digest(path)))
    plt.close(fig)
    rows = [dict(comparison=name, mean_delta=row['mean_delta'], interval_low=row['paired_percentile_97_5_ci'][0],
                 interval_high=row['paired_percentile_97_5_ci'][1], source_denominator=512, primary=False)
            for name, row in zip(names, contrasts)]
    for filename, table in [('confirmation_seed_detail.csv', seeds), ('confirmation_method_contrasts.csv', rows)]:
        path = output / filename
        with path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        files.append(dict(file=path.name, sha256=digest(path)))
    manifest = dict(source_aggregate_sha256=digest(source), source_revision=Path(__file__).name,
        source_revision_sha256=digest(Path(__file__)), files=files, visual_inspection='pending',
        scope='Companion expanded-axis visualization only; saved numerical results unchanged',
        new_statistical_analyses=0, inference_or_scoring_repeated=False)
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(figures=1, numerical_values_changed=False, visual_inspection='pending')))


if __name__ == '__main__':
    main()
