"""Reporting must preserve every saved number and the registered comparison scope."""
import copy
import unittest

from .large_confirmation_reporting_v2 import CONTROLS, MODELS, RECIPES, PUBLIC_FIELDS, qualify
from .large_names import public_labels


def example():
    nominee = 'V7.3.2-seed0'
    pairs = [(model, control) for model in MODELS for control in CONTROLS]
    pairs += [(f'{left}-seed{seed}', f'{right}-seed{seed}')
              for left, right in [('V7.3.1', 'V7.3.0'), ('V7.3.2', 'V7.3.1')] for seed in (0, 1)]
    aggregate = {name: dict(teds=.7 + i * .01, missing_count=20) for i, name in enumerate(list(CONTROLS) + MODELS)}
    seed = {}
    for recipe in RECIPES:
        points = [aggregate[f'{recipe}-seed{s}']['teds'] for s in (0, 1)]
        seed[recipe] = dict(values=points, mean=sum(points)/2, range=[min(points), max(points)])
    return dict(status='complete', cohort='confirmation', denominator=512, nominee=nominee,
                nomination_sha256='a' * 64, aggregate=aggregate, seed_variability=seed,
                comparisons={left + ' - ' + right: dict(
                    primary=left == nominee and right in CONTROLS, source_denominator=512,
                    bootstrap_resamples=10000, bootstrap_seed=0, mean_delta=.002,
                    paired_percentile_97_5_ci=[-.001, .004], gain_count=111, harm_count=87, tie_count=314,
                    repaired_exact_table_count=9, harmed_exact_table_count=7, either_unmatchable_grid_count=20,
                    interval_interpretation='Old coverage wording') for left, right in pairs},
                former_promotion_gate_diagnostic=dict(passed=False), strata={'fixed': dict(denominator=512)},
                nominee_reselected=False, full_benchmark_required=True, limits=['Known limitations'],
                private_operational_field='Must not be exported')


class ConfirmationReportingTests(unittest.TestCase):
    def test_preserves_input_and_all_numerical_results(self):
        source = example()
        original = copy.deepcopy(source)
        result = qualify(source, 'a' * 64, source['nominee'])
        self.assertEqual(source, original)
        self.assertNotIn('private_operational_field', result)
        expected = public_labels(source)
        for key in PUBLIC_FIELDS:
            if key != 'comparisons':
                self.assertEqual(result[key], expected[key])
        self.assertEqual(len(result['comparisons']), 16)
        self.assertEqual(sum(row['primary'] for row in result['comparisons'].values()), 2)
        for name, row in result['comparisons'].items():
            self.assertIn('Nominal 97.5%', row['interval_interpretation'])
            for key, number in expected['comparisons'][name].items():
                if key != 'interval_interpretation':
                    self.assertEqual(row[key], number)
        self.assertFalse(result['interval_reporting_qualification']['numerical_values_changed'])

    def test_rejects_scope_or_binding_changes(self):
        for mutation in ('denominator', 'membership', 'primary', 'nomination', 'resamples',
                         'counts', 'bounds', 'seed_summary'):
            source = example()
            row = source['comparisons']['V7.3.2-seed0 - base']
            if mutation == 'denominator': source['denominator'] = 492
            elif mutation == 'membership': source['comparisons'].pop('V7.3.0-seed0 - base')
            elif mutation == 'primary': row['primary'] = False
            elif mutation == 'nomination': source['nomination_sha256'] = 'b' * 64
            elif mutation == 'resamples': row['bootstrap_resamples'] = 1000
            elif mutation == 'counts': row['tie_count'] -= 20
            elif mutation == 'bounds': row['paired_percentile_97_5_ci'] = [.004, -.001]
            else: source['seed_variability']['V7.3.0']['mean'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(source, 'a' * 64, 'V7.3.2-seed0')


if __name__ == '__main__':
    unittest.main()
