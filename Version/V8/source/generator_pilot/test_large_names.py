"""Naming regressions that could otherwise change package reconstruction."""
import copy
import unittest

from .large_names import CANONICAL_NAMES, identity, mapping, package_recipe, public_labels


class LargeNamesTests(unittest.TestCase):
    def test_all_six_packages_keep_the_original_reconstruction_branch(self):
        for old, canonical in CANONICAL_NAMES.items():
            for seed in (0, 1):
                with self.subTest(recipe=canonical, seed=seed):
                    old_recipe = package_recipe(dict(recipe=old, seed=seed))
                    new_recipe = package_recipe(identity(canonical, seed))
                    self.assertEqual(old_recipe != 'V7.3.0', new_recipe != 'V7.3.0')
                    self.assertEqual(old_recipe == 'V7.3.2', new_recipe == 'V7.3.2')
        self.assertEqual(len({row['run_id'] for row in mapping()['fits']}), 6)

    def test_inconsistent_package_alias_or_seed_is_rejected(self):
        ordinary = identity('V8.1', 0)
        for change in [dict(historical_alias='V7.3.2'), dict(seed=1),
                       dict(internal_run_id='V7.3.1-seed0'), dict(recipe='V8.4')]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                package_recipe(dict(ordinary, **change))

    def test_aggregate_presentation_does_not_mutate_evidence_or_numbers(self):
        evidence = {'nominee': 'V7.3.2-seed0', 'aggregate': {
            'V7.3.0-seed1': {'teds': 0.5, 'missing_count': 7},
            'legacy_simple_rule': {'teds': 0.75}}, 'comparisons': {'V7.3.2-seed0_vs_base': [0.01, -0.02]}}
        original = copy.deepcopy(evidence)
        rendered = public_labels(evidence)
        self.assertEqual(evidence, original)
        self.assertEqual(rendered['nominee'], 'V8.3/seed0')
        self.assertEqual(rendered['aggregate']['V8.1/seed1'], evidence['aggregate']['V7.3.0-seed1'])
        self.assertEqual(rendered['comparisons']['V8.3/seed0_vs_base'], [0.01, -0.02])
        self.assertEqual(public_labels('V7.3.20 V7.3.0-seed12'), 'V7.3.20 V8.1-seed12')


if __name__ == '__main__':
    unittest.main()
