"""Synthetic checks of admission, completeness, pairing and negative comparator."""
import unittest
from .confirmation_gate import REQUIRED_CHECKS, evaluate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.groups = ['synthetic-'+str(i) for i in range(128)]
        self.rows = [{'source_group': g, 'disposition_valid': True,
                      'scores': {'V7.1': .8, 'V7.2': .81, 'V7.3': .82, 'V7.4': .8}} for g in self.groups]
        self.checks = dict.fromkeys(REQUIRED_CHECKS, True)

    def test_positive_constant_effect_and_order_independence(self):
        result = evaluate(self.rows, self.groups, self.checks)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result, evaluate(self.rows[::-1], self.groups, self.checks))
        self.assertEqual(result['pairwise']['V7.3-V7.2']['improved'], 128)
        self.assertEqual(result['pairwise']['V7.3-V7.2']['ci95'][0], .82-.81)

    def test_missing_and_duplicate_are_not_success_subsets(self):
        self.assertEqual(evaluate(self.rows[:-1], self.groups, self.checks)['status'], 'inconclusive')
        bad = self.rows[:-1]+[self.rows[0]]
        self.assertEqual(evaluate(bad, self.groups, self.checks)['status'], 'inconclusive')
        self.rows[0]['scores']['V7.2'] = None
        result = evaluate(self.rows, self.groups, self.checks)
        self.assertEqual(result['status'], 'inconclusive')
        self.assertIsNone(result['means'])
        self.assertEqual(result['pairwise']['V7.3-V7.2']['unmatchable'], 1)
        self.assertEqual(result['pairwise']['V7.3-V7.1']['unmatchable'], 0)

    def test_candidate_must_beat_both_controls(self):
        for row in self.rows:
            row['scores']['V7.2'] = .82
        result = evaluate(self.rows, self.groups, self.checks)
        self.assertEqual(result['status'], 'failed')
        self.assertFalse(result['both_intervals_positive'])

    def test_all_keep_identity_and_prerequisites_enforced(self):
        self.rows[0]['scores']['V7.4'] = .81
        self.assertEqual(evaluate(self.rows, self.groups, self.checks)['status'], 'inconclusive')
        self.rows[0]['scores']['V7.4'] = .8
        self.checks['lineage_verified'] = False
        self.assertEqual(evaluate(self.rows, self.groups, self.checks)['status'], 'inconclusive')


if __name__ == '__main__':
    unittest.main()
