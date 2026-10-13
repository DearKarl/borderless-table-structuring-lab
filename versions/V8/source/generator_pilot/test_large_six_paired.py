"""Conservative saved-score correspondence and missing-page tests."""
import unittest
from .large_six_paired import page_comparison, sample_comparison


def sample(score, indices=None, truth='fixture'):
    return dict(image_name='fixture.jpg', gt_idx=[0] if indices is None else indices,
                gt=truth, metric=dict(TEDS=score, TEDS_structure_only=score, Edit_dist=1-score))


class PairedChecks(unittest.TestCase):
    def test_unique_pair_repair_and_direction(self):
        result = sample_comparison([sample(.8)], [sample(1.)], 'table')
        for metric in result.values():
            self.assertEqual(metric['outcomes']['improved'], 1)
            self.assertEqual(metric['reached_metric_perfect_count'], 1)
        self.assertAlmostEqual(result['Edit_dist']['mean_raw_candidate_minus_baseline_on_unique_pairs'], -.2)

    def test_duplicate_or_changed_GT_identity_is_not_paired(self):
        duplicate = sample_comparison([sample(.8), sample(.8)], [sample(1.)], 'table')['TEDS']
        self.assertEqual(duplicate['outcomes']['unmatchable'], 1)
        self.assertIsNone(duplicate['mean_raw_candidate_minus_baseline_on_unique_pairs'])
        changed = sample_comparison([sample(.8)], [sample(1., truth='different fixture')], 'table')['TEDS']
        self.assertEqual(changed['outcomes']['unmatchable'], 2)
        self.assertEqual(changed['outcomes']['improved'], 0)

    def test_page_absence_and_nonfinite_never_imputed(self):
        result = page_comparison({'a':.8, 'b':.7}, {'a':1., 'b':float('nan')}, {'a','b','c'}, 1)
        self.assertEqual(result['page_denominator'], 3)
        self.assertEqual(result['outcomes']['unmatchable'], 2)
        self.assertEqual(result['outcomes']['improved'], 1)
        with self.assertRaises(ValueError):
            page_comparison({'unregistered':1.}, {}, {'a'}, 1)


if __name__ == '__main__':
    unittest.main()
