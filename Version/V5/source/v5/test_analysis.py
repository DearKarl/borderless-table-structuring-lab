"""Synthetic checks for descriptive paired evaluation summaries."""
import unittest

from .analyze import page_metrics, page_name, paired


class AnalysisTests(unittest.TestCase):
    def test_official_page_name_convention(self):
        self.assertEqual(page_name('paper_name_12.png_3'), 'paper_name_12.png')
        self.assertEqual(page_name('paper_name_12.png'), 'paper_name_12.png')

    def test_page_edit_distance_uses_text_length_weights(self):
        rows = [
            {'img_id': 'p.png_0', 'gt': 'abc', 'pred': 'abc', 'metric': {'Edit_dist': 0.0}},
            {'img_id': 'p.png_1', 'gt': 'abcdefg', 'pred': '', 'metric': {'Edit_dist': 1.0}},
        ]
        self.assertAlmostEqual(page_metrics(rows, 'Edit_dist')['p.png'], 0.7)

    def test_missing_gt_component_page_remains_zero(self):
        rows = [{'img_id': 'p.png_0', 'gt': 'x', 'pred': 'x', 'metric': {'CDM': 1.0}}]
        scores = page_metrics(rows, 'CDM', {'p.png', 'missing.png'})
        self.assertEqual(scores, {'p.png': 1.0, 'missing.png': 0.0})

    def test_quality_direction_and_no_successful_only_substitution(self):
        result = paired({'a': 0.3, 'b': 0.1}, {'a': 0.1, 'b': 0.2}, {'a', 'b'}, False)
        self.assertEqual((result['improved_pages'], result['regressed_pages']), (1, 1))
        self.assertAlmostEqual(result['mean_quality_delta'], 0.05)
        partial = paired({'a': 0.3}, {'a': 0.1, 'b': 0.2}, {'a', 'b'}, False)
        self.assertEqual(partial['status'], 'incomplete_numeric_coverage')
        self.assertNotIn('mean_quality_delta', partial)


if __name__ == '__main__':
    unittest.main()
