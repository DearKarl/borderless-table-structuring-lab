import unittest

from .cell_tokens import full_cell_token_ranges


class FullCellTokensTest(unittest.TestCase):
    def test_all_content_empty_and_span_markers(self):
        text = '<fcel>Ab<lcel><ecel><nl>'
        result = full_cell_token_ranges(text, [(i, i + 1) for i in range(len(text))], sequence_start=12)
        self.assertEqual([c['token_range'] for c in result['cells']], [[12, 20], [26, 32]])
        self.assertEqual([c['empty'] for c in result['cells']], [False, True])
        self.assertEqual(result['adjacent_shared_tokens'], 0)

    def test_bpe_straddling_is_retained_and_reported(self):
        text = '<fcel>A<ecel><nl>'
        result = full_cell_token_ranges(text, [(0, 5), (5, 8), (8, 13), (13, 17)])
        self.assertEqual([c['token_range'] for c in result['cells']], [[0, 2], [1, 3]])
        self.assertEqual(result['adjacent_shared_tokens'], 1)
        self.assertEqual(result['cells'][0]['right_spill_characters'], 1)
        self.assertEqual(result['cells'][1]['left_spill_characters'], 2)

    def test_unicode_multiple_tokens_for_one_character(self):
        text = '<fcel>中<nl>'
        result = full_cell_token_ranges(text, [(0, 6), (6, 7), (6, 7), (7, 11)])
        self.assertEqual(result['cells'][0]['token_range'], [0, 3])

    def test_partial_or_unrepresented_cells_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            full_cell_token_ranges('<fcel>A', [(0, 7)])
        with self.assertRaisesRegex(ValueError, 'omit'):
            full_cell_token_ranges('<fcel>ABC<nl>', [(0, 6), (7, 9), (9, 13)])
        with self.assertRaisesRegex(ValueError, 'non-content'):
            full_cell_token_ranges('<ecel>A<nl>', [(0, 7), (7, 11)])


if __name__ == '__main__':
    unittest.main()
