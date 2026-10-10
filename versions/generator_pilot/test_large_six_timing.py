"""Ensure concurrent GPU allocations are not reported as elapsed wall time."""
import unittest
from .large_six_timing import timestamp, window


class TimingChecks(unittest.TestCase):
    def test_overlapping_nested_and_separated_allocations(self):
        value = window([(0,10), (2,4), (5,15), (20,25)])
        self.assertEqual(value['elapsed_window_seconds'], 25)
        self.assertEqual(value['occupied_union_seconds'], 20)
        self.assertEqual(value['timed_allocations'], 4)

    def test_missing_intervals_are_not_zero_runtime(self):
        value = window([])
        self.assertIsNone(value['elapsed_window_seconds'])
        self.assertIsNone(value['occupied_union_seconds'])
        with self.assertRaises(ValueError):
            window([(5,2)])

    def test_offsets_normalize_and_naive_timestamps_rejected(self):
        self.assertEqual(timestamp('2026-10-10T12:00:00+01:00'), timestamp('2026-10-10T11:00:00Z'))
        with self.assertRaises(ValueError):
            timestamp('2026-10-10T11:00:00')


if __name__ == '__main__':
    unittest.main()
