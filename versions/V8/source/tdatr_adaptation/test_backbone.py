import unittest
import torch

from .backbone import restore_native_maps


class SpatialRestorationTest(unittest.TestCase):
    def test_known_coordinates_survive_windows_and_merging(self):
        # Independent coordinate construction: six 2x2 blocks, shuffled by
        # window order. Channels encode y, x and a unique linear pixel index.
        height, width = 4, 6
        permutation = torch.tensor([2, 0, 5, 1, 4, 3])
        packed = []
        for group in permutation.tolist():
            by, bx = divmod(group, width // 2)
            for dy in range(2):
                for dx in range(2):
                    y, x = by * 2 + dy, bx * 2 + dx
                    packed.append([y, x, y * width + x])
        fine_input = torch.tensor(packed, dtype=torch.float32, requires_grad=True)
        coarse_input = torch.arange(6, dtype=torch.float32)[:, None].requires_grad_()
        fine, coarse = restore_native_maps(fine_input, coarse_input,
            torch.tensor([[1, height, width]]), permutation)
        for y in range(height):
            for x in range(width):
                self.assertEqual(fine[:, y, x].tolist(), [y, x, y * width + x])
                self.assertEqual(coarse[0, y // 2, x // 2], (y // 2) * 3 + x // 2)
        (fine.sum() + coarse.sum()).backward()
        self.assertTrue(torch.equal(fine_input.grad, torch.ones_like(fine_input)))
        self.assertTrue(torch.equal(coarse_input.grad, torch.ones_like(coarse_input)))

    def test_reject_missing_or_repeated_spatial_cells(self):
        with self.assertRaisesRegex(ValueError, 'permutation'):
            restore_native_maps(torch.zeros(24, 3), torch.zeros(6, 2),
                torch.tensor([[1, 4, 6]]), torch.tensor([0, 1, 2, 3, 4, 4]))


if __name__ == '__main__':
    unittest.main()
