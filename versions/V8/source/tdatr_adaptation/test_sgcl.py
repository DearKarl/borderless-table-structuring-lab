"""CPU checks of geometry, annotation isolation and differentiable pathways."""
import inspect
import unittest

import torch

from .sgcl import (CellLocalizer, aligned_giou, cxcywh_to_xyxy,
                   localization_losses, rectangle_masks, relation_targets,
                   xyxy_to_cxcywh)


class CellLocalizationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.boxes = torch.tensor([[0., 0., 1., .5], [0., .5, .5, 1.], [.5, .5, 1., 1.]])
        self.spans = torch.tensor([[0, 0, 0, 1], [1, 1, 0, 0], [1, 1, 1, 1]])

    def test_known_geometry_and_spans(self):
        self.assertTrue(torch.equal(self.boxes, cxcywh_to_xyxy(xyxy_to_cxcywh(self.boxes))))
        self.assertTrue(torch.allclose(aligned_giou(self.boxes, self.boxes), torch.ones(3)))
        row, col = relation_targets(self.spans)
        self.assertEqual(row.tolist(), [[1, 0, 0], [0, 1, 1], [0, 1, 1]])
        self.assertEqual(col.tolist(), [[1, 1, 1], [1, 1, 0], [1, 0, 1]])
        masks = rectangle_masks(self.boxes, 2, 2)
        self.assertEqual(masks.sum((1, 2)).tolist(), [2, 1, 1])
        thin = rectangle_masks(torch.tensor([[.1, .1, .11, .12]]), 2, 2)
        self.assertAlmostEqual(float(thin.sum()), .0008, places=6)

    def test_forward_has_no_label_conditioning(self):
        names = list(inspect.signature(CellLocalizer.forward).parameters)
        self.assertEqual(names, ['self', 'layer_states', 'cell_ranges', 'vision_maps'])

    def test_gradients_reach_complete_cells_both_image_levels_and_heads(self):
        model = CellLocalizer(16, 3, [12, 20], dim=32, heads=4)
        hidden = [torch.randn(9, 16, requires_grad=True) for _ in range(3)]
        images = [torch.randn(12, 4, 6, requires_grad=True), torch.randn(20, 2, 3, requires_grad=True)]
        prediction = model(hidden, [(1, 4), (4, 6), (6, 9)], images)
        self.assertEqual(tuple(prediction['boxes'].shape), (4, 3, 4))
        losses = localization_losses(prediction, self.boxes, self.spans)
        self.assertTrue(all(torch.isfinite(v) and v > 0 for v in losses.values()))
        losses['weighted_localization'].backward()
        for h in hidden:
            self.assertTrue(torch.isfinite(h.grad).all())
            self.assertTrue((h.grad[1:].abs().sum(-1) > 0).all())
            self.assertEqual(float(h.grad[0].abs().sum()), 0.)
        for image in images:
            self.assertTrue(torch.isfinite(image.grad).all())
            self.assertGreater(float(image.grad.abs().sum()), 0.)
        groups = ['layer_weights', 'cell_projection', 'row_projection', 'column_projection',
                  'visual_projections', 'context', 'initial_box', 'refinement', 'mask_cells', 'mask_image']
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for group in groups:
            size = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters() if name.startswith(group))
            self.assertGreater(size, 0., group)
        # Every refinement stage must receive an actual gradient, including its sampler.
        for stage in model.refinement:
            self.assertGreater(float(stage.offsets.weight.grad.abs().sum()), 0.)
            self.assertGreater(float(stage.weights.weight.grad.abs().sum()), 0.)

    def test_invalid_supervision_is_rejected(self):
        model = CellLocalizer(16, 1, [12, 20], dim=32, heads=4)
        hidden = [torch.randn(9, 16)]
        images = [torch.randn(12, 4, 6), torch.randn(20, 2, 3)]
        with self.assertRaisesRegex(ValueError, 'unordered'):
            model(hidden, [(4, 6), (1, 5)], images)
        prediction = model(hidden, [(1, 4), (4, 6), (6, 9)], images)
        invalid = self.boxes.clone()
        invalid[0, 0] = -.01
        with self.assertRaisesRegex(ValueError, 'geometry'):
            localization_losses(prediction, invalid, self.spans)
        invalid_spans = self.spans.float()
        invalid_spans[0, 1] = .5
        with self.assertRaisesRegex(ValueError, 'logical spans'):
            localization_losses(prediction, self.boxes, invalid_spans)


if __name__ == '__main__':
    unittest.main()
