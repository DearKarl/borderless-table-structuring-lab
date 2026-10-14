"""TDATR-style cell localization, independently implemented for a native VLM.

The caller supplies real spatial feature maps and complete cell token intervals.
No annotation enters forward(). Ground-truth geometry is used only by losses.
Backbone integration and full-model validation are separate requirements.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def xyxy_to_cxcywh(boxes):
    return torch.cat(((boxes[..., :2] + boxes[..., 2:]) / 2,
                      boxes[..., 2:] - boxes[..., :2]), dim=-1)


def cxcywh_to_xyxy(boxes):
    return torch.cat((boxes[..., :2] - boxes[..., 2:] / 2,
                      boxes[..., :2] + boxes[..., 2:] / 2), dim=-1)


def aligned_giou(predicted, target):
    """One-to-one GIoU; there is no matching or cross-cell reassignment."""
    intersection = (torch.minimum(predicted[..., 2:], target[..., 2:]) -
                    torch.maximum(predicted[..., :2], target[..., :2])).clamp_min(0).prod(-1)
    p_area = (predicted[..., 2:] - predicted[..., :2]).clamp_min(0).prod(-1)
    t_area = (target[..., 2:] - target[..., :2]).clamp_min(0).prod(-1)
    union = p_area + t_area - intersection
    enclosure = (torch.maximum(predicted[..., 2:], target[..., 2:]) -
                 torch.minimum(predicted[..., :2], target[..., :2])).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-7) - (enclosure - union) / enclosure.clamp_min(1e-7)


def relation_targets(logical_spans):
    """Inclusive row/column ranges [r0,r1,c0,c1]; spanning cells can share rows."""
    if logical_spans.ndim != 2 or logical_spans.shape[-1] != 4 or not len(logical_spans):
        raise ValueError('Expected nonempty N by 4 logical spans')
    if (not torch.isfinite(logical_spans).all() or (logical_spans < 0).any() or
            (logical_spans != logical_spans.floor()).any() or
            (logical_spans[:, 1] < logical_spans[:, 0]).any() or
            (logical_spans[:, 3] < logical_spans[:, 2]).any()):
        raise ValueError('Invalid logical spans')
    return tuple((torch.maximum(logical_spans[:, a, None], logical_spans[None, :, a]) <=
                  torch.minimum(logical_spans[:, a + 1, None], logical_spans[None, :, a + 1])).float()
                 for a in (0, 2))


def rectangle_masks(boxes, height, width):
    """Fractional pixel coverage avoids empty targets for subpixel thin cells."""
    x0 = torch.arange(width, device=boxes.device, dtype=boxes.dtype) / width
    y0 = torch.arange(height, device=boxes.device, dtype=boxes.dtype) / height
    x = (torch.minimum(x0[None] + 1 / width, boxes[:, 2, None]) -
         torch.maximum(x0[None], boxes[:, 0, None])).clamp_min(0) * width
    y = (torch.minimum(y0[None] + 1 / height, boxes[:, 3, None]) -
         torch.maximum(y0[None], boxes[:, 1, None])).clamp_min(0) * height
    return y[:, :, None] * x[:, None, :]


def position_encoding(coords, channels):
    if channels % (2 * coords.shape[-1]):
        raise ValueError('Position encoding dimension is incompatible')
    count = channels // (2 * coords.shape[-1])
    frequencies = 10000 ** (-torch.arange(count, device=coords.device, dtype=coords.dtype) / max(count, 1))
    angles = coords[..., None] * (2 * math.pi) * frequencies
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-3)


class ContextBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(4)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(5)])
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, cells, visual, masks):
        x = cells[None]
        for i, mask in enumerate((*masks, None)):
            value, _ = self.attention[i](x, x, x, attn_mask=mask, need_weights=False)
            x = self.norms[i](x + value)
        value, _ = self.attention[3](x, visual[None], visual[None], need_weights=False)
        x = self.norms[3](x + value)
        return self.norms[4](x + self.ffn(x))[0]


class SpatialRefinement(nn.Module):
    """Two-level, box-anchored deformable sampling without compiled extensions."""
    def __init__(self, dim, heads, points=4):
        super().__init__()
        self.dim, self.heads, self.points = dim, heads, points
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.offsets = nn.Linear(dim, heads * 2 * points * 2)
        self.weights = nn.Linear(dim, heads * 2 * points)
        self.output = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.delta = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        # Radial initial offsets cover the initial cell box. No RNG is reset.
        nn.init.zeros_(self.offsets.weight)
        theta = torch.arange(heads, dtype=torch.float32) * (2 * math.pi / heads)
        directions = torch.stack((theta.cos(), theta.sin()), -1)
        directions = directions / directions.abs().amax(-1, keepdim=True)
        radial = directions[:, None, None, :] * torch.arange(1, points + 1)[None, None, :, None]
        with torch.no_grad():
            self.offsets.bias.copy_(radial.expand(heads, 2, points, 2).reshape(-1))
        nn.init.zeros_(self.weights.weight)
        nn.init.zeros_(self.weights.bias)

    def forward(self, cells, reference, feature_maps):
        pos = position_encoding(reference, self.dim)
        q = (cells + pos)[None]
        value, _ = self.self_attention(q, q, cells[None], need_weights=False)
        cells = self.norms[0](cells + value[0])
        n = len(cells)
        offsets = self.offsets(cells + pos).reshape(n, self.heads, 2, self.points, 2)
        weights = self.weights(cells + pos).reshape(n, self.heads, 2 * self.points).softmax(-1)
        weights = weights.reshape(n, self.heads, 2, self.points)
        locations = (reference[:, None, None, None, :2] +
                     offsets / self.points * reference[:, None, None, None, 2:] * .5)
        sampled_levels = []
        for level, features in enumerate(feature_maps):
            _, height, width = features.shape
            values = features.reshape(self.heads, self.dim // self.heads, height, width)
            grid = locations[:, :, level].permute(1, 0, 2, 3) * 2 - 1
            samples = F.grid_sample(values, grid, align_corners=False, padding_mode='zeros')
            samples = samples.permute(2, 0, 3, 1)
            sampled_levels.append((samples * weights[:, :, level, :, None]).sum(2))
        value = torch.stack(sampled_levels).sum(0).reshape(n, self.dim)
        cells = self.norms[1](cells + self.output(value))
        cells = self.norms[2](cells + self.ffn(cells))
        updated = (torch.logit(reference.clamp(1e-5, 1 - 1e-5)) + self.delta(cells)).sigmoid()
        return cells, updated


class CellLocalizer(nn.Module):
    def __init__(self, language_dim, decoder_layers, vision_channels, dim=256, heads=8,
                 context_layers=2, refinement_layers=3):
        super().__init__()
        if len(vision_channels) != 2 or dim % heads or dim % 8 or decoder_layers < 1:
            raise ValueError('Two feature levels and compatible attention dimensions are required')
        self.language_dim, self.decoder_layers = language_dim, decoder_layers
        # Mirrors linear aggregation along the layer axis, followed by LN.
        self.layer_weights = nn.Parameter(torch.full((decoder_layers,), 1 / decoder_layers))
        self.layer_bias = nn.Parameter(torch.zeros(()))
        self.language_norm = nn.LayerNorm(language_dim)
        self.cell_projection = nn.Linear(language_dim, dim)
        self.visual_projections = nn.ModuleList([nn.Conv2d(c, dim, 1) for c in vision_channels])
        self.level_embeddings = nn.Parameter(torch.empty(2, dim))
        nn.init.normal_(self.level_embeddings, std=.02)
        self.row_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.column_projection = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.context = nn.ModuleList([ContextBlock(dim, heads) for _ in range(context_layers)])
        self.initial_box = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))
        self.refinement = nn.ModuleList([SpatialRefinement(dim, heads) for _ in range(refinement_layers)])
        self.mask_cells = nn.Linear(dim, dim)
        self.mask_image = nn.Conv2d(dim, dim, 1)

    def pool_cells(self, layer_states, cell_ranges):
        if len(layer_states) != self.decoder_layers:
            raise ValueError('Wrong number of decoder layers')
        shape = layer_states[0].shape
        if len(shape) != 2 or shape[1] != self.language_dim or any(x.shape != shape for x in layer_states):
            raise ValueError('Decoder states must have shape tokens by language_dim')
        previous_start, previous_end = 0, 0
        if not cell_ranges:
            raise ValueError('No cells to localize')
        for start, end in cell_ranges:
            if not (0 <= previous_start <= start < end <= shape[0] and previous_end <= end):
                raise ValueError('Invalid or unordered full-cell token ranges')
            # Native BPE tokens can straddle boundaries. The token mapper audits
            # that shared token; both complete cell ranges retain it.
            previous_start, previous_end = start, end
        hidden = self.layer_bias + sum(w * h for w, h in zip(self.layer_weights, layer_states))
        hidden = self.language_norm(hidden)
        return self.cell_projection(torch.stack([hidden[start:end].mean(0) for start, end in cell_ranges]))

    def forward(self, layer_states, cell_ranges, vision_maps):
        if len(vision_maps) != 2:
            raise ValueError('Exactly two actual spatial feature levels are required')
        cells = self.pool_cells(layer_states, cell_ranges)
        features, visual_tokens = [], []
        for level, (projection, image) in enumerate(zip(self.visual_projections, vision_maps)):
            if image.ndim != 3 or min(image.shape[-2:]) < 1:
                raise ValueError('Expected a channels by height by width spatial map')
            value = projection(image[None])[0]
            features.append(value)
            _, h, w = value.shape
            y, x = torch.meshgrid((torch.arange(h, device=value.device, dtype=value.dtype) + .5) / h,
                                  (torch.arange(w, device=value.device, dtype=value.dtype) + .5) / w, indexing='ij')
            positions = position_encoding(torch.stack((x, y), -1), value.shape[0]).reshape(h * w, -1)
            visual_tokens.append(value.flatten(1).T + positions + self.level_embeddings[level])
        visual = torch.cat(visual_tokens)
        logits, blocked = [], []
        for projection in (self.row_projection, self.column_projection):
            encoded = projection(cells)
            similarity = encoded @ encoded.T / math.sqrt(encoded.shape[-1])
            allowed = similarity.detach().sigmoid() > .5
            allowed = allowed | torch.eye(len(cells), device=cells.device, dtype=torch.bool)
            logits.append(similarity)
            blocked.append(~allowed)
        for layer in self.context:
            cells = layer(cells, visual, blocked)
        references = [self.initial_box(cells).sigmoid()]
        for layer in self.refinement:
            cells, reference = layer(cells, references[-1], features)
            references.append(reference)
        masks = torch.einsum('nd,dhw->nhw', self.mask_cells(cells), self.mask_image(features[0][None])[0])
        masks = masks / math.sqrt(cells.shape[-1])
        return dict(boxes=torch.stack([cxcywh_to_xyxy(x) for x in references]),
                    row_logits=logits[0], column_logits=logits[1], mask_logits=masks)


def localization_losses(prediction, target_boxes, logical_spans):
    """Mean per cell, per coordinate, per refinement stage; then per example.

    The caller averages examples over its effective batch. Soft mask BCE and
    Dice are averaged equally. Relation loss averages row and column BCE.
    """
    boxes = target_boxes.float()
    n = len(boxes)
    if (boxes.shape != (n, 4) or n == 0 or not torch.isfinite(boxes).all() or
            (boxes < 0).any() or (boxes > 1).any() or (boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError('Invalid normalized target geometry')
    predicted = prediction['boxes'].float()
    if predicted.ndim != 3 or predicted.shape[1:] != (n, 4) or len(logical_spans) != n:
        raise ValueError('Cell correspondence differs')
    rows, columns = relation_targets(logical_spans)
    target = boxes.unsqueeze(0).expand_as(predicted)
    box = F.l1_loss(predicted, target)
    iou = (1 - aligned_giou(predicted, target)).mean()
    mask_logits = prediction['mask_logits'].float()
    if mask_logits.ndim != 3 or mask_logits.shape[0] != n:
        raise ValueError('Mask cell correspondence differs')
    masks = rectangle_masks(boxes, *mask_logits.shape[-2:])
    bce = F.binary_cross_entropy_with_logits(mask_logits, masks, reduction='none').flatten(1).mean(1)
    probabilities = mask_logits.sigmoid().flatten(1)
    target_mask = masks.flatten(1)
    dice = 1 - (2 * (probabilities * target_mask).sum(1) + 1) / (probabilities.sum(1) + target_mask.sum(1) + 1)
    mask = ((bce + dice) / 2).mean()
    relation = (F.binary_cross_entropy_with_logits(prediction['row_logits'].float(), rows) +
                F.binary_cross_entropy_with_logits(prediction['column_logits'].float(), columns)) / 2
    total = .05 * box + .03 * iou + .03 * mask + .05 * relation
    return dict(box=box, iou=iou, mask=mask, relation=relation, weighted_localization=total)
