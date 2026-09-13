from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn


D_MODEL = 128
CELL_FEATURES = 16
TOKEN_FEATURES = 12
MAX_GRID_DIMENSION = 64

REPRESENTATION_PREFIXES = (
    "visual_encoder.",
    "visual_projection.",
    "cell_projection.",
    "token_projection.",
    "layout_transformer.",
    "token_transformer.",
    "cell_to_token_attention.",
    "token_to_cell_attention.",
    "cell_fusion_norm.",
    "token_fusion_norm.",
    "encoder_norm.",
    "token_norm.",
)
DECISION_PREFIXES = (
    "decision_fusion.",
    "gate_head.",
    "benefit_head.",
    "risk_head.",
    "uncertainty_head.",
)
TOPOLOGY_PREFIXES = (
    "boundary_coordinate.",
    "split_axis_embedding.",
    "split_boundary_head.",
    "merge_left.",
    "merge_right.",
    "grid_shape_head.",
)
OWNERSHIP_PREFIXES = ("owner_token_projection.", "owner_cell_projection.", "unresolved_owner.")
TEXT_PREFIXES = ("text_token_projection.", "text_cell_projection.", "raw_text_bias.")
ALL_PREFIXES = REPRESENTATION_PREFIXES + DECISION_PREFIXES + TOPOLOGY_PREFIXES + OWNERSHIP_PREFIXES + TEXT_PREFIXES


class LocalCellVisualEncoder(nn.Module):
    def __init__(self, output_width: int = 64, crop_size: int = 5) -> None:
        super().__init__()
        self.crop_size = crop_size
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, output_width),
            nn.GELU(),
        )

    def forward(self, image: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or boxes.ndim != 3 or boxes.shape[-1] != 4:
            raise ValueError("LOCAL_VISUAL_INPUT_SHAPE_MISMATCH")
        batch, cells, _ = boxes.shape
        steps = torch.linspace(0.0, 1.0, self.crop_size, device=image.device, dtype=image.dtype)
        yy, xx = torch.meshgrid(steps, steps, indexing="ij")
        x0, y0, x1, y1 = boxes.unbind(-1)
        gx = x0[..., None, None] + (x1 - x0)[..., None, None] * xx
        gy = y0[..., None, None] + (y1 - y0)[..., None, None] * yy
        grid = torch.stack((gx * 2 - 1, gy * 2 - 1), dim=-1).reshape(batch, cells * self.crop_size, self.crop_size, 2)
        sampled = torch.nn.functional.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
        crops = sampled.reshape(batch, image.shape[1], cells, self.crop_size, self.crop_size)
        crops = crops.permute(0, 2, 1, 3, 4).reshape(batch * cells, image.shape[1], self.crop_size, self.crop_size)
        return self.network(crops).reshape(batch, cells, -1)


class ExplicitV2LayoutTransformer(nn.Module):
    """Phase-C successor with safe decision, ownership, text, and grid heads."""

    def __init__(self, seed: int = 20260818) -> None:
        super().__init__()
        self.visual_encoder = LocalCellVisualEncoder(64)
        self.visual_projection = nn.Linear(64, D_MODEL)
        self.cell_projection = nn.Linear(CELL_FEATURES, D_MODEL)
        self.token_projection = nn.Linear(TOKEN_FEATURES, D_MODEL)
        cell_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=8,
            dim_feedforward=384,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        token_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layout_transformer = nn.TransformerEncoder(cell_layer, num_layers=4, enable_nested_tensor=False)
        self.token_transformer = nn.TransformerEncoder(token_layer, num_layers=2, enable_nested_tensor=False)
        self.encoder_norm = nn.LayerNorm(D_MODEL)
        self.token_norm = nn.LayerNorm(D_MODEL)

        # The v1 successor encoded cells and OCR tokens independently.  These
        # two masked attention directions make ownership and text decisions
        # depend on a shared cell-token representation rather than a late
        # projected dot product.
        self.cell_to_token_attention = nn.MultiheadAttention(
            D_MODEL, num_heads=8, dropout=0.0, batch_first=True
        )
        self.token_to_cell_attention = nn.MultiheadAttention(
            D_MODEL, num_heads=8, dropout=0.0, batch_first=True
        )
        self.cell_fusion_norm = nn.LayerNorm(D_MODEL)
        self.token_fusion_norm = nn.LayerNorm(D_MODEL)

        # Four model-free diagnostics expose candidate-owner coverage and
        # ambiguity to the global KEEP/edit decision.  This is deliberately a
        # compact safety context, not a generic capacity increase.
        self.decision_fusion = nn.Sequential(
            nn.Linear(2 * D_MODEL + 4, D_MODEL),
            nn.GELU(),
            nn.LayerNorm(D_MODEL),
        )

        self.gate_head = nn.Linear(D_MODEL, 2)
        self.benefit_head = nn.Linear(D_MODEL, 1)
        self.risk_head = nn.Linear(D_MODEL, 1)
        self.uncertainty_head = nn.Linear(D_MODEL, 1)

        self.boundary_coordinate = nn.Linear(3, D_MODEL)
        self.split_axis_embedding = nn.Embedding(2, D_MODEL)
        self.split_boundary_head = nn.Sequential(nn.GELU(), nn.Linear(D_MODEL, 1))
        self.merge_left = nn.Linear(D_MODEL, 64)
        self.merge_right = nn.Linear(D_MODEL, 64)
        self.grid_shape_head = nn.Linear(D_MODEL, 2 * MAX_GRID_DIMENSION)

        self.owner_token_projection = nn.Linear(D_MODEL, 64)
        self.owner_cell_projection = nn.Linear(D_MODEL, 64)
        self.unresolved_owner = nn.Linear(D_MODEL, 1)

        self.text_token_projection = nn.Linear(D_MODEL, 64)
        self.text_cell_projection = nn.Linear(D_MODEL, 64)
        self.raw_text_bias = nn.Linear(D_MODEL, 1)

        self.reset_frozen_initialization(seed)
        self.cross_modal_fusion_enabled = True
        self.grounding_aware_decision_fusion_enabled = True
        self.assert_parameter_policy()

    def set_cross_modal_fusion_enabled(self, enabled: bool) -> None:
        """Enable or disable only the bidirectional cell-token fusion path.

        This non-persistent switch exists solely for the contract-authorized
        one-factor diagnostic.  It does not alter parameters or state-dict
        identity and defaults to the production architecture.
        """
        self.cross_modal_fusion_enabled = bool(enabled)

    def set_grounding_aware_decision_fusion_enabled(self, enabled: bool) -> None:
        """Enable only token context and grounding diagnostics at decision time.

        This non-persistent switch is the contract-authorized one-factor
        diagnostic for the global KEEP/edit path.  Disabling it leaves the
        learned cell representation and all ownership, text, and topology
        paths unchanged while zeroing only the additional decision inputs.
        """
        self.grounding_aware_decision_fusion_enabled = bool(enabled)

    def reset_frozen_initialization(self, seed: int) -> None:
        prior = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            for name, parameter in self.named_parameters():
                with torch.no_grad():
                    if name.endswith("bias"):
                        parameter.zero_()
                    elif parameter.ndim == 1 and "norm" in name.casefold():
                        parameter.fill_(1.0)
                    elif parameter.ndim == 1:
                        parameter.zero_()
                    else:
                        nn.init.xavier_uniform_(parameter)
            # Sequential containers do not necessarily expose "norm" in a
            # LayerNorm parameter name (for example decision_fusion.2.weight).
            # Restore every LayerNorm scale by module type so new decision
            # fusion cannot be deterministically collapsed to zero.
            for module in self.modules():
                if isinstance(module, nn.LayerNorm):
                    with torch.no_grad():
                        module.weight.fill_(1.0)
                        if module.bias is not None:
                            module.bias.zero_()
        finally:
            torch.random.set_rng_state(prior)

    def assert_parameter_policy(self) -> None:
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        if not names or not all(name.startswith(ALL_PREFIXES) for name in names):
            raise RuntimeError("TRAINABLE_ALLOWLIST_MISMATCH")
        for prefix in ALL_PREFIXES:
            if not any(name.startswith(prefix) for name in names):
                raise RuntimeError(f"EMPTY_PARAMETER_FAMILY:{prefix}")

    def gradient_family_inventory(self) -> dict[str, bool]:
        result = {prefix.rstrip("."): False for prefix in ALL_PREFIXES}
        for name, parameter in self.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not parameter.grad.ne(0).any():
                continue
            for prefix in ALL_PREFIXES:
                if name.startswith(prefix):
                    result[prefix.rstrip(".")] = True
        return result

    def parameter_inventory(self) -> dict[str, Any]:
        families: dict[str, int] = {prefix.rstrip("."): 0 for prefix in ALL_PREFIXES}
        for name, parameter in self.named_parameters():
            for prefix in ALL_PREFIXES:
                if name.startswith(prefix):
                    families[prefix.rstrip(".")] += parameter.numel()
                    break
        return {
            "total_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "families": families,
        }

    def forward(
        self,
        image: torch.Tensor,
        cell_features: torch.Tensor,
        cell_boxes: torch.Tensor,
        cell_mask: torch.Tensor,
        grid_shape: torch.Tensor,
        token_features: torch.Tensor,
        token_mask: torch.Tensor,
        token_owner_candidate_mask: torch.Tensor,
        text_candidate_token_weights: torch.Tensor,
        text_candidate_mask: torch.Tensor,
        text_candidate_is_raw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if cell_mask.dtype != torch.bool or token_mask.dtype != torch.bool:
            raise ValueError("MASK_DTYPE_MISMATCH")
        if cell_features.shape[:2] != cell_mask.shape or cell_boxes.shape[:2] != cell_mask.shape:
            raise ValueError("CELL_TENSOR_SHAPE_MISMATCH")
        if token_features.shape[:2] != token_mask.shape:
            raise ValueError("TOKEN_TENSOR_SHAPE_MISMATCH")
        batch, cells = cell_mask.shape
        tokens = token_mask.shape[1]
        if cells < 1 or not cell_mask.any(1).all() or tokens < 1:
            raise ValueError("EMPTY_CELL_BATCH_OR_TOKEN_AXIS")
        if grid_shape.shape != (batch, 2):
            raise ValueError("GRID_SHAPE_TENSOR_MISMATCH")
        if (grid_shape < 1).any() or (grid_shape > MAX_GRID_DIMENSION).any():
            raise ValueError("GRID_SHAPE_OUTSIDE_BOUNDED_CLASSES")
        if token_owner_candidate_mask.shape != (batch, tokens, cells + 1):
            raise ValueError("OWNER_CANDIDATE_MASK_SHAPE_MISMATCH")
        if text_candidate_token_weights.ndim != 4 or text_candidate_token_weights.shape[:2] != (batch, cells) or text_candidate_token_weights.shape[-1] != tokens:
            raise ValueError("TEXT_CANDIDATE_TOKEN_WEIGHT_SHAPE_MISMATCH")
        candidates = text_candidate_token_weights.shape[2]
        if text_candidate_mask.shape != (batch, cells, candidates) or text_candidate_is_raw.shape != (batch, cells, candidates):
            raise ValueError("TEXT_CANDIDATE_MASK_SHAPE_MISMATCH")
        if token_owner_candidate_mask.dtype != torch.bool or text_candidate_mask.dtype != torch.bool or text_candidate_is_raw.dtype != torch.bool:
            raise ValueError("CANDIDATE_MASK_DTYPE_MISMATCH")
        if not token_owner_candidate_mask[..., -1].all():
            raise ValueError("UNRESOLVED_OWNER_CLASS_MUST_BE_AVAILABLE")
        has_raw_fallback = (text_candidate_mask & text_candidate_is_raw).any(-1)
        if not (has_raw_fallback | ~cell_mask).all():
            raise ValueError("RAW_TEXT_FALLBACK_CANDIDATE_MUST_BE_AVAILABLE")
        if (token_owner_candidate_mask[..., :-1] & ~cell_mask.unsqueeze(1)).any():
            raise ValueError("OWNER_CANDIDATE_REFERENCES_PADDED_CELL")
        if ((text_candidate_token_weights != 0) & ~token_mask.unsqueeze(1).unsqueeze(1)).any():
            raise ValueError("TEXT_CANDIDATE_REFERENCES_PADDED_TOKEN")
        if not torch.isfinite(text_candidate_token_weights).all() or (text_candidate_token_weights < 0).any():
            raise ValueError("TEXT_CANDIDATE_TOKEN_WEIGHTS_INVALID")
        if (text_candidate_is_raw.unsqueeze(-1) & (text_candidate_token_weights != 0)).any():
            raise ValueError("RAW_TEXT_CANDIDATE_MUST_NOT_COPY_OCR_TOKENS")
        candidate_weight = text_candidate_token_weights.sum(-1)
        if (text_candidate_mask & ~text_candidate_is_raw & (candidate_weight <= 0)).any():
            raise ValueError("GROUNDED_TEXT_CANDIDATE_HAS_NO_TOKEN")

        local = self.visual_encoder(image, cell_boxes)
        cell_hidden = self.cell_projection(cell_features) + self.visual_projection(local)
        cell_hidden = self.encoder_norm(self.layout_transformer(cell_hidden, src_key_padding_mask=~cell_mask))
        cell_hidden = cell_hidden * cell_mask.unsqueeze(-1)
        safe_token_mask = token_mask.clone()
        empty_token_rows = ~safe_token_mask.any(1)
        safe_token_mask[empty_token_rows, 0] = True
        token_hidden = self.token_norm(self.token_transformer(self.token_projection(token_features), src_key_padding_mask=~safe_token_mask))
        token_hidden = token_hidden * token_mask.unsqueeze(-1)

        if self.cross_modal_fusion_enabled:
            cell_delta, _ = self.cell_to_token_attention(
                query=cell_hidden,
                key=token_hidden,
                value=token_hidden,
                key_padding_mask=~safe_token_mask,
                need_weights=False,
            )
            token_delta, _ = self.token_to_cell_attention(
                query=token_hidden,
                key=cell_hidden,
                value=cell_hidden,
                key_padding_mask=~cell_mask,
                need_weights=False,
            )
            cell_hidden = self.cell_fusion_norm(cell_hidden + cell_delta) * cell_mask.unsqueeze(-1)
            token_hidden = self.token_fusion_norm(token_hidden + token_delta) * token_mask.unsqueeze(-1)

        pooled_cell = cell_hidden.sum(1) / cell_mask.sum(1, keepdim=True).clamp_min(1)
        token_count = token_mask.sum(1, keepdim=True).to(cell_hidden.dtype)
        pooled_token = token_hidden.sum(1) / token_count.clamp_min(1)

        owner_candidates = token_owner_candidate_mask[..., :-1] & cell_mask.unsqueeze(1)
        candidate_count = owner_candidates.sum(-1).to(cell_hidden.dtype)
        valid_token = token_mask.to(cell_hidden.dtype)
        safe_token_count = token_count.clamp_min(1)
        cell_count = cell_mask.sum(1, keepdim=True).to(cell_hidden.dtype).clamp_min(1)
        owner_coverage = ((candidate_count > 0).to(cell_hidden.dtype) * valid_token).sum(1, keepdim=True) / safe_token_count
        owner_ambiguity = ((candidate_count > 1).to(cell_hidden.dtype) * valid_token).sum(1, keepdim=True) / safe_token_count
        mean_candidate_fraction = (candidate_count * valid_token).sum(1, keepdim=True) / (safe_token_count * cell_count)
        token_cell_ratio = torch.tanh(token_count / cell_count)
        grounding_diagnostics = torch.cat(
            (owner_coverage, owner_ambiguity, mean_candidate_fraction, token_cell_ratio), dim=-1
        )
        grounding_diagnostics = grounding_diagnostics * (~empty_token_rows).unsqueeze(-1).to(cell_hidden.dtype)
        if self.grounding_aware_decision_fusion_enabled:
            decision_token_context = pooled_token
            decision_grounding_diagnostics = grounding_diagnostics
        else:
            decision_token_context = torch.zeros_like(pooled_token)
            decision_grounding_diagnostics = torch.zeros_like(grounding_diagnostics)
        decision_context = self.decision_fusion(
            torch.cat((pooled_cell, decision_token_context, decision_grounding_diagnostics), dim=-1)
        )

        left, right = self.merge_left(cell_hidden), self.merge_right(cell_hidden)
        merge_logits = torch.matmul(left, right.transpose(1, 2)) / math.sqrt(left.shape[-1])
        merge_logits = 0.5 * (merge_logits + merge_logits.transpose(1, 2))

        boundaries = max(1, int(grid_shape.max().item()) - 1)
        boundary_index = torch.arange(1, boundaries + 1, device=cell_hidden.device, dtype=cell_hidden.dtype)
        denominators = grid_shape.to(cell_hidden.dtype).clamp_min(1).unsqueeze(-1)
        fractions = (boundary_index.view(1, 1, -1) / denominators).clamp(0.0, 1.0)
        coordinate = torch.stack((fractions, torch.sin(math.pi * fractions), torch.cos(math.pi * fractions)), dim=-1)
        coordinate_hidden = self.boundary_coordinate(coordinate)
        axes = self.split_axis_embedding(torch.arange(2, device=cell_hidden.device)).view(1, 2, 1, D_MODEL)
        boundary_hidden = cell_hidden.unsqueeze(2).unsqueeze(3) + coordinate_hidden.unsqueeze(1) + axes.unsqueeze(1)
        split_boundary_logits = self.split_boundary_head(boundary_hidden).squeeze(-1)

        owner_tokens = self.owner_token_projection(token_hidden)
        owner_cells = self.owner_cell_projection(cell_hidden)
        owner_cell_logits = torch.matmul(owner_tokens, owner_cells.transpose(1, 2)) / math.sqrt(owner_tokens.shape[-1])
        owner_unmasked_logits = torch.cat((owner_cell_logits, self.unresolved_owner(token_hidden)), dim=-1)
        owner_logits = owner_unmasked_logits.masked_fill(~token_owner_candidate_mask, torch.finfo(owner_unmasked_logits.dtype).min)

        text_cells = self.text_cell_projection(cell_hidden)
        token_weights = text_candidate_token_weights.to(token_hidden.dtype)
        normalizer = token_weights.sum(-1, keepdim=True).clamp_min(1.0)
        candidate_hidden = torch.einsum("bckt,btd->bckd", token_weights, token_hidden) / normalizer
        candidate_hidden = candidate_hidden + text_candidate_is_raw.unsqueeze(-1).to(candidate_hidden.dtype) * cell_hidden.unsqueeze(2)
        text_candidates = self.text_token_projection(candidate_hidden)
        text_unmasked_logits = (text_cells.unsqueeze(2) * text_candidates).sum(-1) / math.sqrt(text_cells.shape[-1])
        text_unmasked_logits = text_unmasked_logits + text_candidate_is_raw.to(text_unmasked_logits.dtype) * self.raw_text_bias(cell_hidden)
        text_logits = text_unmasked_logits.masked_fill(~text_candidate_mask, torch.finfo(text_unmasked_logits.dtype).min)

        grid_logits = self.grid_shape_head(pooled_cell).reshape(batch, 2, MAX_GRID_DIMENSION)
        return {
            "gate_logits": self.gate_head(decision_context),
            "benefit_logit": self.benefit_head(decision_context).squeeze(-1),
            "risk_logit": self.risk_head(decision_context).squeeze(-1),
            "uncertainty_logit": self.uncertainty_head(decision_context).squeeze(-1),
            "split_boundary_logits": split_boundary_logits,
            "merge_logits": merge_logits,
            "token_owner_logits": owner_logits,
            "token_owner_unmasked_logits": owner_unmasked_logits,
            "grounded_text_logits": text_logits,
            "grounded_text_unmasked_logits": text_unmasked_logits,
            "grid_shape_logits": grid_logits,
            "fused_cell_hidden": cell_hidden,
            "fused_token_hidden": token_hidden,
            "grounding_diagnostics": grounding_diagnostics,
            "decision_context": decision_context,
        }
