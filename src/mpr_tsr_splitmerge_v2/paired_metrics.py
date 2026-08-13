from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PairedSummary:
    baseline_correct: int
    candidate_correct: int
    fixed_denominator: int
    delta_correct: int
    delta_pp: float
    help_count: int
    hurt_count: int
    same_count: int
    largest_absolute_delta: int
    leave_one_out_delta_pp: float


def summarize_paired(
    baseline: Sequence[tuple[int, int]],
    candidate: Sequence[tuple[int, int]],
) -> PairedSummary:
    """Summarize paired fixed-denominator cell results.

    Each item is ``(correct_cells, fixed_gt_denominator)``. The two sequences
    must describe the same examples in the same order.
    """
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired results must be non-empty and equally sized")
    pairs = list(zip(baseline, candidate))
    if any(base_den <= 0 or candidate_den <= 0 for (_, base_den), (_, candidate_den) in pairs):
        raise ValueError("fixed denominators must be positive")
    if any(base_den != candidate_den for (_, base_den), (_, candidate_den) in pairs):
        raise ValueError("baseline and candidate denominators must match")

    deltas = [candidate_item[0] - base_item[0] for base_item, candidate_item in pairs]
    denominator = sum(den for _, den in baseline)
    baseline_correct = sum(correct for correct, _ in baseline)
    candidate_correct = sum(correct for correct, _ in candidate)
    largest_index = max(range(len(deltas)), key=lambda index: abs(deltas[index]))
    largest = deltas[largest_index]
    loo_denominator = denominator - baseline[largest_index][1]
    if loo_denominator <= 0:
        raise ValueError("leave-one-out denominator must be positive")
    loo_delta = candidate_correct - baseline_correct - largest
    return PairedSummary(
        baseline_correct=baseline_correct,
        candidate_correct=candidate_correct,
        fixed_denominator=denominator,
        delta_correct=candidate_correct - baseline_correct,
        delta_pp=100.0 * (candidate_correct - baseline_correct) / denominator,
        help_count=sum(delta > 0 for delta in deltas),
        hurt_count=sum(delta < 0 for delta in deltas),
        same_count=sum(delta == 0 for delta in deltas),
        largest_absolute_delta=largest,
        leave_one_out_delta_pp=100.0 * loo_delta / loo_denominator,
    )
