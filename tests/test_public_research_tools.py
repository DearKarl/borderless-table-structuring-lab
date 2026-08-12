from __future__ import annotations

from mpr_tsr_splitmerge_v2.otsl import (
    count_structure_tokens,
    normalize_decoded_otsl,
    to_native_otsl,
)
from mpr_tsr_splitmerge_v2.paired_metrics import summarize_paired


def test_native_otsl_preserves_one_literal_newline_per_row_boundary() -> None:
    assert to_native_otsl("<fcel>A<nl>\n<fcel>B<nl>") == "<fcel>A<nl>\n<fcel>B<nl>"


def test_decoded_otsl_removes_only_chat_controls() -> None:
    text = normalize_decoded_otsl("<|im_end|><fcel>A<nl><fcel>B<nl>")
    assert text == "<fcel>A<nl>\n<fcel>B<nl>"
    assert count_structure_tokens(text)["<fcel>"] == 2


def test_missing_structure_tokens_are_rejected() -> None:
    try:
        normalize_decoded_otsl("plain text")
    except ValueError as error:
        assert "structure token" in str(error)
    else:
        raise AssertionError("plain text was accepted as OTSL")


def test_paired_metrics_uses_the_removed_item_denominator_for_loo() -> None:
    baseline = [(8, 10), (90, 100)]
    candidate = [(10, 10), (89, 100)]
    summary = summarize_paired(baseline, candidate)
    assert summary.delta_correct == 1
    assert summary.largest_absolute_delta == 2
    assert summary.leave_one_out_delta_pp == -1.0
