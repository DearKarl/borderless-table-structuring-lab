from __future__ import annotations

import re


CONTROL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")
STRUCTURE_TOKENS = ("<fcel>", "<ecel>", "<lcel>", "<ucel>", "<xcel>", "<nl>")


def to_native_otsl(text: str) -> str:
    """Normalize OTSL row boundaries to the literal-newline dialect."""
    normalized = text.replace("<nl>\n", "<nl>")
    return normalized.replace("<nl>", "<nl>\n").rstrip("\n")


def strip_chat_controls(text: str) -> str:
    """Remove only chat wrapper tokens after structure-preserving decoding."""
    for token in CONTROL_TOKENS:
        text = text.replace(token, "")
    return text


def validate_structure_tokens(text: str) -> None:
    """Reject missing or non-atomic structural markers in a decoded OTSL string."""
    if not any(token in text for token in STRUCTURE_TOKENS):
        raise ValueError("decoded output contains no OTSL structure token")
    if "<nl>\n" not in text and "<nl>" in text:
        raise ValueError("OTSL newline token is not followed by a literal newline")


def normalize_decoded_otsl(text: str) -> str:
    """Apply the safe post-decode normalization used by the public examples."""
    normalized = to_native_otsl(strip_chat_controls(text))
    validate_structure_tokens(normalized)
    return normalized


def count_structure_tokens(text: str) -> dict[str, int]:
    """Count exact OTSL markers without interpreting cell text."""
    return {token: len(re.findall(re.escape(token), text)) for token in STRUCTURE_TOKENS}
