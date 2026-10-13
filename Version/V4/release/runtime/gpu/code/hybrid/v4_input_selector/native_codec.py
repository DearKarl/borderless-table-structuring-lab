"""Serialize frozen Tele final blocks; no inference, GT, or content repair.

Caller must capture blocks AFTER native post_process and BEFORE middle-json or
Markdown conversion, with abandon_paratext=False and abandon_list=False. Boxes
must already be inverse-mapped to normalized original upright page coordinates.
This module does not implement that capture or inverse transform. Malformed
boxes remain prediction instances with bbox=None, as in local_quality.
"""
from collections.abc import Mapping
from types import MappingProxyType

from .local_quality import InvalidMeasurement, prediction_from_native_blocks

FINAL_STAGE = "tele_final_post_process_before_middle_json"
CANONICAL_FRAME = "canonical_upright_original_page"
# Frozen TeleOCR/vlm_utils/structs.py BLOCK_TYPES, excluding the equation_block
# type which native post_process unconditionally removes. Unknown is a real
# native type; an unrecognized string is a contract error.
CATEGORY_MAP = MappingProxyType({
    "text": "text", "title": "text", "table": "table", "image": "text",
    "code": "text", "algorithm": "text", "header": "text", "footer": "text",
    "page_number": "text", "page_footnote": "text", "aside_text": "text",
    "equation": "formula", "ref_text": "text", "list": "text",
    "phonetic": "text", "table_caption": "text", "image_caption": "text",
    "code_caption": "text", "table_footnote": "text", "image_footnote": "text",
    "unknown": "text", "seal": "text", "char": "text",
})


def final_native_prediction(blocks, *, page_id, input_sha256, stage, frame):
    """Project each final native block in its own order, without GT arguments.

    stage/frame are required caller assertions, not evidence of a live tap.
    Content is str or None per ContentBlock. Table/formula empty content and bad
    geometry are retained for existing scorer failure handling, never dropped.
    Native inline TeX, equation labels and HTML are passed through unchanged.
    """
    if stage != FINAL_STAGE or frame != CANONICAL_FRAME:
        raise InvalidMeasurement("Final native stage and normalized original frame required")
    final_blocks = list(blocks)
    for block in final_blocks:
        if not isinstance(block, Mapping):
            raise InvalidMeasurement("Final native block must be a mapping")
        kind = block.get("type")
        if kind == "equation_block":
            raise InvalidMeasurement("equation_block contradicts completed native post_process")
        if not isinstance(kind, str) or kind not in CATEGORY_MAP:
            raise InvalidMeasurement("Unrecognized final native block type")
        if block.get("content") is not None and not isinstance(block["content"], str):
            raise InvalidMeasurement("Native content must be str or None")
    return prediction_from_native_blocks(
        final_blocks, page_id=page_id, input_sha256=input_sha256,
        category_map=CATEGORY_MAP,
    )
