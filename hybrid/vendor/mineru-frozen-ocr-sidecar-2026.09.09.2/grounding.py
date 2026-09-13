"""Reference-blind, conservative DOM-row to detected-text-box grounding.

No OCR is executed here. A proposal is NOT a commit: the caller must check
image bounds, run the two frozen recognizers on the actual detected box, and
require both results to equal detector_text before a structure-safe writeback.
"""

from collections import Counter, defaultdict
import hashlib
import html
from html.parser import HTMLParser
import importlib.util
import math
from pathlib import Path
import re


CORE_SHA256 = "07c5fc2e82ae4136d36dc52b722c1dda686eee7a83ac79a54e609ef3a39c5334"
CORE_PATH = Path(__file__).resolve().parent.parent / "mineru-frozen-ocr-sidecar-2026.09.09.1" / "core.py"
_MULTILINE = re.compile(r"[\r\n\v\f\x85\u2028\u2029]")
_FORMULA = re.compile(r"[$\\^{}=<>*/+]")


def _core():
    if hashlib.sha256(CORE_PATH.read_bytes()).hexdigest() != CORE_SHA256:
        raise ValueError("FROZEN_CORE_SHA_MISMATCH")
    spec = importlib.util.spec_from_file_location("_frozen_sidecar_grounding_core", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Rows(HTMLParser):
    """Only assign DOM row membership after the frozen strict parser admits it."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.rows = []
        self.index = 0
        self.stack = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            if not self.stack or self.stack[-1] not in {"table", "thead", "tbody", "tfoot"}:
                raise ValueError("NONSTRUCTURAL_DOM_ROW")
            self.rows.append([])
        elif tag in {"td", "th"}:
            self.rows[-1].append(self.index)
            self.index += 1
        if tag not in {"area", "base", "br", "col", "embed", "hr", "img", "input",
                       "link", "meta", "param", "source", "track", "wbr"}:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        self.stack.pop()

    def handle_startendtag(self, tag, attrs):
        # The frozen strict parser already permits only void self-closing tags.
        # HTMLParser's default start->end dispatch would incorrectly pop a row
        # or colgroup here, because a void start never pushes the stack.
        pass


def _dom_rows(raw_html):
    core = _core()
    parsed = core._parse(raw_html)
    if any(c["merged"] or c["nested"] for c in parsed.cells):
        raise ValueError("COMPLEX_CELL_STRUCTURE")
    rows = _Rows()
    rows.feed(raw_html)
    rows.close()
    if (not rows.rows or rows.index != len(parsed.cells)
            or len(rows.rows[0]) < 3
            or any(len(row) != len(rows.rows[0]) for row in rows.rows)):
        raise ValueError("NONRECTANGULAR_OR_TOO_FEW_COLUMNS")
    result = []
    for row_number, indices in enumerate(rows.rows):
        row = []
        for column, index in enumerate(indices):
            cell = parsed.cells[index]
            text = html.unescape(raw_html[cell["start"]:cell["end"]])
            row.append({"cell_index": index, "row": row_number, "col": column,
                        "original": text, "text": text.strip()})
        result.append(row)
    return result


def _tokens(tokens):
    if not isinstance(tokens, list):
        raise ValueError("INVALID_TOKEN_LIST")
    seen = set()
    result = []
    for token in tokens:
        if not isinstance(token, dict):
            raise ValueError("INVALID_TOKEN")
        index, text, box = token.get("token_index"), token.get("text"), token.get("bbox")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("INVALID_OR_DUPLICATE_TOKEN_INDEX")
        if not isinstance(text, str):
            raise ValueError("INVALID_TOKEN_TEXT")
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or box[0] < 0 or box[1] < 0 or box[2] <= box[0] or box[3] <= box[1]):
            raise ValueError("INVALID_DETECTED_BOX")
        seen.add(index)
        result.append({"token_index": index, "text": text.strip(),
                       "raw_text": text, "bbox": list(box)})
    return result


def _same_y_row(left, right):
    a, b = left["bbox"], right["bbox"]
    overlap = min(a[3], b[3]) - max(a[1], b[1])
    return overlap >= 0.5 * min(a[3] - a[1], b[3] - b[1])


def _physical_rows(tokens):
    """Require pairwise-compatible y groups with nonoverlapping row extents.

    A tall bridging token cannot merge two lines through transitive overlap.
    Sub-threshold overlap between different groups is also ambiguous, not an
    excuse to invent a physical row boundary.
    """
    adjacency = [set() for _ in tokens]
    for i, left in enumerate(tokens):
        for j in range(i + 1, len(tokens)):
            if _same_y_row(left, tokens[j]):
                adjacency[i].add(j)
                adjacency[j].add(i)
    unseen, components = set(range(len(tokens))), []
    while unseen:
        seed = min(unseen)
        stack, component = [seed], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        unseen -= component
        if any((component - {i}) - adjacency[i] for i in component):
            raise ValueError("AMBIGUOUS_Y_ROW_BRIDGE")
        components.append(sorted((tokens[i] for i in component),
                                 key=lambda t: (t["bbox"][0], t["token_index"])))
    components.sort(key=lambda row: (min(t["bbox"][1] for t in row), row[0]["token_index"]))
    for left, right in zip(components, components[1:]):
        if max(t["bbox"][3] for t in left) > min(t["bbox"][1] for t in right):
            raise ValueError("AMBIGUOUS_Y_ROW_OVERLAP")
    return components


def _row_veto(row):
    for cell in row:
        text = cell["original"]
        if not text.strip():
            return "BLANK_CELL_IN_ROW"
        if _MULTILINE.search(text):
            return "MULTILINE_DOM_ROW"
        if _FORMULA.search(text):
            return "FORMULA_OR_MARKUP_IN_ROW"
    return None


def _candidate_text_ok(text):
    return (0 < len(text) <= 64 and all(32 <= ord(c) <= 126 for c in text)
            and not _FORMULA.search(text))


def ground_table(raw_html, ocr_tokens):
    """Return only unambiguous one-mismatch row proposals; never edit raw_html.

    Confidence and OCR alternatives are deliberately not consumed. bbox denotes
    a real detected text-token rectangle, NOT a detected whole-cell rectangle.
    Rows with split/multiple OCR tokens per DOM cell fail the exact row count.
    """
    output = {"schema": "reference-blind-text-token-grounding/2026.09.09.2",
              "status": "PROPOSALS_ONLY_NOT_WRITEBACK", "proposals": [],
              "rejections": [], "source_core_sha256": CORE_SHA256,
              "counts": {"dom_rows": 0, "dom_cells": 0, "ocr_tokens": 0,
                         "physical_rows": 0, "proposals": 0},
              "model_executed": False, "reference_or_metric_used": False}
    try:
        if not isinstance(raw_html, str):
            raise ValueError("INVALID_RAW_HTML")
        dom = _dom_rows(raw_html)
        tokens = _tokens(ocr_tokens)
        physical = _physical_rows(tokens)
        output["counts"].update(dom_rows=len(dom), dom_cells=sum(map(len, dom)),
                                ocr_tokens=len(tokens), physical_rows=len(physical))
        dom_counts = Counter(cell["text"] for row in dom for cell in row)
        ocr_counts = Counter(token["text"] for token in tokens)
        pairs, dom_matches = [], defaultdict(list)
        for physical_index, row in enumerate(physical):
            reason = None
            if len(row) != len(dom[0]):
                reason = "PHYSICAL_DOM_TOKEN_COUNT_MISMATCH"
            elif any(a["bbox"][2] > b["bbox"][0] for a, b in zip(row, row[1:])):
                reason = "OVERLAPPING_X_BOXES"
            elif any(_MULTILINE.search(token["raw_text"]) for token in row):
                reason = "MULTILINE_OCR_ROW"
            if reason:
                output["rejections"].append({"physical_row": physical_index, "reason": reason})
                continue
            matches = []
            for dom_index, cells in enumerate(dom):
                differences = [i for i, (cell, token) in enumerate(zip(cells, row))
                               if cell["text"] != token["text"]]
                if len(differences) <= 1:
                    matches.append((dom_index, differences))
            if len(matches) != 1:
                output["rejections"].append({"physical_row": physical_index,
                                             "reason": "AMBIGUOUS_DOM_ROW" if matches else "NO_ONE_MISMATCH_ROW"})
                continue
            dom_index, differences = matches[0]
            pairs.append((physical_index, dom_index, differences))
            dom_matches[dom_index].append(physical_index)
        for physical_index, dom_index, differences in pairs:
            row, cells = physical[physical_index], dom[dom_index]
            reason = _row_veto(cells)
            if len(dom_matches[dom_index]) != 1:
                reason = "MULTIPLE_PHYSICAL_ROWS_FOR_DOM_ROW"
            if not differences:
                reason = reason or "NO_CHANGE"
            if reason:
                output["rejections"].append({"physical_row": physical_index, "dom_row": dom_index,
                                             "reason": reason})
                continue
            column = differences[0]
            cell, token = cells[column], row[column]
            anchors = [i for i in range(len(cells)) if i != column]
            unique_anchors = [i for i in anchors if dom_counts[cells[i]["text"]] == 1
                              and ocr_counts[row[i]["text"]] == 1]
            if len(unique_anchors) < 2:
                reason = "FEWER_THAN_TWO_TABLE_UNIQUE_ANCHORS"
            elif not _candidate_text_ok(cell["original"]) or not _candidate_text_ok(token["text"]):
                reason = "CANDIDATE_OUTSIDE_ASCII_SINGLE_LINE_SCOPE"
            if reason:
                output["rejections"].append({"physical_row": physical_index, "dom_row": dom_index,
                                             "reason": reason})
                continue
            output["proposals"].append({"cell_index": cell["cell_index"], "row": dom_index,
                "col": column, "original": cell["original"], "original_trimmed": cell["text"],
                "detector_text": token["text"], "bbox": token["bbox"],
                "token_index": token["token_index"], "geometry": "grounded-text-token-box",
                "physical_row": physical_index,
                "anchor_cell_indices": [cells[i]["cell_index"] for i in anchors],
                "anchor_token_indices": [row[i]["token_index"] for i in anchors],
                "unique_anchor_cell_indices": [cells[i]["cell_index"] for i in unique_anchors],
                "unique_anchor_token_indices": [row[i]["token_index"] for i in unique_anchors]})
        output["counts"]["proposals"] = len(output["proposals"])
    except (ValueError, TypeError, UnicodeError, IndexError) as exc:
        output["status"] = "REJECTED_GROUNDING_INPUT"
        output["proposals"] = []
        output["counts"]["proposals"] = 0
        output["rejections"].append({"reason": str(exc)})
    output["rejection_counts"] = dict(Counter(item["reason"] for item in output["rejections"]))
    return output
