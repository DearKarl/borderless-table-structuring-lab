"""Non-training, shadow-first OCR sidecar for one frozen HTML table.

Only explicit DOM-cell bindings are considered. No confidence, reference label,
learned selector, or performance metric is used. Strings are UTF-8 bound; every
byte outside an approved cell's inner text is retained, without reserialization.
"""

import hashlib
import html
from html.parser import HTMLParser
import json
import math
import re


SHADOW = "SHADOW"
NUMERIC_CONFUSABLE_CONSENSUS = "NUMERIC_CONFUSABLE_CONSENSUS"
_MODES = {SHADOW, NUMERIC_CONFUSABLE_CONSENSUS}
_CONFUSABLES = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"})
_NUMERIC = re.compile(
    r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)"
    r"(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?%?\Z"
)
_MULTILINE = re.compile(r"[\r\n\v\f\x85\u2028\u2029]")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
         "link", "meta", "param", "source", "track", "wbr"}
_OUTER_CHILDREN = {
    "table": {"caption", "colgroup", "thead", "tbody", "tfoot", "tr"},
    "colgroup": {"col"}, "thead": {"tr"}, "tbody": {"tr"},
    "tfoot": {"tr"}, "tr": {"td", "th"},
}


class InvalidInput(ValueError):
    """Invalid binding or structure; callers must preserve the raw table."""


def _sha(text):
    return hashlib.sha256(text.encode("utf-8", errors="strict")).hexdigest()


def _json_sha(value):
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"), allow_nan=False))


class _TableParser(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.line_starts = [0] + [m.end() for m in re.finditer("\n", source)]
        self.stack = []
        self.cells = []
        self.active_cell = None
        self.markup = []
        self.table_count = 0

    def _position(self):
        line, column = self.getpos()
        return self.line_starts[line - 1] + column

    def _markup(self, start, end):
        self.markup.append(self.source[start:end])

    def _start(self, tag, attrs, self_closing=False):
        start = self._position()
        raw = self.get_starttag_text()
        self._markup(start, start + len(raw))
        if tag == "table":
            if self.stack or self.table_count:
                raise InvalidInput("exactly one unnested table is required")
            self.table_count += 1
        elif not self.stack:
            raise InvalidInput("element outside the table")
        elif self.active_cell is None:
            parent = self.stack[-1]
            if parent != "caption" and tag not in _OUTER_CHILDREN.get(parent, set()):
                raise InvalidInput("invalid explicit table hierarchy")
        if tag in {"td", "th"}:
            if self.active_cell is not None or not self.stack or self.stack[-1] != "tr":
                raise InvalidInput("cell must be a direct child of tr")
            if self_closing:
                raise InvalidInput("self-closing table cell")
            names = [name for name, _ in attrs]
            if len(names) != len(set(names)):
                raise InvalidInput("duplicate cell attributes")
            spans = dict(attrs)
            merged = False
            for key in ("rowspan", "colspan"):
                if key in spans:
                    value = spans[key]
                    if value is None or not re.fullmatch(r"[0-9]+", value):
                        raise InvalidInput("invalid span attribute")
                    if int(value) != 1:
                        merged = True  # rowspan=0 is also a merged/vetoed cell.
            self.active_cell = {
                "cell_index": len(self.cells), "tag": tag,
                "start": start + len(raw), "end": None,
                "nested": False, "merged": merged,
            }
            self.cells.append(self.active_cell)
        elif self.active_cell is not None:
            self.active_cell["nested"] = True
        if not self_closing and tag not in _VOID:
            self.stack.append(tag)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        if tag not in _VOID:
            raise InvalidInput("non-void self-closing element")
        self._start(tag, attrs, True)

    def handle_endtag(self, tag):
        start = self._position()
        end = self.source.find(">", start)
        raw = self.source[start:end + 1]
        if end < 0 or not re.fullmatch(r"</\s*" + re.escape(tag) + r"\s*>",
                                     raw, re.IGNORECASE):
            raise InvalidInput("malformed closing tag")
        self._markup(start, end + 1)
        if not self.stack or self.stack[-1] != tag:
            raise InvalidInput("unbalanced or implicitly closed HTML")
        if tag in {"td", "th"}:
            if self.active_cell is None:
                raise InvalidInput("cell close without open")
            self.active_cell["end"] = start
            self.active_cell = None
        self.stack.pop()

    def handle_data(self, data):
        if "<" in data:
            raise InvalidInput("unparsed markup")
        if data.strip() and (not self.stack or
                             self.stack[-1] in _OUTER_CHILDREN):
            raise InvalidInput("non-cell text in table structure")

    def handle_entityref(self, name):
        if not self.stack or self.stack[-1] in _OUTER_CHILDREN:
            raise InvalidInput("entity outside cell/caption")

    def handle_charref(self, name):
        self.handle_entityref(name)

    def handle_comment(self, data):
        start = self._position()
        end = self.source.find("-->", start)
        if end < 0:
            raise InvalidInput("unterminated comment")
        self._markup(start, end + 3)
        if self.active_cell is not None:
            self.active_cell["nested"] = True

    def handle_decl(self, decl):
        raise InvalidInput("declaration is not a table fragment")

    def handle_pi(self, data):
        raise InvalidInput("processing instruction is not a table fragment")

    def unknown_decl(self, data):
        raise InvalidInput("unknown declaration")


def _parse(raw):
    parser = _TableParser(raw)
    parser.feed(raw)
    parser.close()
    if parser.stack or parser.active_cell is not None or parser.table_count != 1:
        raise InvalidInput("incomplete table fragment")
    if any(cell["end"] is None for cell in parser.cells):
        raise InvalidInput("incomplete cell")
    return parser


def _numeric_confusion(original, replacement):
    """Pure whitelist, not evidence that a correction is actually beneficial."""
    return (
        bool(original)
        and any(c in "OoIl" for c in original)
        and any(c in "0123456789" for c in original)
        and original.translate(_CONFUSABLES) == replacement
        and len(original) == len(replacement)
        and _NUMERIC.fullmatch(replacement) is not None
    )


def _validate_bindings(cells, total):
    if type(cells) is not list:
        raise InvalidInput("cells must be a list")
    seen = set()
    for cell in cells:
        if type(cell) is not dict:
            raise InvalidInput("cell binding must be a dictionary")
        index = cell.get("cell_index")
        if type(index) is not int or not 0 <= index < total or index in seen:
            raise InvalidInput("invalid or duplicate DOM cell_index")
        seen.add(index)
        for key in ("blank_corner", "is_formula"):
            if type(cell.get(key)) is not bool:
                raise InvalidInput(key + " must be explicit bool")
        bbox = cell.get("bbox")
        if type(bbox) not in (list, tuple) or len(bbox) != 4:
            raise InvalidInput("bbox must be [x0,y0,x1,y1] in pixels")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox):
            raise InvalidInput("bbox coordinates must be finite numbers")
        if bbox[0] < 0 or bbox[1] < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            raise InvalidInput("bbox must have positive dimensions and nonnegative origin")
        observations = cell.get("ocr")
        if type(observations) is not list or len(observations) != 2:
            raise InvalidInput("exactly two OCR observations are required")
        engines = []
        for observation in observations:
            if type(observation) is not dict:
                raise InvalidInput("OCR observation must be a dictionary")
            engine = observation.get("engine_id")
            text = observation.get("text")
            if type(engine) is not str or not engine.strip() or engine != engine.strip():
                raise InvalidInput("engine_id must be a nonempty trimmed string")
            if type(text) is not str:
                raise InvalidInput("OCR text must be a string")
            text.encode("utf-8", errors="strict")
            engines.append(engine)
        if len(set(engines)) != 2:
            raise InvalidInput("OCR observations must have distinct engine IDs")


def _patch(raw, replacements):
    result = raw
    for start, end, text in sorted(replacements, reverse=True):
        result = result[:start] + text + result[end:]
    return result


def process_table(raw_html, cells, mode=SHADOW):
    """Return raw/proposed/final strings and deterministic post-commit receipts.

    `cells` may bind a subset of DOM td/th indices. Each binding supplies bbox,
    blank_corner, is_formula, and `ocr=[{engine_id, text}, {engine_id, text}]`.
    Engine IDs identify supplied observations, not engines executed by this core.
    SHADOW may propose escaped consensus text but always returns raw as final.
    The optional numeric policy is a conservative syntactic rule, not a score.
    Malformed structure or bindings roll the entire table back to raw; a non-str
    raw input raises TypeError because no original HTML can be preserved.
    """
    if type(raw_html) is not str:
        raise TypeError("raw_html must be a UTF-8 encodable str")
    raw_sha = _sha(raw_html)
    records = []
    commits = []
    counts = {"engine_invocations": 0, "distinct_engine_ids": 0,
              "engine_observations": 0, "bound_cells": 0,
              "proposals": 0, "policy_approved": 0, "committed": 0,
              "vetoed": 0, "no_change": 0, "rolled_back": 0}
    result = {"schema": "frozen-html-ocr-sidecar/2026.09.09.1",
              "mode": mode if type(mode) is str else "INVALID_MODE_TYPE",
              "status": "PENDING", "raw": raw_html, "proposed": raw_html,
              "final": raw_html, "raw_sha256": raw_sha,
              "proposed_sha256": raw_sha, "final_sha256": raw_sha,
              "skeleton_sha256": None, "final_skeleton_sha256": None,
              "proposals": records, "post_commit_receipts": commits,
              "counts": counts, "model_or_metric_used": False}
    try:
        if type(mode) is not str or mode not in _MODES:
            raise InvalidInput("unsupported sidecar mode")
        parser = _parse(raw_html)
        skeleton = _json_sha(parser.markup)
        result["skeleton_sha256"] = skeleton
        result["final_skeleton_sha256"] = skeleton
        _validate_bindings(cells, len(parser.cells))
        counts["bound_cells"] = len(cells)
        counts["engine_observations"] = 2 * len(cells)
        counts["distinct_engine_ids"] = len({o["engine_id"] for c in cells
                                              for o in c["ocr"]})
        proposals = []
        approved = []
        for binding in sorted(cells, key=lambda c: c["cell_index"]):
            cell = parser.cells[binding["cell_index"]]
            inner = raw_html[cell["start"]:cell["end"]]
            text = html.unescape(inner)
            observations = binding["ocr"]
            trimmed = [observation["text"].strip() for observation in observations]
            record = {"cell_index": cell["cell_index"], "bbox": list(binding["bbox"]),
                      "original_inner_char_span": [cell["start"], cell["end"]],
                      "original_inner_utf8_byte_span": [
                          len(raw_html[:cell["start"]].encode("utf-8")),
                          len(raw_html[:cell["end"]].encode("utf-8"))],
                      "original_inner_sha256": _sha(inner), "original_text": text,
                      "engine_ids": [o["engine_id"] for o in observations],
                      "engine_texts": [o["text"] for o in observations],
                      "candidate_text": None, "status": "VETO", "reason": None,
                      "committed": False}
            records.append(record)
            if binding["blank_corner"]:
                record["reason"] = "BLANK_CORNER_VETO"
            elif binding["is_formula"]:
                record["reason"] = "FORMULA_VETO"
            elif cell["merged"]:
                record["reason"] = "MERGED_CELL_VETO"
            elif cell["nested"]:
                record["reason"] = "NESTED_MARKUP_VETO"
            elif _MULTILINE.search(text) or any(_MULTILINE.search(o["text"])
                                               for o in observations):
                record["reason"] = "MULTILINE_VETO"
            elif not text.strip():
                record["reason"] = "EMPTY_ORIGINAL_VETO"
            elif trimmed[0] != trimmed[1]:
                record["reason"] = "ENGINE_DISAGREEMENT"
            elif not trimmed[0]:
                record["reason"] = "EMPTY_CONSENSUS_VETO"
            elif text.strip() == trimmed[0]:
                record.update(status="NO_CHANGE", reason="EXACT_TEXT_NO_CHANGE")
                counts["no_change"] += 1
            else:
                candidate = trimmed[0]
                record.update(candidate_text=candidate, status="SHADOW_PROPOSAL",
                              reason="SHADOW_ONLY")
                replacement = html.escape(candidate, quote=True)
                proposals.append((cell["start"], cell["end"], replacement))
                counts["proposals"] += 1
                if _numeric_confusion(text.strip(), candidate):
                    counts["policy_approved"] += 1
                    if mode == NUMERIC_CONFUSABLE_CONSENSUS:
                        approved.append((cell["start"], cell["end"], replacement))
                        record.update(status="PENDING_COMMIT", reason="NUMERIC_WHITELIST")
                elif mode == NUMERIC_CONFUSABLE_CONSENSUS:
                    record["reason"] = "OUTSIDE_NUMERIC_WHITELIST"
            if record["status"] == "VETO":
                counts["vetoed"] += 1
        proposed_html = _patch(raw_html, proposals)
        final_html = _patch(raw_html, approved)
        for candidate_html in (proposed_html, final_html):
            candidate_parser = _parse(candidate_html)
            if (len(candidate_parser.cells) != len(parser.cells) or
                    _json_sha(candidate_parser.markup) != skeleton):
                raise InvalidInput("structural identity check failed; entire table rolled back")
        final_sha = _sha(final_html)
        for record in records:
            if record["status"] == "PENDING_COMMIT":
                record.update(status="COMMITTED", committed=True)
                counts["committed"] += 1
                commits.append({"cell_index": record["cell_index"],
                                "decision": "COMMITTED_AFTER_STRUCTURE_CHECK",
                                "raw_table_sha256": raw_sha,
                                "final_table_sha256": final_sha,
                                "structure_sha256_before": skeleton,
                                "structure_sha256_after": skeleton,
                                "original_inner_sha256": record["original_inner_sha256"],
                                "committed_inner_sha256": _sha(html.escape(
                                    record["candidate_text"], quote=True)),
                                "candidate_text": record["candidate_text"],
                                "engine_ids": record["engine_ids"]})
        result.update(status="SHADOW_COMPLETE" if mode == SHADOW else "APPLY_COMPLETE",
                      proposed=proposed_html, final=final_html,
                      proposed_sha256=_sha(proposed_html), final_sha256=final_sha)
    except (InvalidInput, UnicodeError, OverflowError) as exc:
        counts["rolled_back"] = 1
        counts["committed"] = 0
        commits.clear()
        for record in records:
            record.update(committed=False)
            if record["status"] in {"PENDING_COMMIT", "COMMITTED"}:
                record.update(status="ROLLED_BACK", reason="TABLE_ROLLBACK")
        result.update(status="ROLLED_BACK_INVALID_INPUT", error=str(exc))
    result["receipt_sha256"] = _json_sha(result)
    return result
