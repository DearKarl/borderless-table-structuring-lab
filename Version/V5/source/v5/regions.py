"""Reference regional rules, adapted from ki-OCR-v1 at ac1ab4f.

Sources: archive/dapp_assemble.py, stack_reread_v1.py, stack_guard_v1.py.
Audit history lives outside middle JSON so old spans cannot become new targets.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Callable

from lxml import html
from PIL import Image

TEXT_BLOCKS = {
    "text", "title", "list", "ref_text", "image_caption", "table_caption",
    "image_footnote", "table_footnote", "code_caption", "phonetic",
    "aside_text", "page_footnote",
}
LEAK = re.compile(r"<nl>|<box[:>]|<\|[a-z_]+\|>|<ref>|<quad>|<[felux]cel>")


def repeated(text: str) -> bool:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 256:
        return False
    tail = text[-256:]
    return any(tail[:-period] == tail[period:] for period in range(1, 33))


def runaway(text, kind: str):
    if text is None or not str(text).strip():
        return "empty"
    text = str(text)
    if LEAK.search(text):
        return "token_leak"
    if kind == "table":
        if "<table" in text and not text.rstrip().endswith("</table>"):
            return "truncated"
        rows = re.findall(r"<tr.*?</tr>", text, re.S)
        if not rows:
            return "no_rows"
        cols = [len(re.findall(r"<t[dh][\s>]", row)) for row in rows]
        if max(cols) > 60 or len(rows) > 400:
            return "table_explosion"
        if repeated(re.sub(r"<[^>]+>", " ", text)):
            return "repetition"
        try:
            html.fromstring(text)
        except Exception:
            return "unparsable_html"
        return None
    return "repetition" if repeated(text) else None


def typed_spans(node):
    out = []
    if isinstance(node, dict):
        if node.get("type") == "table" and "html" in node and "bbox" in node:
            out.append((node, "table", "html"))
        elif node.get("type") == "interline_equation" and "content" in node and "bbox" in node:
            out.append((node, "equation", "content"))
        for key, value in node.items():
            if key != "dapp":
                out.extend(typed_spans(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(typed_spans(value))
    return out


def text_blocks(node):
    out = []
    if isinstance(node, dict):
        if node.get("type") in TEXT_BLOCKS and isinstance(node.get("lines"), list) and "bbox" in node:
            return [node]
        for key, value in node.items():
            if key not in ("dapp", "dapp_lines_orig"):
                out.extend(text_blocks(value))
    elif isinstance(node, list):
        for value in node:
            out.extend(text_blocks(value))
    return out


def block_text(block):
    return "\n".join("".join(str(span.get("content", "")) for span in line.get("spans", []))
                     for line in block.get("dapp_lines_orig", block["lines"]))


def crop_of(image, bbox):
    x0, y0, x1, y1 = (int(value) for value in bbox)
    x0, y0 = max(0, x0), max(0, y0)
    return image.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


def region_crop(image, page, obj, factor):
    sx, sy = image.width / page["page_size"][0], image.height / page["page_size"][1]
    x0, y0, x1, y1 = obj["bbox"]
    crop = crop_of(image, (x0 * sx, y0 * sy, x1 * sx, y1 * sy))
    if abs(factor - 1.0) >= 1e-6:
        crop = crop.resize((max(1, round(crop.width * factor)), max(1, round(crop.height * factor))),
                           Image.Resampling.LANCZOS)
    return crop


def reread(middle, image, client, kind, factor, emit: Callable):
    result = copy.deepcopy(middle)
    page = result["pdf_info"][0]
    spans = [(obj, k, key) for obj, k, key in typed_spans(page["para_blocks"]) if k == kind]
    if not spans:
        emit({"event": "stage_coverage", "eligible": 0})
        return result, False
    crops = [region_crop(image, page, obj, factor) for obj, _, _ in spans]
    texts = client.batch_content_extract(crops, [kind] * len(spans))
    if len(texts) != len(spans):
        raise ValueError("Regional output cardinality differs")
    changed = False
    for index, ((obj, _, key), text, crop) in enumerate(zip(spans, texts, crops)):
        before = obj[key]
        rejection = runaway(text, kind)
        accepted = text is not None and rejection is None
        emit({"event": "region", "index": index, "kind": kind, "bbox": obj["bbox"],
              "factor": factor, "crop_size": list(crop.size), "before": before, "after": text,
              "before_reason": runaway(before, kind), "rejection": rejection, "accepted": accepted,
              "changed": accepted and before != text})
        if accepted:
            obj[key] = text
            changed |= before != text
    return result, changed


def guard(middle, image, client, emit: Callable, scales=(1.0, 0.75, 1.5)):
    result = copy.deepcopy(middle)
    page = result["pdf_info"][0]
    bad = [(obj, kind, key) for obj, kind, key in typed_spans(page["para_blocks"])
           if runaway(obj.get(key), kind)]
    bad += [(obj, "text", None) for obj in text_blocks(page["para_blocks"])
            if runaway(block_text(obj), "text") not in (None, "empty")]
    emit({"event": "stage_coverage", "eligible": len(bad)})
    changed = False
    for index, (obj, kind, key) in enumerate(bad):
        before = obj.get(key) if key else block_text(obj)
        for attempt, nominal_scale in enumerate(scales, 1):
            factor = nominal_scale * (200 / 72 if kind == "equation" else 1.0)
            crop = region_crop(image, page, obj, factor)
            outputs = client.batch_content_extract([crop], [kind])
            if len(outputs) != 1:
                raise ValueError("Guard output cardinality differs")
            text = outputs[0]
            reason = runaway(text, kind)
            accepted = bool(text) and reason is None
            emit({"event": "guard_attempt", "index": index, "kind": kind, "bbox": obj["bbox"],
                  "attempt": attempt, "nominal_scale": nominal_scale, "factor": factor,
                  "crop_size": list(crop.size), "before": before, "after": text,
                  "before_reason": runaway(before, kind), "rejection": reason, "accepted": accepted,
                  "changed": accepted and before != text})
            if accepted:
                if key:
                    obj[key] = text
                else:
                    obj["lines"] = [{"bbox": obj["bbox"], "spans": [
                        {"bbox": obj["bbox"], "type": "text", "content": text}]}]
                changed |= before != text
                break
    return result, changed
