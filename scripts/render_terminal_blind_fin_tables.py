from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


CN_ITEMS = ["营业收入", "营业成本", "流动资产", "固定资产", "应收账款", "存货", "短期借款", "所有者权益", "净利润", "现金及等价物"]
EN_ITEMS = ["Revenue", "Cost of sales", "Current assets", "Fixed assets", "Receivables", "Inventories", "Borrowings", "Total equity", "Net profit", "Cash equivalents"]
CN_TITLES = ["财务状况表", "经营成果表", "现金流量表", "主要财务指标"]
EN_TITLES = ["Financial Position", "Operating Results", "Cash Flows", "Key Financial Indicators"]
FONT_CANDIDATES = (
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simsun.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
)


def _font(path: Path | None, size: int) -> tuple[ImageFont.FreeTypeFont, dict[str, Any]]:
    candidates = [path] if path is not None else list(FONT_CANDIDATES)
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            payload = candidate.read_bytes()
            return ImageFont.truetype(str(candidate), size), {
                "source": "system_or_user_provided",
                "file_name": candidate.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "redistributed": False,
            }
    raise RuntimeError("no usable font found; pass --font")


def _money(rng: random.Random) -> str:
    value = rng.randint(1, 9_999_999)
    text = f"{value:,}.{rng.randint(0, 99):02d}"
    return f"({text})" if rng.random() < 0.12 else text


def _build_cells(rng: random.Random, spec: dict[str, Any], language: str) -> tuple[list[dict[str, Any]], int, int]:
    density = str(spec["density"])
    n_body = rng.randint(int(spec["body_rows"][0]), int(spec["body_rows"][1]))
    n_cols = int(spec["columns"])
    header_depth = int(spec["header_depth"])
    merged = bool(spec["merged"])
    items = CN_ITEMS if language == "cn" else EN_ITEMS
    item_head = "项目" if language == "cn" else "Item"
    periods = ("本期", "上期") if language == "cn" else ("Current", "Prior")
    cells: list[dict[str, Any]] = []

    if header_depth == 2:
        cells.append({"r0": 0, "r1": 1, "c0": 0, "c1": 0, "text": item_head, "is_header": True})
        split = 1 + (n_cols - 1) // 2
        cells.append({"r0": 0, "r1": 0, "c0": 1, "c1": split - 1, "text": periods[0], "is_header": True})
        cells.append({"r0": 0, "r1": 0, "c0": split, "c1": n_cols - 1, "text": periods[1], "is_header": True})
        for col in range(1, n_cols):
            cells.append({"r0": 1, "r1": 1, "c0": col, "c1": col, "text": "金额" if language == "cn" else "Amount", "is_header": True})
    else:
        cells.append({"r0": 0, "r1": 0, "c0": 0, "c1": 0, "text": item_head, "is_header": True})
        for col in range(1, n_cols):
            cells.append({"r0": 0, "r1": 0, "c0": col, "c1": col, "text": periods[(col - 1) % 2], "is_header": True})

    row = header_depth
    item_index = 0
    while row < header_depth + n_body:
        remaining = header_depth + n_body - row
        if merged and remaining >= 3 and item_index % 6 == 3:
            text = "其中：核心业务" if language == "cn" else "Of which: core business"
            cells.append({"r0": row, "r1": row, "c0": 0, "c1": n_cols - 1, "text": text, "is_header": False})
            row += 1
            continue
        if merged and remaining >= 2 and item_index % 7 == 4:
            cells.append({"r0": row, "r1": row + 1, "c0": 0, "c1": 0, "text": items[item_index % len(items)], "is_header": False})
            for offset in range(2):
                for col in range(1, n_cols):
                    cells.append({"r0": row + offset, "r1": row + offset, "c0": col, "c1": col, "text": _money(rng), "is_header": False})
            row += 2
            item_index += 2
            continue
        cells.append({"r0": row, "r1": row, "c0": 0, "c1": 0, "text": items[item_index % len(items)], "is_header": False})
        for col in range(1, n_cols):
            cells.append({"r0": row, "r1": row, "c0": col, "c1": col, "text": "" if rng.random() < 0.05 else _money(rng), "is_header": False})
        row += 1
        item_index += 1
    return cells, row, n_cols


def _html(cells: list[dict[str, Any]], rows: int, cols: int) -> str:
    origins = {(cell["r0"], cell["c0"]): cell for cell in cells}
    owner = {}
    for cell in cells:
        for row in range(cell["r0"], cell["r1"] + 1):
            for col in range(cell["c0"], cell["c1"] + 1):
                if (row, col) in owner:
                    raise ValueError("overlapping cells")
                owner[(row, col)] = (cell["r0"], cell["c0"])
    if len(owner) != rows * cols:
        raise ValueError("incomplete grid")
    output = []
    for row in range(rows):
        nodes = []
        for col in range(cols):
            if owner[(row, col)] != (row, col):
                continue
            cell = origins[(row, col)]
            rowspan = cell["r1"] - cell["r0"] + 1
            colspan = cell["c1"] - cell["c0"] + 1
            attrs = (f' rowspan="{rowspan}"' if rowspan > 1 else "") + (f' colspan="{colspan}"' if colspan > 1 else "")
            tag = "th" if cell["is_header"] else "td"
            text = cell["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            nodes.append(f"<{tag}{attrs}>{text}</{tag}>")
        output.append("<tr>" + "".join(nodes) + "</tr>")
    return "<table>" + "".join(output) + "</table>"


def _otsl(cells: list[dict[str, Any]], rows: int, cols: int) -> str:
    origins = {(cell["r0"], cell["c0"]): cell for cell in cells}
    owner = {}
    for cell in cells:
        for row in range(cell["r0"], cell["r1"] + 1):
            for col in range(cell["c0"], cell["c1"] + 1):
                owner[(row, col)] = (cell["r0"], cell["c0"])
    lines = []
    for row in range(rows):
        tokens = []
        for col in range(cols):
            origin = owner[(row, col)]
            if origin == (row, col):
                text = str(origins[origin]["text"])
                tokens.append(("<ecel>" if not text else "<fcel>") + text)
            elif origin[0] == row:
                tokens.append("<lcel>")
            elif origin[1] == col:
                tokens.append("<ucel>")
            else:
                tokens.append("<xcel>")
        lines.append("".join(tokens) + "<nl>")
    return "\n".join(lines)


def _render_table(rng: random.Random, spec: dict[str, Any], language: str, width: int, font_path: Path | None) -> tuple[Image.Image, dict[str, Any], dict[str, Any]]:
    density = str(spec["density"])
    font_size = {"sparse": 23, "medium": 19, "dense": 15}[density]
    font, font_info = _font(font_path, font_size)
    title_font, _ = _font(font_path, font_size + 5)
    cells, rows, cols = _build_cells(rng, spec, language)
    title = rng.choice(CN_TITLES if language == "cn" else EN_TITLES)
    unit = "单位：人民币元" if language == "cn" else "Unit: currency"
    top = font_size * 4
    row_h = font_size + {"sparse": 18, "medium": 12, "dense": 8}[density]
    col_w = max(60, width // cols)
    width = col_w * cols
    image = Image.new("RGB", (width, top + rows * row_h + 2), "white")
    draw = ImageDraw.Draw(image)
    draw.text((4, 2), title, fill="black", font=title_font)
    draw.text((width - 4, font_size + 10), unit, fill="black", font=font, anchor="ra")
    grid_top = top
    for cell in cells:
        x0 = cell["c0"] * col_w
        y0 = grid_top + cell["r0"] * row_h
        x1 = (cell["c1"] + 1) * col_w
        y1 = grid_top + (cell["r1"] + 1) * row_h
        cell["bbox"] = [float(x0), float(y0), float(x1), float(y1)]
        text = str(cell["text"])
        if text:
            draw.text((x0 + 4, y0 + 3), text, fill="black", font=font)
        if spec["border"] == "full":
            draw.rectangle((x0, y0, x1, y1), outline="black", width=1)
    if spec["border"] == "three_line":
        draw.line((0, grid_top, width, grid_top), fill="black", width=2)
        draw.line((0, grid_top + int(spec["header_depth"]) * row_h, width, grid_top + int(spec["header_depth"]) * row_h), fill="black", width=1)
        draw.line((0, grid_top + rows * row_h, width, grid_top + rows * row_h), fill="black", width=2)
    table = {
        "cells": cells,
        "html": _html(cells, rows, cols),
        "otsl": _otsl(cells, rows, cols),
        "n_rows": rows,
        "n_cols": cols,
        "meta": {
            "header_depth": int(spec["header_depth"]),
            "title": title,
            "unit": unit,
            "border": str(spec["border"]),
            "lang": language,
        },
    }
    return image, table, font_info


def _apply_profile(image: Image.Image, profile: str) -> tuple[Image.Image, list[str], int]:
    quality = 95
    operations = []
    if profile == "blur_light":
        image = image.filter(ImageFilter.GaussianBlur(0.45))
        operations.append("gaussian_blur_0.45")
    elif profile == "downsample_light":
        original = image.size
        image = image.resize((max(1, int(original[0] * 0.82)), max(1, int(original[1] * 0.82))), Image.Resampling.BICUBIC).resize(original, Image.Resampling.BICUBIC)
        operations.append("downsample_0.82")
    elif profile == "low_contrast":
        image = ImageEnhance.Contrast(image).enhance(0.82)
        operations.append("contrast_0.82")
    elif profile == "jpeg_light":
        quality = 88
        operations.append("jpeg_88")
    elif profile == "jpeg_medium":
        quality = 76
        operations.append("jpeg_76")
    elif profile != "clean":
        raise ValueError(f"unknown renderer profile: {profile}")
    return image, operations, quality


def _page(entry: dict[str, Any], font_path: Path | None) -> tuple[Image.Image, list[dict[str, Any]], dict[str, Any], list[str], int]:
    rng = random.Random(int(entry["generation_seed"]))
    spec = dict(entry["template_spec"])
    language = str(entry["language"])
    portrait = spec["orientation"] == "portrait"
    page_w, page_h = ((1240, 1754) if portrait else (1754, 1240))
    margin, gap = 70, 36
    table_count = int(spec["tables"])
    side = spec["layout"] == "side_by_side"
    table_width = (page_w - 2 * margin - gap * (table_count - 1)) // table_count if side else page_w - 2 * margin
    rendered = [_render_table(rng, spec, language, table_width, font_path) for _ in range(table_count)]
    required_height = (
        75 + max(image.height for image, _, _ in rendered) + margin
        if side
        else 75 + sum(image.height for image, _, _ in rendered) + gap * (table_count - 1) + margin
    )
    page_h = max(page_h, required_height)
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    body_font, font_info = _font(font_path, 18)
    heading = "示例财务报告" if language == "cn" else "Illustrative Financial Report"
    draw.text((margin, 24), heading, fill="black", font=body_font)
    tables = []
    x, y = margin, 75
    for index, (image, table, _) in enumerate(rendered):
        page.paste(image, (x, y))
        moved_cells = []
        for cell in table["cells"]:
            value = dict(cell)
            box = cell["bbox"]
            value["bbox"] = [box[0] + x, box[1] + y, box[2] + x, box[3] + y]
            value["row_start"] = int(value["r0"])
            value["row_end"] = int(value["r1"])
            value["col_start"] = int(value["c0"])
            value["col_end"] = int(value["c1"])
            moved_cells.append(value)
        table_bbox = [float(x), float(y), float(x + image.width), float(y + image.height)]
        tables.append({**table, "table_index": index, "bbox": table_bbox, "cells": moved_cells, "structure_sha256": hashlib.sha256(json.dumps([(c["r0"], c["r1"], c["c0"], c["c1"]) for c in moved_cells], separators=(",", ":")).encode()).hexdigest()})
        if side:
            x += image.width + gap
        else:
            y += image.height + gap
    profile = str(entry["renderer_spec"]["profile"])
    page, operations, quality = _apply_profile(page, profile)
    return page, tables, font_info, operations, quality


def main() -> int:
    parser = argparse.ArgumentParser(description="Render terminal-blind synthetic financial tables from a frozen family plan.")
    parser.add_argument("--family-plan", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    image_dir = args.out_dir / "images"
    image_dir.mkdir()
    entries = [json.loads(line) for line in args.family_plan.open(encoding="utf-8") if line.strip()]
    manifest = args.out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for entry in entries:
            page, tables, font_info, operations, quality = _page(entry, args.font)
            image_name = f"{entry['document_cluster_id']}.jpg"
            image_path = image_dir / image_name
            buffer = io.BytesIO()
            page.save(buffer, format="JPEG", quality=quality)
            payload = buffer.getvalue()
            image_path.write_bytes(payload)
            source_id = str(entry["document_cluster_id"])
            record = {
                "schema_version": "terminal_blind_fin_page_v1",
                "task": "table_page_and_crop",
                "source_dataset": "TerminalBlindFinSynthetic",
                "source_document_id": source_id,
                "source_document_hash": hashlib.sha256(source_id.encode()).hexdigest(),
                "source_document_hash_kind": "synthetic_id_sha256",
                "license": "self-generated-code-output",
                "deployment_status": "research_only_pending_license_review",
                "renderer_version": "terminal_blind_fin_renderer_v1",
                "generator_policy": "terminal_blind_v1",
                "terminal_data_used": False,
                "role": entry["role"],
                "pair_group_id": entry["pair_group_id"],
                "document_cluster_id": entry["document_cluster_id"],
                "template_family_id": entry["template_family_id"],
                "content_family_id": entry["content_family_id"],
                "renderer_family_id": entry["renderer_family_id"],
                "generation_seed": entry["generation_seed"],
                "family_plan_entry_sha256": entry["plan_entry_sha256"],
                "template_spec": entry["template_spec"],
                "renderer_spec": entry["renderer_spec"],
                "image": f"images/{image_name}",
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "image_size": [page.width, page.height],
                "font": font_info,
                "degradations": operations,
                "language": entry["language"],
                "n_tables": len(tables),
                "tables": tables,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"accepted_pages": len(entries), "rejected": 0, "manifest": str(manifest)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
