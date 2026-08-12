from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from jsonschema import Draft202012Validator
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .canonical import canonical_cells, occupied_slots, table_shape, validate_cells


RELEASE = "2026.08.12.4"
DATASET_RELEASE = f"data-smoke-{RELEASE}"
SCHEMA_RELEASE = f"synthetic-table-record-{RELEASE}"
LICENSE_DECISION = "APPROVED_REBUILD_ONLY"
ZERO_SHA256 = "0" * 64


def _deterministic_rng(namespace: str, *parts: object) -> random.Random:
    digest = hashlib.sha256(
        "|".join([namespace, *(str(part) for part in parts)]).encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def _deterministic_label(base_seed: int, row: int, col: int) -> str:
    digest = hashlib.sha256(
        f"cell-text|{base_seed}|{row}|{col}".encode("utf-8")
    ).hexdigest()
    return f"S{digest[:10].upper()}-R{row + 1}-C{col + 1}"


def _geometry_profile(base_seed: int, rows: int, columns: int) -> dict[str, Any]:
    rng = _deterministic_rng("geometry-2026.08.12.6", base_seed, rows, columns)
    width = rng.randrange(820, 1241)
    margin_x = rng.randrange(22, 59)
    margin_y = rng.randrange(20, 53)
    available_width = width - 2 * margin_x
    weights = [rng.randrange(65, 151) for _ in range(columns)]
    weight_sum = sum(weights)
    raw_widths = [max(48, round(available_width * weight / weight_sum)) for weight in weights]
    raw_widths[-1] += available_width - sum(raw_widths)
    if raw_widths[-1] < 42:
        transfer = 42 - raw_widths[-1]
        donor = max(range(columns - 1), key=lambda index: raw_widths[index])
        raw_widths[donor] -= transfer
        raw_widths[-1] += transfer
    row_heights = [rng.randrange(30, 52) for _ in range(rows)]
    return {
        "width": width,
        "margin_x": margin_x,
        "margin_y": margin_y,
        "column_widths": raw_widths,
        "row_heights": row_heights,
        "cell_padding_x": rng.randrange(4, 10),
        "cell_padding_y": rng.randrange(3, 8),
    }


@lru_cache(maxsize=1)
def _font_candidates() -> tuple[tuple[str, str, str], ...]:
    candidates = (
        ("arial", "/System/Library/Fonts/Supplemental/Arial.ttf"),
        ("courier-new", "/System/Library/Fonts/Supplemental/Courier New.ttf"),
        ("georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf"),
        ("times-new-roman", "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        ("dejavu-sans", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ("dejavu-serif", "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
        ("dejavu-mono", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    )
    available: list[tuple[str, str, str]] = []
    for family, raw_path in candidates:
        path = Path(raw_path)
        if path.is_file():
            available.append((family, str(path), _sha256(path.read_bytes())))
    if not available:
        available.append(("pillow-default", "PILLOW_EMBEDDED_DEFAULT", "embedded"))
    return tuple(available)


def _render_profile(base_seed: int, phenomenon: str) -> dict[str, Any]:
    rng = _deterministic_rng("renderer-2026.08.12.6", base_seed, phenomenon)
    fonts = _font_candidates()
    family, source, source_sha256 = fonts[rng.randrange(len(fonts))]
    return {
        "font_family_id": family,
        "font_source": source,
        "font_source_sha256": source_sha256,
        "font_size": rng.randrange(9, 15),
        "text_rgb": [rng.randrange(18, 47), rng.randrange(24, 55), rng.randrange(30, 67)],
        "text_spacing": rng.randrange(1, 4),
    }


@dataclass(frozen=True)
class CoverageRequest:
    category: str
    gate_label: str
    phenomenon: str
    role: str


def _stable_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _table_state(table: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": table["rows"],
        "columns": table["columns"],
        "cells": [
            {
                "row": cell["row"],
                "col": cell["col"],
                "rowspan": cell["rowspan"],
                "colspan": cell["colspan"],
                "text": cell["text"],
                "tag": cell["tag"],
            }
            for cell in sorted(
                table["cells"], key=lambda x: (x["row"], x["col"])
            )
        ],
    }


def _finalize_table(table: dict[str, Any]) -> dict[str, Any]:
    table = copy.deepcopy(table)
    table["cells"] = sorted(table["cells"], key=lambda x: (x["row"], x["col"]))
    table["tokens"] = sorted(table["tokens"], key=lambda x: x["token_id"])
    table["semantic_state_sha256"] = _sha256(_stable_json(_table_state(table)))
    payload = copy.deepcopy(table)
    payload.pop("full_state_sha256", None)
    table["full_state_sha256"] = _sha256(_stable_json(payload))
    return table


def _cell_id(row: int, col: int, rowspan: int = 1, colspan: int = 1) -> str:
    return f"cell-{row:02d}-{col:02d}-{rowspan:02d}-{colspan:02d}"


def _primitive_table(
    base_seed: int,
    rows: int,
    columns: int,
    phenomenon: str,
    header_depth: int,
) -> dict[str, Any]:
    geometry = _geometry_profile(base_seed, rows, columns)
    margin_x = geometry["margin_x"]
    margin_y = geometry["margin_y"]
    x_edges = [margin_x]
    for col_width in geometry["column_widths"]:
        x_edges.append(x_edges[-1] + col_width)
    y_edges = [margin_y]
    for row_height in geometry["row_heights"]:
        y_edges.append(y_edges[-1] + row_height)
    tokens: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    greek = ("alpha", "beta", "gamma", "delta", "sigma")
    for row in range(rows):
        for col in range(columns):
            token_id = f"tok-{row:02d}-{col:02d}"
            if "formula" in phenomenon and (row + col) % 4 == 0:
                text = f"{greek[(row + col) % len(greek)]}_{row + 1}={col + 2}x"
            elif "empty" in phenomenon and (row * columns + col) % 7 == 0:
                text = ""
            elif "multiline" in phenomenon and (row + col) % 3 == 0:
                text = f"Group {row + 1}\nMeasure {col + 1}"
            else:
                text = _deterministic_label(base_seed, row, col)
            x0, x1 = x_edges[col], x_edges[col + 1]
            y0, y1 = y_edges[row], y_edges[row + 1]
            owner = _cell_id(row, col)
            token = {
                "token_id": token_id,
                "text": text,
                "bbox": [
                    x0 + geometry["cell_padding_x"],
                    y0 + geometry["cell_padding_y"],
                    x1 - geometry["cell_padding_x"],
                    y1 - geometry["cell_padding_y"],
                ],
                "owner_cell_id": owner,
                "confidence": 1.0,
                "grid_row": row,
                "grid_col": col,
            }
            tokens.append(token)
            cells.append(
                {
                    "cell_id": owner,
                    "row": row,
                    "col": col,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": text,
                    "tag": "th" if row < header_depth else "td",
                    "bbox": [x0, y0, x1, y1],
                    "token_ids": [token_id],
                }
            )
    return _finalize_table(
        {"rows": rows, "columns": columns, "cells": cells, "tokens": tokens}
    )


def _merge_region(
    table: dict[str, Any], row: int, col: int, rowspan: int, colspan: int
) -> dict[str, Any]:
    result = copy.deepcopy(table)
    slots = {
        (r, c)
        for r in range(row, row + rowspan)
        for c in range(col, col + colspan)
    }
    selected = [
        cell
        for cell in result["cells"]
        if (cell["row"], cell["col"]) in slots
        and cell["rowspan"] == 1
        and cell["colspan"] == 1
    ]
    if len(selected) != len(slots):
        return result
    selected_ids = {cell["cell_id"] for cell in selected}
    token_ids = [token for cell in selected for token in cell["token_ids"]]
    bbox = [
        min(cell["bbox"][0] for cell in selected),
        min(cell["bbox"][1] for cell in selected),
        max(cell["bbox"][2] for cell in selected),
        max(cell["bbox"][3] for cell in selected),
    ]
    merged_id = _cell_id(row, col, rowspan, colspan)
    merged = {
        "cell_id": merged_id,
        "row": row,
        "col": col,
        "rowspan": rowspan,
        "colspan": colspan,
        "text": " | ".join(cell["text"] for cell in selected if cell["text"]),
        "tag": "th" if any(cell["tag"] == "th" for cell in selected) else "td",
        "bbox": bbox,
        "token_ids": token_ids,
    }
    result["cells"] = [
        cell for cell in result["cells"] if cell["cell_id"] not in selected_ids
    ] + [merged]
    for token in result["tokens"]:
        if token["token_id"] in token_ids:
            token["owner_cell_id"] = merged_id
    return _finalize_table(result)


def _split_spanning_cell(table: dict[str, Any], cell_id: str) -> dict[str, Any]:
    result = copy.deepcopy(table)
    target = next(cell for cell in result["cells"] if cell["cell_id"] == cell_id)
    if target["rowspan"] == 1 and target["colspan"] == 1:
        return result
    result["cells"] = [cell for cell in result["cells"] if cell["cell_id"] != cell_id]
    for row in range(target["row"], target["row"] + target["rowspan"]):
        for col in range(target["col"], target["col"] + target["colspan"]):
            owned = [
                token
                for token in result["tokens"]
                if token["grid_row"] == row and token["grid_col"] == col
            ]
            owner = _cell_id(row, col)
            for token in owned:
                token["owner_cell_id"] = owner
            result["cells"].append(
                {
                    "cell_id": owner,
                    "row": row,
                    "col": col,
                    "rowspan": 1,
                    "colspan": 1,
                    "text": " ".join(token["text"] for token in owned if token["text"]),
                    "tag": target["tag"],
                    "bbox": [
                        min(token["bbox"][0] - 5 for token in owned),
                        min(token["bbox"][1] - 5 for token in owned),
                        max(token["bbox"][2] + 5 for token in owned),
                        max(token["bbox"][3] + 5 for token in owned),
                    ],
                    "token_ids": [token["token_id"] for token in owned],
                }
            )
    return _finalize_table(result)


def _gold_table(
    base_seed: int,
    rows: int,
    columns: int,
    phenomenon: str,
    category: str,
) -> dict[str, Any]:
    header_depth = 2 if "header" in phenomenon or category == "complex_correction" else 1
    table = _primitive_table(base_seed, rows, columns, phenomenon, header_depth)
    needs_span = (
        "span" in phenomenon
        or category in {"single_minimal_edit", "complex_correction"}
    )
    if needs_span and columns >= 2:
        table = _merge_region(table, 0, 0, 1, 2)
    if ("mixed" in phenomenon or category == "complex_correction") and rows >= 4:
        table = _merge_region(table, 1, columns - 1, 2, 1)
    # Realized topology diversity is generated from the complete seed rather
    # than from a small repeating shape pool.  These legal, non-overlapping
    # body spans make the final Canonical state the source of truth while
    # providing enough structural families for split isolation at 40k scale.
    rng = _deterministic_rng(
        "topology-2026.08.12.6", base_seed, rows, columns, phenomenon, category
    )
    candidates = [
        (row, col, 1, 2)
        for row in range(max(1, header_depth), rows)
        for col in range(columns - 1)
    ] + [
        (row, col, 2, 1)
        for row in range(max(1, header_depth), rows - 1)
        for col in range(columns)
    ]
    rng.shuffle(candidates)
    target_spans = min(
        len(candidates),
        2 + rng.randrange(0, min(9, max(2, rows * columns // 10))),
    )
    added = 0
    for row, col, rowspan, colspan in candidates:
        if added >= target_spans:
            break
        updated = _merge_region(table, row, col, rowspan, colspan)
        if updated["semantic_state_sha256"] != table["semantic_state_sha256"]:
            table = updated
            added += 1
    return table


def _first_spanning_cell(table: dict[str, Any]) -> dict[str, Any] | None:
    return next(
        (
            cell
            for cell in table["cells"]
            if cell["rowspan"] > 1 or cell["colspan"] > 1
        ),
        None,
    )


def _merge_first_available_pair(
    table: dict[str, Any], reverse_rows: bool = True
) -> tuple[dict[str, Any], tuple[int, int] | None]:
    row_order: Iterable[int] = range(table["rows"] - 1, -1, -1) if reverse_rows else range(table["rows"])
    for row in row_order:
        for col in range(table["columns"] - 1):
            updated = _merge_region(table, row, col, 1, 2)
            if updated["semantic_state_sha256"] != table["semantic_state_sha256"]:
                return updated, (row, col)
    return copy.deepcopy(table), None


def _prior_for_edit(
    gold: dict[str, Any], phenomenon: str, category: str
) -> tuple[dict[str, Any], list[str]]:
    prior = copy.deepcopy(gold)
    operations: list[str] = []
    spanning = _first_spanning_cell(prior)
    if phenomenon in {"extra_split", "missing_merge", "span_extent_error"}:
        if spanning is not None:
            prior = _split_spanning_cell(prior, spanning["cell_id"])
            operations.append("split_spanning_cell")
        if phenomenon == "span_extent_error" and prior["columns"] >= 3:
            prior = _merge_region(prior, 0, 1, 1, 2)
            operations.append("merge_shifted_span")
    elif phenomenon == "row_or_column_assignment_error":
        if spanning is not None:
            prior = _split_spanning_cell(prior, spanning["cell_id"])
            operations.append("split_gold_span")
        prior = _merge_region(prior, 0, 0, 2, 1)
        operations.append("merge_orthogonal_span")
    else:
        prior, merged_at = _merge_first_available_pair(prior)
        if merged_at is None:
            raise ValueError(f"no deterministic merge region available: {phenomenon}")
        operations.append("merge_body_pair")

    if category == "complex_correction":
        spanning = _first_spanning_cell(prior)
        if spanning is not None:
            prior = _split_spanning_cell(prior, spanning["cell_id"])
            operations.append("split_additional_span")
        prior, merged_at = _merge_first_available_pair(prior)
        if merged_at is None:
            raise ValueError("no additional merge region available")
        operations.append("merge_additional_pair")
    if prior["semantic_state_sha256"] == gold["semantic_state_sha256"]:
        raise ValueError(f"edit phenomenon produced identity: {phenomenon}")
    return prior, operations


def _difference(prior: dict[str, Any], gold: dict[str, Any]) -> list[dict[str, Any]]:
    prior_cells = {
        (cell["row"], cell["col"], cell["rowspan"], cell["colspan"])
        for cell in prior["cells"]
    }
    gold_cells = {
        (cell["row"], cell["col"], cell["rowspan"], cell["colspan"])
        for cell in gold["cells"]
    }
    return [
        {"kind": kind, "row": key[0], "col": key[1], "rowspan": key[2], "colspan": key[3]}
        for kind, keys in (
            ("remove_prior_cell", sorted(prior_cells - gold_cells)),
            ("add_gold_cell", sorted(gold_cells - prior_cells)),
        )
        for key in keys
    ]


def _draw_table(
    table: dict[str, Any], phenomenon: str, base_seed: int
) -> tuple[bytes, str, dict[str, Any]]:
    rng = _deterministic_rng("draw-2026.08.12.6", base_seed, phenomenon)
    width = int(max(cell["bbox"][2] for cell in table["cells"]) + rng.randrange(22, 61))
    height = int(max(cell["bbox"][3] for cell in table["cells"]) + 30)
    background = (248, 250, 252) if "color" in phenomenon else (255, 255, 255)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    render_profile = _render_profile(base_seed, phenomenon)
    if render_profile["font_source"] == "PILLOW_EMBEDDED_DEFAULT":
        font = ImageFont.load_default(size=render_profile["font_size"])
    else:
        font = ImageFont.truetype(
            render_profile["font_source"], render_profile["font_size"]
        )
    weak = "weak" in phenomenon or "missing" in phenomenon
    border = (210, 216, 224) if weak else (88, 98, 112)
    border_width = 1 if weak else 2
    for cell in table["cells"]:
        box = tuple(round(value) for value in cell["bbox"])
        if not ("missing" in phenomenon and (cell["row"] + cell["col"]) % 3 == 0):
            draw.rectangle(box, outline=border, width=border_width)
        if cell["text"]:
            draw.multiline_text(
                (box[0] + 6, box[1] + 6),
                cell["text"][:48],
                fill=tuple(render_profile["text_rgb"]),
                font=font,
                spacing=render_profile["text_spacing"],
            )
    blur = 0.0
    if any(key in phenomenon for key in ("blur", "noise", "compression")):
        blur = round(rng.uniform(0.3, 1.2), 3)
        image = image.filter(ImageFilter.GaussianBlur(radius=blur))
    rotation = 0.0
    if "rotation" in phenomenon:
        rotation = round(rng.uniform(-3.0, 3.0), 3)
        image = image.rotate(rotation, expand=True, fillcolor=background)
    normalized = image.convert("L")
    normalized_pixel_sha256 = _sha256(normalized.tobytes())
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return (
        buffer.getvalue(),
        normalized_pixel_sha256,
        {
            "width_px": image.width,
            "height_px": image.height,
            "rotation_degrees": rotation,
            "gaussian_blur_radius": blur,
            "border_condition": "weak_or_partial" if weak else "visible",
            "background_rgb": list(background),
            **render_profile,
        },
    )


def _generative_identity(
    base_seed: int,
    rows: int,
    columns: int,
    phenomenon: str,
    category: str,
    realized_structure_sha256: str | None = None,
) -> dict[str, Any]:
    geometry = _geometry_profile(base_seed, rows, columns)
    renderer = _render_profile(base_seed, phenomenon)
    header_depth = 2 if "header" in phenomenon or category == "complex_correction" else 1
    components = {
        "structure": {
            "rows": rows,
            "columns": columns,
            "header_depth": header_depth,
            "span_profile": {
                "horizontal": "span" in phenomenon
                or category in {"single_minimal_edit", "complex_correction"},
                "vertical": "mixed" in phenomenon or category == "complex_correction",
            },
            "realized_structure_sha256": realized_structure_sha256,
        },
        "content": {
            "namespace": _sha256(f"content|{base_seed}".encode("utf-8")),
            "phenomenon": phenomenon,
        },
        "geometry": geometry,
        "renderer": renderer,
    }
    return {
        "generative_family_key": _sha256(_stable_json(components)),
        "generative_components": components,
        "template_family_key": _sha256(_stable_json(components["structure"])),
        "content_family_key": _sha256(_stable_json(components["content"])),
        "geometry_family_key": _sha256(_stable_json(components["geometry"])),
        "renderer_family_key": _sha256(_stable_json(components["renderer"])),
    }


def _expand_coverage(path: Path) -> list[CoverageRequest]:
    requests: list[CoverageRequest] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            roles = (
                ["train"] * int(row["train_count"])
                + ["development"] * int(row["development_count"])
                + ["holdout"] * int(row["holdout_count"])
            )
            assert len(roles) == int(row["requested_count"])
            requests.extend(
                CoverageRequest(row["category"], row["gate_label"], row["phenomenon"], role)
                for role in roles
            )
    order = {"exact_keep": 0, "hard_keep": 1, "single_minimal_edit": 2, "complex_correction": 3}
    return sorted(requests, key=lambda item: (order[item.category], item.role, item.phenomenon))


@lru_cache(maxsize=None)
def _shape_pool(shuffle_seed: int = 2026081204) -> tuple[tuple[int, int], ...]:
    values = [(rows, columns) for rows in range(3, 29) for columns in range(2, 15) if rows * columns <= 240]
    random.Random(shuffle_seed).shuffle(values)
    return tuple(values)


def _sample_shape(base_seed: int, config: dict[str, Any]) -> tuple[int, int]:
    structure = config.get("structure", {})
    minimum_rows = int(structure.get("rows", {}).get("minimum", 3))
    maximum_rows = int(structure.get("rows", {}).get("maximum", 28))
    minimum_columns = int(structure.get("columns", {}).get("minimum", 2))
    maximum_columns = int(structure.get("columns", {}).get("maximum", 14))
    maximum_cells = int(structure.get("maximum_cells", 240))
    choices = [
        (rows, columns)
        for rows in range(minimum_rows, maximum_rows + 1)
        for columns in range(minimum_columns, maximum_columns + 1)
        if rows * columns <= maximum_cells
    ]
    if not choices:
        raise ValueError("no valid table shapes under the frozen configuration")
    rng = _deterministic_rng("shape-2026.08.12.6", base_seed)
    return choices[rng.randrange(len(choices))]


def _make_record(
    request: CoverageRequest,
    sample_index: int,
    base_family_index: int,
    pair_index: int | None,
    config: dict[str, Any],
    license_manifest_sha256: str,
    schema: dict[str, Any],
    shared_phenomenon: str | None = None,
    dataset_release: str = DATASET_RELEASE,
    schema_release: str = SCHEMA_RELEASE,
    generator_release: str = RELEASE,
    shape_shuffle_seed: int = 2026081204,
    base_seed_override: int | None = None,
    family_nonce: int = 0,
) -> tuple[dict[str, Any], bytes]:
    seed = int(config["seed_start"]) + sample_index
    base_seed = (
        base_seed_override
        if base_seed_override is not None
        else int(config["seed_start"]) + 100000 + base_family_index
    )
    if base_seed_override is None:
        shape_pool = _shape_pool(shape_shuffle_seed)
        rows, columns = shape_pool[base_family_index % len(shape_pool)]
    else:
        rows, columns = _sample_shape(base_seed, config)
    pair_id = f"pair-{pair_index:06d}" if pair_index is not None else None
    family = pair_id or f"family-{base_family_index:06d}"
    rendering_phenomenon = shared_phenomenon or request.phenomenon
    # A counterfactual KEEP/EDIT pair must share a Gold state that actually
    # supports the requested structural edit.  The rendering phenomenon still
    # carries the hard-KEEP appearance, while the Gold builder receives an
    # edit-capable structural profile.
    gold_category = "single_minimal_edit" if pair_index is not None else request.category
    gold = _gold_table(base_seed, rows, columns, rendering_phenomenon, gold_category)
    realized_structure = {
        "rows": gold["rows"],
        "columns": gold["columns"],
        "cells": [
            {
                "row": cell["row"],
                "col": cell["col"],
                "rowspan": cell["rowspan"],
                "colspan": cell["colspan"],
                "tag": cell["tag"],
            }
            for cell in gold["cells"]
        ],
    }
    realized_structure_sha256 = _sha256(_stable_json(realized_structure))
    generative = _generative_identity(
        base_seed,
        rows,
        columns,
        rendering_phenomenon,
        gold_category,
        realized_structure_sha256,
    )
    if request.gate_label == "KEEP":
        prior = copy.deepcopy(gold)
        operations: list[str] = []
    else:
        prior, operations = _prior_for_edit(gold, request.phenomenon, request.category)
    difference = _difference(prior, gold)
    image_bytes, pixel_sha, render_parameters = _draw_table(gold, rendering_phenomenon, base_seed)
    image_name = f"images/{dataset_release}-{sample_index:06d}.png"
    identity = {
        "generation_seed": seed,
        "document_cluster_id": f"document-{generative['generative_family_key']}",
        "source_family_id": f"project-authored-{generative['generative_family_key']}",
        "template_family_id": f"template-{generative['template_family_key']}",
        "content_family_id": f"content-{generative['content_family_key']}",
        "renderer_family_id": f"renderer-{generative['renderer_family_key']}",
        "font_family_id": render_parameters["font_family_id"],
        "counterfactual_pair_id": pair_id,
    }
    allowed_identity = schema["properties"]["identity"].get("properties", {})
    if "counterfactual_group_id" in allowed_identity:
        identity["counterfactual_group_id"] = pair_id
    if "generative_family_key" in allowed_identity:
        identity["generative_family_key"] = generative["generative_family_key"]
    if "generative_components" in allowed_identity:
        identity["generative_components"] = generative["generative_components"]
    if "family_nonce" in allowed_identity:
        identity["family_nonce"] = family_nonce
    if "base_seed" in allowed_identity:
        identity["base_seed"] = base_seed
    record = {
        "schema_release": schema_release,
        "dataset_release": dataset_release,
        "sample_id": f"{dataset_release}-{sample_index:06d}",
        "role": request.role,
        "gate_label": request.gate_label,
        "category": request.category,
        "identity": identity,
        "provenance": {
            "generator_release": generator_release,
            "source_id": "project-authored-synthetic-content",
            "license_decision": LICENSE_DECISION,
            "license_manifest_sha256": license_manifest_sha256,
            "terminal_inputs_used": False,
            "phenomenon": request.phenomenon,
            "shared_rendering_phenomenon": rendering_phenomenon,
            "corruption_operations": operations,
        },
        "rendering": {
            "parameters": render_parameters,
            "font_sources": ["pillow-default-embedded-font"],
            "normalized_pixel_sha256": pixel_sha,
        },
        "payloads": {
            "image_path": image_name,
            "image_sha256": _sha256(image_bytes),
            "record_sha256": ZERO_SHA256,
        },
        "gold": gold,
        "prior": prior,
        "views": {
            "gate": {"label": request.gate_label, "false_edit_cost": 4.0, "missed_edit_cost": 1.0},
            "explicit": {
                "prior_state_sha256": prior["semantic_state_sha256"],
                "order_invariant_difference": difference,
                "raw_text_frozen": True,
            },
            "lora": {"complete_canonical_target": gold, "table_only": True},
        },
        "audits": {
            "schema_valid": True,
            "canonical_legal": not validate_cells(gold["cells"]),
            "prior_canonical_legal": not validate_cells(prior["cells"]),
            "grid_complete": not validate_cells(gold["cells"]),
            "prior_grid_complete": not validate_cells(prior["cells"]),
            "gold_recompiled": True,
            "token_ownership_valid": _token_ownership_valid(gold) and _token_ownership_valid(prior),
            "geometry_valid": _geometry_valid(gold) and _geometry_valid(prior),
            "semantic_replay_valid": True,
            "normalized_pixel_replay_valid": True,
            "cross_role_overlap_clear": True,
        },
    }
    record["payloads"]["record_sha256"] = _sha256(_stable_json(record))
    Draft202012Validator(schema).validate(record)
    return record, image_bytes


def _token_ownership_valid(table: dict[str, Any]) -> bool:
    cell_ids = {cell["cell_id"] for cell in table["cells"]}
    token_ids = [token["token_id"] for token in table["tokens"]]
    if len(token_ids) != len(set(token_ids)):
        return False
    return all(token["owner_cell_id"] in cell_ids for token in table["tokens"])


def _geometry_valid(table: dict[str, Any]) -> bool:
    for item in [*table["cells"], *table["tokens"]]:
        bbox = item.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if not all(isinstance(value, (int, float)) for value in bbox):
            return False
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return False
    return True


def _perceptual_hash(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("L").resize((9, 8))
    pixels = list(image.getdata())
    bits = []
    for row in range(8):
        offset = row * 9
        bits.extend(pixels[offset + col] > pixels[offset + col + 1] for col in range(8))
    return f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"


def _hamming(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _overlap_report(records: list[dict[str, Any]], images: dict[str, bytes]) -> dict[str, Any]:
    signatures: list[dict[str, Any]] = []
    for record in records:
        image = images[record["sample_id"]]
        signatures.append(
            {
                "sample_id": record["sample_id"],
                "role": record["role"],
                "image_sha256": record["payloads"]["image_sha256"],
                "pixel_sha256": record["rendering"]["normalized_pixel_sha256"],
                "perceptual_hash": _perceptual_hash(image),
                "structure_sha256": _sha256(
                    _stable_json(
                        {
                            "rows": record["gold"]["rows"],
                            "columns": record["gold"]["columns"],
                            "cells": [
                                {
                                    "row": cell["row"],
                                    "col": cell["col"],
                                    "rowspan": cell["rowspan"],
                                    "colspan": cell["colspan"],
                                    "tag": cell["tag"],
                                }
                                for cell in record["gold"]["cells"]
                            ],
                        }
                    )
                ),
                "text_sha256": _sha256(_stable_json([token["text"] for token in record["gold"]["tokens"]])),
                "geometry_sha256": _sha256(_stable_json([cell["bbox"] for cell in record["gold"]["cells"]])),
                "family_signatures": {
                    key: record["identity"][key]
                    for key in (
                        "document_cluster_id",
                        "source_family_id",
                        "template_family_id",
                        "content_family_id",
                        "renderer_family_id",
                    )
                },
            }
        )
    exact: list[dict[str, str]] = []
    near: list[dict[str, Any]] = []
    for index, first in enumerate(signatures):
        for second in signatures[index + 1 :]:
            if first["role"] == second["role"]:
                continue
            equal_keys = [
                key
                for key in ("image_sha256", "pixel_sha256", "structure_sha256", "text_sha256", "geometry_sha256")
                if first[key] == second[key]
            ]
            exact_payload_match = any(
                key in equal_keys for key in ("image_sha256", "pixel_sha256")
            )
            exact_table_match = all(
                key in equal_keys
                for key in ("structure_sha256", "text_sha256", "geometry_sha256")
            )
            shared_families = [
                key
                for key in first["family_signatures"]
                if first["family_signatures"][key]
                == second["family_signatures"][key]
            ]
            if exact_payload_match or exact_table_match or shared_families:
                exact.append({"first": first["sample_id"], "second": second["sample_id"], "signals": equal_keys})
            distance = _hamming(first["perceptual_hash"], second["perceptual_hash"])
            near_evidence = (
                "text_sha256" in equal_keys
                or all(
                    key in equal_keys
                    for key in ("structure_sha256", "geometry_sha256")
                )
            )
            if distance <= 2 and near_evidence:
                near.append({"first": first["sample_id"], "second": second["sample_id"], "distance": distance, "signals": equal_keys})
    return {
        "release": RELEASE,
        "cross_role_exact_overlaps": exact,
        "cross_role_near_overlaps": near,
        "status": "PASS" if not exact and not near else "FAIL",
    }


def _license_manifest() -> dict[str, Any]:
    return {
        "release": RELEASE,
        "decision": LICENSE_DECISION,
        "sources": [
            {
                "source_id": "project-authored-synthetic-content",
                "origin": "PROJECT_AUTHORED",
                "redistribution": False,
                "local_research_use": True,
                "terminal_derivation": False,
            },
            {
                "source_id": "pillow-default-embedded-font",
                "origin": "PILLOW_RUNTIME",
                "redistribution": False,
                "local_research_use": True,
                "terminal_derivation": False,
            },
            {
                "source_id": "local-system-font-rebuild-only",
                "origin": "LOCAL_SYSTEM_FONT_RUNTIME",
                "redistribution": False,
                "local_research_use": True,
                "terminal_derivation": False,
            },
        ],
    }


def _validate_record_contract(record: dict[str, Any]) -> None:
    gold_errors = validate_cells(record["gold"]["cells"])
    prior_errors = validate_cells(record["prior"]["cells"])
    if gold_errors or prior_errors:
        raise ValueError(f"canonical validation failed: gold={gold_errors}, prior={prior_errors}")
    equal = record["gold"]["semantic_state_sha256"] == record["prior"]["semantic_state_sha256"]
    if record["gate_label"] == "KEEP" and not equal:
        raise ValueError("KEEP_NOT_IDENTITY")
    if record["gate_label"] == "EDIT" and equal:
        raise ValueError("EDIT_NOT_REPLAYABLE")
    difference = record["views"]["explicit"]["order_invariant_difference"]
    if record["gate_label"] == "KEEP" and difference:
        raise ValueError("KEEP_EXPLICIT_DIFF_NOT_EMPTY")
    if record["gate_label"] == "EDIT" and not difference:
        raise ValueError("EDIT_EXPLICIT_DIFF_EMPTY")
    if not _token_ownership_valid(record["gold"]) or not _token_ownership_valid(record["prior"]):
        raise ValueError("TOKEN_OWNERSHIP_INVALID")
    if not _geometry_valid(record["gold"]) or not _geometry_valid(record["prior"]):
        raise ValueError("GEOMETRY_INVALID")


def _negative_grid_fixture() -> str:
    cells = [
        {"row": 0, "col": 0, "rowspan": 1, "colspan": 1, "text": "A", "tag": "td"},
        {"row": 1, "col": 1, "rowspan": 1, "colspan": 1, "text": "B", "tag": "td"},
    ]
    errors = validate_cells(cells)
    if not any("holes" in error for error in errors):
        raise AssertionError("negative fixture was not rejected")
    return "GRID_INCOMPLETE"


def generate_smoke(
    output: Path,
    config_path: Path,
    coverage_path: Path,
    schema_path: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"non-overwrite path already exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    requests = _expand_coverage(coverage_path)
    if len(requests) != config["requested_records"]:
        raise ValueError("coverage count does not match configuration")
    output.mkdir(parents=True)
    for name in ("images", "manifests", "records", "audits", "reports", "quarantine"):
        (output / name).mkdir()
    input_hashes = {
        "config_sha256": _sha256(config_path.read_bytes()),
        "coverage_sha256": _sha256(coverage_path.read_bytes()),
        "schema_sha256": _sha256(schema_path.read_bytes()),
    }
    (output / "PREREGISTRATION.md").write_text(
        "# Data Smoke Preregistration 2026.08.12.4\n\n"
        "This run generates exactly 256 records under the frozen four-category "
        "coverage matrix. It performs no model training and no benchmark "
        "evaluation. Every seed must pass schema, Canonical legality, complete "
        "Gold and prior grids, Gold recompilation, token ownership, geometry, "
        "two-run semantic and pixel replay, source/license, counterfactual-pair, "
        "and cross-role overlap checks. Any failure stops the run; failed seeds "
        "are not replaced.\n",
        encoding="utf-8",
    )
    run_config = {
        "release": RELEASE,
        "dataset_release": DATASET_RELEASE,
        "purpose": "BOUNDED_GENERATOR_SMOKE",
        "training": False,
        "benchmark_evaluation": False,
        "terminal_inputs_used": False,
        "output_path": str(output.resolve()),
        **input_hashes,
    }
    (output / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "input_manifest.json").write_text(
        json.dumps(
            {
                "release": RELEASE,
                "inputs": [
                    {"path": str(config_path), "sha256": input_hashes["config_sha256"]},
                    {"path": str(coverage_path), "sha256": input_hashes["coverage_sha256"]},
                    {"path": str(schema_path), "sha256": input_hashes["schema_sha256"]},
                ],
                "terminal_inputs": [],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    license_manifest = _license_manifest()
    license_bytes = _stable_json(license_manifest)
    license_sha = _sha256(license_bytes)
    (output / "manifests" / "source_license.json").write_bytes(
        json.dumps(license_manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    )

    grouped: dict[str, list[CoverageRequest]] = defaultdict(list)
    for request in requests:
        grouped[request.category].append(request)
    hard = grouped["hard_keep"]
    single = grouped["single_minimal_edit"]
    records: list[dict[str, Any]] = []
    images: dict[str, bytes] = {}
    base_family_index = 0
    assignments: dict[int, tuple[int, int | None, str | None]] = {}
    for pair_index, (keep_request, edit_request) in enumerate(zip(hard, single)):
        if keep_request.role != edit_request.role:
            raise ValueError("counterfactual pair roles do not match")
        shared_phenomenon = (
            f"{keep_request.phenomenon} {edit_request.phenomenon}"
        )
        assignments[id(keep_request)] = (
            base_family_index,
            pair_index,
            shared_phenomenon,
        )
        assignments[id(edit_request)] = (
            base_family_index,
            pair_index,
            shared_phenomenon,
        )
        base_family_index += 1
    for category in ("exact_keep", "complex_correction"):
        for request in grouped[category]:
            assignments[id(request)] = (base_family_index, None, None)
            base_family_index += 1

    for sample_index, request in enumerate(requests):
        family_index, pair_index, shared_phenomenon = assignments[id(request)]
        record, image = _make_record(
            request,
            sample_index,
            family_index,
            pair_index,
            config,
            license_sha,
            schema,
            shared_phenomenon,
        )
        replay_record, replay_image = _make_record(
            request,
            sample_index,
            family_index,
            pair_index,
            config,
            license_sha,
            schema,
            shared_phenomenon,
        )
        if record != replay_record:
            raise ValueError(f"NONDETERMINISTIC_SEMANTICS: {record['sample_id']}")
        if image != replay_image:
            raise ValueError(f"NONDETERMINISTIC_PIXELS: {record['sample_id']}")
        _validate_record_contract(record)
        image_path = output / record["payloads"]["image_path"]
        image_path.write_bytes(image)
        records.append(record)
        images[record["sample_id"]] = image

    manifest_path = output / "records" / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    assignments_path = output / "manifests" / "role_assignment.jsonl"
    assignments_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": record["sample_id"],
                    "role": record["role"],
                    "template_family_id": record["identity"]["template_family_id"],
                    "content_family_id": record["identity"]["content_family_id"],
                    "renderer_family_id": record["identity"]["renderer_family_id"],
                    "counterfactual_pair_id": record["identity"]["counterfactual_pair_id"],
                },
                sort_keys=True,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    overlap = _overlap_report(records, images)
    (output / "audits" / "overlap_report.json").write_text(
        json.dumps(overlap, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    negative_reason = _negative_grid_fixture()
    category_counts = Counter(record["category"] for record in records)
    role_counts = Counter(record["role"] for record in records)
    pair_counts = Counter(
        record["identity"]["counterfactual_pair_id"]
        for record in records
        if record["identity"]["counterfactual_pair_id"] is not None
    )
    complete_pairs = sum(count == 2 for count in pair_counts.values())
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        pair_id = record["identity"]["counterfactual_pair_id"]
        if pair_id is not None:
            by_pair[pair_id].append(record)
    pair_identity_valid = all(
        len(members) == 2
        and members[0]["role"] == members[1]["role"]
        and members[0]["gold"]["full_state_sha256"]
        == members[1]["gold"]["full_state_sha256"]
        and members[0]["payloads"]["image_sha256"]
        == members[1]["payloads"]["image_sha256"]
        for members in by_pair.values()
    )
    acceptance = {
        "release": RELEASE,
        "dataset_release": DATASET_RELEASE,
        "requested": len(requests),
        "generated": len(records),
        "accepted": len(records),
        "quarantined": 0,
        "failed": 0,
        "category_counts": dict(sorted(category_counts.items())),
        "role_counts": dict(sorted(role_counts.items())),
        "complete_counterfactual_pairs": complete_pairs,
        "counterfactual_pair_identity_valid": pair_identity_valid,
        "negative_grid_fixture_reason": negative_reason,
        "overlap_status": overlap["status"],
        "terminal_inputs_used": False,
    }
    expected_categories = {name: value["count"] for name, value in config["categories"].items()}
    expected_roles = {role: count * 4 for role, count in config["roles_per_category"].items()}
    checks = {
        "count": acceptance["accepted"] == 256,
        "categories": dict(category_counts) == expected_categories,
        "roles": dict(role_counts) == expected_roles,
        "pairs": complete_pairs >= config["counterfactual_pairs"]["minimum_pairs"],
        "pair_identity": pair_identity_valid,
        "overlap": overlap["status"] == "PASS",
        "negative_fixture": negative_reason == "GRID_INCOMPLETE",
        "terminal_nonuse": not acceptance["terminal_inputs_used"],
    }
    acceptance["checks"] = checks
    acceptance["status"] = "PASS" if all(checks.values()) else "FAIL"
    (output / "reports" / "acceptance_report.json").write_text(
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "EXPERIMENT_RECORD.md").write_text(
        "# Experiment Record: Data Smoke 2026.08.12.4\n\n"
        f"Status: `{acceptance['status']}`\n\n"
        "## Scope\n\n"
        "A bounded deterministic generator smoke over newly authored synthetic "
        "content. No model training, current benchmark evaluation, or terminal "
        "content access occurred.\n\n"
        "## Accounting\n\n"
        f"- Requested: `{acceptance['requested']}`\n"
        f"- Generated: `{acceptance['generated']}`\n"
        f"- Accepted: `{acceptance['accepted']}`\n"
        f"- Quarantined: `{acceptance['quarantined']}`\n"
        f"- Failed: `{acceptance['failed']}`\n"
        f"- Complete counterfactual pairs: `{acceptance['complete_counterfactual_pairs']}`\n"
        f"- Cross-role overlap audit: `{acceptance['overlap_status']}`\n"
        f"- Negative grid fixture: `{acceptance['negative_grid_fixture_reason']}`\n\n"
        "## Boundary\n\n"
        "A pass is data-generation evidence only. It does not authorize model "
        "training or support a model-performance claim.\n",
        encoding="utf-8",
    )
    (output / "quarantine" / "quarantine.jsonl").write_text("", encoding="utf-8")
    if acceptance["status"] != "PASS":
        raise RuntimeError(f"data smoke failed: {checks}")

    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (output / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path.read_bytes())}  {path.relative_to(output)}\n" for path in files),
        encoding="utf-8",
    )
    return acceptance


def verify_smoke(output: Path, schema_path: Path) -> dict[str, Any]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    records = [
        json.loads(line)
        for line in (output / "records" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for record in records:
        validator.validate(record)
        _validate_record_contract(record)
        hashed_record = copy.deepcopy(record)
        expected_record_sha256 = hashed_record["payloads"]["record_sha256"]
        hashed_record["payloads"]["record_sha256"] = ZERO_SHA256
        if _sha256(_stable_json(hashed_record)) != expected_record_sha256:
            raise ValueError(f"record hash mismatch: {record['sample_id']}")
        image_path = output / record["payloads"]["image_path"]
        if _sha256(image_path.read_bytes()) != record["payloads"]["image_sha256"]:
            raise ValueError(f"image hash mismatch: {record['sample_id']}")
    for line in (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected_sha256, relative_path = line.split("  ", 1)
        payload_path = output / relative_path
        if _sha256(payload_path.read_bytes()) != expected_sha256:
            raise ValueError(f"sealed payload hash mismatch: {relative_path}")
    acceptance = json.loads((output / "reports" / "acceptance_report.json").read_text())
    if acceptance["status"] != "PASS" or len(records) != 256:
        raise ValueError("smoke acceptance report is not sealed-pass ready")
    return acceptance
