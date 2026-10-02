"""Frozen consumer projection. Upstream Apache-2.0; see vendor NOTICE.
Only the source folder binding differs from assembler003. No Gold or metrics.
"""
import ast
import re
from pathlib import Path
from functools import lru_cache
from .assembly import AssemblyAbstain, require, digest
import json

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


@lru_cache(maxsize=1)
def official_tools():
    """Load only frozen, side-effect-free format functions, never evaluator imports.

    The AST nodes below are executed unchanged. No metric, matching, annotation,
    image, LaTeX subprocess, BeautifulSoup, matplotlib or model code is loaded.
    Official md_tex_filter extracts HTML before its code-block branch; a fence
    therefore is not an independent reason to reject a page.
    """
    folder = Path(__file__).parent / "vendor/omnidocbench"
    specs = {
        "extract.py": ("6c10a9a9982c3f10436e0ccde73c216688fe129588f49ba6121170bd82b421eb",
            {"extract_html_table", "extract_tex_table", "extract_tabular",
             "_looks_like_image_description_block", "_strip_inline_formula_delimiters",
             "_suppress_formula_delimiters_in_image_descriptions"},
            {"img_pattern", "display_reg", "md_table_reg", "html_table_reg"}),
        "data_preprocess.py": ("d108d0726f0cc36eaf1fed2786a0405531549b3b32879e64fd4690ff88ba9dbd",
            {"remove_markdown_fences", "replace_repeated_chars"}, set()),
        "table_utils.py": ("febb662ee1be4e97343833c3844321e8ac3d4f29452fd2ceeef5f965add9b731",
            {"markdown_to_html", "convert_markdown_to_html", "convert_table", "replace_table_with_placeholder",
             "merge_tables", "delete_table_and_body", "find_md_table_mode"}, set())}
    namespace, bindings = {"re": re}, {}
    for name, (expected, functions, variables) in specs.items():
        path = folder / name
        source = path.read_bytes()
        require(digest(source) == expected, "OFFICIAL_FORMAT_SOURCE_CHANGED: " + name)
        bindings[str(path)] = expected
        selected = []
        for node in ast.parse(source, filename=str(path)).body:
            if isinstance(node, ast.FunctionDef) and node.name in functions:
                selected.append(node)
            elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id in variables for target in node.targets):
                selected.append(node)
        require(functions <= {node.name for node in selected if isinstance(node, ast.FunctionDef)},
                "MISSING_FROZEN_FORMAT_FUNCTION")
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, bindings


def consumer_tables(text):
    """Table-only format projection following frozen md_tex_filter order.

    Any detected LaTeX table makes this limited projection unprovable; abstain
    rather than invoking a LaTeX converter or inventing an equivalent HTML body.
    Formula rescue/text branches cannot append tables and are not executed.
    """
    f, _ = official_tools()
    content = re.sub(f["img_pattern"], lambda match: " " * len(match.group(0)), text)
    content = f["remove_markdown_fences"](content)
    content = f["replace_repeated_chars"](content)
    content = content.replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", "")
    content = f["_suppress_formula_delimiters_in_image_descriptions"](content)
    if f["extract_tex_table"](content)[0]:
        raise AssemblyAbstain("OFFICIAL_CONSUMER_HAS_UNPROVEN_LATEX_TABLE")
    tables, positions = f["extract_html_table"](content)
    for html, position in zip(tables, positions):
        start, end = position[0], position[0] + len(html)
        content = content[:start] + " " * (end - start) + content[end:]
    # Follow md_tex_filter's exact distinction: display math is blanked,
    # but inline $...$/\\(...\\) is retained for the later Markdown-table branch.
    original = content
    for match in f["display_reg"].finditer(original):
        matched = match.group(0)
        if matched:
            single_line = " ".join(matched.strip().split("\n"))
            dollar_pattern = re.compile(r"\$\$(.*?)\$\$|\$(.*?)\$|\\\((.*?)\\\)", re.DOTALL)
            sub_match = dollar_pattern.search(single_line)
            if sub_match is None or sub_match.group(1):
                start, end = match.span()
                content = content[:start] + " " * (end - start) + content[end:]
    if len(f["md_table_reg"].findall(content + "\n")) >= 2:
        converted = f["convert_markdown_to_html"](content)
        tables.extend(match.group(0).strip() for match in f["html_table_reg"].finditer(converted))
    return tables


def consumer_closure(text, native):
    try:
        observed = consumer_tables(text)
        expected = []
        for html in native:
            isolated = consumer_tables(html)
            if len(isolated) != 1:
                raise AssemblyAbstain("NATIVE_TABLE_NOT_SINGLY_CONSUMABLE")
            expected.extend(isolated)
        if observed != expected:
            raise AssemblyAbstain("OFFICIAL_CONSUMER_TABLE_COLLECTION_DIFFERS")
        return {"status": "EXACT_FROZEN_FORMAT_PROJECTION", "table_count": len(expected),
                "normalized_collection_sha256": digest(canonical(expected)),
                "scores_or_answers_read": False, "prediction_bytes_rewritten": False}
    except RecursionError as exc:
        raise AssemblyAbstain("OFFICIAL_CONSUMER_RECURSION_UNSUPPORTED") from exc
