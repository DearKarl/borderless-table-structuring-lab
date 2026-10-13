"""Whole-page assembly from evaluated assembler003, with a lazy local consumer binding."""
import hashlib
from html.parser import HTMLParser

def consumer_closure(text, native):
    from .consumer import consumer_closure as check
    return check(text, native)

POLICY = "NAVIDC_NATIVE_TABLE_COLLECTION_WITH_MINERU_NONTABLE_V1"


POLICY_DETAILS = {"name": "WHOLE_NATIVE_TABLE_COLLECTION_OR_WHOLE_PAGE_EXACT_RAW",
          "successful_zero_tables": "remove all old tables; no quality fallback",
          "append_separator_hex": "0a0a", "append_terminal_hex": "0a",
          "non_table_chunks_preserved": True, "original_table_interleaving_preserved": False,
          "structure_may_change": True, "count_or_text_disagreement_gate": False}


class BindingError(RuntimeError):
    pass


class AssemblyAbstain(ValueError):
    pass


def require(ok, reason):
    if not ok:
        raise BindingError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class TableSpans(HTMLParser):
    """Locate complete outer table literals without normalizing any Markdown."""
    def __init__(self, text):
        super().__init__(convert_charrefs=False)
        self.text, self.depth, self.start, self.spans = text, 0, None, []
        self.lines = [0]
        self.lines.extend(i + 1 for i, char in enumerate(text) if char == "\n")
        self.feed(text)
        self.close()
        if self.depth:
            raise AssemblyAbstain("UNCLOSED_RAW_TABLE")

    def char_position(self):
        row, col = self.getpos()
        return self.lines[row - 1] + col

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if not self.depth:
                self.start = self.char_position()
            self.depth += 1

    def handle_startendtag(self, tag, attrs):
        if tag == "table":
            raise AssemblyAbstain("SELF_CLOSING_TABLE")

    def handle_endtag(self, tag):
        if tag == "table":
            if not self.depth:
                raise AssemblyAbstain("ORPHAN_TABLE_END")
            self.depth -= 1
            if not self.depth:
                end = self.text.find(">", self.char_position())
                if end < 0:
                    raise AssemblyAbstain("UNCLOSED_TABLE_END")
                self.spans.append((self.start, end + 1))


def raw_chunks(raw, tables):
    text = raw.decode("utf-8", errors="strict")
    spans, identities = [], set()
    for row in tables:
        identity = row.get("table_id")
        if not isinstance(identity, str) or identity in identities:
            raise AssemblyAbstain("INVALID_OR_DUPLICATE_RAW_TABLE_ID")
        identities.add(identity)
        literal = row.get("raw_html")
        if not isinstance(literal, str) or digest(literal.encode("utf-8")) != row.get("raw_html_sha256"):
            raise AssemblyAbstain("RAW_TABLE_LITERAL_SERIALIZATION_INVALID")
        start, end = row.get("start_char"), row.get("end_char")
        if (row.get("mapping_reason") != "UNIQUE_EXACT_RAW_SPAN"
                or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(text) or text[start:end] != literal
                or text.count(literal) != 1):
            raise AssemblyAbstain("RAW_TABLE_SPAN_NOT_UNIQUE_EXACT")
        spans.append((start, end))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise AssemblyAbstain("OVERLAPPING_RAW_TABLE_SPANS")
    if TableSpans(text).spans != spans:
        raise AssemblyAbstain("RAW_TABLE_COLLECTION_NOT_CLOSED")
    chunks, cursor = [], 0
    for start, end in spans:
        byte_start, byte_end = len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))
        chunks.append((cursor, byte_start, raw[cursor:byte_start]))
        cursor = byte_end
    chunks.append((cursor, len(raw), raw[cursor:]))
    return chunks


def assemble_page(raw, tables, native_status, native_tables, failure_reason=None):
    """Pure operation. Input provenance mismatch must be caught before this call."""
    require(native_status in ("NATIVE_PAGE_COMPLETE", "PAGE_ABSTAIN_EXACT_RAW"),
            "NONTERMINAL_NATIVE_PAGE")
    note = {"raw_sha256": digest(raw), "raw_table_count": len(tables),
            "native_table_count": len(native_tables), "policy": POLICY, "policy_details": POLICY_DETAILS}
    if native_status == "PAGE_ABSTAIN_EXACT_RAW":
        return raw, {**note, "status": "EXACT_RAW_NATIVE_ABSTENTION", "reason": failure_reason,
                     "used_native_tables": 0, "final_sha256": digest(raw), "chunks": []}
    try:
        chunks = raw_chunks(raw, tables)
        native = []
        previous = -1
        for table in native_tables:
            index, html = table.get("layout_index"), table.get("html")
            if type(index) is not int or index <= previous or not isinstance(html, str):
                raise AssemblyAbstain("NATIVE_COLLECTION_ORDER_OR_SERIALIZATION_INVALID")
            encoded = html.encode("utf-8", errors="strict")
            if (digest(encoded) != table.get("html_sha256")
                    or TableSpans(html).spans != [(0, len(html))]):
                raise AssemblyAbstain("NATIVE_HTML_NOT_EXACT_SINGLE_TABLE")
            native.append(encoded)
            previous = index
        complement = b"".join(chunk[2] for chunk in chunks)
        suffix = b"\n\n" + b"\n\n".join(native) + b"\n" if native else b""
        final = complement + suffix
        out = final.decode("utf-8", errors="strict")
        found = [out[start:end].encode("utf-8") for start, end in TableSpans(out).spans]
        if found != native or not final.startswith(complement):
            raise AssemblyAbstain("FINAL_TABLE_COLLECTION_SERIALIZATION_INVALID")
        consumer = consumer_closure(out, [value.decode("utf-8") for value in native])
        return final, {**note, "status": "NATIVE_COLLECTION_ASSEMBLED", "reason": None,
            "used_native_tables": len(native), "final_sha256": digest(final),
            "non_table_complement_sha256": digest(complement), "append_bytes": len(suffix),
            "official_consumer_format_check": consumer,
            "chunks": [{"start_byte": start, "end_byte": end, "bytes": len(chunk),
                        "sha256": digest(chunk)} for start, end, chunk in chunks]}
    except (AssemblyAbstain, UnicodeError, ValueError) as exc:
        return raw, {**note, "status": "EXACT_RAW_ASSEMBLY_ABSTENTION", "reason": str(exc),
                     "used_native_tables": 0, "final_sha256": digest(raw), "chunks": []}
