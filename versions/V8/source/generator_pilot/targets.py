"""Lossless cell-text/grid serialization for the saved TeleOCR OTSL decoder.

No predicted text or quality scores enter this module. Unsupported inputs fail
explicitly rather than being silently padded, normalized, or repaired.
"""
import importlib.util
import re
import sys
from pathlib import Path
from html.parser import HTMLParser
from .common import ROOT

TOKENS = ('<fcel>', '<ecel>', '<lcel>', '<ucel>', '<xcel>', '<nl>')

class Parser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows = []
        self.row = self.cell = None
        self.opened = self.closed = False
        self.table_count = 0
        self.decorations = set()

    def handle_starttag(self, tag, attrs):
        if tag == 'table':
            if self.opened or self.table_count:
                raise ValueError('multiple_or_nested_tables')
            self.opened = True
            self.table_count += 1
        elif tag == 'tr':
            if not self.opened or self.row is not None:
                raise ValueError('invalid_row_boundary')
            self.row = []
        elif tag in ('td', 'th'):
            if self.row is None or self.cell is not None:
                raise ValueError('invalid_cell_boundary')
            if tag == 'th':
                raise ValueError('header_tag_not_representable_by_saved_decoder')
            attrs = dict(attrs)
            spans = []
            for name in ('rowspan', 'colspan'):
                value = attrs.get(name, '1')
                if not re.fullmatch('[1-9][0-9]*', value) or int(value) > 400:
                    raise ValueError('invalid_span')
                spans.append(int(value))
            self.cell = dict(rowspan=spans[0], colspan=spans[1], text='')
        elif tag in ('thead', 'tbody', 'tfoot', 'html', 'body'):
            pass
        elif self.cell is not None:
            if tag not in ('b', 'i', 'sup', 'sub'):
                raise ValueError('unsupported_cell_markup:' + tag)
            self.decorations.add(tag)
        else:
            raise ValueError('unsupported_tag_outside_cell:' + tag)

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            if self.cell is None:
                raise ValueError('invalid_cell_close')
            self.row.append(self.cell)
            self.cell = None
        elif tag == 'tr':
            if self.row is None or self.cell is not None:
                raise ValueError('invalid_row_close')
            self.rows.append(self.row)
            self.row = None
        elif tag == 'table':
            if not self.opened or self.row is not None:
                raise ValueError('invalid_table_close')
            self.opened = False
            self.closed = True

    def handle_data(self, data):
        if self.cell is not None:
            self.cell['text'] += data
        elif data.strip():
            raise ValueError('text_outside_cell')

def parse_html(value):
    p = Parser()
    p.feed(value)
    p.close()
    if p.opened or not p.closed or not p.rows or p.cell is not None:
        raise ValueError('incomplete_table')
    # The saved decoder exports plain cell text. Preserve all characters but
    # retain the original rich HTML separately for evaluation and audit.
    cells, grid = [], {}
    for ri, row in enumerate(p.rows):
        ci = 0
        for cell in row:
            while (ri, ci) in grid:
                ci += 1
            text = cell['text']
            if text != text.strip():
                raise ValueError('boundary_whitespace_stripped_by_saved_decoder')
            if any(token in text for token in TOKENS):
                raise ValueError('reserved_otsl_token_in_cell_text')
            rs, cs = cell['rowspan'], cell['colspan']
            if ri + rs > len(p.rows):
                raise ValueError('rowspan_outside_table')
            index = len(cells)
            cells.append(dict(row=ri, col=ci, **cell))
            for r in range(ri, ri + rs):
                for c in range(ci, ci + cs):
                    if (r, c) in grid:
                        raise ValueError('overlapping_spans')
                    grid[r, c] = index
            ci += cs
    if not cells:
        raise ValueError('empty_table')
    cols = max(c for r, c in grid) + 1
    if len(grid) != len(p.rows) * cols:
        raise ValueError('ragged_grid_no_implicit_empty_cells')
    return dict(rows=len(p.rows), cols=cols, cells=cells)

def serialize(table):
    grid = {}
    for cell in table['cells']:
        r, c = cell['row'], cell['col']
        for y in range(r, r + cell['rowspan']):
            for x in range(c, c + cell['colspan']):
                if (y, x) in grid:
                    raise ValueError('overlapping_spans')
                if y == r and x == c:
                    grid[y, x] = '<fcel>' + cell['text'] if cell['text'] else '<ecel>'
                else:
                    grid[y, x] = '<lcel>' if y == r else '<ucel>' if x == c else '<xcel>'
    if set(grid) != {(r, c) for r in range(table['rows']) for c in range(table['cols'])}:
        raise ValueError('nonrectangular_grid')
    return ''.join(''.join(grid[r, c] for c in range(table['cols'])) + '<nl>' for r in range(table['rows']))

def load_saved_decoder(path=None):
    import os
    path = path or os.environ.get('TELEOCR_OTSL_DECODER')
    if not path:
        raise ValueError('Supply the preserved TeleOCR otsl2html.py path')
    path = Path(path)
    spec = importlib.util.spec_from_file_location('generator_saved_otsl_decoder', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.convert_otsl_to_html

def roundtrip(html, decoder):
    table = parse_html(html)
    target = serialize(table)
    decoded = parse_html(decoder(target))
    if table != decoded:
        raise ValueError('saved_decoder_roundtrip_mismatch')
    return table, target
