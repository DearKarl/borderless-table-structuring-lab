"""Independent, strict native-format utilities. Pure Python; no model or file I/O.

Format references (not imported/copied pipeline implementations):
https://github.com/caipeng328/NaviDC-OCR/blob/2483d473ae272fd137a744a37dcdb7b201578ece/NaviOCR/vlm_utils/NaviOCR_client.py
https://github.com/caipeng328/NaviDC-OCR/blob/2483d473ae272fd137a744a37dcdb7b201578ece/NaviOCR/vlm_utils/structs.py
https://huggingface.co/StarDoc-AI/NaviDC-OCR/blob/710ea2e26d794fe89cbf3ece0402707c332a8671/README.md

The first two references specify the layout wire grammar, orientation and label
vocabulary. The Apache-2.0 HF card specifies the six OTSL markers. Our owner-grid
validator is independently implemented and deliberately does NOT copy permissive
padding, skipped malformed blocks, HTML passthrough, or text stripping.

Validity means format validity, not image correctness or reproduction of the
published pipeline/97.05 result. No reading-order sorting, geometry repair,
inference, scoring, cropping, or policy adoption occurs here. Polygon layouts
are explicitly unsupported in this axis-aligned first implementation.
"""
import html
from pathlib import PurePosixPath
import re
from urllib.parse import quote

LABELS = frozenset(('text', 'title', 'table', 'image', 'code', 'algorithm',
                   'header', 'footer', 'page_number', 'page_footnote', 'aside_text',
                   'equation', 'equation_block', 'ref_text', 'list', 'phonetic',
                   'table_caption', 'image_caption', 'code_caption', 'table_footnote',
                   'image_footnote', 'unknown', 'seal', 'char'))
ANGLES = {'up': 0, 'right': 90, 'down': 180, 'left': 270}
EOS = ('<|im_end|>', '<|endoftext|>')
TOKENS = frozenset(('fcel', 'ecel', 'lcel', 'ucel', 'xcel', 'nl'))
MAX_CHARS = 2_000_000
MAX_SLOTS = 100_000


class FormatError(ValueError):
    def __init__(self, code, detail):
        self.code, self.detail = code, detail
        super().__init__(f'{code}: {detail}')


def _require(condition, code, detail):
    if not condition:
        raise FormatError(code, detail)


def _body(text, terminal_eos_present, truncated):
    _require(isinstance(text, str), 'NOT_TEXT', 'Expected decoded string')
    _require(len(text) <= MAX_CHARS, 'FORMAT_SIZE_LIMIT', 'Input exceeds parser bound')
    _require(not truncated, 'TRUNCATED_GENERATION', 'Caller reports token-budget truncation')
    _require(terminal_eos_present is None or isinstance(terminal_eos_present, bool),
             'BAD_EOS_METADATA', 'Expected bool or None')
    body = text
    terminal = text.rstrip()
    marker = next((token for token in EOS if terminal.endswith(token)), None)
    if marker:
        body = terminal[:-len(marker)]
    _require(not any(token in body for token in EOS), 'NONTERMINAL_EOS', 'EOS inside content')
    _require(terminal_eos_present is not False, 'INCOMPLETE_GENERATION', 'Caller reports no terminal EOS')
    return body, {'terminal_marker': marker,
                  'generation_complete': True if marker or terminal_eos_present is True else None,
                  'completion_scope': 'Syntax alone does not prove generation completion'}


def _result(operation):
    try:
        return {'valid': True, 'error': None, **operation()}
    except FormatError as exc:
        return {'valid': False, 'error': {'code': exc.code, 'detail': exc.detail}}


def parse_layout(text, *, terminal_eos_present=None, truncated=False):
    """Return ordered blocks; bbox is normalized xyxy, bbox_1000 is unchanged.

    Empty separator lines are counted. Any malformed nonempty line invalidates
    the whole output rather than silently omitting a detection. No /999 clipping.
    """
    def parse():
        body, end = _body(text, terminal_eos_present, truncated)
        blocks, blanks = [], 0
        pattern = r'<box:([0-9\s]+)><label:([A-Za-z_]+)><(up|right|down|left)>'
        for line_number, raw in enumerate(body.splitlines(), 1):
            line = raw.strip()
            if not line:
                blanks += 1
                continue
            match = re.fullmatch(pattern, line)
            _require(match is not None, 'MALFORMED_LAYOUT_LINE', f'Line {line_number}')
            coordinates, label, direction = match.groups()
            values = [int(value) for value in coordinates.split()]
            if len(values) > 4 and len(values) % 2 == 0:
                raise FormatError('UNSUPPORTED_POLYGON_LAYOUT', f'Line {line_number}: {len(values)} coordinates')
            _require(len(values) == 4, 'BOX_ARITY', f'Line {line_number}')
            _require(label in LABELS, 'UNKNOWN_LAYOUT_LABEL', label)
            x0, y0, x1, y1 = values
            _require(0 <= x0 < x1 <= 1000 and 0 <= y0 < y1 <= 1000,
                     'INVALID_BOX', f'Line {line_number}: {values}')
            blocks.append({'index': len(blocks), 'source_line': line_number,
                           'type': label, 'bbox_1000': values,
                           'bbox': [value / 1000 for value in values],
                           'direction': direction, 'angle': ANGLES[direction]})
        _require(bool(blocks), 'EMPTY_LAYOUT', 'No complete layout blocks')
        return {'blocks': blocks, 'blank_separator_lines': blanks, **end}
    return _result(parse)


def _lex_otsl(body):
    """A cell marker owns following literal text only when it is fcel."""
    rows, row, previous, end = [], [], None, 0
    count = 0
    for match in re.finditer(r'<[^<>]*>', body):
        payload = body[end:match.start()]
        _require(not re.search(r'<(?:fcel|ecel|lcel|ucel|xcel|nl)(?=[^A-Za-z]|$)', payload),
                 'MALFORMED_OTSL_TOKEN', f'Character {end}')
        if previous is not None and previous['token'] == 'fcel':
            previous['text'] = payload
        else:
            _require(not payload.strip(), 'TEXT_OUTSIDE_FCEL', f'Character {end}')
        token = match.group()[1:-1]
        _require(token in TOKENS, 'UNKNOWN_OTSL_TOKEN', match.group())
        if token == 'nl':
            _require(bool(row), 'EMPTY_OTSL_ROW', f'Row {len(rows)}')
            rows.append(row)
            row = []
            previous = {'token': 'nl'}
        else:
            previous = {'token': token, 'text': ''}
            row.append(previous)
            count += 1
            _require(count <= MAX_SLOTS, 'FORMAT_SIZE_LIMIT', 'Too many table positions')
        end = match.end()
    tail = body[end:]
    _require(not row, 'UNTERMINATED_OTSL_ROW', 'Each row must end with <nl>')
    _require(not tail.strip(), 'TRAILING_OTSL_TEXT', f'Character {end}')
    _require(bool(rows), 'EMPTY_OTSL', 'No complete rows')
    width = len(rows[0])
    _require(all(len(row) == width for row in rows), 'RAGGED_OTSL', 'No invented padding is permitted')
    for row in rows:
        for item in row:
            if item['token'] == 'fcel':
                _require(bool(item['text'].strip()), 'EMPTY_FCEL', 'Use ecel for an empty cell')
    return rows


def parse_otsl(text, *, terminal_eos_present=None, truncated=False):
    """Validate a complete rectangular topology, returning classical escaped HTML.

    lcel extends an owner right on its first row; ucel extends it downward on
    its first column; xcel must agree with both neighboring owner IDs. A final
    rectangle check rejects incomplete compound spans. Literal text is retained.
    """
    def parse():
        body, end = _body(text, terminal_eos_present, truncated)
        rows = _lex_otsl(body)
        owners, cells = [], []
        for r, row in enumerate(rows):
            owners.append([])
            for c, item in enumerate(row):
                token = item['token']
                if token in ('fcel', 'ecel'):
                    owner = len(cells)
                    cells.append({'row': r, 'col': c, 'text': item['text'], 'positions': []})
                elif token == 'lcel':
                    _require(c > 0, 'ORPHAN_LCEL', f'{r},{c}')
                    owner = owners[r][c - 1]
                    _require(cells[owner]['row'] == r, 'WRONG_LCEL_TOPOLOGY', f'{r},{c}')
                elif token == 'ucel':
                    _require(r > 0, 'ORPHAN_UCEL', f'{r},{c}')
                    owner = owners[r - 1][c]
                    _require(cells[owner]['col'] == c, 'WRONG_UCEL_TOPOLOGY', f'{r},{c}')
                else:  # xcel
                    _require(r > 0 and c > 0, 'ORPHAN_XCEL', f'{r},{c}')
                    owner = owners[r - 1][c]
                    _require(owner == owners[r][c - 1] and cells[owner]['row'] < r
                             and cells[owner]['col'] < c, 'CROSS_OWNER_XCEL', f'{r},{c}')
                owners[r].append(owner)
                cells[owner]['positions'].append((r, c))
        for owner, cell in enumerate(cells):
            bottom = max(r for r, _ in cell['positions'])
            right = max(c for _, c in cell['positions'])
            cell['rowspan'] = bottom - cell['row'] + 1
            cell['colspan'] = right - cell['col'] + 1
            _require(len(cell['positions']) == cell['rowspan'] * cell['colspan'],
                     'NONRECTANGULAR_SPAN', f"Cell {cell['row']},{cell['col']}")
            for r in range(cell['row'], bottom + 1):
                for c in range(cell['col'], right + 1):
                    _require(owners[r][c] == owner, 'OVERLAPPING_SPAN', f'{r},{c}')
            del cell['positions']
        markup = ['<table>']
        by_row = {r: [] for r in range(len(rows))}
        for cell in cells:
            attrs = ''.join(f' {key}="{cell[key]}"' for key in ('rowspan', 'colspan') if cell[key] > 1)
            by_row[cell['row']].append('<td' + attrs + '>' + html.escape(cell['text'], quote=True) + '</td>')
        for r in range(len(rows)):
            markup.append('<tr>' + ''.join(by_row[r]) + '</tr>')
        markup.append('</table>')
        return {'rows': len(rows), 'cols': len(rows[0]), 'cells': cells,
                'html': ''.join(markup), **end}
    return _result(parse)


def assemble_markdown(blocks):
    """Assemble ALL supplied blocks in order, or explicitly fail the whole page.

    Each block needs known type and content (tables/char: native OTSL). Image
    blocks instead need a caller-supplied relative image_ref; no image is fetched
    or fabricated. Ordinary text labels remain paragraphs without invented title
    levels/list hierarchy. Equations are not repaired, parsed, or normalized.
    Table HTML is produced only by parse_otsl, never passed through from a model.
    Optional terminal_eos_present/truncated fields are honored per content block.
    """
    def assemble():
        _require(isinstance(blocks, (list, tuple)) and bool(blocks), 'EMPTY_PAGE', 'Expected ordered blocks')
        fragments, audit = [], []
        for index, block in enumerate(blocks):
            _require(isinstance(block, dict), 'BAD_BLOCK', f'Block {index}')
            kind = block.get('type')
            _require(kind in LABELS, 'UNKNOWN_LAYOUT_LABEL', str(kind))
            if kind == 'image':
                ref = block.get('image_ref')
                _require(isinstance(ref, str) and bool(ref), 'MISSING_IMAGE_REFERENCE', f'Block {index}')
                path = PurePosixPath(ref)
                _require(not path.is_absolute() and '..' not in path.parts and ':' not in ref
                         and '\\' not in ref and not any(ord(ch) < 32 for ch in ref),
                         'UNSAFE_IMAGE_REFERENCE', ref)
                fragment = '![](' + quote(ref, safe='/.-_') + ')'
                completion = None
            elif kind in ('table', 'char'):
                parsed = parse_otsl(block.get('content'),
                                    terminal_eos_present=block.get('terminal_eos_present'),
                                    truncated=block.get('truncated', False))
                _require(parsed['valid'], 'INVALID_TABLE_BLOCK', f"Block {index}: {parsed['error']}")
                fragment, completion = parsed['html'], parsed['generation_complete']
            else:
                content, end = _body(block.get('content'), block.get('terminal_eos_present'),
                                     block.get('truncated', False))
                _require(bool(content.strip()), 'MISSING_BLOCK_CONTENT', f'Block {index}')
                if kind in ('code', 'algorithm'):
                    length = max([len(match.group()) for match in re.finditer(r'`+', content)] + [2]) + 1
                    fence = '`' * length
                    fragment = fence + '\n' + content + '\n' + fence
                else:
                    fragment = html.escape(content, quote=True)
                completion = end['generation_complete']
            fragments.append(fragment)
            audit.append({'index': index, 'type': kind, 'generation_complete': completion})
        return {'markdown': '\n\n'.join(fragments), 'blocks': audit,
                'all_content_generation_complete': all(b['generation_complete'] is True
                                                       for b in audit if b['type'] != 'image'),
                'scope': 'Project-authored assembly; not published-pipeline reproduction'}
    return _result(assemble)
