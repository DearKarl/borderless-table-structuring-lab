"""Source-derived detail tasks. No prediction, sampling, or cohort selection occurs here."""
from html import escape
import json
import math

TASKS = ('ordered_text_spotting', 'ordered_text', 'box_text_spotting', 'box_text',
         'markdown', 'html', 'cell_detection', 'span_detection', 'row_column_detection')


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def box1000(box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid box values')
    if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
        raise ValueError('Box is outside the native crop')
    return [round(v * 1000) for v in box]


def validate_grid(table):
    occupied = set()
    if table['rows'] < 1 or table['cols'] < 1:
        raise ValueError('Empty table grid')
    order = []
    for c in table['cells']:
        order.append((c['row'], c['col']))
        if not isinstance(c['text'], str) or c['rowspan'] < 1 or c['colspan'] < 1:
            raise ValueError('Invalid cell')
        for r in range(c['row'], c['row'] + c['rowspan']):
            for col in range(c['col'], c['col'] + c['colspan']):
                if not (0 <= r < table['rows'] and 0 <= col < table['cols']) or (r, col) in occupied:
                    raise ValueError('Invalid or overlapping cell spans')
                occupied.add((r, col))
    if order != sorted(set(order)) or len(occupied) != table['rows'] * table['cols']:
        raise ValueError('Unordered or incomplete grid')


def table_html(table):
    validate_grid(table)
    rows = []
    for r in range(table['rows']):
        cells = []
        for c in table['cells']:
            if c['row'] != r:
                continue
            attrs = ''.join(f' {k}="{c[k]}"' for k in ['rowspan', 'colspan'] if c[k] != 1)
            cells.append('<td' + attrs + '>' + escape(c['text'], quote=False) + '</td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return '<table>' + ''.join(rows) + '</table>'


def table_markdown(table):
    validate_grid(table)
    # Standard Markdown cannot encode spans. Embedded HTML preserves them exactly.
    if any(c['rowspan'] != 1 or c['colspan'] != 1 for c in table['cells']):
        return table_html(table)
    lookup = {(c['row'], c['col']): c['text'] for c in table['cells']}
    def cell(text):
        return escape(text, quote=False).replace('\\', '\\\\').replace('|', '\\|').replace('\n', '<br>')
    rows = ['| ' + ' | '.join(cell(lookup[r, col]) for col in range(table['cols'])) + ' |' for r in range(table['rows'])]
    rows.insert(1, '| ' + ' | '.join('---' for _ in range(table['cols'])) + ' |')
    return '\n'.join(rows)


def verified_lines(table, annotation):
    if annotation.get('status') != 'processed' or not annotation.get('full_table_text_verified'):
        raise ValueError('Complete source-verified text lines are required')
    if len(annotation['cells']) != len(table['cells']):
        raise ValueError('Line/cell membership differs')
    output = []
    for i, (cell, lines) in enumerate(zip(table['cells'], annotation['cells'])):
        if lines['cell_index'] != i or lines['status'] != 'verified':
            raise ValueError('Unverified or reordered cell text')
        rebuilt = []
        for line in lines['lines']:
            start, end = line['text_start'], line['text_end']
            if not 0 <= start < end <= len(cell['text']):
                raise ValueError('Invalid line character range')
            text = cell['text'][start:end]
            box1000(line['normalized_box'])
            output.append(dict(text=text, box=line['normalized_box'], row=cell['row'], col=cell['col']))
            rebuilt.append(text)
        if ' '.join(rebuilt) != cell['text']:
            raise ValueError('Line spans do not reconstruct the frozen cell text')
    return sorted(output, key=lambda x: (x['box'][1], x['box'][0], x['row'], x['col']))


def render(task, table, boxes, *, line_annotation=None, query_box=None, row_boxes=None, column_boxes=None):
    if task not in TASKS:
        raise ValueError('Unknown detail task')
    validate_grid(table)
    if len(boxes) != len(table['cells']):
        raise ValueError('Cell/box count differs')
    quantized = [box1000(b) for b in boxes]
    coordinate_rule = ' Boxes use [left,top,right,bottom], normalized to integers from 0 to 1000.'
    if task in ('ordered_text_spotting', 'ordered_text', 'box_text_spotting', 'box_text'):
        lines = verified_lines(table, line_annotation or {})
        prompt = 'Read the text lines inside the table cells in spatial reading order.'
        if task.startswith('box_'):
            if query_box is None:
                raise ValueError('Box-query task requires an explicit source-derived query')
            query = box1000(query_box)
            selected = []
            for line in lines:
                b = line['box']
                overlaps = min(b[2], query_box[2]) > max(b[0], query_box[0]) and min(b[3], query_box[3]) > max(b[1], query_box[1])
                inside = b[0] >= query_box[0] and b[1] >= query_box[1] and b[2] <= query_box[2] and b[3] <= query_box[3]
                if overlaps and not inside:
                    raise ValueError('Query boundary cuts a text line')
                if inside:
                    selected.append(line)
            lines = selected
            prompt = 'Read the table-cell text lines inside query box ' + compact(query) + ' in spatial reading order.'
        if task.endswith('spotting'):
            return prompt + coordinate_rule + ' Return JSON objects with text and box.', compact([dict(text=x['text'], box=box1000(x['box'])) for x in lines])
        return prompt + ' Return a JSON list of text lines.', compact([x['text'] for x in lines])
    if task == 'markdown':
        return 'Convert the table into Markdown. Use an embedded HTML table if cells span rows or columns.', table_markdown(table)
    if task == 'html':
        return 'Convert the complete table structure and cell content into HTML.', table_html(table)
    if task == 'cell_detection':
        return 'List every table cell box in logical row/column order, including empty cells.' + coordinate_rule, compact(quantized)
    if task == 'span_detection':
        target = [dict(box=b, rows=[c['row'], c['row'] + c['rowspan'] - 1], columns=[c['col'], c['col'] + c['colspan'] - 1])
                  for c, b in zip(table['cells'], quantized) if c['rowspan'] > 1 or c['colspan'] > 1]
        return 'List spanning cells, their boxes, and inclusive zero-based row and column ranges.' + coordinate_rule, compact(target)
    if row_boxes is None or column_boxes is None or len(row_boxes) != table['rows'] or len(column_boxes) != table['cols']:
        raise ValueError('Original verified row/column geometry is required')
    result = {}
    for name, cell_axis, span, geometry in [('rows', 'row', 'rowspan', row_boxes), ('columns', 'col', 'colspan', column_boxes)]:
        result[name] = [dict(index=i, box=box1000(b), cells=[j for j,c in enumerate(table['cells']) if c[cell_axis] <= i < c[cell_axis] + c[span]]) for i,b in enumerate(geometry)]
    result['cell_boxes'] = quantized
    return 'List all row and column boxes and their zero-based cell memberships. Include each cell box once in cell_boxes.' + coordinate_rule, compact(result)
