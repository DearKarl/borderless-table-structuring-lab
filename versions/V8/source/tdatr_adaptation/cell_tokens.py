"""Full native OTSL cell intervals, from training or generated text without GT.

BPE may straddle a cell boundary. Both cells include a straddling token rather
than discarding part of either cell; the overlap and character spill are audited.
The caller must also validate the table structure and exact tokenizer roundtrip.
"""
import re

MARKERS = re.compile(r'<(?:fcel|ecel|lcel|ucel|xcel|nl)>')


def full_cell_token_ranges(text, offsets, *, sequence_start=0):
    if sequence_start < 0 or not isinstance(sequence_start, int):
        raise ValueError('Invalid sequence offset')
    markers = list(MARKERS.finditer(text))
    if not markers or markers[0].start() != 0 or markers[-1].group() != '<nl>' or markers[-1].end() != len(text):
        raise ValueError('Incomplete native structure string')
    previous_start = -1
    for start, end in offsets:
        if not 0 <= start < end <= len(text) or start < previous_start:
            raise ValueError('Invalid or unordered tokenizer offsets')
        previous_start = start
    if not offsets:
        raise ValueError('No target tokens')
    cells = []
    for i, marker in enumerate(markers):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(text)
        payload = text[marker.end():end]
        if marker.group() != '<fcel>' and payload:
            raise ValueError('Text follows a non-content structural marker')
        if marker.group() == '<fcel>' and not payload:
            raise ValueError('Empty cell must use its native empty marker')
        if marker.group() not in ('<fcel>', '<ecel>'):
            continue
        start = marker.start()
        matched = [j for j, (a, b) in enumerate(offsets) if a < end and b > start]
        if not matched or matched != list(range(matched[0], matched[-1] + 1)):
            raise ValueError('Cell does not have a contiguous token interval')
        covered = start
        for j in matched:
            a, b = offsets[j]
            if a > covered:
                raise ValueError('Tokenizer offsets omit cell characters')
            covered = max(covered, b)
        if covered < end:
            raise ValueError('Tokenizer offsets truncate cell text')
        cells.append(dict(token_range=[sequence_start + matched[0], sequence_start + matched[-1] + 1],
                          character_range=[start, end], empty=marker.group() == '<ecel>',
                          left_spill_characters=max(0, start - offsets[matched[0]][0]),
                          right_spill_characters=max(0, offsets[matched[-1]][1] - end)))
    if not cells:
        raise ValueError('No origin cells')
    shared = sum(max(0, a['token_range'][1] - b['token_range'][0]) for a, b in zip(cells, cells[1:]))
    return dict(cells=cells, adjacent_shared_tokens=shared,
                schema='all-intersecting-marker-and-content-tokens-v1')
