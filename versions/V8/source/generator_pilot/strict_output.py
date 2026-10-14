"""Reject malformed OTSL before the frozen decoder can pad or repair its grid."""
import re
from .targets import Parser,TOKENS


def html_grid(value):
    parser=Parser();parser.feed(value);parser.close()
    if parser.opened or not parser.closed or parser.cell is not None or parser.row is not None or not parser.rows:
        raise ValueError('incomplete_table')
    occupied={};cells=[]
    for r,row in enumerate(parser.rows):
        c=0
        for cell in row:
            while (r,c) in occupied:c+=1
            if r+cell['rowspan']>len(parser.rows):raise ValueError('rowspan_outside_table')
            if c+cell['colspan']>400:raise ValueError('column_limit')
            if re.search(r'<\|[^>]*|<(?:[felux]cel|nl|box|loc)[^>]*>',cell['text']):raise ValueError('structural_token_leak')
            cells.append(dict(row=r,col=c,**cell))
            for y in range(r,r+cell['rowspan']):
                for x in range(c,c+cell['colspan']):
                    if (y,x) in occupied:raise ValueError('overlapping_span')
                    occupied[y,x]=len(cells)-1
            c+=cell['colspan']
    if not occupied:raise ValueError('empty_table')
    cols=max(x for y,x in occupied)+1
    if len(occupied)!=len(parser.rows)*cols:raise ValueError('ragged_grid')
    return dict(rows=len(parser.rows),cols=cols,cells=cells)


def validate_otsl(raw):
    raw=raw.strip()
    if not raw:raise ValueError('empty_output')
    if raw.startswith('<table'):
        return html_grid(raw)
    parts=re.split('('+ '|'.join(re.escape(t) for t in TOKENS)+')',raw)
    if parts[0].strip():raise ValueError('text_before_grid')
    rows=[];row=[]
    for i in range(1,len(parts),2):
        token,text=parts[i],parts[i+1]
        if token=='<nl>':
            if text.strip() or not row:raise ValueError('invalid_row_boundary')
            rows.append(row);row=[];continue
        if token!='<fcel>' and text.strip():raise ValueError('text_attached_to_non_content_token')
        if re.search(r'<[A-Za-z_|][^>]*>',text):raise ValueError('unknown_or_leaked_markup')
        row.append((token,text.strip()))
    if row or not rows:raise ValueError('incomplete_final_row')
    width=len(rows[0])
    if any(len(row)!=width for row in rows):raise ValueError('ragged_otsl_grid')
    occupied={};origins={}
    for r,row in enumerate(rows):
        for c,(token,text) in enumerate(row):
            if token in ('<fcel>','<ecel>'):
                owner=(r,c);origins[owner]=text
            elif token=='<lcel>':
                owner=occupied.get((r,c-1))
                if owner is None or owner[0]!=r:raise ValueError('invalid_horizontal_span')
            elif token=='<ucel>':
                owner=occupied.get((r-1,c))
                if owner is None or owner[1]!=c:raise ValueError('invalid_vertical_span')
            elif token=='<xcel>':
                owner=occupied.get((r-1,c-1))
                if owner is None or owner!=occupied.get((r-1,c)) or owner!=occupied.get((r,c-1)):raise ValueError('invalid_cross_span')
            occupied[r,c]=owner
    cells=[]
    for (r,c),text in origins.items():
        owned=[pos for pos,owner in occupied.items() if owner==(r,c)]
        rs=max(y for y,x in owned)-r+1;cs=max(x for y,x in owned)-c+1
        if set(owned)!={(y,x) for y in range(r,r+rs) for x in range(c,c+cs)}:raise ValueError('nonrectangular_span')
        cells.append(dict(row=r,col=c,rowspan=rs,colspan=cs,text=text))
    return dict(rows=len(rows),cols=width,cells=cells)


def validate_prediction(record):
    if not record or record.get('decode_error'):raise ValueError('missing_or_failed_decode')
    raw=validate_otsl(record['raw_otsl'])
    decoded=html_grid(record['decoded_html'])
    if raw!=decoded:raise ValueError('decoder_changed_grid_or_content')
    return decoded
