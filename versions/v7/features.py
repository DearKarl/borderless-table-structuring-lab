"""Frozen post-reread table features; no quality labels or identifiers."""
import difflib
import json
import math
import re
import unicodedata
from html.parser import HTMLParser

BASE_NAMES=['log_width','log_height','log_area','aspect_ratio','gray_mean','gray_std',
            'ink_fraction','gradient_mean','horizontal_ink_runs','vertical_ink_runs']
STRUCT_NAMES=['rows','columns','cells','empty_fraction','duplicate_fraction',
              'log_canonical_length','valid','spanned_fraction']
DIFF_NAMES=STRUCT_NAMES[:6]
FEATURE_NAMES=BASE_NAMES+[p+'_'+x for p in ('native','candidate') for x in STRUCT_NAMES]+[
    'difference_'+x for x in DIFF_NAMES]+['canonical_disagreement','common_valid']


def normalize(text):
    return re.sub(r'\s+',' ',unicodedata.normalize('NFC',text)).strip()


class TableParser(HTMLParser):
    """Strict explicit table/row/cell boundaries; decorative tags carry text."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables=0;self.inside=False;self.closed=False;self.rows=[];self.row=None;self.cell=None

    def handle_starttag(self,tag,attrs):
        if tag=='table':
            if self.inside or self.tables:raise ValueError('nested_or_multiple_table')
            self.tables+=1;self.inside=True
        elif tag=='tr':
            if not self.inside or self.row is not None or self.cell is not None:raise ValueError('invalid_row_boundary')
            self.row=[]
        elif tag in ('td','th'):
            if not self.inside or self.row is None or self.cell is not None:raise ValueError('invalid_cell_boundary')
            attrs=dict(attrs);spans=[]
            for name in ('rowspan','colspan'):
                raw=attrs.get(name,'1')
                if not isinstance(raw,str) or not re.fullmatch(r'[0-9]+',raw):raise ValueError('invalid_span')
                n=int(raw)
                if not 1<=n<=1000:raise ValueError('span_limit')
                spans.append(n)
            self.cell={'tag':tag,'text':[],'rowspan':spans[0],'colspan':spans[1]}
        elif tag=='br' and self.cell is not None:self.cell['text'].append(' ')

    def handle_endtag(self,tag):
        if tag in ('td','th'):
            if self.cell is None or self.cell['tag']!=tag or self.row is None:raise ValueError('invalid_cell_close')
            self.cell['text']=normalize(''.join(self.cell['text']));self.row.append(self.cell);self.cell=None
        elif tag=='tr':
            if self.row is None or self.cell is not None:raise ValueError('invalid_row_close')
            self.rows.append(self.row);self.row=None
        elif tag=='table':
            if not self.inside or self.row is not None or self.cell is not None:raise ValueError('invalid_table_close')
            self.inside=False;self.closed=True

    def handle_data(self,data):
        if self.cell is not None:self.cell['text'].append(data)
        elif data.strip():raise ValueError('unassigned_text')


def structure(text):
    """Unknown structures return explicit flags and zero placeholders, never guesses."""
    fallback={k:0. for k in STRUCT_NAMES}
    try:
        if not isinstance(text,str) or len(text)>100000:raise ValueError('text_limit_or_type')
        p=TableParser();p.feed(text);p.close()
        if p.tables!=1 or not p.closed or p.inside or p.cell is not None or p.row is not None or not p.rows:
            raise ValueError('incomplete_table')
        if len(p.rows)>400:raise ValueError('row_limit')
        occupied=set();columns=0;cells=[];pairs=duplicates=0;canonical=[]
        for ri,row in enumerate(p.rows):
            ci=0;canon_row=[]
            for cell in row:
                while (ri,ci) in occupied:ci+=1
                rs,cs=cell['rowspan'],cell['colspan']
                if ri+rs>len(p.rows) or ci+cs>1000:raise ValueError('span_outside_table')
                for r in range(ri,ri+rs):
                    for c in range(ci,ci+cs):
                        if (r,c) in occupied:raise ValueError('overlapping_span')
                        occupied.add((r,c))
                columns=max(columns,ci+cs);ci+=cs;cells.append(cell)
                canon_row.append([rs,cs,cell['text']])
            for left,right in zip(row,row[1:]):
                if left['text'] and right['text']:
                    pairs+=1;duplicates+=left['text']==right['text']
            canonical.append(canon_row)
        if not cells:raise ValueError('no_cells')
        body=json.dumps(canonical,ensure_ascii=False,separators=(',',':'))
        stats={'rows':len(p.rows),'columns':columns,'cells':len(cells),
            'empty_fraction':sum(not c['text'] for c in cells)/len(cells),
            'duplicate_fraction':duplicates/pairs if pairs else 0.,'log_canonical_length':math.log1p(len(body)),
            'valid':1.,'spanned_fraction':sum(c['rowspan']>1 or c['colspan']>1 for c in cells)/len(cells)}
        return {**stats,'canonical':body,'reason':None}
    except (ValueError,TypeError,OverflowError) as exc:
        return {**fallback,'canonical':'','reason':str(exc)}


def extract(native_features,native,candidate,common_valid):
    n,c=structure(native),structure(candidate)
    base=list(native_features[:10])
    base_valid=len(base)==10 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in base)
    if not base_valid:base=[0.]*10
    disagreement=1-difflib.SequenceMatcher(None,n['canonical'],c['canonical'],autojunk=True).ratio() if n['valid'] and c['valid'] else 0.
    values=base+[x[k] for x in (n,c) for k in STRUCT_NAMES]+[c[k]-n[k] for k in DIFF_NAMES]+[disagreement,float(common_valid)]
    assert len(values)==len(FEATURE_NAMES)==34
    return {'values':values,'required_features_known':bool(base_valid and n['valid'] and c['valid']),
            'native_structure':n,'candidate_structure':c,'common_valid':bool(common_valid)}


def simple_rule(features):
    n,c=features['native_structure'],features['candidate_structure']
    return bool(features['common_valid'] and features['required_features_known']
        and n['rows']==c['rows'] and n['columns']==c['columns']
        and c['empty_fraction']<=n['empty_fraction']+1e-12
        and c['duplicate_fraction']<=n['duplicate_fraction']+1e-12)
