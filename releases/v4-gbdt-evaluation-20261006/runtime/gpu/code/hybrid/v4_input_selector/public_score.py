"""Inert, bounded public silver-transcript agreement; never an official metric.

Characters and structural tuples are distinct symbols. No browser, TeX, network,
native model, GT geometry, or external edit-distance dependency is used.
"""
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import math
import re
import time
import unicodedata

VERSION = 'PublicTranscriptEdit_v1'
MAX_SYMBOLS = 65536
MAX_RAW_CHARS = 1048576
MAX_WORD_WORK = 16000000
MIN_REFERENCE_SYMBOLS = 32
MATH_TEXT = re.compile(r'\\(?:text|textrm|textsf|texttt|mbox|operatorname)\s*\{')
TEX_COMMAND = re.compile(r'\\[A-Za-z]+')


class ScoreResourceLimit(RuntimeError):
    pass


class MalformedMarkup(ValueError):
    pass


@dataclass(frozen=True)
class Canonical:
    symbols: tuple
    issues: tuple = ()

    @property
    def non_whitespace(self):
        return sum(not isinstance(x, str) or not x.isspace() for x in self.symbols)


def _check(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise ScoreResourceLimit('scoring_deadline')


def _literal(text):
    return list(re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip())


def _math(text, deadline=None):
    # TeX spacing around commands is formatting. Spaces inside text-like groups
    # can be semantic and are collapsed, not removed. Commands/signs stay exact.
    out = []
    i = 0
    while i < len(text):
        if i%256==0:_check(deadline)
        m = MATH_TEXT.match(text,i)
        if m:
            start = m.end(); depth = 1; j = start
            while j < len(text) and depth:
                if j%256==0:_check(deadline)
                if text[j] == '\\':
                    j += 2; continue
                if text[j] == '{': depth += 1
                elif text[j] == '}': depth -= 1
                j += 1
            if depth: raise MalformedMarkup('unclosed_math_text')
            command = re.match(r'\\[A-Za-z]+',m.group()).group()
            out.extend(command);out.append(('tex_command_end',))
            out.extend('{'+re.sub(r'\s+', ' ', text[start:j-1])+'}')
            i = j
        else:
            command=TEX_COMMAND.match(text,i)
            if command:
                out.extend(command.group());out.append(('tex_command_end',));i=command.end()
            elif text[i]=='\\' and i+1<len(text):
                out.append(('tex_control',text[i+1]));i+=2
            else:
                if not text[i].isspace(): out.append(text[i])
                i += 1
    return [('math',)] + out + [('/math',)]


def _bracket(text, start, opening, closing):
    depth = 1; i = start + 1
    while i < len(text):
        if text[i] == '\\': i += 2; continue
        if text[i] == opening: depth += 1
        elif text[i] == closing:
            depth -= 1
            if depth == 0: return i
        if depth>64:raise ScoreResourceLimit('link_nesting')
        i += 1
    raise MalformedMarkup('unclosed_link_or_image')


def _inline(text, *, depth=0, deadline=None):
    if depth > 32: raise ScoreResourceLimit('markup_nesting')
    out=[]; i=0
    while i < len(text):
        if i%256==0:_check(deadline)
        image = text.startswith('![', i)
        if image or text[i] == '[':
            start = i+1 if image else i
            end = _bracket(text, start, '[', ']') if ']' in text[start+1:] else None
            if end is not None and text[end+1:end+2] == '(':
                last = _bracket(text, end+1, '(', ')')
                if not image: out.extend(_inline(text[start+1:end], depth=depth+1,deadline=deadline))
                i=last+1; continue
            if image: raise MalformedMarkup('unsupported_image_syntax')
        opener = None
        for left,right in ((r'\[',r'\]'),(r'\(',r'\)'),('$$','$$'),('$','$')):
            if text.startswith(left,i): opener=(left,right); break
        if opener:
            left,right=opener; end=text.find(right,i+len(left))
            # Unpaired currency dollar remains literal; explicit TeX delimiters
            # or $$ indicate malformed markup. No truncation or content drop.
            if end < 0:
                if left!='$': raise MalformedMarkup('unclosed_math')
            else:
                out.extend(_math(text[i+len(left):end],deadline));i=end+len(right);continue
        if text.startswith('\\$',i): out.append('$');i+=2;continue
        # Only balanced simple Markdown emphasis/code wrappers are styling.
        wrapper = next((w for w in ('**','__','`','*','_') if text.startswith(w,i)),None)
        if wrapper:
            end=text.find(wrapper,i+len(wrapper))
            if end>i+len(wrapper) and '\n' not in text[i+len(wrapper):end]:
                out.extend(_inline(text[i+len(wrapper):end],depth=depth+1,deadline=deadline));i=end+len(wrapper);continue
        out.append(text[i]);i+=1
    return out


def _cells(line):
    # Supported Markdown tables: pipe rows with unescaped cell delimiters.
    line=line.strip()
    if line.startswith('|'):line=line[1:]
    if line.endswith('|') and not line.endswith(r'\|'):line=line[:-1]
    return [x.strip().replace(r'\|','|') for x in re.split(r'(?<!\\)\|',line)]


def _markdown(text,deadline=None):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        _check(deadline)
        if i+1<len(lines) and '|' in lines[i]:
            header=_cells(lines[i]);seps=_cells(lines[i+1])
            if len(header)==len(seps) and len(header)>=2 and all(re.fullmatch(r':?-{3,}:?',x) for x in seps):
                rows=[header];i+=2
                while i<len(lines) and '|' in lines[i] and lines[i].strip():
                    cells=_cells(lines[i])
                    if len(cells)!=len(header):raise MalformedMarkup('ragged_markdown_table')
                    rows.append(cells);i+=1
                out.append(('table',))
                for cells in rows:
                    out.append(('row',))
                    for cell in cells:out.extend([('cell',1,1),*_inline(cell,deadline=deadline),('/cell',)])
                    out.append(('/row',))
                out.append(('/table',));continue
        line=re.sub(r'^\s{0,3}#{1,6}\s+','',lines[i])
        out.extend(_inline(line,deadline=deadline));i+=1
        if i<len(lines):out.append(' ')
    return out


class _HTML(HTMLParser):
    VOID={'br','hr','img','wbr','meta','link','input','source','area','base','col','embed','param','track'}
    BLOCK={'div','p','h1','h2','h3','h4','h5','h6','li','ul','ol','pre','blockquote','section','article'}
    STYLE={'b','strong','i','em','span','u','font','code','a','html','body','thead','tbody','tfoot','caption'}

    def __init__(self,deadline=None):
        super().__init__(convert_charrefs=True);self.stack=[];self.out=[];self.buffer=[];self.deadline=deadline

    def flush(self):
        _check(self.deadline)
        if self.buffer:self.out.extend(_markdown(''.join(self.buffer),self.deadline));self.buffer=[]

    def handle_starttag(self,tag,attrs):
        self.flush()
        if len(self.stack)>64:raise ScoreResourceLimit('html_nesting')
        if tag=='img':return  # ALT/src/title are identified image payloads.
        if tag in ('script','style'):raise MalformedMarkup('unsupported_active_markup')
        if tag=='table':self.out.append(('table',))
        elif tag=='tr':
            if 'table' not in self.stack:raise MalformedMarkup('row_outside_table')
            self.out.append(('row',))
        elif tag in ('td','th'):
            if not self.stack or self.stack[-1]!='tr':raise MalformedMarkup('cell_outside_row')
            attrs=dict(attrs)
            try:rs,cs=int(attrs.get('rowspan','1')),int(attrs.get('colspan','1'))
            except (TypeError,ValueError) as e:raise MalformedMarkup('invalid_span') from e
            if not 1<=rs<=65536 or not 1<=cs<=65536:raise MalformedMarkup('invalid_span')
            self.out.append(('cell',rs,cs))
        elif tag in ('sup','sub'):self.out.append((tag,))
        elif tag in self.BLOCK or tag in self.VOID:self.out.append(' ')
        elif tag not in self.STYLE:raise MalformedMarkup('unsupported_html_tag:'+tag)
        if tag not in self.VOID:self.stack.append(tag)

    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
        if tag not in self.VOID:self.handle_endtag(tag)

    def handle_endtag(self,tag):
        self.flush()
        if tag in self.VOID:return
        if not self.stack or self.stack.pop()!=tag:raise MalformedMarkup('unbalanced_html')
        if tag in ('td','th'):self.out.append(('/cell',))
        elif tag=='tr':self.out.append(('/row',))
        elif tag=='table':self.out.append(('/table',))
        elif tag in ('sup','sub'):self.out.append(('/'+tag,))
        elif tag in self.BLOCK:self.out.append(' ')

    def handle_data(self,data):self.buffer.append(data)
    def handle_comment(self,data):pass
    def handle_decl(self,decl):pass
    def unknown_decl(self,data):raise MalformedMarkup('unsupported_declaration')


def _spaces(symbols):
    # Whitespace next to structural boundaries carries no extra symbol.
    out=[];pending=False
    for s in symbols:
        if isinstance(s,str) and s.isspace():pending=True;continue
        if pending and out and isinstance(out[-1],str) and isinstance(s,str):out.append(' ')
        out.append(s);pending=False
    return tuple(out)


def canonicalize(text, *, deadline=None):
    if not isinstance(text,str):raise TypeError('Transcript must be text')
    _check(deadline)
    if len(text)>MAX_RAW_CHARS:raise ScoreResourceLimit('raw_text_limit')
    text=unicodedata.normalize('NFC',text.replace('\r\n','\n').replace('\r','\n'))
    issues=()
    try:
        parser=_HTML(deadline);parser.feed(text);parser.close();parser.flush()
        if parser.stack:raise MalformedMarkup('unclosed_html')
        symbols=_spaces(parser.out)
    except MalformedMarkup as exc:
        # Literal source remains measurable, with an atomic error marker.
        issues=(str(exc),);symbols=(('malformed_markup',),*_literal(text))
    _check(deadline)
    if len(symbols)>MAX_SYMBOLS:raise ScoreResourceLimit('canonical_symbol_limit')
    return Canonical(tuple(symbols),issues)


def edit_distance(a,b, *, deadline=None, max_word_work=MAX_WORD_WORK):
    """Exact Myers bit-vector Levenshtein; bounded bigint work and memory.

    This replaces the legacy quadratic Python character loop, retaining exact
    insertion/deletion/substitution semantics. No approximate/truncated score.
    """
    _check(deadline)
    if len(a)>len(b):a,b=b,a
    m,n=len(a),len(b)
    if not m:return n
    if math.ceil(m/30)*n>max_word_work:raise ScoreResourceLimit('edit_word_work_limit')
    masks={}
    for i,c in enumerate(a):masks[c]=masks.get(c,0)|(1<<i)
    mask=(1<<m)-1;vp=mask;vn=0;distance=m;top=1<<(m-1)
    for i,c in enumerate(b):
        if i%128==0:_check(deadline)
        eq=masks.get(c,0);x=eq|vn
        d0=(((eq&vp)+vp)^vp)|eq|vn
        hp=vn|~(d0|vp);hn=vp&d0
        if hp&top:distance+=1
        elif hn&top:distance-=1
        hp=((hp<<1)|1)&mask;hn=(hn<<1)&mask
        vp=(hn|~(d0|hp))&mask;vn=d0&hp
    _check(deadline)
    return distance


def score(reference, prediction, *, status='completed', deadline=None):
    """Resource/parser implementation faults yield null, never model failure 0."""
    started=time.monotonic()
    base=dict(score_version=VERSION,status='invalid_measurement',quality=None,reference_symbols=None,prediction_symbols=None,reference_issues=[],prediction_issues=[])
    try:
        ref=canonicalize(reference,deadline=deadline)
        base.update(reference_symbols=len(ref.symbols),reference_issues=list(ref.issues))
        if ref.non_whitespace<MIN_REFERENCE_SYMBOLS:
            base['reason']='ineligible_short_reference';return base
        if status in ('not_run','uncertain','interrupted'):
            base.update(status='not_run',reason='no_terminal_model_result');return base
        if status in ('failed','timeout','model_failure'):
            base.update(status='model_failure',quality=0.,reason='terminal_model_failure');return base
        if status!='completed':raise ValueError('Unknown model terminal status')
        pred=canonicalize(prediction,deadline=deadline)
        base.update(prediction_symbols=len(pred.symbols),prediction_issues=list(pred.issues))
        if pred.non_whitespace==0:
            base.update(status='model_failure',quality=0.,reason='empty_terminal_output');return base
        distance=edit_distance(ref.symbols,pred.symbols,deadline=deadline)
        quality=1-distance/max(len(ref.symbols),len(pred.symbols),1)
        if not 0<=quality<=1:raise ArithmeticError('Edit score outside [0,1]')
        base.update(status='completed',quality=quality,distance=distance,reason='public_silver_reference_agreement')
    except ScoreResourceLimit as exc:base['reason']=str(exc)
    except Exception as exc:base.update(reason='scorer_error',error=type(exc).__name__+': '+str(exc))
    finally:base['scoring_seconds']=time.monotonic()-started
    return base


def native_transcript(blocks):
    """Final blocks in native reading order; no category concatenation or GT.

    Image block content is NOT automatically discarded: only explicit ALT syntax
    is removed by the common canonicalizer. Ordinary captions remain visible.
    """
    from .native_codec import CATEGORY_MAP
    if not isinstance(blocks,list):raise ValueError('Expected final native block list')
    parts=[]
    for block in blocks:
        if not isinstance(block,dict) or block.get('type') not in CATEGORY_MAP:raise ValueError('Unknown final native block')
        text=block.get('content')
        if text is None:text=''
        if not isinstance(text,str):raise ValueError('Native content must be text or null')
        if block['type']=='equation' and text.strip():
            stripped=text.strip()
            if not any(stripped.startswith(x) for x in ('$','\\[','\\(')):text=r'\['+text+r'\]'
        parts.append(text)
    return '\n'.join(parts)
