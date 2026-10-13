"""GT-free M+P formula byte splicing; no model loading or score-based choices."""
import hashlib
import json
import math
import re

VERSION = 'HYBRID-FORMULA-v0'
IOU_THRESHOLD = .50


def sha(data): return hashlib.sha256(data).hexdigest()
def objsha(value): return sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())


def escaped(text, i):
    n=0;i-=1
    while i>=0 and text[i]=='\\':n+=1;i-=1
    return n%2==1


def unwrap(text, required=False):
    """Return body and exact character span, stripping only one outer wrapper."""
    start=len(text)-len(text.lstrip());end=len(text.rstrip());value=text[start:end]
    for opening,closing in [('$$','$$'),('\\[','\\]'),('\\(','\\)'),('$','$')]:
        if value.startswith(opening) and value.endswith(closing) and len(value)>=len(opening)+len(closing):
            if escaped(value,len(value)-len(closing)):continue
            lo=start+len(opening);hi=end-len(closing)
            while lo<hi and text[lo].isspace():lo+=1
            while hi>lo and text[hi-1].isspace():hi-=1
            body=text[lo:hi]
            # One mathematical wrapper, not two inline equations or nested wrappers.
            if any(c=='$' and not escaped(body,i) for i,c in enumerate(body)):return None
            if any(body.startswith(a) and body.endswith(b) for a,b in [('\\[','\\]'),('\\(','\\)')]):return None
            return {'body':body,'start':lo,'end':hi,'wrapper':[opening,closing]}
    if required:return None
    return {'body':value,'start':start,'end':end,'wrapper':['','']}


def latex_valid(body):
    if not body.strip():return False,'empty_formula'
    if any(c=='$' and not escaped(body,i) for i,c in enumerate(body)):return False,'residual_math_delimiter'
    if any(not escaped(body,m.start()) for m in re.finditer(r'\\[()\[\]]',body)):return False,'residual_math_delimiter'
    if '```' in body or '~~~' in body or re.search(r'<\s*/?\s*[A-Za-z!][^>]*>',body):return False,'non_formula_payload'
    forbidden=r'\\(?:documentclass|usepackage|input|include(?:graphics)?|write(?:18)?|openout|openin|read|catcode|csname|def|edef|gdef|newcommand|directlua)\b'
    if any(not escaped(body,m.start()) for m in re.finditer(forbidden,body)):return False,'non_formula_command'
    balance=0
    for i,c in enumerate(body):
        if c in '{} ' and not escaped(body,i):
            if c=='{':balance+=1
            elif c=='}':balance-=1
            if balance<0:return False,'unbalanced_braces'
    if balance:return False,'unbalanced_braces'
    stack=[]
    pattern=r'\\(begin|end)\s*\{([^{}]+)\}'
    for m in re.finditer(pattern,body):
        if escaped(body,m.start()):continue
        kind,env=m.groups()
        if env in ['document','tabular','tabular*','tabularx','longtable','longtabu','tabu','table','table*']:return False,'non_formula_environment'
        if kind=='begin':stack.append(env)
        elif not stack or stack.pop()!=env:return False,'environment_mismatch'
    if stack:return False,'environment_mismatch'
    if any(not escaped(body,m.start()) and not re.match(pattern,body[m.start():]) for m in re.finditer(r'\\(?:begin|end)\b',body)):return False,'invalid_environment_command'
    return True,None


def tags(body):
    signature=[]
    for match in re.finditer(r'\\(?:tag\*?|label)(?![A-Za-z])\s*\{',body):
        if escaped(body,match.start()):continue
        i=match.end();level=1
        while i<len(body) and level:
            if not escaped(body,i):
                if body[i]=='{':level+=1
                elif body[i]=='}':level-=1
            i+=1
        if level:return None
        signature.append(body[match.start():i])
    return signature


def bbox(value,width,height,normalized=False):
    if not isinstance(value,(list,tuple)) or len(value)!=4:return None
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in value):return None
    x0,y0,x1,y1=value
    if normalized:
        if not (0<=x0<x1<=1 and 0<=y0<y1<=1):return None
        return [x0*width,y0*height,x1*width,y1*height]
    if not (0<=x0<x1<=width and 0<=y0<y1<=height):return None
    return list(value)


def iou(a,b):
    intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union


def math_paragraphs(text):
    """Find standalone math paragraphs, carrying exclusion state across blank lines."""
    lines=text.splitlines(keepends=True);ranges=[];offset=0;fence=None;html_table=False;current=[]
    def flush():
        if current:
            a=current[0][0];b=current[-1][1];raw=text[a:b];parsed=unwrap(raw,required=True)
            blocked=any(x[2] for x in current) or bool(re.search(r'<\s*/?\s*[A-Za-z!]',raw))
            pipe=any(re.match(r'^\s*\|',x[3]) or re.match(r'^\s*\|?\s*:?-{3,}',x[3]) for x in current)
            if parsed:ranges.append({'body':parsed['body'],'start':a+parsed['start'],'end':a+parsed['end'],'wrapper':parsed['wrapper'],'eligible':not(blocked or pipe)})
            current.clear()
    for line in lines:
        end=offset+len(line);stripped=line.lstrip();match=re.match(r'(`{3,}|~{3,})',stripped)
        blocked=fence is not None or html_table or line.startswith('    ') or line.startswith('\t')
        if match:
            token=match.group(1);blocked=True
            if fence is None:fence=(token[0],len(token))
            elif token[0]==fence[0] and len(token)>=fence[1]:fence=None
        if re.search(r'<\s*table\b',line,re.I):html_table=True;blocked=True
        if html_table:blocked=True
        if re.search(r'<\s*/\s*table\s*>',line,re.I):html_table=False
        if not line.strip():flush()
        else:current.append((offset,end,blocked,line))
        offset=end
    flush();return ranges


def raw_math_occurrences(text,body):
    """Count matching wrapped math everywhere, including excluded code/table contexts."""
    matches=[]
    pattern=r'\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|(?<!\$)\$(?!\$)[\s\S]*?(?<!\$)\$(?!\$)'
    for m in re.finditer(pattern,text):
        if escaped(text,m.start()):continue
        value=unwrap(m.group(),required=True)
        if value and value['body']==body:matches.append((m.start()+value['start'],m.start()+value['end']))
    return matches


def splice(raw,replacements):
    chunks=[];cursor=0;outpos=0;unchanged=[]
    for item in sorted(replacements,key=lambda x:x['span'][0]):
        a,b=item['span'];assert cursor<=a<=b<=len(raw)
        part=raw[cursor:a];chunks.append(part)
        unchanged.append({'input_span':[cursor,a],'output_span':[outpos,outpos+len(part)],'sha256':sha(part)})
        outpos+=len(part);chunks.append(item['replacement']);outpos+=len(item['replacement']);cursor=b
    tail=raw[cursor:];chunks.append(tail);unchanged.append({'input_span':[cursor,len(raw)],'output_span':[outpos,outpos+len(tail)],'sha256':sha(tail)})
    output=b''.join(chunks)
    for r in unchanged:
        a,b=r['input_span'];c,d=r['output_span'];assert raw[a:b]==output[c:d]
    return output,unchanged


def assemble(raw,m_structure,p_native,width,height,m_status='success',p_status='success'):
    receipt={'version':VERSION,'input_sha256':{'M_raw':sha(raw),'M_structured':objsha(m_structure),'P_native':objsha(p_native)},'dimensions':[width,height],
        'model_status':{'M':m_status,'P':p_status},'pairs':[],'abstentions':[],'replacements':[],'GT_or_score_used':False,'page_id_used_for_decision':False,
        'M_primary_baseline_sha256':sha(raw if m_status=='success' else b''),'invariant_baseline':'M raw on success; empty primary on M failure/truncation/missing'}
    def finish(output,reason=None,unchanged=None):
        if reason:receipt['page_abstention']=reason
        receipt.update(output_sha256=sha(output),replacement_count=len(receipt['replacements']),unchanged_intervals=unchanged or [],nonreplacement_bytes_verified=True)
        return output,receipt
    if m_status!='success':return finish(b'','M_non_success_primary_empty')
    if p_status!='success' or not isinstance(p_native,dict):return finish(raw,'P_unavailable',splice(raw,[])[1])
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<=0 for x in [width,height]):return finish(raw,'invalid_image_dimensions',splice(raw,[])[1])
    if p_native.get('width')!=width or p_native.get('height')!=height:return finish(raw,'P_dimensions_mismatch',splice(raw,[])[1])
    try:text=raw.decode('utf-8')
    except UnicodeDecodeError:return finish(raw,'M_raw_invalid_UTF8',splice(raw,[])[1])
    pages=m_structure.get('pages',[]) if isinstance(m_structure,dict) else []
    if not isinstance(pages,list) or len(pages)!=1 or not isinstance(pages[0],dict) or not isinstance(pages[0].get('blocks'),list):return finish(raw,'M_structure_missing_or_multipage',splice(raw,[])[1])
    if any(not isinstance(x,dict) for x in pages[0]['blocks']):return finish(raw,'M_block_schema_invalid',splice(raw,[])[1])
    pblocks=p_native.get('parsing_res_list')
    if not isinstance(pblocks,list) or any(not isinstance(x,dict) for x in pblocks):return finish(raw,'P_block_schema_invalid',splice(raw,[])[1])
    ms=[];ps=[]
    for index,block in enumerate(pages[0]['blocks']):
        kind=block.get('type');content=block.get('content');parsed=unwrap(content,required=kind=='text') if isinstance(content,str) and kind in ['equation','text'] else None
        if parsed is None or not parsed['body']:continue
        box=bbox(block.get('bbox'),width,height,True)
        if box is None:receipt['abstentions'].append({'M_index':index,'reason':'invalid_M_bbox'});continue
        ms.append({'index':index,'bbox':box,'body':parsed['body']})
    for index,block in enumerate(pblocks):
        if block.get('block_label')!='display_formula':continue
        box=bbox(block.get('block_bbox'),width,height)
        if box is None:receipt['abstentions'].append({'P_index':index,'reason':'invalid_P_bbox'});continue
        ps.append({'index':index,'bbox':box,'content':block.get('block_content')})
    edges=[(m,p,iou(m['bbox'],p['bbox'])) for m in ms for p in ps if iou(m['bbox'],p['bbox'])>=IOU_THRESHOLD]
    paragraphs=math_paragraphs(text);pending=[]
    for m,p,overlap in edges:
        pair={'M_index':m['index'],'P_index':p['index'],'M_bbox_pixels':m['bbox'],'P_bbox_pixels':p['bbox'],'IoU':overlap,'selected':False}
        receipt['pairs'].append(pair)
        if sum(x[0]['index']==m['index'] for x in edges)!=1 or sum(x[1]['index']==p['index'] for x in edges)!=1:pair['reason']='ambiguous_IoU_degree';continue
        matches=[v for v in paragraphs if v['body']==m['body'] and v['eligible']]
        if len(matches)!=1 or len(raw_math_occurrences(text,m['body']))!=1:pair['reason']='raw_target_not_unique_standalone_math';continue
        target=matches[0]
        value=unwrap(p['content']) if isinstance(p['content'],str) else None
        if not value:pair['reason']='P_content_missing_or_wrapper_invalid';continue
        valid,reason=latex_valid(value['body'])
        if not valid:pair['reason']=reason;continue
        if tags(m['body']) is None or tags(m['body'])!=tags(value['body']):pair['reason']='tag_label_signature_mismatch';continue
        a=len(text[:target['start']].encode());b=len(text[:target['end']].encode());replacement=value['body'].encode()
        pair.update(target_byte_span=[a,b],target_span_sha256=sha(raw[a:b]),replacement_span_sha256=sha(replacement),M_wrapper=target['wrapper'])
        pending.append({'span':[a,b],'replacement':replacement,'pair':pair})
    rejected=set()
    for i,a in enumerate(pending):
        for j,b in enumerate(pending[:i]):
            if max(a['span'][0],b['span'][0])<min(a['span'][1],b['span'][1]):rejected.update([i,j])
    chosen=[]
    for i,item in enumerate(pending):
        pair=item['pair']
        if i in rejected:pair['reason']='overlapping_byte_spans';continue
        pair.update(selected=True,reason='unique_geometry_and_raw_span_valid_formula');chosen.append(item)
        receipt['replacements'].append({k:v for k,v in pair.items() if k!='selected'})
    for m in ms:
        if not any(x[0]['index']==m['index'] for x in edges):receipt['abstentions'].append({'M_index':m['index'],'reason':'no_unique_P_formula_match'})
    output,unchanged=splice(raw,chosen)
    return finish(output,unchanged=unchanged)
