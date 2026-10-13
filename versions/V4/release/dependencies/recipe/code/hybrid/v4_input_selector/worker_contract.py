"""Inference-only contract and atomic records. Imports only the standard library."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid
import time

SCHEMA='v4_tele_worker_v1'
BOUND_SCHEMA='v4_tele_worker_v2'
PAGE_SECONDS=600
CLEANUP_SECONDS=60
WORK_SECONDS=PAGE_SECONDS-CLEANUP_SECONDS
ROOT_KEYS={'schema','job_id','wall_seconds','vendor_files','model_files','aux_files','runtime_versions','expected','items'}
ITEM_KEYS={'item_id','page_id','file','input_sha256','action','source'}
EXPECTED_KEYS={'model_class','processor_class','dtype','model_max_length','processor_merge_size'}
SOURCE_KEYS={'original_file_sha256','source_type','original_page_ordinal','oriented_size','pdf_cropbox','pdf_rotation','pdf_content_type'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def hash_value(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def safe_id(value):
    return isinstance(value,str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,99}',value) is not None


def bound_file(root,name):
    root=Path(root).resolve();rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts or not rel.parts:raise ValueError('Relative owned path required')
    p=root/rel
    if p.is_symlink() or not p.resolve().is_relative_to(root):raise ValueError('Path escapes staged root')
    return p


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    if os.name=='posix':
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    return digest(raw)


def validate_contract(c):
    if not isinstance(c,dict):raise ValueError('Unknown or malformed inference contract')
    schema=c.get('schema');keys=ROOT_KEYS|{'runtime_binding'} if schema==BOUND_SCHEMA else ROOT_KEYS
    if set(c)!=keys or schema not in (SCHEMA,BOUND_SCHEMA) or not safe_id(c['job_id']):raise ValueError('Unknown or malformed inference contract')
    if type(c['wall_seconds']) is not int or not 600<=c['wall_seconds']<=43200:raise ValueError('Unbounded job wall time')
    for name in ('vendor_files','model_files','aux_files'):
        if not isinstance(c[name],dict) or not c[name]:raise ValueError('Missing frozen asset inventory: '+name)
        for rel,h in c[name].items():
            if not isinstance(rel,str) or not hash_value(h) or Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Invalid asset binding')
    required={'TeleOCR/config.py','TeleOCR/src/vlm_analyze.py','TeleOCR/vlm_utils/TeleOCR_model.py','TeleOCR/vlm_utils/TeleOCR_client.py','TeleOCR/vlm_utils/vlm_client/transformers_client.py','TeleOCR/tools/pdf_reader.py','TeleOCR/tools/pdf_image_tools_pdfium.py','TeleOCR/tools/read_file.py','TeleOCR/src/model_output_to_middle_json.py','TeleOCR/vlm_utils/post_process/__init__.py'}
    if not required<=set(c['vendor_files']):raise ValueError('Native call-chain files not frozen')
    if not {'config.json','preprocessor_config.json','tokenizer_config.json','model.safetensors'}<=set(c['model_files']):raise ValueError('Incomplete frozen model inventory')
    if not {'torch','transformers','pypdfium2','Pillow'}<=set(c['runtime_versions']) or any(not isinstance(v,str) or not v for v in c['runtime_versions'].values()):raise ValueError('Missing runtime package freeze')
    e=c['expected']
    if not isinstance(e,dict) or set(e)!=EXPECTED_KEYS or not all(isinstance(e[k],str) and e[k] for k in ('model_class','processor_class','dtype')) or e['model_max_length']!=128000 or e['processor_merge_size']!=2:raise ValueError('Unfrozen loaded identities')
    if not isinstance(c['items'],list) or not 1<=len(c['items'])<=180:raise ValueError('Empty or excessive input queue')
    ids=set();page_actions=set()
    for item in c['items']:
        if not isinstance(item,dict) or set(item)!=ITEM_KEYS or not safe_id(item['item_id']) or not safe_id(item['page_id']) or item['item_id'] in ids:raise ValueError('Invalid/repeated inference item')
        ids.add(item['item_id'])
        if not hash_value(item['input_sha256']) or not isinstance(item['file'],str) or Path(item['file']).is_absolute() or '..' in Path(item['file']).parts:raise ValueError('Invalid input binding')
        s=item['source']
        if not isinstance(s,dict) or set(s)!=SOURCE_KEYS or s['original_file_sha256']!=item['input_sha256']:raise ValueError('Invalid original source binding')
        if s['source_type'] not in ('original_pdf','raster') or type(s['original_page_ordinal']) is not int or s['original_page_ordinal']<0:raise ValueError('Invalid input provenance')
        if not isinstance(s['oriented_size'],list) or len(s['oriented_size'])!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in s['oriented_size']):raise ValueError('Invalid upright geometry')
        if item['action'] not in ('A','B','C') or (s['source_type']=='raster' and (item['action']=='C' or s['original_page_ordinal']!=0)):raise ValueError('Unavailable action')
        page_action=(item['input_sha256'],s['original_page_ordinal'],item['action'])
        if page_action in page_actions:raise ValueError('Repeated original page/action in one job')
        page_actions.add(page_action)
        if s['source_type']=='original_pdf' and (not isinstance(s['pdf_cropbox'],list) or len(s['pdf_cropbox'])!=4 or not all(type(v) in (int,float) and math.isfinite(v) for v in s['pdf_cropbox']) or s['pdf_rotation'] not in (0,90,180,270)):raise ValueError('Missing PDF geometry')
    if schema==BOUND_SCHEMA:
        from .runtime_binding import validate_binding
        validate_binding(c)
    return c


def read_contract(path,expected_sha):
    raw=Path(path).read_bytes()
    if not hash_value(expected_sha) or digest(raw)!=expected_sha:raise ValueError('Reviewed contract hash differs')
    return validate_contract(json.loads(raw))


def verify_assets(c,vendor,model,aux):
    for key,root in [('vendor_files',vendor),('model_files',model),('aux_files',aux)]:
        root=Path(root).resolve()
        observed={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and (key!='vendor_files' or p.suffix=='.py')}
        if observed-set(c[key]):raise ValueError('Unfrozen staged assets: '+key)
        for rel,h in c[key].items():
            if file_sha(bound_file(root,rel))!=h:raise ValueError('Frozen asset mismatch: '+key+'/'+rel)


def send_message(sock,message):
    raw=json.dumps(message,allow_nan=False,separators=(',',':')).encode()
    if len(raw)>1024*1024:raise ValueError('Oversized control message')
    sock.sendall(len(raw).to_bytes(4,'big')+raw)


def recv_message(sock,*,deadline=None):
    def exact(n):
        parts=[]
        while n:
            if deadline is not None:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise TimeoutError("Control receive deadline")
                sock.settimeout(remaining)
            x=sock.recv(n)
            if not x:raise EOFError('Worker control channel closed')
            parts.append(x);n-=len(x)
        return b''.join(parts)
    n=int.from_bytes(exact(4),'big')
    if not 0<n<=1024*1024:raise ValueError('Invalid control frame')
    return json.loads(exact(n))
