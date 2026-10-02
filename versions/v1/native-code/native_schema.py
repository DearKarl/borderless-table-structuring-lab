"""Structural native-output validation; never judge content correctness."""
import hashlib,json,math,re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote,urlsplit
from PIL import Image
from image_only_audit import SETTINGS

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def asset(folder,name):
 assert isinstance(name,str) and name and not urlsplit(name).scheme and not name.startswith('/') and '\\' not in name
 rel=Path(unquote(name));assert '..' not in rel.parts and not rel.is_absolute()
 p=(folder/rel).resolve();assert p.is_relative_to(folder.resolve()) and p.is_file() and not p.is_symlink() and p.stat().st_size>0
 return {'reference':name,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
class ImageReferences(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):
  if tag=='img':
   src=dict(attrs).get('src');assert src;self.refs.append(src)
 def handle_startendtag(self,tag,attrs):self.handle_starttag(tag,attrs)
def markdown_assets(folder,filename='markdown.md'):
 text=(folder/filename).read_bytes().decode('utf-8');parser=ImageReferences();parser.feed(text);parser.close()
 refs=parser.refs+re.findall(r'!\[[^\]]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)',text)
 return [asset(folder,n) for n in refs]
def validate_schema(folder,arm,page,input_file):
 folder=Path(folder);assert sha(input_file)==page['input_sha256']
 with Image.open(input_file) as im:assert im.size==(page['width'],page['height']) and im.format in ('PNG','JPEG')
 if arm=='mineru':
  middle=read(folder/'middle_json.json');structured=read(folder/'structured_content.json')
  assert isinstance(middle,dict) and middle['schema']=='docvortex.middle' and middle['schema_version']=='2.0'
  assert middle['metadata']['producer']['name']=='mineru' and middle['metadata']['producer']['version']=='4.0.5'
  assert isinstance(structured,dict) and isinstance(structured['metadata'],dict) and isinstance(structured['extensions'],dict)
  assert type(middle['is_full_document']) is bool and type(structured['is_full_document']) is bool
  for document in (middle,structured):
   assert isinstance(document['pages'],list) and len(document['pages'])==1
   p=document['pages'][0];assert type(p['page_idx']) is int and p['page_idx']==0 and isinstance(p['blocks'],list)
  layout=middle['extensions']['docvortex_layout']['pages'];assert len(layout)==1 and layout[0]['page_idx']==0
  assert all(finite(layout[0][k]) and layout[0][k]>0 for k in ('width_pt','height_pt'))
  assets=[]
  def walk(value):
   if isinstance(value,dict):
    if 'type' in value:
     assert isinstance(value['type'],str) and value['type']
     if 'bbox' in value:assert isinstance(value['bbox'],list) and len(value['bbox'])==4 and all(finite(v) for v in value['bbox'])
    for key,child in value.items():
     if key in ('image_source','image_path') and child:assets.append(asset(folder,child))
     else:walk(child)
   elif isinstance(value,list):
    for child in value:walk(child)
  walk(middle['pages']);walk(structured['pages']);assets+=markdown_assets(folder)
  assert (folder/'markdown.md').is_file()
  return {'valid':True,'arm':arm,'page_id':page['page_id'],'input_sha256':page['input_sha256'],'saved_markdown_sha256':sha(folder/'markdown.md'),'schema_artifacts':[{'path':str(folder/n),'sha256':sha(folder/n)} for n in ['middle_json.json','structured_content.json']],'assets':assets,'block_count':len(structured['pages'][0]['blocks']),'native_page_dimensions_pt':{k:layout[0][k] for k in ['width_pt','height_pt']}}
 assert arm=='paddle'
 native=read(folder/'native.json');assert isinstance(native,dict)
 assert native.get('page_index') in (None,0) and native.get('page_count') in (None,1)
 assert native['input_path']==page['image_path'] and type(native['width']) is int and type(native['height']) is int
 assert (native['width'],native['height'])==(page['width'],page['height']) and native['model_settings']==SETTINGS
 assert isinstance(native['parsing_res_list'],list) and isinstance(native['layout_det_res'],dict) and isinstance(native['layout_det_res']['boxes'],list)
 for block in native['parsing_res_list']:
  assert isinstance(block,dict) and isinstance(block['block_label'],str) and isinstance(block['block_content'],str)
  box=block['block_bbox'];assert isinstance(box,list) and len(box)==4 and all(finite(v) for v in box)
 for box in native['layout_det_res']['boxes']:
  assert isinstance(box,dict) and isinstance(box['label'],str) and type(box['cls_id']) is int and finite(box['score'])
  assert isinstance(box['coordinate'],list) and len(box['coordinate'])==4 and all(finite(v) for v in box['coordinate'])
 return {'valid':True,'arm':arm,'page_id':page['page_id'],'input_sha256':page['input_sha256'],'native_sha256':sha(folder/'native.json'),'block_count':len(native['parsing_res_list']),'assets':markdown_assets(folder,'prediction.md')}

def paddle_context(folder,page,input_file,manifest_file,adapter_sha256,runtime_key):
 folder=Path(folder);attempt=folder.parent
 expected=read(Path(__file__).with_name('PADDLE_PARAMETERS_TEMPLATE.json'))
 expected['manifest_sha256']=sha(manifest_file);expected['adapter_sha256']=adapter_sha256
 names=['parameters.json','effective-pipeline.yaml','loaded-model-details.json','model-generation-config.json','loaded.json']
 pins={n:sha(attempt/n) for n in names}
 pins.update({n:sha(attempt/n) for n in ['START.json','EXIT.json','ACTUAL_GPU_DEVICE.json'] if (attempt/n).exists()})
 assets={p.relative_to(folder).as_posix():sha(p) for p in (folder/'imgs').rglob('*') if p.is_file()}
 return {'page':page,'input_file':str(input_file),'manifest_file':str(manifest_file),'expected_parameters':expected,'adapter_sha256':adapter_sha256,'runtime_key':runtime_key,'attempt_pins':pins,'asset_pins':assets}
