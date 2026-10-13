"""GT-free generic native image-only predicate, ported from accepted static audit."""
import hashlib,json,math,re
from html.parser import HTMLParser
from pathlib import Path
from page_paths import component_name
from PIL import Image
SETTINGS={'use_doc_preprocessor':False,'use_layout_detection':True,'use_chart_recognition':False,
'use_seal_recognition':False,'use_ocr_for_image_block':False,'format_block_content':False,
'merge_layout_blocks':True,'markdown_ignore_labels':['number','footnote','header','header_image','footer','footer_image','aside_text'],
'return_layout_polygon_points':True}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def bbox(value,width,height):
    assert isinstance(value,list) and len(value)==4 and all(finite(x) for x in value)
    x0,y0,x1,y1=value
    assert 0<=x0<x1<=width and 0<=y0<y1<=height


class ImageOnlyHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images=[]
    def handle_starttag(self,tag,attrs):
        assert tag in ('div','img'), 'Non-image Markdown content'
        if tag=='img':
            pairs=dict(attrs)
            assert pairs.get('src')
            self.images.append(pairs['src'])
    def handle_startendtag(self,tag,attrs):
        self.handle_starttag(tag,attrs)
    def handle_endtag(self,tag):
        assert tag in ('div','img')
    def handle_data(self,data):
        assert not data.strip(), 'Nonempty text in image-only output'
    def handle_decl(self,decl):
        raise AssertionError("Unexpected declaration")
    def handle_pi(self,data):
        raise AssertionError("Unexpected processing instruction")
    def handle_comment(self,data):
        raise AssertionError('Unexpected comment')


def validate_zero(folder,receipt,context):
    folder=Path(folder).resolve();attempt=folder.parent;page=context['page'];page_id=page['page_id']
    assert folder.name==component_name(page_id) and canonical(read(folder/'receipt.json'))==canonical(receipt)
    assert receipt['page_id']==page_id and receipt['status'] in ('returned','success')
    assert type(receipt['generation_calls']) is int and receipt['generation_calls']==0
    assert read(folder/'generation-calls.json')==[] and not receipt.get('error') and not receipt.get('arm_paused')
    params=read(attempt/'parameters.json')
    assert canonical(params)==canonical(context['expected_parameters'])
    assert params['revision']=='c5630abae1d940eafe0697512a0325494b02ab42'
    assert params['adapter_sha256']==context['adapter_sha256']
    assert receipt['params_sha256']==sha(attempt/'parameters.json')
    manifest=Path(context['manifest_file']);assert params['manifest_sha256']==sha(manifest)
    assert [p for p in read(manifest)['pages'] if p['page_id']==page_id]==[page]
    actual=Path(context['input_file']);assert receipt['input_sha256']==page['input_sha256']==sha(actual)
    assert receipt['prediction_sha256']==sha(folder/'prediction.md')
    required={'parameters.json','effective-pipeline.yaml','loaded-model-details.json','model-generation-config.json','loaded.json'}
    assert required<=set(context['attempt_pins']) and context['runtime_key']
    for name,digest in context['attempt_pins'].items():
        assert Path(name).name==name and re.fullmatch('[0-9a-f]{64}',digest) and sha(attempt/name)==digest
    loaded=read(attempt/'loaded.json');assert finite(loaded['seconds']) and loaded['seconds']>=0
    assert read(attempt/'loaded-model-details.json')['dtype']=='bfloat16'
    generation=read(attempt/'model-generation-config.json');assert generation['eos_token_id']==2 and generation['pad_token_id']==0
    native=read(folder/'native.json');assert native['input_path']==page['image_path']
    assert type(native['width']) is int and type(native['height']) is int
    assert native['width']==page['width'] and native['height']==page['height']
    with Image.open(actual) as image:assert image.size==(page['width'],page['height']) and image.format in ('PNG','JPEG')
    assert canonical(native['model_settings'])==canonical(SETTINGS), 'Unknown or changed native settings'
    blocks=native['parsing_res_list']; boxes=native['layout_det_res']['boxes']
    assert isinstance(blocks,list) and isinstance(boxes,list) and blocks and boxes and len(blocks)==len(boxes)
    assert all(b['label']=='image' for b in boxes)
    for b in boxes:
        bbox(b['coordinate'],page['width'],page['height'])
        assert b['cls_id']==14 and finite(b['score']) and 0<=b['score']<=1 and b.get('order') is None
    used=set()
    for index,block in enumerate(blocks):
        assert block['block_label']=='image' and block['block_content']==''
        assert type(block['block_id']) is int and block['block_id']==index
        assert block.get('block_order') is None and type(block['group_id']) is int and block['group_id']>=0
        bbox(block['block_bbox'],page['width'],page['height'])
        matches=[i for i,b in enumerate(boxes) if b['coordinate']==block['block_bbox']]
        assert len(matches)==1 and matches[0] not in used
        used.add(matches[0])
        polygon=block['block_polygon_points']
        assert isinstance(polygon,list) and len(polygon)>=3 and all(isinstance(x,list) and len(x)==2 and all(finite(v) for v in x) for x in polygon)
        assert polygon==boxes[matches[0]]['polygon_points']
    prediction=(folder/'prediction.md').read_text(encoding='utf-8')
    assert receipt['nonempty'] is True and receipt['characters']==len(prediction)
    parser=ImageOnlyHTML(); parser.feed(prediction); parser.close()
    assert len(parser.images)==len(blocks)
    assert len(set(parser.images))==len(parser.images)
    images=[]
    for name in parser.images:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and ':' not in name and '\\' not in name
        image=(folder/name).resolve()
        assert image.is_relative_to(folder/'imgs') and image.is_file() and image.stat().st_size>0
        relative=image.relative_to(folder).as_posix()
        assert context['asset_pins'][relative]==sha(image)
        images.append({'path':str(image),'sha256':sha(image)})
    return {'kind':'native_image_only_zero_generation','reason':'no_generation_required',
            'page_id':page_id,'runtime_key':context['runtime_key'],'GT_or_quality_used':False,
            'native_omission_not_claimed_correct':True,'native_sha256':sha(folder/'native.json'),
            'receipt_sha256':sha(folder/'receipt.json'),'prediction_sha256':sha(folder/'prediction.md'),
            'input_sha256':page['input_sha256'],'parameters_sha256':sha(attempt/'parameters.json'),
            'generation_calls_sha256':sha(folder/'generation-calls.json'),'image_artifacts':images,
            'native_block_count':len(blocks),'attempt_pins':context['attempt_pins']}


"""Appended to image_only_audit after its unchanged common validation prefix."""


def finish_empty_validation(folder, receipt, context, native, blocks, boxes):
    # Called only after all existing validate_zero identity/config/input checks.
    attempt = folder.parent
    page = context['page']
    assert blocks == [] and boxes == []
    assert receipt['nonempty'] is False and type(receipt['characters']) is int and receipt['characters'] == 0
    assert (folder / 'prediction.md').read_bytes() == b'' and context['asset_pins'] == {}
    event = attempt / 'events' / component_name(page['page_id'], '.started.json')
    assert read(event)['page_id'] == page['page_id']
    start, ex = read(attempt / 'START.json'), read(attempt / 'EXIT.json')
    assigned = read(Path(context['manifest_file']))
    assert start['runtime_key'] == assigned['runtime_key'] == context['runtime_key']
    assert start['cid'] == ex['cid'] and ex['timeout'] is None
    assert ex['state']['ExitCode'] == 0 and ex['state']['OOMKilled'] is False and ex['state']['Running'] is False
    assert ex['all_owned_container_processes_stopped'] is True
    paths = [folder / n for n in ('native.json', 'receipt.json', 'prediction.md', 'generation-calls.json')]
    paths += [event, Path(context['manifest_file'])]
    paths += [attempt / n for n in context['attempt_pins']]
    return {'policy': 'HYBRID-NATIVE-EMPTY-v1', 'kind': 'verified_native_empty_return',
            'page_id': page['page_id'], 'input_sha256': page['input_sha256'], 'runtime_key': context['runtime_key'],
            'status_remains': 'failed', 'quality_correctness_claimed': False, 'GT_used': False,
            'generation_stop_audit_complete': False,
            'artifact_pins': {str(p): sha(p) for p in paths}}


def validate_empty(folder,receipt,context):
    folder=Path(folder).resolve();attempt=folder.parent;page=context['page'];page_id=page['page_id']
    assert folder.name==component_name(page_id) and canonical(read(folder/'receipt.json'))==canonical(receipt)
    assert receipt['page_id']==page_id and receipt['status'] in ('returned','success')
    assert type(receipt['generation_calls']) is int and receipt['generation_calls']==0
    assert read(folder/'generation-calls.json')==[] and not receipt.get('error') and not receipt.get('arm_paused')
    params=read(attempt/'parameters.json')
    assert canonical(params)==canonical(context['expected_parameters'])
    assert params['revision']=='c5630abae1d940eafe0697512a0325494b02ab42'
    assert params['adapter_sha256']==context['adapter_sha256']
    assert receipt['params_sha256']==sha(attempt/'parameters.json')
    manifest=Path(context['manifest_file']);assert params['manifest_sha256']==sha(manifest)
    assert [p for p in read(manifest)['pages'] if p['page_id']==page_id]==[page]
    actual=Path(context['input_file']);assert receipt['input_sha256']==page['input_sha256']==sha(actual)
    assert receipt['prediction_sha256']==sha(folder/'prediction.md')
    required={'parameters.json','effective-pipeline.yaml','loaded-model-details.json','model-generation-config.json','loaded.json'}
    assert required<=set(context['attempt_pins']) and context['runtime_key']
    for name,digest in context['attempt_pins'].items():
        assert Path(name).name==name and re.fullmatch('[0-9a-f]{64}',digest) and sha(attempt/name)==digest
    loaded=read(attempt/'loaded.json');assert finite(loaded['seconds']) and loaded['seconds']>=0
    assert read(attempt/'loaded-model-details.json')['dtype']=='bfloat16'
    generation=read(attempt/'model-generation-config.json');assert generation['eos_token_id']==2 and generation['pad_token_id']==0
    native=read(folder/'native.json');assert native['input_path']==page['image_path']
    assert type(native['width']) is int and type(native['height']) is int
    assert native['width']==page['width'] and native['height']==page['height']
    with Image.open(actual) as image:assert image.size==(page['width'],page['height']) and image.format in ('PNG','JPEG')
    assert canonical(native['model_settings'])==canonical(SETTINGS), 'Unknown or changed native settings'
    blocks=native['parsing_res_list']; boxes=native['layout_det_res']['boxes']
    return finish_empty_validation(folder,receipt,context,native,blocks,boxes)
