"""Candidate image-only pre-inference features; no OCR/model/PDF text access.

Whole-page grid geometry follows Transformers 4.57.6 Qwen2-VL smart_resize
(Apache-2.0; https://github.com/huggingface/transformers), checked by source SHA.
It describes hypothetical whole-page preprocessing, not native crop tokens.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

from .inputs import CAP,image_fingerprint,pdf_geometry
from .schema import ACTION_CODES,FEATURES,SourceRecord,choose_action,numeric_features

DEFINITION_ID='v4_input_features_candidate_001'
ACCEPTANCE_DOMAIN_REVISION='v4_input_features_pdf_single_ulp_ceil_002'
PROBE_LONG_SIDE=1024
MIN_GLYPHS=8
MAX_RELATIVE_HEIGHT_MAD=.5
GRID_HASHES=(
    'f2058c716eef96ccaed1cc1e2d0c08306b62586d535b28d9d08e691b2fab7ca0',
    '7820a0fcca107e75605e08d9b774285ca2b0316f857bc0225c779794705ecf4f',
    '09bfa9b17df7c3f0c6159bc34008ee50f21d2472cd5bae7e5c21ba1ca13a423c',
)
_GRID_SEAL=object()


@dataclass(frozen=True)
class PreflightBinding:
    """Created by bind_preflight_files after reading exact local frozen assets."""
    hashes: tuple
    _seal: object


def bind_preflight_files(config_path,resize_source_path,fast_source_path):
    """Read/hash three non-weight assets; never import or execute their modules."""
    raw=[Path(p).read_bytes() for p in (config_path,resize_source_path,fast_source_path)]
    hashes=tuple(hashlib.sha256(x).hexdigest() for x in raw)
    if hashes!=GRID_HASHES:raise ValueError('Unbound preflight configuration/source bytes')
    config=json.loads(raw[0])
    expected=dict(min_pixels=3136,max_pixels=12845056,patch_size=14,temporal_patch_size=2,merge_size=2,processor_class='Qwen2_5_VLProcessor',image_processor_type='Qwen2VLImageProcessor')
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('Preflight processor configuration differs')
    return PreflightBinding(hashes,_GRID_SEAL)


def preflight_grid(width,height,binding):
    """Pure dimension descriptor. Does not materialize tensors or count real tokens."""
    if not isinstance(binding,PreflightBinding) or binding._seal is not _GRID_SEAL or binding.hashes!=GRID_HASHES:
        raise ValueError('Verified preflight binding required')
    if any(type(v) is not int or v<=0 for v in (width,height)) or max(width,height)/min(width,height)>200:
        raise ValueError('Unsupported preflight geometry')
    factor=28;minimum=3136;maximum=12845056
    h=round(height/factor)*factor;w=round(width/factor)*factor
    if h*w>maximum:
        scale=math.sqrt(height*width/maximum)
        h=max(factor,math.floor(height/scale/factor)*factor)
        w=max(factor,math.floor(width/scale/factor)*factor)
    elif h*w<minimum:
        scale=math.sqrt(minimum/(height*width))
        h=math.ceil(height*scale/factor)*factor;w=math.ceil(width*scale/factor)*factor
    if min(h,w)<=0 or h%factor or w%factor:raise ValueError('Invalid frozen grid result')
    return (1,h//14,w//14)


def _source_binding(source):
    return dict(original_file_sha256=source.original_file_sha256,original_page_ordinal=source.original_page_ordinal,oriented_size=list(source.oriented_size))


def extract_features(prepared,source,action,render_audit,*,grid_binding=None):
    """Only a matched native prepared RGB page is accepted; metadata is not a feature."""
    start=time.perf_counter();values={name:None for name in FEATURES};reasons=[]
    audit=dict(definition_id=DEFINITION_ID,acceptance_domain_revision=ACCEPTANCE_DOMAIN_REVISION,pdf_text_access=False,grid_semantics='whole_page_preprocessing_descriptor_not_layout_or_recognition_tokens',recognition_used=False)
    result=dict(definition_id=DEFINITION_ID,action=action,source_type=getattr(source,'source_type',None),ready=False,features=values,vector=None,reasons=reasons,audit=audit)
    def finish():
        if not reasons:
            try:result['vector']=numeric_features(values)
            except (ValueError,TypeError,OverflowError):reasons.append('missing_or_nonfinite_required_feature')
        result['ready']=not reasons
        audit['elapsed_seconds']=time.perf_counter()-start
        return result
    if type(source) is not SourceRecord:
        reasons.append('unsupported_source_record');return finish()
    audit['source_binding']=_source_binding(source)
    values['is_original_pdf']=int(source.source_type=='original_pdf')
    if action not in source.available_actions():
        reasons.append('unavailable_action');return finish()
    values['action_code']=ACTION_CODES[action]
    try:
        from PIL import Image
    except ImportError:
        reasons.append('image_dependency_unavailable');return finish()
    if not isinstance(prepared,Image.Image) or prepared.mode!='RGB':
        reasons.append('unsupported_rgb_page');return finish()
    width,height=prepared.size
    fingerprint=image_fingerprint(prepared)
    expected_affine=[width/source.oriented_size[0],height/source.oriented_size[1]]
    if not isinstance(render_audit,dict) or render_audit.get('action')!=action or render_audit.get('original_source_type')!=source.source_type or render_audit.get('prepared')!=fingerprint:
        reasons.append('unbound_native_render_audit');return finish()
    affine=render_audit.get('page_affine_xy')
    if not isinstance(affine,(list,tuple)) or len(affine)!=2 or any(type(a) not in (int,float) or not math.isfinite(a) or abs(a-b)>1e-9 for a,b in zip(affine,expected_affine)):
        reasons.append('unbound_upright_page_geometry');return finish()
    audit['prepared']=fingerprint
    # Bind the observation before rejecting its dimensions so an unavailable
    # feature row still refers to the pixels actually supplied to recognition.
    rounding=None
    if source.source_type=='original_pdf' and max(width,height)==CAP+1:
        dpi=300 if action=='B' else 200
        expected_size,expected_scale=pdf_geometry(source.oriented_size,dpi)
        continuous_long_side=max(source.oriented_size)*expected_scale
        limit=math.nextafter(float(CAP),math.inf)
        actual_scale=render_audit.get('scale')
        if (prepared.size==expected_size and type(actual_scale) in (int,float)
                and actual_scale==expected_scale and continuous_long_side<=limit):
            rounding=dict(branch='pdf_single_ulp_ceil',dpi=dpi,
                          expected_size=list(expected_size),scale=expected_scale,
                          continuous_long_side=continuous_long_side,limit=limit)
    if min(width,height)<4 or (max(width,height)>CAP and rounding is None):
        reasons.append('unsupported_candidate_dimensions');return finish()
    audit['dimension_acceptance']=rounding or dict(branch='within_cap')
    values.update(width=width,height=height,aspect=width/height)
    try:
        gt,gh,gw=preflight_grid(width,height,grid_binding)
        values.update(preflight_grid_t=gt,preflight_grid_h=gh,preflight_grid_w=gw)
        audit['preflight_binding_hashes']=list(grid_binding.hashes)
    except (ValueError,TypeError,OverflowError):reasons.append('missing_or_unsupported_preflight_grid')
    try:
        import cv2
        import numpy as np
    except ImportError:
        reasons.append('image_statistics_dependency_unavailable');return finish()
    scale=min(1.,PROBE_LONG_SIDE/max(width,height))
    size=(max(1,math.floor(width*scale+.5)),max(1,math.floor(height*scale+.5)))
    gray=prepared.convert('L')
    try:
        if gray.size!=size:
            smaller=gray.resize(size,Image.Resampling.BOX);gray.close();gray=smaller
        raw=np.asarray(gray,dtype=np.uint8)
        if min(raw.shape)<3:
            reasons.append('probe_too_thin');return finish()
        normalized=raw.astype(np.float64)/255.
        lap=(normalized[:-2,1:-1]+normalized[2:,1:-1]+normalized[1:-1,:-2]+normalized[1:-1,2:]-4*normalized[1:-1,1:-1])
        threshold,mask=cv2.threshold(raw,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,connectivity=8)
        heights=[];probe_h,probe_w=raw.shape
        for x,y,w,h,area in stats[1:]:
            if x>0 and y>0 and x+w<probe_w and y+h<probe_h and 2<=h<=probe_h/10 and w<=4*h and area>=2:
                heights.append(int(h))
        median=float(np.median(heights)) if heights else None
        relative_mad=float(np.median(np.abs(np.asarray(heights)-median))/median) if heights else None
        reliable=len(heights)>=MIN_GLYPHS and relative_mad<=MAX_RELATIVE_HEIGHT_MAD
        values.update(blur=float(1/(1+np.var(lap))),contrast=float(np.std(normalized)),whitespace=float(np.mean(raw>=243)),density=float(np.mean(mask!=0)),glyph_reliability=int(reliable))
        if reliable:values['glyph_height']=median*height/probe_h
        else:reasons.append('unreliable_glyph_height')
        audit.update(probe_size=list(size),probe_resample='Pillow BOX; no upsampling',probe_scale_xy=[probe_w/width,probe_h/height],otsu_threshold=float(threshold),component_count=int(count-1),retained_component_count=len(heights),component_height_median=median,relative_height_mad=relative_mad,thresholds=dict(minimum_components=MIN_GLYPHS,maximum_relative_height_mad=MAX_RELATIVE_HEIGHT_MAD),runtime_versions=dict(numpy=np.__version__,opencv=cv2.__version__,pillow=Image.__version__))
    finally:gray.close()
    return finish()


def choose_with_feature_rows(rows,source,fixed_action,gains,margin):
    """Conservative existing F* rule; neither fitting nor gain prediction occurs here."""
    if type(source) is not SourceRecord:raise ValueError('Valid source record required for F* availability')
    available=source.available_actions();valid=isinstance(rows,dict) and set(rows)==set(available)
    if valid:
        for action in available:
            row=rows[action]
            try:
                vector=numeric_features(row['features'])
                valid=(set(row)=={'definition_id','action','source_type','ready','features','vector','reasons','audit'} and row['audit']['source_binding']==_source_binding(source) and row['audit'].get('preflight_binding_hashes')==list(GRID_HASHES) and row['definition_id']==DEFINITION_ID and row['ready'] is True and row['reasons']==[] and row['source_type']==source.source_type and row['action']==action and row['vector']==vector and row['features']['action_code']==ACTION_CODES[action] and row['features']['is_original_pdf']==int(source.source_type=='original_pdf') and row['features']['glyph_reliability']==1)
            except (KeyError,TypeError,ValueError,OverflowError,AttributeError):valid=False
            if not valid:break
    return choose_action(gains,available,fixed_action,margin,features_valid=valid)
