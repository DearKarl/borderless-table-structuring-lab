"""Load-once frozen native Tele worker. Heavy imports occur only in load_native()."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import importlib
import importlib.metadata
import io
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

from .worker_contract import (atomic_json,bound_file,digest,file_sha,read_contract,
                              recv_message,send_message,verify_assets)
from .schema import SourceRecord
from .tele_bridge import NativeObserver,page_input_hook
from .native_codec import CANONICAL_FRAME,FINAL_STAGE,final_native_prediction
from .runtime_binding import RuntimeBinding
from .input_features import bind_preflight_files,extract_features


def deadline_check(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Page work/finally deadline exhausted')


def load_native(contract,vendor_root,model_root,aux_root,output_root=None):
    """Actual native singleton factory; never called by local mock/AST checks."""
    binding=RuntimeBinding(contract,vendor_root,model_root,aux_root)
    if output_root is None:raise RuntimeError('Owned output root required for native source cache')
    versions=binding.preflight()
    verify_assets(contract,vendor_root,model_root,aux_root)
    binding.report['staged_verification']['status']='all frozen staged assets freshly hashed'
    for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE'):
        if os.environ.get(key)!='1':raise RuntimeError('Offline staged-assets execution required')
    binding.install(output_root)
    binding.observe_device()
    sys.path.insert(0,str(Path(vendor_root).resolve()))
    config=importlib.import_module('TeleOCR.config')
    # Runtime backend/path overrides only; no vendor/model files or decoding changed.
    config.BACKEND='transformers';config.model_path=str(Path(model_root).resolve())
    if config.LAYOUT_MODE!='Detection' or config.PDF_TOOLS!='pypdfium2' or config.MAX_PIXELS!=64000000:raise RuntimeError('Frozen native configuration differs')
    analyze=importlib.import_module('TeleOCR.src.vlm_analyze')
    client_module=importlib.import_module('TeleOCR.vlm_utils.TeleOCR_client')
    service=importlib.import_module('TeleOCR.vlm_utils.TeleOCR_model').TeleOCRMODEL_SERVICE
    predictor=service.get_model('transformers',config.model_path,None)
    helper= predictor.helper;backend=predictor.client
    observed=dict(model_class=type(backend.model).__name__,processor_class=type(backend.processor).__name__,dtype=str(backend.model.dtype),model_max_length=backend.model_max_length,processor_merge_size=backend.processor.image_processor.merge_size)
    if observed!=contract['expected']:raise RuntimeError('Loaded native model/processor identity differs')
    if predictor.backend!='transformers' or predictor.batching_mode!='stepping' or predictor.executor is not None or max(1,backend.batch_size)!=1:raise RuntimeError('Serial native stepping/batch-one required')
    if helper.abandon_paratext or helper.abandon_list or tuple(helper.layout_image_size)!=(1036,1036):raise RuntimeError('Native content inclusion/layout settings differ')
    read_fn=importlib.import_module('TeleOCR.tools.read_file').read_fn
    pdf_tools=importlib.import_module('TeleOCR.tools.pdf_image_tools')
    writer=importlib.import_module('TeleOCR.data_reader_writer').ImageDataWriter
    pdfium=importlib.import_module('pypdfium2')
    binding.observe_native(backend,pdfium)
    return SimpleNamespace(config=config,analyze=analyze,client_module=client_module,predictor=predictor,
        read_fn=read_fn,convert=pdf_tools.convert_pdf_bytes_to_bytes,ImageWriter=writer,pdfium=pdfium,
        runtime_binding=binding,feature_binding=None,
        identity=dict(loaded=observed,versions=versions,model_object_id=id(backend.model),processor_object_id=id(backend.processor),factory_calls=1,native_effective_batch_size=max(1,backend.batch_size),native_declared_max_concurrency=predictor.max_concurrency,actual_page_concurrency=1,consumption_binding=binding.report))


def source_pdf(native,path,item):
    """Validate original source and use original read/selected-page conversion."""
    raw=path.read_bytes()
    if digest(raw)!=item['input_sha256']:raise ValueError('Original input changed')
    source=SourceRecord(**item['source'])
    raster=None
    if source.source_type=='raster':
        from PIL import Image,ImageOps
        with Image.open(io.BytesIO(raw)) as im:
            raw_size=list(im.size);orientation=im.getexif().get(274,1)
            oriented=ImageOps.exif_transpose(im)
            try:raster=dict(raw_size=raw_size,exif_orientation=orientation,oriented_size=list(oriented.size))
            finally:oriented.close()
        if tuple(raster['oriented_size'])!=tuple(source.oriented_size):raise ValueError('Raster EXIF/upright size mismatch')
    pdf_bytes=native.read_fn(path)
    if file_sha(path)!=item['input_sha256']:raise ValueError('Input changed during native read')
    if source.source_type=='original_pdf' and pdf_bytes!=raw:raise ValueError('Native read changed original PDF bytes')
    doc=native.pdfium.PdfDocument(pdf_bytes)
    try:
        if source.original_page_ordinal>=len(doc):raise ValueError('Selected page outside original source')
        page=doc[source.original_page_ordinal]
        try:geometry=dict(display_size=list(page.get_size()),cropbox=list(page.get_cropbox()),rotation=page.get_rotation())
        finally:page.close()
        if any(abs(a-b)>1e-6 for a,b in zip(geometry['display_size'],source.oriented_size)):raise ValueError('Canonical source size mismatch')
        if source.source_type=='original_pdf' and (geometry['cropbox']!=source.pdf_cropbox or geometry['rotation']!=source.pdf_rotation):raise ValueError('Original cropbox/rotation mismatch')
    finally:doc.close()
    selected=native.convert(pdf_bytes,[source.original_page_ordinal])
    doc=native.pdfium.PdfDocument(selected)
    try:
        if len(doc)!=1:raise ValueError('Native selection did not produce one page')
        page=doc[0]
        try:
            if list(page.get_size())!=geometry['display_size'] or list(page.get_cropbox())!=geometry['cropbox'] or page.get_rotation()!=geometry['rotation']:raise ValueError('Native page selection altered displayed geometry')
        finally:page.close()
    finally:doc.close()
    return source,selected,dict(original_sha256=item['input_sha256'],selected_pdf_sha256=digest(selected),original_page_ordinal=source.original_page_ordinal,original_pdf=geometry,raster=raster,canonical_frame=CANONICAL_FRAME)


def canonical_blocks(blocks,source,render_audit):
    """Explicit inverse of native whole-page resize; crop rotation is not page rotation."""
    prepared=render_audit['prepared']['size'];dims=list(source.oriented_size)
    if len(prepared)!=2 or not all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in prepared):raise ValueError('Missing actual prepared canvas')
    expected=[prepared[i]/dims[i] for i in (0,1)]
    if len(render_audit['page_affine_xy'])!=2 or any(not math.isfinite(a) or abs(a-b)>1e-9 for a,b in zip(render_audit['page_affine_xy'],expected)):raise ValueError('Unexplained rendered-page transform')
    out=deepcopy(blocks)
    # Native _convert_bbox divided both axes by 1000 before crop extraction.
    # x_norm * prepared_width / (prepared_width / upright_width) / upright_width
    # equals x_norm exactly. Preserve original floats/polygons rather than add noise.
    transforms=dict(canonical_size=dims,prepared_size=prepared,canonical_to_prepared_scale_xy=expected,
        layout_size=[1036,1036],prepared_to_layout_scale_xy=[1036/prepared[0],1036/prepared[1]],
        native_final_frame='normalized_prepared_upright_page',normalized_inverse='identity after explicit per-axis resize cancellation',
        intrinsic_pdf_rotation_already_applied=source.source_type=='original_pdf',recognition_crop_angles=[b.get('angle') for b in out],
        crop_angle_changes_page_geometry=False,malformed_polygons='Preserved for accepted native codec; never dropped or repaired')
    return out,transforms


@contextmanager
def final_block_tap(native):
    """Prove successful original post_process and snapshot before native middle mutation."""
    helper=native.predictor.helper;module=native.client_module;analyze=native.analyze
    old_post=helper.post_process;old_raw=module.post_process;old_middle=analyze.result_to_middle_json
    tap=dict(raw_calls=0,raw_successes=0,helper_calls=0,middle_calls=0,blocks=None,raw_error=None)
    final_object=None
    def raw(*args,**kwargs):
        tap['raw_calls']+=1
        try:
            result=old_raw(*args,**kwargs);tap['raw_successes']+=1;return result
        except BaseException as exc:tap['raw_error']=repr(exc);raise
    def post(*args,**kwargs):
        nonlocal final_object
        tap['helper_calls']+=1;result=old_post(*args,**kwargs)
        if tap['helper_calls']!=1 or tap['raw_calls']!=1 or tap['raw_successes']!=1 or tap['raw_error']:raise RuntimeError('Native post_process failed or swallowed failure')
        final_object=result;return result
    def middle(results,images,pdf_doc,writer):
        tap['middle_calls']+=1
        if tap['middle_calls']!=1 or len(results)!=1 or results[0] is not final_object or tap['raw_successes']!=1:raise RuntimeError('Unproven final native/middle boundary')
        tap['blocks']=deepcopy(results[0]);tap['rendered_size']=list(images[0]['img_pil'].size)
        return old_middle(results,images,pdf_doc,writer)
    try:
        module.post_process=raw;helper.post_process=post;analyze.result_to_middle_json=middle
        yield tap
        if tap['middle_calls']!=1 or tap['blocks'] is None:raise RuntimeError('Missing final native block tap')
    finally:
        analyze.result_to_middle_json=old_middle;helper.post_process=old_post;module.post_process=old_raw


class PageWorker:
    def __init__(self,native):
        self.native=native;self.seen=set()

    def run(self,item,input_root,output_root,deadline):
        if item['item_id'] in self.seen:raise RuntimeError('No repeated worker item')
        self.seen.add(item['item_id']);started=time.monotonic();out=bound_file(output_root,item['item_id'])
        out.mkdir(exist_ok=False);audit=dict(item_id=item['item_id'],page_id=item['page_id'],status='RUNNING',page_starts=0,
            source=None,active_stage='runtime_guard',features=dict(status='not_observed',row=None,policy='collect the preregistered action; no selector pruning'),runtime=self.native.identity)
        tap=None;observer=None;render=None
        binding=getattr(self.native,'runtime_binding',None)
        try:
            if binding is not None:binding.assert_clean()
            deadline_check(deadline);before=time.monotonic()
            audit['active_stage']='source_preparation'
            source,selected,geometry=source_pdf(self.native,bound_file(input_root,item['file']),item)
            audit.update(source=geometry,source_preparation_seconds=time.monotonic()-before)
            deadline_check(deadline);writer=self.native.ImageWriter(str(out/'native_images'))
            observer=NativeObserver(merge_size=2,page_id=item['page_id'])
            before=time.monotonic()
            def observe_prepared(prepared,render_audit):
                audit['active_stage']='pre_inference_features'
                deadline_check(deadline)
                if binding is not None and getattr(self.native,'feature_binding',None) is None:
                    self.native.feature_binding=bind_preflight_files(binding.model/'preprocessor_config.json',
                        binding.site/'transformers/models/qwen2_vl/image_processing_qwen2_vl.py',
                        binding.site/'transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py')
                row=extract_features(prepared,source,item['action'],render_audit,grid_binding=getattr(self.native,'feature_binding',None))
                audit['features'].update(status='ready' if row['ready'] else 'unavailable',row=row)
                deadline_check(deadline)
                audit['active_stage']='native_pipeline'
            audit['active_stage']='native_pipeline'
            with page_input_hook(self.native.analyze,source,item['action'],deadline_seconds=max(.001,deadline-time.monotonic()),on_prepared=observe_prepared) as render:
                with observer.observe(self.native.predictor.helper,self.native.predictor.client):
                    with final_block_tap(self.native) as tap:
                        audit['page_starts']=1
                        middle=self.native.analyze.doc_analyze(selected,image_writer=writer,predictor=self.native.predictor)
                        # Original native image writer; never substitutes inference output.
                        writer.save_all_images()
            audit['native_pipeline_seconds']=time.monotonic()-before
            audit['active_stage']='runtime_guard'
            if binding is not None:binding.assert_clean()
            deadline_check(deadline)
            if tap['rendered_size']!=render['prepared']['size']:raise RuntimeError('Final tap and prepared canvas differ')
            audit['active_stage']='output_projection'
            blocks,transforms=canonical_blocks(tap['blocks'],source,render)
            prediction=final_native_prediction(blocks,page_id=item['page_id'],input_sha256=item['input_sha256'],stage=FINAL_STAGE,frame=CANONICAL_FRAME)
            audit['active_stage']='serialization'
            before=time.monotonic();pred_sha=atomic_json(out/'prediction.json',prediction)
            final_sha=atomic_json(out/'final_native_blocks.json',tap['blocks']);middle_sha=atomic_json(out/'native_middle.json',middle)
            image_hashes={str(p.relative_to(out)).replace('\\','/'):file_sha(p) for p in (out/'native_images').rglob('*') if p.is_file()}
            audit.update(status='completed',native_image_files=image_hashes,coordinate_transform=transforms,prediction_sha256=pred_sha,final_native_sha256=final_sha,native_middle_sha256=middle_sha,serialization_seconds=time.monotonic()-before)
        except BaseException as exc:
            audit.update(status='timeout' if isinstance(exc,TimeoutError) else 'failed',failure_stage=audit['active_stage'],error=repr(exc),traceback=traceback.format_exc())
        finally:
            # All reversible contexts have exited before the supervisor can see success.
            if binding is not None:
                try:binding.assert_clean()
                except BaseException as exc:audit.update(status='failed',failure_stage='runtime_guard',error=repr(exc))
            audit.update(elapsed_seconds=time.monotonic()-started,render=render,
                native_stage_diagnostics=dict(requests=list(observer.requests.values()),generate_events=observer.events,visual_tokens=observer.visual_tokens) if observer else None,
                final_tap={k:v for k,v in tap.items() if k!='blocks'} if tap else None)
            if time.monotonic()>=deadline:audit.update(status='timeout',failure_stage=audit.get('failure_stage','serialization'),error='Deadline during page finally/serialization')
            audit_sha=atomic_json(out/'audit.json',audit)
        deadline_check(deadline)
        return dict(item_id=item['item_id'],status=audit['status'],page_starts=audit['page_starts'],audit_file=item['item_id']+'/audit.json',audit_sha256=audit_sha,
            prediction_file=item['item_id']+'/prediction.json' if audit['status']=='completed' else None,prediction_sha256=audit.get('prediction_sha256'))


def main():
    parser=argparse.ArgumentParser()
    for name in ('contract','contract-sha256','vendor-root','model-root','aux-root','input-root','output-root','owner-token'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--socket-fd',type=int,required=True);args=parser.parse_args()
    sock=socket.socket(fileno=args.socket_fd)
    try:
        contract=read_contract(args.contract,args.contract_sha256)
        started=time.monotonic();native=load_native(contract,args.vendor_root,args.model_root,args.aux_root,args.output_root);worker=PageWorker(native)
        send_message(sock,dict(kind='ready',owner_token=args.owner_token,contract_sha256=args.contract_sha256,pid=os.getpid(),pgid=os.getpgrp(),initialization_seconds=time.monotonic()-started,runtime=native.identity))
        items={x['item_id']:x for x in contract['items']}
        while True:
            msg=recv_message(sock)
            if msg=={'kind':'stop'}:return 0
            if set(msg)!={'kind','item_id','deadline'} or msg['kind']!='page' or msg['item_id'] not in items or not isinstance(msg['deadline'],(int,float)) or not math.isfinite(msg['deadline']):raise ValueError('Invalid worker request')
            result=worker.run(items[msg['item_id']],args.input_root,args.output_root,msg['deadline'])
            send_message(sock,result)
            if result['status']!='completed':return 1
    except BaseException as exc:
        try:send_message(sock,dict(kind='fatal',error=repr(exc)))
        except BaseException:pass
        return 1
    finally:sock.close()

if __name__=='__main__':raise SystemExit(main())
