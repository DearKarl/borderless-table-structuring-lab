"""Shared direct actors: one prediction and at most one reread per region/family."""
import hashlib
import math
import time

from .direct_controller import DirectScaleModel
from .features import extract,scaled
from .native import install_input_audit,pixel_identity,realized_view
from .regions import runaway


def text_identity(value):
    text=str(value or '')
    canonical=text.replace('\r\n','\n').replace('\r','\n').strip()
    return {'raw_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'canonical_sha256':hashlib.sha256(canonical.encode()).hexdigest(),
            'canonicalization':'Normalize line endings and strip outer whitespace only'}


class InputObserver:
    def __init__(self,client,emit):
        self.client=client;self.emit=emit;self.request_ids=[];self.submitted_request_ids=[];self.recent=[];self.cache={}
        self.engine=client.client.vllm_llm.llm_engine
        real=self.engine.add_request
        def add(request_id,*args,**kwargs):
            self.request_ids.append(request_id)
            result=real(request_id,*args,**kwargs)
            self.submitted_request_ids.append(request_id)
            self.emit({'event':'model_request_submitted','request_id':request_id})
            return result
        self.engine.add_request=add
        def observed(row):
            if row.get('event')=='model_input':
                for feature in row['features']:
                    if feature['fields']:self.cache[feature['identifier']]=feature['fields']
                    self.recent.append(self.cache.get(feature['identifier']))
            self.emit(row)
        install_input_audit(client,observed)

    def begin_page(self):
        self.request_ids.clear();self.submitted_request_ids.clear();self.recent.clear()

    def reread(self,view,kind,preview):
        start=time.monotonic();before=len(self.submitted_request_ids);attempts=len(self.request_ids);self.recent.clear()
        values=self.client.batch_content_extract([view],[kind])
        count=len(self.submitted_request_ids)-before
        if len(values)!=1 or count!=1:raise RuntimeError('One actor action must issue exactly one model request')
        if len(self.recent)!=1 or not self.recent[0]:raise RuntimeError('Actual regional tensor identity unavailable')
        for key in ('pixel_values','image_grid_thw'):
            if self.recent[0][key]['sha256']!=preview['tensors'][key]['sha256']:
                raise RuntimeError('Preview and actual regional tensors differ')
        return {'output':values[0],'physical_requests':count,'ocr_seconds':time.monotonic()-start,
                'actual_tensor_identity_verified':True,'request_ids':self.submitted_request_ids[before:],
                'request_attempts':len(self.request_ids)-attempts}

    def cancel(self):
        if self.request_ids:self.engine.abort_request(list(self.request_ids))
        if self.engine.has_unfinished_requests():raise RuntimeError('Requests remain after page cancellation')


class DirectActors:
    def __init__(self,client,observer,bindings):
        self.client=client;self.observer=observer
        self.models={key:DirectScaleModel(value['path'],value['sha256']) for key,value in bindings.items()}
        if set(self.models)!={'gbdt_table','gbdt_equation','mlp_table','mlp_equation'}:
            raise ValueError('Four matched family/task checkpoints are required')
        for key,model in self.models.items():
            if key!=model.model['family']+'_'+model.model['kind']:
                raise ValueError('Controller routed to the wrong family or task')
        self.bank={}

    def begin_page(self):self.bank={}

    def prepare(self,crop,native,kind):
        start=time.monotonic();features=extract(crop,native,kind);feature_seconds=time.monotonic()-start
        start=time.monotonic();reference=realized_view(self.client,crop,kind)
        return {'features':features,'feature_seconds':feature_seconds,'reference':reference,
                'reference_preprocess_seconds':time.monotonic()-start,'native_crop':pixel_identity(crop)}

    def apply(self,crop,native,kind,prepared,family,region_id):
        start=time.monotonic();model=self.models[family+'_'+kind]
        calls_before=model.calls;rows_before=model.rows_predicted
        decision=model.choose(prepared['features'],crop.width,crop.height)
        row={'family':family,'kind':kind,'region_id':region_id,'decision':decision,
             'controller_seconds':time.monotonic()-start,'predict_api_calls':model.calls-calls_before,
             'predicted_rows':model.rows_predicted-rows_before,'predict_batches':model.calls-calls_before,
             'checkpoint_sha256':model.sha256,'reference_1x':prepared['reference'],
             'reference_scope':'Matched 1x regional preprocessing; not asserted identical to native page extraction',
             'before':text_identity(native),'accepted':False,'output':native,'physical_requests':0,
             'logical_ocr_requested':False,'reused':False,'render_dpi':200,'render_dpi_changed':False}
        self.emit({'event':'controller_evaluated','family':family,'kind':kind,'region_id':region_id,
                   'decision':decision,'checkpoint_sha256':model.sha256,'controller_seconds':row['controller_seconds'],
                   'predict_api_calls':row['predict_api_calls'],'predict_batches':row['predict_batches'],'predicted_rows':row['predicted_rows']})
        return self.execute(crop,native,kind,prepared,family,region_id,row)

    def emit(self,row):
        if hasattr(self.observer,'emit'):self.observer.emit(row)

    def apply_fixed(self,crop,native,kind,prepared,scale,region_id):
        if not math.isfinite(scale) or not .5<=scale<=3:raise ValueError('Frozen fixed control is outside the common interval')
        size=[max(1,round(crop.width*scale)),max(1,round(crop.height*scale))]
        valid=all(math.isfinite(x) for x in prepared['features'])
        feasible=size[0]*size[1]<=64000000
        decision={'action':'reread' if valid and feasible else 'no_change','scale':scale,
                  'reason':'validation_fixed_scale' if valid and feasible else ('invalid_features' if not valid else 'pixel_limit'),
                  'requested_dimensions':size,'controller_calls':0,'runtime_search':False}
        row={'family':'fixed','kind':kind,'region_id':region_id,'decision':decision,'scale':scale,
             'controller_seconds':0.,'predict_api_calls':0,'predicted_rows':0,'predict_batches':0,
             'reference_1x':prepared['reference'],'reference_scope':'Matched 1x regional preprocessing',
             'before':text_identity(native),'accepted':False,'output':native,'physical_requests':0,
             'logical_ocr_requested':False,'reused':False,'render_dpi':200,'render_dpi_changed':False}
        return self.execute(crop,native,kind,prepared,'fixed',region_id,row)

    def execute(self,crop,native,kind,prepared,family,region_id,row):
        decision=row['decision']
        if decision['action']!='reread':
            row.update(rejection=decision['reason'],canonical_output_changed=False,input_changed_vs_1x=None)
            return row
        start=time.monotonic();view=scaled(crop,decision['scale']);row['resize_seconds']=time.monotonic()-start
        row['nominal_scale_changed']=abs(decision['scale']-1)>1e-6
        row['requested_dimensions_changed']=view.size!=crop.size
        start=time.monotonic();preview=realized_view(self.client,view,kind);row['preview_seconds']=time.monotonic()-start
        row.update(realized=preview,input_changed_vs_1x=preview['identity']!=prepared['reference']['identity'],
                   input_comparison_verified=True,logical_ocr_requested=True)
        self.emit({'event':'actor_action_prepared','family':family,'kind':kind,'region_id':region_id,
                   **{k:row[k] for k in ('decision','nominal_scale_changed','requested_dimensions_changed','input_changed_vs_1x','realized','resize_seconds','preview_seconds')}})
        if preview['identity'] in self.bank:
            cached=self.bank[preview['identity']]
            row.update(output=cached['output'],physical_requests=0,ocr_seconds=0.,reused=True,
                       reuse_from=cached['owner'],actual_tensor_identity_verified=cached['actual_tensor_identity_verified'])
        else:
            attempts=len(getattr(self.observer,'request_ids',[]));submitted=len(getattr(self.observer,'submitted_request_ids',[]))
            self.emit({'event':'regional_ocr_requested','family':family,'kind':kind,'region_id':region_id})
            try:measured=self.observer.reread(view,kind,preview)
            except BaseException as exc:
                self.emit({'event':'regional_ocr_failed','family':family,'kind':kind,'region_id':region_id,
                           'error_type':type(exc).__name__,'attempted_requests':len(getattr(self.observer,'request_ids',[]))-attempts,
                           'submitted_requests':len(getattr(self.observer,'submitted_request_ids',[]))-submitted})
                raise
            self.bank[preview['identity']]={**measured,'owner':{'family':family,'region_id':region_id}}
            row.update(measured)
        self.emit({'event':'regional_ocr_completed','family':family,'kind':kind,'region_id':region_id,
                   'physical_requests':row['physical_requests'],'reused':row['reused'],'actual_tensor_identity_verified':row['actual_tensor_identity_verified']})
        rejection=runaway(row['output'],kind)
        row.update(rejection=rejection,accepted=rejection is None,after=text_identity(row['output']))
        row['canonical_output_changed']=row['accepted'] and row['after']['canonical_sha256']!=row['before']['canonical_sha256']
        row['raw_output_changed']=row['accepted'] and row['after']['raw_sha256']!=row['before']['raw_sha256']
        return row
