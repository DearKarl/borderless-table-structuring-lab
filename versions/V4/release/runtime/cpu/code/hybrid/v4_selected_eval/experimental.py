"""Explicit experimental policy; never a production eligibility certificate."""
import hashlib
import json
from pathlib import Path
import time
from hybrid.v4_input_selector.scale_policy import MODEL_FILE,canonical
from hybrid.v4_input_selector.selector_input import select_and_run
from hybrid.v4_input_selector.tele_worker import PageWorker
from hybrid.v4_input_selector.worker_contract import atomic_json,bound_file,file_sha,safe_id
from .contracts import (checked_json,verify_code,prepare_frozen_model,MODEL_SHA,
                        ORIGINAL_CONFIG_SHA,ORIGINAL_MANIFEST_SHA)
from .dependencies import need
from .native_export import export_middle

LIMITS=dict(item_starts=1651,gpu_seconds=96*3600,wall_seconds=120*3600,
    cpu_eval_seconds=24*3600,batch_items=64,batch_seconds=12*3600,page_work=540,page_cleanup=60)
MODEL_FILES={MODEL_FILE:MODEL_SHA,'config.json':ORIGINAL_CONFIG_SHA,'manifest.json':ORIGINAL_MANIFEST_SHA}
CALIBRATION_SHA='317a2267187bcdab354b8e877925dee66f270fa8ea8c772f8c684e10de1ae922'

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def validate_auth(auth):
    need(auth.get('schema')=='v4_experiment_auth_v1' and auth.get('experimental_only') is True
        and auth.get('adaptive_eligible') is False and auth.get('calibration_accepted') is False
        and auth.get('training_allowed') is False,'Explicit experimental-only authorization required')
    need(auth.get('user_authorized_full1651_despite_failed_calibration') is True,'Explicit full experimental evaluation authorization required')
    need(auth['model_files']==MODEL_FILES and auth['margin']==0.0 and auth['fixed_action']=='B'
        and auth['limits']==LIMITS and auth['calibration_report_sha256']==CALIBRATION_SHA,'Frozen experimental policy/budget differs')
    need(safe_id(auth['run_id']) and len(auth['item_hashes'])==1651 and len(set(auth['item_hashes']))==1651,'Full unique experimental identity lock required')
    return auth

def validate_batch(projection,auth,auth_sha):
    need(projection.get('schema')=='v4_experimental_batch_v1' and projection['experiment_auth_sha256']==auth_sha
        and projection['run_id']==auth['run_id'] and projection['full_input_sha256']==auth['full_input_sha256'],'Batch authorization differs')
    items=projection['items'];indices=projection['indices']
    need(1<=len(items)<=64 and len(items)==len(indices) and indices==sorted(set(indices)),'Invalid batch indices/count')
    for i,item in zip(indices,items):
        need(type(i) is int and 0<=i<1651 and digest(item)==auth['item_hashes'][i],'Batch is not a member of frozen full input')
        need(item['item_id']==item['page_id'] and item['action']=='policy_selected','One selected item per page required')
    return items

def prepare_experimental(directory,auth_path,auth_sha,*,code_root):
    auth=validate_auth(checked_json(auth_path,auth_sha));verify_code(Path(code_root),auth['code_files'])
    prepared=prepare_frozen_model(directory);config=json.loads(prepared.config_json)
    need(config['margin']==0.0,'Experimental margin must equal the pre-calibration original preset')
    config.update(learned_enabled=True,margin=0.0)
    binding=dict(experimental_only=True,adaptive_eligible=False,calibration_accepted=False,
        experiment_auth_sha256=auth_sha,original_manifest_sha256=ORIGINAL_MANIFEST_SHA,
        original_config_sha256=ORIGINAL_CONFIG_SHA,model_sha256=MODEL_SHA,margin=0.0,
        rounding_evidence_sha256=auth['rounding_evidence_sha256'])
    identity=digest(dict(schema='v4_effective_experimental_policy_v1',binding=binding,config=config))
    prepared.config_json=canonical(config);prepared.identity=identity
    # Deliberately no deployment_binding: this cannot stand in for a passed certificate.
    return prepared,dict(binding,effective_policy_identity=identity),auth

class ExperimentalWorker(PageWorker):
    def __init__(self,native,*,policy,experiment):
        need(experiment.get('experimental_only') is True and experiment.get('adaptive_eligible') is False
            and experiment.get('calibration_accepted') is False and policy.identity==experiment['effective_policy_identity']
            and policy.config['learned_enabled'] is True and policy.config['margin']==0.0
            and not hasattr(policy,'deployment_binding'),'Experimental policy identity required')
        need(native.selector_rounding_evidence_sha256==experiment['rounding_evidence_sha256'],'Experimental rounding differs')
        self.policy,self.experiment=policy,experiment;super().__init__(native)

    def run(self,item,input_root,output_root,deadline):
        need(item['item_id']==item['page_id'] and item['action']=='policy_selected','Single experimental item required')
        started=time.monotonic()
        selected=select_and_run(self,item,input_root,output_root,self.policy,deadline=deadline)
        if selected['result']['status']=='completed':export_middle(Path(output_root)/item['item_id'],item['item_id'])
        if time.monotonic()>=deadline:raise TimeoutError('Experimental export exhausted original page deadline')
        selected_path=bound_file(output_root,'_selected/'+item['item_id']+'.json')
        atomic_json(bound_file(output_root,'_experimental/'+item['item_id']+'.json'),
            dict(schema='v4_experimental_page_v1',item_id=item['item_id'],experiment=self.experiment,
                selected_file=selected_path.relative_to(Path(output_root).resolve()).as_posix(),
                selected_sha256=file_sha(selected_path),result=selected['result'],
                end_to_end_seconds_including_export=time.monotonic()-started,original_deadline=deadline))
        if time.monotonic()>=deadline:raise TimeoutError('Experimental final receipt exceeded original page deadline')
        return selected['result']
