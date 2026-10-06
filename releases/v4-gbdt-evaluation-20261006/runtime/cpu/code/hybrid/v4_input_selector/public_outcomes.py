"""Host-only public outcome export; never mounted into recognition or prediction.

References are consumed only for explicit fit/calibration splits by default.
Held export requires the independently frozen model/config claim before opening
any held reference. Final held comparison remains a separate one-time operation.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time

from .bank import BankLedger, FrozenBank, checked_json, require, value_sha
from .input_features import DEFINITION_ID
from .public_score import VERSION, canonicalize, native_transcript, score
from .schema import FEATURES, numeric_features
from .supervisor import validated_result
from .worker_contract import atomic_json, bound_file, digest, file_sha


def reference_for(row):
    ref=row['reference']
    require(ref.get('score_version')==VERSION,'Wrong public score binding')
    value=checked_json({k:ref[k] for k in ('path','sha256')},max_bytes=8*1024**2)
    require(value.get('schema')=='nju_public_reference_v1' and value.get('score_version')==VERSION
            and value.get('page_id')==row['page_id'] and value.get('source_group')==row['source_family']
            and value.get('split')==row['split'] and value.get('input_sha256')==row['input']['sha256'],
            'Public reference identity/split differs')
    require(isinstance(value.get('natural_text'),str),'Missing public transcript')
    return value


def consume(row, item, slot_state, reference, *, deadline):
    out=dict(status='not_run',quality=None,seconds=None,features=None,audit={})
    if slot_state['status'] in ('pending','uncertain') or slot_state.get('result') is None:
        out['audit']['reason']='unstarted_or_uncertain_not_a_zero';return out
    try:
        saved=slot_state['result'];result=saved['payload'];root=Path(saved['output_root'])
        validated_result(item,result,root)
        audit=checked_json(dict(path=str(bound_file(root,result['audit_file'])),sha256=result['audit_sha256']))
        elapsed=audit.get('elapsed_seconds')
        if type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0:out['seconds']=elapsed
        feature=audit.get('features',{}).get('row')
        if isinstance(feature,dict) and feature.get('ready') is True:
            require(feature.get('definition_id')==DEFINITION_ID and feature.get('action')==item['action']
                    and feature.get('audit',{}).get('recognition_used') is False and feature['audit'].get('pdf_text_access') is False,
                    'Features are not frozen pre-recognition observations')
            vector=numeric_features(feature['features'])
            require(vector==feature.get('vector'),'Feature vector/order differs')
            out['features']=vector
        status=result['status']
        if status in ('failed','timeout'):
            require(result.get('page_starts')==1 and audit.get('failure_stage')=='native_pipeline',
                    'Infrastructure/source/feature/serialization fault is not a model quality zero')
            transcript=None
        else:
            path=bound_file(root,item['item_id']+'/final_native_blocks.json')
            require(file_sha(path)==audit.get('final_native_sha256'),'Native block hash differs')
            require(path.stat().st_size<=8*1024**2,'Native block payload resource cap')
            transcript=native_transcript(json.loads(path.read_bytes()))
        measured=score(reference['natural_text'],transcript,status=status,deadline=deadline)
        require(measured.get('reference_symbols')==reference['canonical_symbols'],'Canonical reference count/source version differs')
        out.update(status=measured['status'],quality=measured['quality'])
        diagnostics=audit.get('native_stage_diagnostics') or {}
        out['audit'].update(measured,result_audit_sha256=result['audit_sha256'],
                            result_audit_path=str(bound_file(root,result['audit_file'])),
                            prediction_sha256=result.get('prediction_sha256'),
                            failure_stage=audit.get('failure_stage'),
                            native_stage_summary=dict(visual_tokens=diagnostics.get('visual_tokens'),
                                requests=len(diagnostics.get('requests',[])),generate_events=len(diagnostics.get('generate_events',[]))),
                            render=audit.get('render'),source_preparation_seconds=audit.get('source_preparation_seconds'))
    except Exception as exc:
        out.update(status='invalid_measurement',quality=None)
        out['audit'].update(reason='host_evaluator_or_binding_failure',error=type(exc).__name__+': '+str(exc))
    return out


def export_outcomes(bank, ledger, output, *, splits=('fit','calibration'), model_freeze=None, seconds=1800):
    """One bounded export per new directory. No terminal inference is rerun.

    Root must subtract previous scoring/fit CPU receipts from the 1800-second
    total before passing seconds. A failed or interrupted claim is never reused.
    All processes are serial and CPU worker thread counts must be set to one.
    """
    require(getattr(bank,'public',False),'Public bank profile required')
    require(tuple(splits) in (('fit','calibration'),('fit',),('held_out',)),'Unsupported reference scope')
    require(type(seconds) in (int,float) and math.isfinite(seconds) and 0<seconds<=1800,'Bounded remaining analysis time required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    held=tuple(splits)==('held_out',)
    if held:
        # The model freeze is an integrity receipt emitted after fit/cal; it must
        # bind its actual immutable saved bundle/config and the same roster.
        freeze=checked_json(model_freeze)
        require(freeze.get('schema')=='nju_model_freeze_v1' and freeze.get('fit_cal_complete') is True
                and freeze.get('roster_sha256')==bank.master['roster']['sha256'], 'Held requires frozen completed fit/cal')
        for name in ('model','configuration'):
            descriptor=freeze[name]
            require(set(descriptor)=={'path','sha256'} and file_sha(descriptor['path'])==descriptor['sha256'],'Frozen model/config changed')
        claim=bank.root/'held_score_claim.json'
        with claim.open('x',encoding='utf-8') as f:
            json.dump(dict(model_freeze=model_freeze,output=str(output),opened_once=True),f)
    started=time.monotonic();cpu=time.process_time();deadline=started+seconds
    atomic_json(output/'CLAIM.json',dict(schema='nju_outcome_export_claim_v1',master_sha256=bank.sha,
                splits=list(splits),seconds_limit=seconds,held=held,model_freeze=model_freeze,retry_allowed=False))
    try:
        state=ledger.snapshot();require(state['active'] is None,'Export only after owned allocation release')
        pages=[]
        for opaque,row in bank.host_pages.items():
            if row['split'] not in splits:continue  # Before reference_for/open.
            require(time.monotonic()<deadline,'Analysis deadline exhausted')
            reference=reference_for(row)
            canon=canonicalize(reference['natural_text'],deadline=deadline)
            require(len(canon.symbols)==reference['canonical_symbols'],'Reference normalization binding stale')
            items={it['action']:(slot,it) for slot,it in bank.items.items() if it['page_id']==opaque}
            actions={}
            for action,(slot,it) in items.items():
                actions[action]=consume(row,it,state['slots'][slot],reference,deadline=deadline)
            preparation=[];source_preparation=[]
            for value in actions.values():
                observed=value.get('audit',{});render=observed.get('render') or {}
                costs=[render.get('render_and_transform_seconds'),render.get('input_feature_seconds')]
                original=observed.get('source_preparation_seconds')
                if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in costs+[original]):
                    preparation=None;break
                preparation.append(sum(costs));source_preparation.append(original)
            selection_cost=sum(preparation)+max(source_preparation) if preparation is not None and source_preparation else None
            pages.append(dict(page_id=row['page_id'],source_group=row['source_family'],split=row['split'],
                              reference_symbols=reference['canonical_symbols'],available_actions=sorted(items),actions=actions,
                              selection_preparation_seconds=selection_cost,
                              selection_cost_kind='cached render+feature sum plus max single source preparation; conservative preselection proxy, not a measured deployed selector'))
        result=dict(schema='nju_public_outcomes_v1',roster_sha256=bank.master['roster']['sha256'],score_version=VERSION,
                    feature_definition_id=DEFINITION_ID,feature_names=list(FEATURES),pages=pages,
                    master_sha256=bank.sha,bank_reserved_starts=state['reserved_count'],held_consumed=held,
                    score_source_sha256=file_sha(Path(__file__).with_name('public_score.py')),
                    wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
        acceptance=checked_json(bank.master['roster_acceptance'])
        result['smoke_passed']=state.get('gate') is not None
        result['bank_status']='bank_complete' if all(s['status'] in ('completed','failed','timeout') for s in state['slots'].values()) else 'bank_partial'
        result['bindings']=dict(
            source_sha256=value_sha([dict(page_id=r['page_id'],source_group=r['source_family'],input_sha256=r['input']['sha256']) for r in bank.host_pages.values()]),
            model_sha256=value_sha(bank.template['model_files']),
            processor_sha256=bank.template['model_files']['preprocessor_config.json'],
            actions_sha256=value_sha({n:file_sha(Path(__file__).with_name(n)) for n in ('inputs.py','tele_bridge.py')}),
            score_sha256=result['score_source_sha256'],plan_sha256=bank.master['plan_sha256'],
            addendum_sha256=acceptance['protocol_addendum_sha256'])
        h=atomic_json(output/'OUTCOMES.json',result)
        atomic_json(output/'RECEIPT.json',dict(status='completed',outcomes_sha256=h,pages=len(pages),held_consumed=held,
                    cpu_seconds=time.process_time()-cpu,wall_seconds=time.monotonic()-started))
        return result
    except BaseException as exc:
        atomic_json(output/'FAILURE.json',dict(status='failed',error=repr(exc),cpu_seconds=time.process_time()-cpu,
                    wall_seconds=time.monotonic()-started,claim_preserved=True,retry_allowed=False))
        raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--master',required=True);p.add_argument('--master-sha256',required=True)
    p.add_argument('--output',required=True);p.add_argument('--remaining-seconds',type=float,required=True)
    p.add_argument('--scope',choices=('fit','fit-cal','held'),default='fit-cal')
    p.add_argument('--model-freeze');p.add_argument('--model-freeze-sha256')
    a=p.parse_args();bank=FrozenBank(a.master,a.master_sha256);ledger=BankLedger(bank)
    splits={'fit':('fit',),'fit-cal':('fit','calibration'),'held':('held_out',)}[a.scope]
    freeze=dict(path=a.model_freeze,sha256=a.model_freeze_sha256) if a.model_freeze else None
    value=export_outcomes(bank,ledger,a.output,splits=splits,model_freeze=freeze,seconds=a.remaining_seconds)
    print(json.dumps(dict(pages=len(value['pages']),held_consumed=value['held_consumed'],cpu_seconds=value['cpu_seconds'])))


if __name__=='__main__':main()
