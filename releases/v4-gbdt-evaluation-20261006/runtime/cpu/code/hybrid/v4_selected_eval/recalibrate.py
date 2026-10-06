"""Host-only frozen-model calibration. Never fits/retrains; never reads held.

This module is not imported by the production worker. Only hash-bound native
receipts from the preregistered sixteen raster items enter the gate.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from hybrid.v4_input_selector.scale_policy import MODEL_FILE, ScalePolicy, canonical
from hybrid.v4_input_selector.worker_contract import bound_file, file_sha
from .contracts import (MARGINS, MODEL_SHA, ORIGINAL_MANIFEST_SHA, checked_json,
                        frozen_model, verify_code)
from .prepare import write_new


def gate(pages):
    """Same formal public_fit gate and tie-break, with measured preparation.

    Tolerance remains group mean 2/reference_symbols on the NEW paired pages;
    the old 0.003850609624790562 is historical, not a retuned absolute threshold.
    """
    if len(pages)!=8 or len({p['source_group'] for p in pages})!=8:
        raise ValueError('Exactly eight distinct frozen groups required')
    paired=[p for p in pages if all(r['status'] in ('completed','model_failure') for r in p['actions'].values())]
    reports=[]
    for index,margin in enumerate(MARGINS):
        choices=[p['decisions'][index]['action'] for p in paired]
        baseline=[p['actions']['B'] for p in paired]
        chosen=[p['actions'][a] for p,a in zip(paired,choices)]
        gain=statistics.mean(r['quality']-b['quality'] for r,b in zip(chosen,baseline)) if paired else None
        tol=statistics.mean(2/p['reference_symbols'] for p in paired) if paired else None
        added=sum(r['status']=='model_failure' and b['status']!='model_failure' for r,b in zip(chosen,baseline))
        # All terms observed around the same deployed preparation and base call.
        selected=statistics.mean(r['recognition_seconds']+r['preparation_seconds']+r['prediction_seconds'][index]
                                 for r in chosen) if paired else None
        fixed=statistics.mean(r['recognition_seconds'] for r in baseline) if paired else None
        passed=(len(paired)==8 and gain>tol and added==0 and selected<=1.10*fixed)
        reports.append(dict(margin=margin,valid_paired_pages=len(paired),improvement=gain,tolerance=tol,
            added_terminal_failures=added,selected_seconds=selected,baseline_seconds=fixed,
            quality=statistics.mean(r['quality'] for r in chosen) if paired else None,
            choices=choices,accepted=passed))
    accepted=[r for r in reports if r['accepted']]
    winner=min(accepted,key=lambda r:(-r['quality'],r['selected_seconds'],-r['margin'])) if accepted else None
    return dict(accepted=winner is not None,winner=winner,candidates=reports,
                valid_paired_pages=len(paired),reason='gate_passed' if winner else 'not_adaptive_eligible',
                cost_semantics='actual candidate preparation + observed margin prediction + actual selected-arm recognition; paired counterfactual comparison')


def collect_pages(plan, roster, data_root, results_root, policy, *, deadline):
    from hybrid.v4_input_selector.public_score import VERSION, score, native_transcript
    from hybrid.v4_input_selector.selector_input import verify_selected_render
    from hybrid.v4_input_selector.supervisor import validated_result
    if roster.get('schema')!='v4_raster_roster_v1' or len(roster['pages'])!=8 or len(roster['items'])!=16:
        raise ValueError('Invalid frozen raster roster')
    selected={p['page_id']:p for p in plan['pages']}
    pages=[]; evidence={}
    for row in roster['pages']:
        if time.monotonic()>=deadline:raise TimeoutError('Calibration CPU wall deadline')
        pid=row['page_id']; original=selected.get(pid)
        if (original is None or original['group_id']!=row['source_group']
                or original['pdf_sha256']!=row['source_pdf_sha256'] or row['source']['source_type']!='raster'):
            raise ValueError('Raster page provenance differs from frozen sample')
        if file_sha(bound_file(data_root,row['file']))!=row['input_sha256']:
            raise ValueError('Raster input changed')
        refpath=bound_file(data_root,row['reference_file'])
        ref=checked_json(refpath,row['reference_sha256'])
        if (ref['page_id']!=pid or ref['source_group']!=row['source_group'] or ref['split']!='calibration'
                or ref['input_sha256']!=row['input_sha256'] or ref['score_version']!=VERSION):
            raise ValueError('Reference binding differs')
        page=dict(page_id=pid,source_group=row['source_group'],reference_symbols=ref['canonical_symbols'],actions={})
        maps=[]; all_decisions=[]
        for action in ('A','B'):
            item=dict(item_id=pid+'_'+action,page_id=pid,file=row['file'],input_sha256=row['input_sha256'],source=row['source'],action=action)
            if item not in roster['items']:raise ValueError('Frozen arm item absent')
            receiptpath=bound_file(results_root,'_calibration_results/'+item['item_id']+'.json')
            if not receiptpath.is_file():
                page['actions'][action]=dict(status='not_run',quality=None)
                continue
            receipt=json.loads(receiptpath.read_bytes()); evidence[receiptpath.relative_to(results_root).as_posix()]=file_sha(receiptpath)
            if receipt.get('schema')!='v4_calibration_result_v1' or receipt['item_id']!=item['item_id']:
                raise ValueError('Unexpected collection receipt')
            prepath=bound_file(results_root,receipt['predecision_file'])
            pre=checked_json(prepath,receipt['predecision_sha256'])
            if (pre['item']!=item or pre['model_sha256']!=MODEL_SHA
                    or pre['original_manifest_sha256']!=ORIGINAL_MANIFEST_SHA
                    or pre['preparation']['recognition_calls']!=0
                    or set(pre['features'])!={'A','B'}):
                raise ValueError('Unbound candidate collection')
            prep=pre['preparation']; vectors=pre['features']; maps.append(vectors)
            for a in ('A','B'):
                feature=prep['actions'][a]['features']
                if feature['source_type']!='raster' or feature['action']!=a or vectors[a]!=(feature['vector'] if feature['ready'] else None):
                    raise ValueError('Feature audit mismatch')
                if vectors[a] is not None and vectors[a][13]!=0:raise ValueError('PDF relabelled as raster')
            decisions=[]; times=[]
            if len(pre['predictions'])!=3:raise ValueError('Three frozen margins required')
            for margin,observed in zip(MARGINS,pre['predictions']):
                probe=ScalePolicy(policy.estimator,dict(policy.config,learned_enabled=True,margin=margin),policy.identity)
                decision=probe.predict(vectors)
                if observed['margin']!=margin or observed['decision']!=decision:
                    raise ValueError('Saved GBDT prediction does not replay')
                expected_calls=1 if decision['predicted_gains'] else 0
                if observed['timing']['estimator_predict_calls']!=expected_calls:
                    raise ValueError('Estimator call count mismatch')
                decisions.append(decision); times.append(observed['timing']['seconds'])
            all_decisions.append(decisions)
            result=receipt['result']; validated_result(item,result,results_root)
            audit=checked_json(bound_file(results_root,result['audit_file']),result['audit_sha256'])
            if audit.get('render') is not None:
                verify_selected_render(prep['actions'][action]['render'],audit['render'])
            if result['status']=='completed':
                if receipt['native_page_pipeline_calls']!=1:raise ValueError('Native call count mismatch')
                blocks=checked_json(bound_file(results_root,item['item_id']+'/final_native_blocks.json'),audit['final_native_sha256'])
                transcript=native_transcript(blocks)
            elif (result['status'] in ('failed','timeout') and receipt['native_page_pipeline_calls']==1
                    and audit.get('failure_stage')=='native_pipeline'):
                transcript=None
            else:
                page['actions'][action]=dict(status='invalid_measurement',quality=None)
                continue
            measured=score(ref['natural_text'],transcript,status=result['status'],deadline=deadline)
            if measured.get('reference_symbols')!=ref['canonical_symbols']:
                raise ValueError('Reference symbol count changed')
            if receipt['selection_nonprediction_seconds'] < prep['preparation_seconds']:
                raise ValueError('Selection overhead cannot omit candidate preparation')
            numbers=[receipt['recognition_seconds'],receipt['selection_nonprediction_seconds'],*times]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in numbers):
                raise ValueError('Invalid observed stage cost')
            page['actions'][action]=dict(status=measured['status'],quality=measured['quality'],
                recognition_seconds=numbers[0],preparation_seconds=numbers[1],prediction_seconds=times)
        if maps and any(m!=maps[0] for m in maps):raise ValueError('Paired candidate feature replay differs')
        if all_decisions and any(d!=all_decisions[0] for d in all_decisions):raise ValueError('Paired model decisions differ')
        page['decisions']=all_decisions[0] if all_decisions else [dict(action='B')]*3
        pages.append(page)
    return pages,evidence


def recalibrate(args):
    started=time.monotonic(); deadline=started+min(1800,args.seconds)
    root=Path(args.output); root.mkdir(parents=True,exist_ok=False)
    write_new(root/'STARTED.json',dict(model_sha256=MODEL_SHA,fit_calls=0,deadline=deadline))
    plan=checked_json(args.plan,args.plan_sha); roster=checked_json(args.roster,args.roster_sha)
    if (plan['model_sha256']!=MODEL_SHA or plan['margins']!=list(MARGINS) or plan['fixed_action']!='B'
            or plan['known_overlap_reviewed'] is not True or roster['plan_sha256']!=args.plan_sha):
        raise ValueError('Unfrozen or unreviewed calibration inputs')
    verify_code(Path(args.code_root),plan['code_files'])
    policy=frozen_model(args.model)
    pages,evidence=collect_pages(plan,roster,Path(args.data_root),Path(args.results_root),policy,deadline=deadline)
    evidence_sha=write_new(root/'EVIDENCE.json',dict(plan_sha256=args.plan_sha,roster_sha256=args.roster_sha,
        result_receipts=evidence,pages=pages,original_manifest_sha256=policy.identity))
    report=gate(pages); report.update(model_sha256=MODEL_SHA,fit_calls=0,elapsed_seconds=time.monotonic()-started)
    write_new(root/'REPORT.json',report)
    if report['accepted']:
        cert=dict(report['winner'],schema='v4_raster_calibration_certificate_v1',model_sha256=MODEL_SHA,
            original_manifest_sha256=policy.identity,fixed_action='B',plan_sha256=args.plan_sha,
            evidence_sha256=evidence_sha,code_files=plan['code_files'],
            rounding_evidence_sha256=plan.get('rounding_evidence_sha256'))
        certificate_sha=write_new(root/'CERTIFICATE.json',cert)
        config=deepcopy(policy.config); config.update(learned_enabled=True,margin=cert['margin'],
            raster_calibration=dict(certificate_sha256=certificate_sha,plan_sha256=args.plan_sha,evidence_sha256=evidence_sha))
        bundle=root/'policy'; bundle.mkdir()
        # Copy exact original model bytes; no pickle dump, no fit, no mutation.
        modelroot=Path(args.model)
        if (modelroot/'policy').is_dir():modelroot=modelroot/'policy'
        with (bundle/MODEL_FILE).open('xb') as f:
            f.write((modelroot/MODEL_FILE).read_bytes());f.flush();os.fsync(f.fileno())
        if file_sha(bundle/MODEL_FILE)!=MODEL_SHA:raise ValueError('Model copy changed')
        config_sha=write_new(bundle/'config.json',config)
        write_new(bundle/'manifest.json',dict(schema='nju_scale_bundle_v1',trusted_local_only=True,
            files={MODEL_FILE:MODEL_SHA,'config.json':config_sha}))
    return report


def main():
    p=argparse.ArgumentParser()
    for k in ('plan','plan-sha','roster','roster-sha','model','data-root','results-root','output','code-root'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--seconds',type=int,default=1800); p.add_argument('--bounded-child',action='store_true')
    args=p.parse_args()
    if not 0<args.seconds<=1800:raise ValueError('Remaining CPU budget must be <=1800 seconds')
    if not args.bounded_child:
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        result=subprocess.run([sys.executable,'-B','-m','hybrid.v4_selected_eval.recalibrate',*sys.argv[1:],'--bounded-child'],env=env,timeout=args.seconds)
        raise SystemExit(result.returncode)
    if os.name=='posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CPU,(args.seconds,args.seconds))
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):print(json.dumps(recalibrate(args),sort_keys=True))

if __name__=='__main__':main()
