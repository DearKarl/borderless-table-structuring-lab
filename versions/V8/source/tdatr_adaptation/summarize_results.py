"""Reduce completed saved evidence; never fit, score, or select another model."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics
from .portable_load import digest

FIELDS=['overall','text_edit','formula_cdm','formula_edit','table_teds','table_teds_structure','table_edit','reading_order_edit']
VERSIONS=['V8.4','V8.5']
CONTROLS=['base','legacy_simple_rule','V8.1/seed0','V8.1/seed1','V8.2/seed0','V8.2/seed1','V8.3/seed0','V8.3/seed1']


def read(path):return json.loads(Path(path).read_text())


def finite_metrics(values):
    result={}
    for key in FIELDS:
        value=values.get(key)
        if value is not None and (not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=(100 if key=='overall' else 1)):
            raise ValueError('Invalid official metric: '+key)
        result[key]=value
    return result


def paired_confirmation(candidate,baseline):
    if len(candidate)!=512 or len(baseline)!=512 or set(candidate)!=set(baseline):
        raise ValueError('Original512-source correspondence required')
    if any(not isinstance(row['teds'],(int,float)) or not math.isfinite(row['teds']) or not 0<=row['teds']<=1 for rows in [candidate,baseline] for row in rows.values()):
        raise ValueError('Invalid per-source confirmation score')
    differences=[candidate[key]['teds']-baseline[key]['teds'] for key in candidate]
    return dict(source_denominator=512,mean_teds_delta=sum(differences)/512,
        improved=sum(v>1e-9 for v in differences),worsened=sum(v< -1e-9 for v in differences),tied=sum(abs(v)<=1e-9 for v in differences),
        repaired_exact_tables=sum(candidate[key]['exact_table'] and not baseline[key]['exact_table'] for key in candidate),
        harmed_exact_tables=sum(baseline[key]['exact_table'] and not candidate[key]['exact_table'] for key in candidate),
        tie_tolerance=1e-9,descriptive_only=True,optimization_seed_uncertainty_not_estimated=True)


def training_statistics(training,scalars,diagnostics):
    if training['status']!='complete' or training['optimizer_updates']!=7500 or training['exposures']!=120000:
        raise ValueError('Completed single formal fit required')
    if [row['step'] for row in scalars]!=list(range(1,7501)):raise ValueError('Training scalar update coverage differs')
    if any(row['exposures']!=16*row['step'] or row.get('skipped') for row in scalars):raise ValueError('Dropped formal exposures')
    expected={(step,partition) for step in range(0,7501,250) for partition in ['train','dev']}
    if {(row['step'],row['partition']) for row in diagnostics}!=expected or len(diagnostics)!=len(expected):
        raise ValueError('Diagnostic coverage differs')
    stages={}
    for stage in ['detail','fusion']:
        rows=[row for row in scalars if row['stage']==stage]
        if len(rows)!=3750 or {row['epoch'] for row in rows}!={1,2,3}:raise ValueError('Complete3-pass stage missing')
        stages[stage]=dict(updates=len(rows),exposures=16*len(rows),
            optimizer_update_seconds=sum(row['update_seconds'] for row in rows),
            median_update_seconds=statistics.median(row['update_seconds'] for row in rows),
            target_tokens=sum(row['target_tokens'] for row in rows),
            token_weighted_generation_nll=sum(row['generation_nll_numerator'] for row in rows)/sum(row['target_tokens'] for row in rows),
            peak_allocated_bytes=max(row['peak_gpu_allocated_bytes'] for row in rows),
            first_update_objective=rows[0]['losses']['total'],last_update_objective=rows[-1]['losses']['total'])
    return dict(status='complete',optimizer_updates=7500,exposures=120000,stages=stages,
        diagnostics=diagnostics,task_exposures=training['task_exposures'],diagnostic_seconds=training['diagnostic_seconds'],
        elapsed_worker_seconds=training['elapsed_seconds'],loss_interpretation='Detail tasks vary; objectives across stages are not one stationary metric. Native CE diagnostics use the same fixed panels.')


def main():
    parser=argparse.ArgumentParser()
    for name in ['download','protocol','old-benchmark','old-confirmation-endpoints','old-selection','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();download=Path(args.download);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest=read(download/'REMOTE_MANIFEST.json');protocol=read(args.protocol)
    if manifest['protocol_sha256']!=digest(args.protocol) or read(download/'DOWNLOAD_RECEIPT.json')['status']!='verified':
        raise ValueError('Verified completed download required')
    for name,item in manifest['files'].items():
        if digest(download/name)!=item['sha256']:raise ValueError('Saved execution evidence changed: '+name)
    nomination=read(download/'evaluation-controller-001/NOMINATION.json')
    if nomination['protocol_sha256']!=digest(args.protocol):raise ValueError('Selection protocol differs')
    benchmark={}
    with Path(args.old_benchmark).open(newline='',encoding='utf-8-sig') as stream:
        for row in csv.DictReader(stream):
            if row['policy'] not in CONTROLS or row['page_denominator']!='1651':raise ValueError('Old benchmark identity differs')
            benchmark[row['policy']]=dict(status=row['status'],page_denominator=1651,
                values=finite_metrics({key:float(row[key]) if row[key] else None for key in FIELDS}),
                retained_technical_failures=int(row['all_incomplete_pages']),old_control=True,exposures_per_fit=60000 if row['policy'].startswith('V8.') else None)
    if set(benchmark)!=set(CONTROLS):raise ValueError('Preserved old controls incomplete')
    old_selection=read(args.old_selection)
    if old_selection['pages_per_model']!=1651 or len(old_selection['models'])!=6:raise ValueError('Preserved six-model selection required')
    old_epochs={row['run_id']:row for row in old_selection['models']}
    confirmation={};confirmation_rows={};source_hashes={'old_benchmark':digest(args.old_benchmark),'old_selection':digest(args.old_selection)}
    aliases={'V7.3.0':'V8.1','V7.3.1':'V8.2','V7.3.2':'V8.3'}
    for name,item in read(args.old_confirmation_endpoints)['models'].items():
        if digest(item['path'])!=item['sha256']:raise ValueError('Preserved confirmation score changed')
        value=read(item['path'])
        if value['gold_file_sha256']!=protocol['gold_hashes']['confirmation'] or value['denominator']!=512 or value['status']!='complete':
            raise ValueError('Preserved confirmation cohort differs')
        public=name if name in ['base','legacy_simple_rule'] else aliases[name.split('-seed')[0]]+'/seed'+name.split('-seed')[1]
        if public.startswith('V8.'):
            epoch=old_epochs[name]['epoch']
            if epoch not in [1,2,3] or value['epoch']!=epoch:raise ValueError('Preserved selected epoch correspondence differs')
            benchmark[public].update(selected_checkpoint_epoch=epoch,selected_checkpoint_exposures=20000*epoch,
                selected_checkpoint_updates=1250*epoch,selected_checkpoint_sha256=old_epochs[name]['checkpoint_sha256'])
        else:benchmark[public].update(selected_checkpoint_epoch=None,selected_checkpoint_exposures=None,selected_checkpoint_updates=None)
        confirmation[public]=value['aggregate'];confirmation_rows[public]={row['id']:row for row in value['rows']};source_hashes['confirmation:'+public]=item['sha256']
    if set(confirmation)!=set(CONTROLS):raise ValueError('Preserved confirmation controls incomplete')
    development={};training_summary={}
    for version,tag in [('V8.4','v84'),('V8.5','v85')]:
        chosen=nomination['selected_checkpoints'][version]
        stage=download/('train-'+tag+'-001')/'output';training=read(stage/'TRAINING.json')
        scalars=[json.loads(line) for line in (stage/'SCALARS.jsonl').read_text().splitlines()]
        diagnostics=[json.loads(line) for line in (stage/'DIAGNOSTICS.jsonl').read_text().splitlines()]
        training_summary[version]=training_statistics(training,scalars,diagnostics)
        if training['protocol_sha256']!=digest(args.protocol):raise ValueError('Formal fit contract differs')
        dev=[]
        for epoch in [1,2,3]:
            value=read(download/('score-dev-'+tag+'-epoch'+str(epoch))/'work/SCORES.json')
            if value['cohort']!='dev' or value['denominator']!=256 or value['status']!='complete' or value['gold_file_sha256']!=protocol['gold_hashes']['dev'] or value['version']!=version or value['protocol_sha256']!=digest(args.protocol):
                raise ValueError('Full original DEV score required')
            dev.append(dict(epoch=epoch,teds=value['aggregate']['teds'],structure_teds=value['aggregate']['structure_teds'],inference_status=value['inference_status']))
        if max(dev,key=lambda row:(row['teds'],-row['epoch']))['epoch']!=chosen['epoch']:
            raise ValueError('Frozen DEV selection does not match its declared rule; do not reselect')
        development[version]=dict(endpoints=dev,selected_fusion_epoch=chosen['epoch'],selected_checkpoint_sha256=chosen['checkpoint_sha256'])
        confirm=read(download/('score-confirmation-'+tag+'-001')/'work/SCORES.json')
        if confirm['denominator']!=512 or confirm['gold_file_sha256']!=protocol['gold_hashes']['confirmation'] or confirm['epoch']!=chosen['epoch'] or confirm['version']!=version or confirm['protocol_sha256']!=digest(args.protocol):
            raise ValueError('Confirmation binding differs')
        confirmation[version]=confirm['aggregate'];confirmation_rows[version]={row['id']:row for row in confirm['rows']}
        suffix=manifest.get('benchmark_attempt_suffix','001')
        saved=manifest['benchmark_summaries'][version]
        if saved['status']!='complete' or saved['model_calls']<=0 or not saved['first_successful_generation_recorded']:
            raise ValueError('Completed real benchmark inference required before quality reporting')
        full=read(download/('score-benchmark-'+tag+'-'+suffix)/'BENCHMARK_SCORES.json')
        if full['status']!='complete' or full['denominator']!=1651 or full['selection_sha256']!=digest(download/'evaluation-controller-001/NOMINATION.json'):
            raise ValueError('Complete frozen benchmark score required')
        item=full['policies']['candidate']
        benchmark[version]=dict(status=item['status'],page_denominator=1651,values=finite_metrics(item['values']),
            retained_technical_failures=item['input_failures'],empty_predictions=item['empty_predictions'],
            missing_reasons=item['missing_reasons'],category_breakdowns=item['category_breakdowns'],
            cpu_evaluator_elapsed_seconds=item['cpu_seconds'],old_control=False,exposures_per_fit=120000,
            selected_checkpoint_epoch=chosen['epoch'],selected_checkpoint_exposures=60000+20000*chosen['epoch'],
            selected_checkpoint_updates=3750+1250*chosen['epoch'],selected_checkpoint_sha256=chosen['checkpoint_sha256'])
    comparisons={};confirmation_pairs={}
    pairs=[(version,control) for version in VERSIONS for control in CONTROLS]+[('V8.5','V8.4')]
    for left,right in pairs:
        contrasts={}
        for metric in FIELDS:
            a,b=benchmark[left]['values'][metric],benchmark[right]['values'][metric]
            delta=a-b if a is not None and b is not None else None
            contrasts[metric]=dict(raw_left_minus_right=delta,improvement_signed_delta=None if delta is None else delta*(-1 if metric.endswith('_edit') else 1))
        comparisons[left+' - '+right]=dict(metrics=contrasts,descriptive_only=True,
            old_control_exposure_difference=right.startswith(('V8.1','V8.2','V8.3')),
            affected_by_retained_technical_failures=bool(benchmark[left]['retained_technical_failures'] or benchmark[right]['retained_technical_failures']))
        confirmation_pairs[left+' - '+right]=paired_confirmation(confirmation_rows[left],confirmation_rows[right])
    allocations=[];heldout_cpu=[];unlaunched=[]
    for name in manifest['files']:
        if name.count('/')!=1:continue
        stage,filename=name.split('/')
        if filename=='EXECUTION.json':
            value=read(download/name)
            if not value['worker_exited'] or not value['lease_released']:raise ValueError('An allocated worker is not closed')
            allocations.append(dict(stage=stage,status=value['status'],gpu_seconds=value['gpu_seconds'],started_at_utc=value['started_at_utc'],finished_at_utc=value['finished_at_utc']))
        elif filename=='SCORING_EXECUTION.json':
            value=read(download/name);heldout_cpu.append(dict(stage=stage,status=value['status'],wall_seconds=value['elapsed_seconds']))
        elif filename=='NO_IDLE_DEVICE.json':unlaunched.append(stage)
    costs=dict(allocated_gpu_seconds=sum(row['gpu_seconds'] for row in allocations),allocations=allocations,
        GPU_allocations=len(allocations),unlaunched_no_idle_attempts=unlaunched,unlaunched_attempts_consume_GPU_seconds=False,
        heldout_CPU_scoring_elapsed_seconds=sum(row['wall_seconds'] for row in heldout_cpu),heldout_CPU_stages=heldout_cpu,
        full_CPU_scoring_elapsed_seconds=sum(benchmark[v]['cpu_evaluator_elapsed_seconds'] for v in VERSIONS)+manifest.get('recovery',{}).get('failed_full_scoring_cpu_seconds',0.),
        failed_full_CPU_scoring_elapsed_seconds=manifest.get('recovery',{}).get('failed_full_scoring_cpu_seconds',0.),
        original_cohort_preparation_and_old_baseline_compute_reused_not_charged_again=True,
        times_are_not_CPU_core_hours=True,allocated_GPU_time_is_not_campaign_wall_time=True,
        CPU_audit_and_local_preparation_not_in_GPU_total=True)
    result=dict(status='complete_scores_and_descriptive_analysis',at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=digest(args.protocol),nomination_sha256=digest(download/'evaluation-controller-001/NOMINATION.json'),
        benchmark=benchmark,benchmark_comparisons=comparisons,confirmation=confirmation,confirmation_comparisons=confirmation_pairs,
        development=development,training=training_summary,costs=costs,source_hashes=source_hashes,
        saved_inference_summary=manifest['benchmark_summaries'],prediction_bindings=manifest['prediction_bindings'],
        benchmark_recovery=manifest.get('recovery'),
        model_reselected=False,new_scoring_or_inference_performed=False,raw_GT_or_prediction_bodies_included=False,
        limits=['One fit per new version does not measure seed stability.','Old complete fits have half the exposures; actual selected-checkpoint exposures vary and are separately reported.',
            'Previously evaluated confirmation and benchmark data are exposed tests.','All denominators retain missing and failed predictions.',
            'Native Transformers comparison is not historical vLLM equivalence.','No leaderboard submission or significance claim.'])
    target=output/'RESULTS.json'
    if target.exists():raise FileExistsError('Preserve existing completed analysis')
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'METRICS.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['policy','status','page_denominator',*FIELDS,'retained_technical_failures','exposures_per_fit','selected_checkpoint_exposures','selected_checkpoint_updates']);writer.writeheader()
        for name in [*CONTROLS,*VERSIONS]:
            row=benchmark[name];writer.writerow(dict(policy=name,**{k:row[k] for k in ['status','page_denominator','retained_technical_failures','exposures_per_fit','selected_checkpoint_exposures','selected_checkpoint_updates']},**row['values']))
    print(json.dumps(dict(status=result['status'],versions=VERSIONS,comparisons=len(comparisons),raw_prediction_bodies_exported=False)))


if __name__=='__main__':main()
