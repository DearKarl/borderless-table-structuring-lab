"""Write a descriptive report only after saved results and delivery checks pass."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from .portable_load import digest
from .summarize_results import CONTROLS,FIELDS,VERSIONS


def number(value,*,signed=False,scale=1,precision=4):
    if value is None:return 'unavailable'
    return format(value*scale,('+' if signed else '')+'.'+str(precision)+'f')


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(str(value) for value in row)+' |' for row in rows])


def main():
    parser=argparse.ArgumentParser()
    for name in ['results','paired','packages','closure','figures','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    loaded={name:json.loads(Path(getattr(args,name)).read_text()) for name in ['results','paired','packages','closure','figures']}
    results,paired,packages,closure,figures=(loaded[name] for name in ['results','paired','packages','closure','figures'])
    if results['status']!='complete_scores_and_descriptive_analysis' or paired['status']!='complete_saved_paired_aggregates':
        raise ValueError('Completed saved score analysis required')
    if set(results['benchmark'])!=set(CONTROLS+VERSIONS) or set(results['benchmark_comparisons'])!=set(paired['comparisons']) or len(paired['comparisons'])!=17:
        raise ValueError('Preserved controls or fixed comparisons missing')
    if any(row['page_denominator']!=1651 for row in results['benchmark'].values()):raise ValueError('Full page universe required')
    clean_controls=['base']+[name for name in CONTROLS[2:] if not results['benchmark'][name]['retained_technical_failures']]
    if len(clean_controls)!=5:raise ValueError('The interpretation expects four clean older fits plus native')
    for version in VERSIONS:
        for control in clean_controls:
            a,b=results['benchmark'][version]['values'],results['benchmark'][control]['values']
            if not (a['table_teds']<b['table_teds'] and a['table_teds_structure']<b['table_teds_structure'] and a['table_edit']>b['table_edit']):
                raise ValueError('Written full-benchmark interpretation is not supported by these values')
        if results['confirmation'][version]['teds']<=results['confirmation']['base']['teds']:
            raise ValueError('Written confirmation interpretation is not supported')
        if paired['policy_component_page_denominators'][version]['table']['TEDS']['ALL']!=458:
            raise ValueError('Written table-page denominator differs')
        if paired['comparisons'][version+' - base']['official_sample_outcomes']['table']['TEDS']['candidate_scored_rows']!=665:
            raise ValueError('Written table-row denominator differs')
    if paired['protocol_sha256']!=results['protocol_sha256'] or paired['nomination_sha256']!=results['nomination_sha256']:
        raise ValueError('Paired output binding differs')
    if packages['status']!='passed' or set(packages['versions'])!=set(VERSIONS):raise ValueError('Both complete selected packages required')
    if closure['status']!='owned_compute_closed' or closure['protocol_sha256']!=results['protocol_sha256']:
        raise ValueError('Fresh owned compute closure required')
    if figures['status']!='visually_reviewed' or figures['result_sha256']!=digest(args.results):raise ValueError('Actual rendered figure review required')
    for filename,expected in figures['figures'].items():
        path=Path(args.figures).parent/filename
        if not path.resolve().is_relative_to(Path(args.figures).parent.resolve()) or digest(path)!=expected:raise ValueError('Reviewed figure changed')
    curves=Path(args.output).parent/'training_curves';curve_manifest=json.loads((curves/'MANIFEST.json').read_text())
    if curve_manifest['status']!='complete_numeric_history' or curve_manifest['result_sha256']!=digest(args.results):raise ValueError('Complete numeric curves required')
    if set(curve_manifest['files'])!={'V8.4.csv','V8.5.csv'}:raise ValueError('Both training histories required')
    for filename,item in curve_manifest['files'].items():
        if item['rows']!=7500 or digest(curves/filename)!=item['sha256']:raise ValueError('Training history changed')
    primary=[]
    recovery_note=''
    if results.get('benchmark_recovery'):
        recovery=results['benchmark_recovery']
        if recovery['original_model_calls_per_version']!=0 or recovery['successful_model_pages_repeated']!=0:
            raise ValueError('Unexpected recovery scope')
        recovery_note=('The first benchmark attempt for each version failed before any model call because the recording wrapper used incompatible keyword names. '
            'Each attempt contained 462 failed table-containing pages and 1,189 completed non-table pages. '
            'After diagnosis, its outputs and invalid score records were deleted at the user\'s request; compact failure identities and costs were retained. '
            'A separately identified repair kept the frozen DEV checkpoints, page list, native inputs and scoring protocol. '
            'Only previously unexecuted table inference was performed; non-table outputs were recomposed from the same shared cache. '
            'Reported model scores below come from the repaired attempt. No successful model page was repeated.')
    for version in VERSIONS:
        value=results['benchmark'][version]['values']['table_teds']
        native=results['benchmark_comparisons'][version+' - base']['metrics']['table_teds']['raw_left_minus_right']
        legacy=results['benchmark_comparisons'][version+' - legacy_simple_rule']['metrics']['table_teds']['raw_left_minus_right']
        native_text='unavailable' if native is None else number(native,signed=True,scale=100)+' percentage points'
        legacy_text='unavailable' if legacy is None else number(legacy,signed=True,scale=100)+' percentage points'
        primary.append(f"{version} Table TEDS is **{number(value)}**; its difference from the matched native base is {native_text}, and from the legacy rule {legacy_text}.")
    lines=['# V8.4 / V8.5: matched TDATR-style adaptation','',
        '## Material Passport','',
        'Type: computational experiment report. Status: completed saved execution and descriptive analysis. '+
        'Two formal fits, one per version; no quality rerun or new checkpoint selection during reporting.','',
        'V8.4 uses ordinary LoRA; V8.5 uses LoRA-GA with the same TDATR-style detail and fusion supervision. '+
        'Both completed 7,500 updates and 120,000 example exposures. Each DEV-selected checkpoint was evaluated on all 512 confirmation sources and the same 1,651 benchmark pages.','',
        ' '.join(primary),'',
        'In these observed full-benchmark results, both new versions are worse on all three main table metrics than the matched native reference and all four older fits without technical failures. '+
        'The confirmation TEDS gains over native therefore do not carry over to this full benchmark. '+
        'Lower training losses and improved auxiliary localization losses do not establish better full-benchmark table recognition.','',
        'These are local official-protocol measurements. One fit per version does not establish seed stability or a statistically reliable method advantage. '+
        'Old V8.1–V8.3 full fits have 60,000 exposures. The new fitting budget is twice as large; actual selected checkpoints can precede the last epoch, as reported below.','',
        recovery_note,'',
        '## Eight fitted runs: table metrics first','',
        'Every fitted run is a separate row: two runs each for V8.1, V8.2 and V8.3, and one each for V8.4 and V8.5. '+
        'No seed average or best-seed selection is used. In this table TEDS, structure TEDS and Overall use 0–100; Table Edit Distance uses 0–1. '+
        'All eight rows retain the full 1,651-page denominator. Official table scoring covers 458 pages and 665 table rows.','',
        table(['Run','Table TEDS ↑','Structure TEDS ↑','Table Edit Distance ↓','Overall ↑','Technical failures'],
            [[name]+[number(results['benchmark'][name]['values'][field],scale=100 if field in ['table_teds','table_teds_structure'] else 1,precision=6 if field=='table_edit' else 4)
                for field in ['table_teds','table_teds_structure','table_edit','overall']]+
                [results['benchmark'][name]['retained_technical_failures']] for name in CONTROLS[2:]+VERSIONS]),'',
        'Matched native and legacy-rule references are retained in the full table below. '+
        'The two historical seed0 failures and the doubled fitting budget of the new versions prevent an algorithm-only reading of these differences.','',
        '## Full benchmark','',
        'Overall is on a 0–100 scale; other fields are on a 0–1 scale. Higher is better for Overall, CDM and TEDS; lower is better for edit distances. '+
        'Display values are rounded; the CSV retains full precision. Unavailable official fields remain unavailable. All page denominators include retained failures and empty predictions.','',
        table(['Policy','Overall ↑','Text edit ↓','Formula CDM ↑','Formula edit ↓','Table TEDS ↑','Structure TEDS ↑','Table edit ↓','Reading order ↓','Technical failures'],
            [[name]+[number(results['benchmark'][name]['values'][field]) for field in FIELDS]+[results['benchmark'][name]['retained_technical_failures']] for name in CONTROLS+VERSIONS]),'',
        '![Full benchmark table metrics](figures/full_benchmark_table_metrics.png)','',
        'Complete values and category breakdowns: [RESULTS.json](RESULTS.json), [METRICS.csv](METRICS.csv). '+
        'V8.1/seed0 and V8.2/seed0 retain their historical 161 technical failures each; these are not clean algorithm-only controls.','',
        '## Fixed paired comparisons','',
        'The following differences use candidate minus comparator. Positive TEDS differences improve that metric; negative edit differences improve edit distance. '+
        'TEDS differences are percentage points, not relative percentages.','']
    contrasts=[(version,control) for version in VERSIONS for control in ['base','legacy_simple_rule']]+[('V8.5','V8.4')]
    lines+=[table(['Candidate − comparator','Table TEDS Δ (pp)','Structure TEDS Δ (pp)','Table edit Δ','Overall Δ'],
        [[left+' − '+right]+[number(results['benchmark_comparisons'][left+' - '+right]['metrics'][key]['raw_left_minus_right'],signed=True,
            scale=100 if key in ['table_teds','table_teds_structure'] else 1,precision=6 if key=='table_edit' else 4) for key in ['table_teds','table_teds_structure','table_edit','overall']] for left,right in contrasts]),'',
        'All 17 frozen contrasts are retained, including each new version against all eight preserved policies. '+
        'Exact sample correspondence requires the same page, GT-index group and GT-text hash, unique in both outputs. '+
        'Ambiguous or unidentified examples stay unmatchable. No bootstrap, invented page Overall or favorable missing-value imputation is used.','']
    rows=[]
    for left,right in contrasts:
        entry=paired['comparisons'][left+' - '+right]['official_sample_outcomes']['table']['TEDS'];counts=entry['outcomes']
        rows.append([left+' − '+right]+[counts[key] for key in ['improved','worsened','tied','unmatchable']]+[entry['comparison_unit_denominator']])
    lines+=[table(['Table sample contrast','Improved','Worsened','Tied','Unmatchable','Comparison units'],rows),'',
        'Comparison units are official matched/unmatchable components, not independent documents. '+
        'Metric-perfect tables are not necessarily exact output-text matches. Component-page denominators and repair/harm counts are in [PAIRED_OUTCOMES.json](PAIRED_OUTCOMES.json).','',
        '## DEV selection and confirmation','',
        'Only fusion epochs 1, 2 and 3 were eligible. Each version selected the highest mean TEDS across all 256 DEV sources, with earlier-epoch ties. '+
        'The 12 original missing native inputs remain in DEV. Confirmation and benchmark scores did not select checkpoints.','']
    lines+=[table(['Selected policy','Completed fit exposures','Selected checkpoint exposures','Selected optimizer updates'],
        [[name,results['benchmark'][name]['exposures_per_fit'],results['benchmark'][name]['selected_checkpoint_exposures'],
          results['benchmark'][name]['selected_checkpoint_updates']] for name in CONTROLS[2:]+VERSIONS]),'',
        'Selected-checkpoint counts differ from total fitting costs. These contrasts do not isolate the supervision change from training amount.','']
    rows=[]
    for version in VERSIONS:
        row=results['development'][version]
        for endpoint in row['endpoints']:
            rows.append([version,endpoint['epoch'],number(endpoint['teds']),number(endpoint['structure_teds']),endpoint['inference_status'],endpoint['epoch']==row['selected_fusion_epoch']])
    lines+=[table(['Version','Fusion epoch','DEV TEDS','DEV structure TEDS','Inference','Selected'],rows),'',
        table(['Policy','Confirmation TEDS','Structure TEDS','Exact table','Missing','Malformed'],
            [[name]+[number(results['confirmation'][name][key]) for key in ['teds','structure_teds','exact_table']]+
             [results['confirmation'][name][key] for key in ['missing_count','malformed_count']] for name in CONTROLS+VERSIONS]),'',
        '![Confirmation and benchmark contrasts](figures/confirmation_benchmark_contrasts.png)','',
        'The confirmation unit is one source; the full benchmark uses official component matching and aggregation. '+
        'Their deltas are cohort-specific. Confirmation is reported without an advancement gate. Both test sets were exposed in earlier project work.','',
        '## Training evidence','',
        'The frozen cohort has 20,000 tables from 11,217 TRAIN sources. Each version receives three complete detail passes and three complete native OTSL fusion passes, '+
        'with identical realized orders, assignments and localization-head initialization. V8.5 uses an additional 256-example gradient calibration without optimizer updates. '+
        'AdamW and the cosine schedule reset between stages. No user-fixed seed or seed sweep is introduced.','',
        '![Generation losses](figures/generation_loss_curves.png)','',
        'Detail objectives vary across examples; the two stages do not form one stationary loss series. Faint lines are raw updates and solid lines are 50-update trailing means.','',
        '![Native CE diagnostics](figures/native_CE_diagnostics.png)','',
        'Native CE diagnostics use unchanged 128-example TRAIN and DEV panels every 250 updates, with dropout off and RNG restored. These diagnostics do not select the checkpoint.','',
        '![Localization losses](figures/localization_loss_components.png)','',
        '![Training runtime diagnostics](figures/training_runtime_diagnostics.png)','',
        'Complete numeric update histories: [V8.4](training_curves/V8.4.csv), [V8.5](training_curves/V8.5.csv). '+
        'Each contains all 7,500 optimizer updates. Inapplicable detail-stage localization fields remain blank, not zero.','',
        '[Dataset and protocol](DATASET_AND_PROTOCOL.md) records task eligibility, full sequence lengths, optimization and the explicit adaptation from TDATR. '+
        'Native pre-merger and merger feature maps replace the paper backbone; this is not an exact paper reproduction.','',
        '## Cost and reconstruction','']
    groups=defaultdict(float)
    for allocation in results['costs']['allocations']:
        name=allocation['stage'].split('-')[0];groups['preparation' if name in ['smoke','calibration'] else name]+=allocation['gpu_seconds']
    lines+=[table(['Stage','Allocated GPU-hours'],[[name,number(seconds/3600)] for name,seconds in groups.items()]),'',
        f"Total new allocated GPU-hours: **{number(results['costs']['allocated_gpu_seconds']/3600)}**. "+
        f"Held-out CPU scoring elapsed time: {number(results['costs']['heldout_CPU_scoring_elapsed_seconds']/3600)} hours; "+
        f"full official CPU scoring elapsed time: {number(results['costs']['full_CPU_scoring_elapsed_seconds']/3600)} hours. "+
        'These CPU elapsed sums are not CPU-core hours, and summed GPU allocations are not campaign wall time. Shared input preparation and old-control compute are reused, not charged again. '+
        'Unlaunched queue attempts consume no GPU allocation. New calibration, discarded smoke allocations and failed benchmark attempts are included. '+
        f"Discarded failed-attempt CPU scoring contributes {number(results['costs'].get('failed_full_CPU_scoring_elapsed_seconds',0)/3600)} hours to the full scoring total.",'',
        '![Campaign GPU time](figures/new_campaign_compute.png)','',
        'Localization uses an extra forward on the generated OTSL and does not rewrite official text/HTML. '+
        'Saved evidence reports completed replay time and unavailable counts. Failed or unavailable replay time is not separately isolated; all such cost remains inside worker/allocation totals. '+
        'Text-generation call counts therefore exclude successful additional localization forwards.','']
    for version in VERSIONS:
        item=packages['versions'][version]
        if item['execution_protocol_sha256']!=results['protocol_sha256'] or item['nomination_sha256']!=results['nomination_sha256']:
            raise ValueError('Delivered package selection differs')
        lines+=[f"{version}: selected fusion epoch {item['selected_fusion_epoch']}, {item['selected_training_exposures']:,} exposures at selection; complete checkpoint SHA-256 `{item['checkpoint_sha256']}`.",'']
    lines+=['Both packages contain attention adapters, the trained visual merger and full localization head. V8.5 also preserves its initial compensation factors. '+
        'Selected checkpoint/source hashes and complete FP32 tensor storage are verified; final packaging adds no numerical or quality rerun. '+
        'The frozen implementation had already passed fresh numerical reload and annotation-free inference smoke checks. '+
        'Generic final-factor-only PEFT loading and BF16 merging are not equivalent reconstruction claims. '+
        'Weights remain local; [MODEL_PACKAGES.json](MODEL_PACKAGES.json) carries public metadata.','',
        'Fresh read-only checks confirm that the campaign workers, supervisors and score containers exited. '+
        'Other users’ compute was not terminated. [COMPUTE_CLOSURE.json](COMPUTE_CLOSURE.json) records the verified closure scope.','',
        '## Interpretation limits','',
        'The new pair matches full-fit supervision and budgets while changing LoRA initialization; DEV-selected epochs can differ, and one realized fit does not isolate initialization from stochastic variation. '+
        'Comparisons with the older trained controls also change exposure count. Retained technical failures affect some older comparisons. '+
        'Previously exposed confirmation and benchmark sets do not establish fresh generalization. Transformer-native results are distinct from historical vLLM scores because presence/frequency penalties are not applied. '+
        'No external leaderboard rank, statistical significance, causal supervision gain or deployment recommendation follows from these measurements.','']
    output=Path(args.output)
    with output.open('x',encoding='utf-8',newline='\n') as stream:stream.write('\n'.join(lines))
    print(json.dumps(dict(status='completed_report_written',comparisons=17,new_model_calls=0)))


if __name__=='__main__':main()
