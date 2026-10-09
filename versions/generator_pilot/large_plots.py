"""Publish sanitized measured series and PNG/SVG figures, never invented data."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
from .large_names import canonical_recipe, identity, mapping, public_labels

RECIPES=['V7.3.0','V7.3.1','V7.3.2']
COLORS={'V7.3.0':'#1769aa','V7.3.1':'#d97706','V7.3.2':'#16866b'}
SCALAR_FIELDS=['run_id','recipe','seed','step','microsteps','epoch','unique_examples_visited','exposures','valid_target_tokens',
    'cumulative_target_tokens','generation_nll_numerator','generation_nll_denominator','common_generation_ce','geometry_l1',
    'valid_cell_count','total_objective','adapter_learning_rate','projection_learning_rate','pre_clip_gradient_norm','clipped',
    'nonfinite','skipped','update_seconds','training_seconds','target_tokens_per_second','projection_gradient_norm',
    'head_gradient_norm','adapter_update_frobenius','adapter_update_to_attention_base_ratio','allocated_gpu_bytes',
    'peak_gpu_allocated_bytes','utilization_percent','device_memory_used_mib']
DIAGNOSTIC_FIELDS=['run_id','recipe','seed','step','partition','examples','common_generation_ce',
    'generation_nll_numerator','valid_target_tokens','dropout','checkpoint_selection_used']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    # Only complete newline-terminated records can be used from a live log.
    raw=Path(path).read_text(encoding='utf-8');parts=raw.splitlines()
    if raw and not raw.endswith('\n'):raise ValueError('Log snapshot ends with an incomplete line')
    return [json.loads(line) for line in parts if line.strip()]


def numeric_rows(rows,fields):
    result=[]
    for row in rows:
        if row['recipe'] not in RECIPES or row['seed'] not in [0,1] or row['run_id']!=f"{row['recipe']}-seed{row['seed']}":
            raise ValueError('Unregistered training identity')
        selected={key:row.get(key) for key in fields}
        for key,value in selected.items():
            if key in ['run_id','recipe','partition']:continue
            if value is not None and (not isinstance(value,(int,float,bool)) or not math.isfinite(value)):
                raise ValueError('Non-numeric/nonfinite public scalar: '+key)
        if 'partition' in selected and selected['partition'] not in ['train','dev']:raise ValueError('Unknown panel')
        result.append(selected)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();manifest=read(args.manifest);output=Path(args.output)
    figures=output/'figures';metrics=output/'metrics';figures.mkdir(parents=True,exist_ok=True);metrics.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':170})
    created=[]
    def csv_file(name,rows,fields):
        if 'recipe' in fields and 'run_id' in fields:
            rows=[dict(row,**identity(row['recipe'],row['seed'])) for row in rows]
            fields=[*fields,'historical_alias','internal_run_id']
        with (metrics/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def finish(fig,name,title,footnote):
        title=public_labels(title);footnote=public_labels(footnote)
        fig.suptitle(title,fontsize=13);fig.text(.02,.01,footnote,fontsize=8,va='bottom')
        fig.tight_layout(rect=[0,.08,1,.94])
        for extension in ['png','svg']:
            path=figures/(name+'.'+extension);fig.savefig(path,bbox_inches='tight')
            if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8')
        plt.close(fig);created.append(dict(name=name,title=title,footnote=footnote))
    if manifest.get('frozen_inputs'):
        corpus=read(manifest['frozen_inputs']);train=corpus['train'];dev=corpus['dev']
        confirmation=read(manifest['confirmation_inputs'])['rows']
        if corpus['status']!='full_dataset_frozen' or len(train)!=20000 or len(dev)!=256 or len(confirmation)!=512:
            raise ValueError('Complete frozen corpus required for dataset plots')
        groups={'TRAIN':train,'DEV':dev,'Confirmation':confirmation}
        composition=[dict(partition=name,tables=len(rows),source_documents=len({r['source'] for r in rows}),
            input_eligible=sum(r['status']=='input_eligible' for r in rows)) for name,rows in groups.items()]
        csv_file('split_composition',composition,['partition','tables','source_documents','input_eligible'])
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,key,title in zip(axes,['tables','source_documents'],['Distinct table targets','Distinct source documents']):
            bars=ax.bar([r['partition'] for r in composition],[r[key] for r in composition],color=['#1769aa','#d97706','#16866b'])
            ax.bar_label(bars);ax.set(title=title,ylabel='Count');ax.set_ylim(0,max(r[key] for r in composition)*1.15)
        finish(fig,'dataset_partitions','Source-disjoint frozen corpus','TRAIN20,000 tables; DEV256 and confirmation512 sources. Held-out missing inputs remain in their denominators; counts are targets, not exposures.')
        eligible=[r for r in train if r['status']=='input_eligible'];calibration=set(corpus['calibration_ids'])
        strata=sorted({r['features']['selection_stratum'] for r in train});rows=[]
        for key in strata:
            total=sum(r['features']['selection_stratum']==key for r in train)
            subset=sum(r['features']['selection_stratum']==key and r['id'] in calibration for r in train)
            rows.append(dict(stratum=key,train_tables=total,calibration_tables=subset,train_fraction=total/20000,calibration_fraction=subset/256))
        csv_file('structure_calibration_coverage',rows,['stratum','train_tables','calibration_tables','train_fraction','calibration_fraction'])
        fig,ax=plt.subplots(figsize=(12,6));x=list(range(len(rows)))
        ax.bar([i-.2 for i in x],[r['train_fraction'] for r in rows],width=.4,label='TRAIN n=20,000')
        ax.bar([i+.2 for i in x],[r['calibration_fraction'] for r in rows],width=.4,label='GA calibration n=256')
        ax.set(xticks=x,xticklabels=strata,ylabel='Fraction within cohort',
            ylim=(0,max(max(r['train_fraction'],r['calibration_fraction']) for r in rows)*1.3))
        ax.tick_params(axis='x',rotation=35);ax.legend()
        finish(fig,'calibration_coverage','Training/calibration structure coverage','Provider-marked merged headers x long/wide x blank cells. Header flags are not exhaustive semantic-header annotation. Calibration also stratifies sequence length.')
        import numpy as np
        columns=[('rows',[r['features']['rows'] for r in eligible],[0,5,10,25,50,100,200,1000]),
            ('columns',[r['features']['columns'] for r in eligible],[0,2,4,8,16,32,64,1000]),
            ('sequence_tokens',[r['audit']['counts']['sequence_tokens'] for r in eligible],[0,512,1024,2048,4096,8192,16385]),
            ('native_crop_area_pixels',[math.prod(r['audit']['native_crop_size']) for r in eligible],[0,65536,262144,1048576,4194304,16777216,64000001])]
        histograms=[];fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,(key,values,bins) in zip(axes.flat,columns):
            counts,edges=np.histogram(values,bins=bins)
            if int(counts.sum())!=len(values):raise ValueError('Histogram drops accepted records: '+key)
            labels=[f'{int(a)}–{int(b)-1}' for a,b in zip(edges,edges[1:])]
            ax.bar(range(len(counts)),counts,color='#1769aa');ax.set(xticks=range(len(counts)),xticklabels=labels,title=key,ylabel='Tables');ax.tick_params(axis='x',rotation=35)
            histograms.extend(dict(variable=key,lower_inclusive=int(a),upper_exclusive=int(b),tables=int(count)) for a,b,count in zip(edges,edges[1:],counts))
        csv_file('training_distributions',histograms,['variable','lower_inclusive','upper_exclusive','tables'])
        finish(fig,'training_distributions','Accepted TRAIN structure, token length and image size','All 20,000 unique accepted TRAIN tables. Fixed bins; counts, not probability density. Total sequence <=16,384; no silent truncation.')
        if manifest.get('input_dispositions'):
            dispositions=[r for r in read(manifest['input_dispositions'])['rows'] if r['partition']=='train']
            counts=Counter('selected_train' if r['selected_train'] else 'input_excluded' if r['status']!='input_eligible' else 'isolation_excluded' if r['isolation_exclusion'] else 'eligible_not_selected' for r in dispositions)
            reasons=Counter(r.get('reason') or 'isolation_screen' for r in dispositions if r['status']!='input_eligible' or r['isolation_exclusion'])
            public=dict(static_train_candidates=len(dispositions),dispositions=dict(counts),exclusion_reasons=dict(reasons),
                acquisition_candidates_are_separate=True,final_train_tables=20000)
            (metrics/'input_flow.json').write_text(json.dumps(public,indent=2)+'\n')
            fig,axes=plt.subplots(1,2,figsize=(14,5))
            for ax,values,title in [(axes[0],counts,'Static-eligible TRAIN candidate dispositions'),(axes[1],reasons,'Input/isolation exclusions')]:
                labels=list(values);bars=ax.barh(labels,[values[k] for k in labels],color='#1769aa');ax.bar_label(bars);ax.set(title=title,xlabel='Tables')
                ax.set_xlim(0,max(values.values(),default=1)*1.2)
            finish(fig,'input_flow','Data preparation outcomes before model quality','Static-source acceptance precedes this diagram. Every TRAIN candidate receives a disposition; no model-quality filtering or held-out replacement.')
    scalars=[];diagnostics=[];run_meta=[]
    for item in manifest.get('runs',[]):
        rows=numeric_rows(jsonl(item['scalars']),SCALAR_FIELDS);panel=numeric_rows(jsonl(item['diagnostics']),DIAGNOSTIC_FIELDS)
        if not rows:raise ValueError('Empty measured run log')
        if len({r['run_id'] for r in rows+panel})!=1:raise ValueError('Multiple identities in one run')
        steps=[r['step'] for r in rows]
        if steps!=list(range(max(steps)+1)):raise ValueError('Training log has repeated or missing optimizer steps')
        scalars.extend(rows);diagnostics.extend(panel)
        run_meta.append(dict(run_id=rows[0]['run_id'],last_step=max(steps),status=item['status']))
    if len({r['run_id'] for r in run_meta})!=len(run_meta):raise ValueError('Duplicate run snapshot')
    missing=6-len(run_meta);observed='; '.join(f"{r['run_id']}: {r['last_step']}/3750 ({r['status']})" for r in run_meta)
    footer=f'TRAIN:20,000 unique tables/fit, effective batch16. Raw points; no smoothing. Missing run logs:{missing}/6.'
    def curves(ax,records,x,y,*,panel=None):
        for recipe in RECIPES:
            for seed in [0,1]:
                values=[r for r in records if r['recipe']==recipe and r['seed']==seed and r.get(x) is not None and r.get(y) is not None and (panel is None or r.get('partition')==panel)]
                if values:ax.plot([r[x] for r in values],[r[y] for r in values],color=COLORS[recipe],linestyle='-' if seed==0 else '--',linewidth=.8,alpha=.85,label=identity(recipe,seed)['run_id'])
    if scalars:
        csv_file('training_scalars',scalars,SCALAR_FIELDS);csv_file('diagnostic_panels',diagnostics,DIAGNOSTIC_FIELDS)
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,x,label in zip(axes.flat,['step','exposures','cumulative_target_tokens','training_seconds'],
            ['Optimizer updates','Training exposures','Cumulative supervised tokens','Training wall time (seconds)']):
            curves(ax,scalars,x,'common_generation_ce');ax.set(xlabel=label,ylabel='Generation CE (nats; lower)');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        finish(fig,'generation_ce','Full-corpus training: generation loss',footer+' Wall time excludes diagnostics/initialization.')
        if any(r.get('geometry_l1') is not None for r in scalars):
            fig,ax=plt.subplots(figsize=(9,5));curves(ax,scalars,'step','geometry_l1');ax.legend();ax.set(xlabel='Optimizer updates',ylabel='Mean cell-box L1 (normalized coordinates; lower)')
            finish(fig,'cell_geometry','V7.3.2 auxiliary geometry loss',footer+' Training objective coefficient0.1; this is not a generation-quality score.')
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        for ax,key,label in zip(axes.flat,['adapter_learning_rate','pre_clip_gradient_norm','adapter_update_to_attention_base_ratio','projection_gradient_norm'],
            ['Adapter/head learning rate','Pre-clip global gradient L2 norm','Adapter update / attention-base Frobenius norm','Projection gradient L2 norm']):
            curves(ax,scalars,'step',key);ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=8);finish(fig,'optimization','Training schedule and gradient/update diagnostics',footer+' Resource/norm cadence50updates; missing measurements are omitted.')
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,key,label in zip(axes,['target_tokens_per_second','peak_gpu_allocated_bytes','utilization_percent'],
            ['Supervised tokens / training second','Peak allocated GPU bytes','GPU utilization (%)']):
            curves(ax,scalars,'step',key);ax.set(xlabel='Optimizer updates',ylabel=label);ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);finish(fig,'training_efficiency','Training throughput and resources',footer+' Devices shown only by recipe/seed; peak memory is cumulative within each fit.')
    if diagnostics:
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,partition in zip(axes,['train','dev']):
            curves(ax,diagnostics,'step','common_generation_ce',panel=partition);ax.set(title=partition.upper()+' fixed128 examples',xlabel='Optimizer updates',ylabel='Mean example CE (nats; lower)');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);finish(fig,'train_dev_panels','Teacher-forced TRAIN/DEV diagnostics',f'Dropout off; RNG restored. Update0, every250, final. Diagnostic only; not checkpoint selection. Missing runs:{missing}/6.')
    if manifest.get('dev_scores'):
        rows=[]
        for path in manifest['dev_scores']:
            value=read(path)
            if value['cohort']!='dev' or value['denominator']!=256:raise ValueError('DEV plot denominator differs')
            rows.append({k:value[k] for k in ['run_id','recipe','seed','epoch']}|value['aggregate'])
        fields=['run_id','recipe','seed','epoch','teds','structure_teds','character_distance','cell_text_exact_match','exact_grid','exact_table','malformed_count','missing_count','unmatchable_grid_count','truncated_count']
        csv_file('dev_epochs',rows,fields)
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for ax,key,label in zip(axes,['teds','structure_teds','character_distance'],['Full TEDS (higher)','Structure TEDS (higher)','Character distance (lower)']):
            curves(ax,rows,'epoch',key);ax.set(xlabel='Training epoch',ylabel=label,xticks=[1,2,3],ylim=(0,1));ax.grid(alpha=.2)
        axes[0].legend(fontsize=8);finish(fig,'dev_generation','Free generation on all256 DEV sources',f'Full denominator includes missing/malformed outputs. Observed checkpoints:{len(rows)}/18. Only these epochs select checkpoints.')
    if manifest.get('confirmation'):
        value=read(manifest['confirmation'])
        if value['status']!='complete' or value['denominator']!=512:raise ValueError('Complete confirmation analysis required')
        # This analysis is already aggregate; exclude operational hashes and
        # any source identifiers from the public evidence file explicitly.
        public={k:value[k] for k in ['cohort','denominator','nominee','aggregate','comparisons','former_promotion_gate_diagnostic','strata','seed_variability','nominee_reselected','full_benchmark_required','limits']}
        public=public_labels(public);public['version_mapping']=mapping()
        (metrics/'confirmation_aggregate.json').write_text(json.dumps(public,indent=2)+'\n')
        selected=[(name,row) for name,row in value['comparisons'].items() if row['primary']]
        fig,ax=plt.subplots(figsize=(10,5))
        for i,(name,row) in enumerate(selected):
            mean=row['mean_delta'];low,high=row['paired_percentile_97_5_ci']
            ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=5,color='#1769aa')
        ax.set(yticks=range(len(selected)),yticklabels=[public_labels(name) for name,_ in selected],xlabel='Paired mean full-TEDS difference (higher)');ax.axvline(0,color='#555',linewidth=1)
        finish(fig,'confirmation_primary','Prespecified nominee vs both controls:512 sources','10,000 paired source resamples, seed0;97.5% two-sided marginal intervals, Bonferroni family coverage>=95%. Optimization-seed variation is separate.')
        fig,axes=plt.subplots(1,2,figsize=(13,5));names=list(value['aggregate']);x=list(range(len(names)))
        axes[0].scatter(x,[value['aggregate'][name]['teds'] for name in names]);axes[0].set(xticks=x,xticklabels=public_labels(names),ylabel='Mean full TEDS (higher)',ylim=(0,1));axes[0].tick_params(axis='x',rotation=60)
        for recipe,row in value['seed_variability'].items():
            label=canonical_recipe(recipe)
            axes[1].scatter([label]*2,row['values'],color=COLORS[recipe]);axes[1].plot([label,label],row['range'],color=COLORS[recipe])
        axes[1].set(ylabel='Mean full TEDS (higher)',ylim=(0,1))
        finish(fig,'confirmation_all_models','All six DEV-selected models and both controls','All512 sources retained. Right: two individual seeds and their range; not a confidence interval. Nominee fixed from DEV, no confirmation reselection.')
        fig,ax=plt.subplots(figsize=(10,5));left=[0]*len(selected)
        for key,label,color in [('gain_count','TEDS improved','#16866b'),('harm_count','TEDS worsened','#bb3e3e'),('tie_count','Tied','#b7bdc5')]:
            counts=[row[key] for _,row in selected];ax.barh(range(len(selected)),counts,left=left,label=label,color=color);left=[a+b for a,b in zip(left,counts)]
        ax.set(yticks=range(len(selected)),yticklabels=[public_labels(name) for name,_ in selected],xlabel='Number of sources',xlim=(0,512));ax.legend()
        finish(fig,'confirmation_gains_harms','Paired source gains, harms and ties','All512 source pairs, full TEDS. Exact-table repairs/harms and unmatchable grids are separately reported in confirmation_aggregate.json.')
    (metrics/'VERSION_MAPPING.json').write_text(json.dumps(mapping(),indent=2)+'\n')
    (metrics/'PLOT_MANIFEST.json').write_text(json.dumps(dict(measured_runs=public_labels(run_meta),missing_training_runs=missing,figures=created,
        version_mapping='VERSION_MAPPING.json',
        visual_inspection='pending',raw_prediction_or_reference_bodies_included=False,training_trajectories_are_not_dataset_size_learning_curves=True),indent=2)+'\n')
    print(dict(figures=len(created),formats=['png','svg'],measured_runs=len(run_meta),visual_inspection='pending'))


if __name__=='__main__':main()
