"""Render completed saved training and evaluation evidence without recomputation."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from .portable_load import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ['results','download','output','cache']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--plotting-deps');args=parser.parse_args()
    if args.plotting_deps:sys.path.insert(0,args.plotting_deps)
    os.environ['MPLCONFIGDIR']=args.cache
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    result=json.loads(Path(args.results).read_text())
    if result['status']!='complete_scores_and_descriptive_analysis':raise ValueError('Completed saved result reduction required')
    download=Path(args.download);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((download/'REMOTE_MANIFEST.json').read_text());curves={};colors={'V8.4':'#15616D','V8.5':'#DB7C26'};files={}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(fig,name,rect=None):
        fig.tight_layout(pad=1.7,rect=rect)
        for suffix in ['png','svg']:
            path=output/(name+'.'+suffix)
            if path.exists():raise FileExistsError('Preserve existing result figure: '+name)
            fig.savefig(path,dpi=180,bbox_inches='tight',facecolor='white');files[path.name]=digest(path)
        plt.close(fig)
    def smooth(values,width=50):
        values=np.asarray(values,dtype=float);sums=np.r_[0.,values.cumsum()];indices=np.arange(len(values));starts=np.maximum(0,indices-width+1)
        return (sums[indices+1]-sums[starts])/(indices-starts+1)
    for version,tag in [('V8.4','v84'),('V8.5','v85')]:
        name='train-'+tag+'-001/output/SCALARS.jsonl';path=download/name
        if digest(path)!=manifest['files'][name]['sha256']:raise ValueError('Frozen scalar history changed')
        curves[version]=[json.loads(line) for line in path.read_text().splitlines()]
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,stage in zip(axes,['detail','fusion']):
        for version,rows in curves.items():
            rows=[row for row in rows if row['stage']==stage];x=[row['stage_step'] for row in rows];y=[row['losses']['ce'] for row in rows]
            ax.plot(x,y,color=colors[version],alpha=.12,linewidth=.5);ax.plot(x,smooth(y),label=version,color=colors[version],linewidth=1.5)
        ax.set(title=stage.capitalize()+' stage',xlabel='Optimizer updates within stage',ylabel='Generation CE');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Generation losses: raw values and 50-update trailing means\nDetail tasks vary; the two stages have different objectives',fontsize=12)
    save(fig,'generation_loss_curves')
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,partition in zip(axes,['train','dev']):
        for version in colors:
            rows=[row for row in result['training'][version]['diagnostics'] if row['partition']==partition]
            ax.plot([row['step'] for row in rows],[row['native_generation_ce'] for row in rows],marker='.',label=version,color=colors[version])
        ax.axvline(3750,color='#777777',linestyle='--',linewidth=1);ax.set(title=partition.upper()+' fixed panel (128 examples)',xlabel='Total optimizer updates',ylabel='Native generation CE');ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Native CE diagnostics do not select checkpoints; dashed line marks stage transition',fontsize=12)
    save(fig,'native_CE_diagnostics')
    fig,axes=plt.subplots(2,2,figsize=(11,7));weights={'box':.05,'iou':.03,'mask':.03,'relation':.05}
    for ax,(name,weight) in zip(axes.flat,weights.items()):
        for version,rows in curves.items():
            rows=[row for row in rows if row['stage']=='fusion'];ax.plot([row['stage_step'] for row in rows],smooth([row['losses'][name]*weight for row in rows]),label=version,color=colors[version])
        label='GIoU' if name=='iou' else name.capitalize()
        ax.set(title=f'{label} × {weight:g}',xlabel='Fusion optimizer updates',ylabel='Weighted loss, trailing 50 updates');ax.grid(alpha=.2);ax.legend()
    save(fig,'localization_loss_components')
    fig,axes=plt.subplots(1,3,figsize=(13,4.2))
    for version,rows in curves.items():
        x=[row['step'] for row in rows]
        axes[0].plot(x,smooth([row['pre_clip_gradient_norm'] for row in rows]),label=version,color=colors[version])
        axes[1].plot(x,smooth([row['update_seconds'] for row in rows]),label=version,color=colors[version])
        axes[2].plot(x,[row['peak_gpu_allocated_bytes']/1024**3 for row in rows],label=version,color=colors[version])
    for ax,title,ylabel in zip(axes,['Gradient norm before clipping','Update execution time','Cumulative peak allocation'],['Norm, trailing 50 updates','Seconds, trailing 50 updates','GiB']):
        ax.axvline(3750,color='#777777',linestyle='--',linewidth=1);ax.set(title=title,xlabel='Total optimizer updates',ylabel=ylabel);ax.grid(alpha=.2);ax.legend()
    axes[0].set_yscale('log');save(fig,'training_runtime_diagnostics')
    policies=['base','legacy_simple_rule','V8.1/seed0','V8.1/seed1','V8.2/seed0','V8.2/seed1','V8.3/seed0','V8.3/seed1','V8.4','V8.5']
    fig,axes=plt.subplots(1,3,figsize=(13,6.3),sharey=True)
    for ax,field,title in zip(axes,['table_teds','table_teds_structure','table_edit'],['Table TEDS ↑','Structure TEDS ↑','Table edit ↓']):
        for index,policy in enumerate(policies):
            row=result['benchmark'][policy];value=row['values'][field]
            if value is None:ax.text(.01,index,'unavailable',va='center',fontsize=8);continue
            ax.barh(index,value,color=colors.get(policy,'#A8B3B8'),hatch='//' if row['retained_technical_failures'] else None,height=.7)
            ax.text(value+(.003 if field!='table_edit' else .001),index,f'{value:.4f}',va='center',fontsize=8)
        ax.set(title=title,xlabel='Official metric (0–1)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        maximum=max(row['values'][field] or 0 for row in result['benchmark'].values());ax.set_xlim(0,max(.01,maximum)*1.2)
    axes[0].set_yticks(range(len(policies)),policies);axes[0].invert_yaxis()
    fig.suptitle('All 1,651 pages retained; hatched bars include technical failures\nCompleted fits: new 120,000 / old 60,000 exposures; selected checkpoint budgets vary',fontsize=11)
    save(fig,'full_benchmark_table_metrics')
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for ax,control in zip(axes,['base','legacy_simple_rule']):
        x=[];y=[]
        for version in colors:
            x.append(100*(result['confirmation'][version]['teds']-result['confirmation'][control]['teds']))
            left,right=result['benchmark'][version]['values']['table_teds'],result['benchmark'][control]['values']['table_teds']
            y.append(100*(left-right) if left is not None and right is not None else np.nan)
        positions=np.arange(2);ax.bar(positions-.18,x,width=.36,color='#679436',label='Confirmation (512 sources)');ax.bar(positions+.18,y,width=.36,color='#82589F',label='Full benchmark')
        for index,value in enumerate(y):
            if not np.isfinite(value):ax.text(index+.18,0,'unavailable',rotation=90,ha='center',va='bottom',fontsize=8)
        ax.axhline(0,color='#333333',linewidth=.8);ax.set(title='Difference from '+control,xticks=positions,xticklabels=list(colors),ylabel='TEDS percentage-point difference');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Cohort-specific effects; confirmation sources and official benchmark components differ',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.01),ncol=2,fontsize=9)
    save(fig,'confirmation_benchmark_contrasts',rect=(0,.10,1,1))
    groups=defaultdict(float)
    for row in result['costs']['allocations']:
        prefix=row['stage'].split('-')[0];groups['preparation' if prefix in ['smoke','calibration'] else prefix]+=row['gpu_seconds']/3600
    labels=[name for name in ['preparation','train','dev','confirmation','benchmark'] if name in groups]
    fig,ax=plt.subplots(figsize=(8.2,4.6));values=[groups[name] for name in labels];ax.barh(labels,values,color='#15616D');ax.invert_yaxis()
    for i,value in enumerate(values):ax.text(value+max(values)*.015,i,f'{value:.3f}',va='center')
    ax.set_xlim(0,max(values)*1.2);ax.set(title='New campaign allocated device time, including preparation',xlabel='GPU-hours; shared calibration counted once');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    save(fig,'new_campaign_compute')
    (output/'RESULT_FIGURES.json').write_text(json.dumps(dict(status='rendered_pending_visual_review',result_sha256=digest(args.results),figures=files,
        smoothing='Trailing50 update mean, separately within stages for generation loss',no_model_calls=True),indent=2)+'\n')
    print(json.dumps(dict(figures=len(files),status='rendered_pending_visual_review')))


if __name__=='__main__':main()
