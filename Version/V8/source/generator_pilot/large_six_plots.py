"""Plot measured full-benchmark outputs, preserving missing values explicitly."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ('overall','text_edit','formula_cdm','formula_edit','table_teds',
          'table_teds_structure','table_edit','reading_order_edit')
NAMES = ['base','legacy_simple_rule'] + [f'V8.{recipe}/seed{seed}' for recipe in (1,2,3) for seed in (0,1)]
LABELS = ['Base','Legacy','8.1 s0','8.1 s1','8.2 s0','8.2 s1','8.3 s0','8.3 s1']
COLORS = ['#64748b','#a855a1','#1769aa','#1769aa','#d97706','#d97706','#16866b','#16866b']


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--aggregate',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); source=Path(args.aggregate); value=json.loads(source.read_text(encoding='utf-8'))
    if value['distinct_pages']!=1651 or set(value['versions'])!=set(NAMES):
        raise ValueError('Both controls and all six full-model dispositions required')
    output=Path(args.output); output.mkdir(parents=True,exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.dpi':170})
    files=[]
    def finish(fig,name,title,footer):
        fig.suptitle(title,fontsize=14); fig.text(.02,.01,footer,fontsize=8,va='bottom')
        fig.tight_layout(rect=[0,.06,1,.96])
        for suffix in ('png','svg'):
            path=output/f'{name}.{suffix}'; fig.savefig(path,bbox_inches='tight')
            if suffix=='svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
            files.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)
    fig,axes=plt.subplots(2,4,figsize=(17,9))
    for ax,metric in zip(axes.flat,FIELDS):
        for i,name in enumerate(NAMES):
            number=value['versions'][name]['values'][metric]
            if number is None:
                ax.text(i,.025,'missing',rotation=90,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7,color='#8b1e22')
            else:
                ax.bar(i,number,color=COLORS[i],alpha=.9,hatch='//' if i>1 and name.endswith('seed1') else None)
        ax.set(title=metric+' ('+value['directions'][metric]+')',xticks=range(8),xticklabels=LABELS,
            ylim=(0,100 if metric=='overall' else 1))
        ax.tick_params(axis='x',rotation=65,labelsize=8); ax.grid(axis='y',alpha=.15)
    finish(fig,'official_metrics','All eight official fields: six models and matched controls',
        'Full axes; all1,651 pages retained. Missing scores are labeled, never plotted as zero. Overall0–100; other fields0–1. No significance or leaderboard-rank claim.')
    fig,axes=plt.subplots(2,4,figsize=(17,9))
    for ax,metric in zip(axes.flat,FIELDS):
        for i,name in enumerate(NAMES[2:]):
            delta=value['comparisons'][name+' - base']['metrics'][metric]['improvement_signed_delta']
            if delta is None:
                ax.text(i,.02,'missing',rotation=90,transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=7,color='#8b1e22')
            else:
                ax.bar(i,delta,color=COLORS[i+2],alpha=.9,hatch='//' if name.endswith('seed1') else None)
        ax.axhline(0,color='#333',linewidth=.8); ax.grid(axis='y',alpha=.15)
        ax.set(title=metric,xticks=range(6),xticklabels=LABELS[2:],ylabel='Improvement over matched base')
        ax.tick_params(axis='x',rotation=55,labelsize=8)
    finish(fig,'matched_base_differences','Full-benchmark changes against the shared native base',
        'Positive is better: edit-distance differences are sign-reversed. Each panel uses its own units/axis. Descriptive paired aggregate differences; no confidence intervals.')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,metric in zip(axes,['table_teds','table_teds_structure']):
        for i,recipe in enumerate(['V8.1','V8.2','V8.3']):
            points=value['seed_variability'][recipe][metric]['individual_values']
            for seed,point in enumerate(points):
                if point is not None:
                    ax.scatter(i+(-.08 if seed==0 else .08),point,color=COLORS[2+i*2],marker='o' if seed==0 else '^',s=55,
                        label='seed'+str(seed) if i==0 else None)
            if None not in points:
                ax.plot([i,i],[min(points),max(points)],color=COLORS[2+i*2],linewidth=2)
                ax.scatter(i,sum(points)/2,color=COLORS[2+i*2],marker='_',s=160)
        for name,color,style in [('base','#64748b','--'),('legacy_simple_rule','#a855a1',':')]:
            score=value['versions'][name]['values'][metric]
            if score is not None:
                ax.axhline(score,color=color,linestyle=style,label='Base' if name=='base' else 'Legacy')
        ax.set(title=metric,xticks=range(3),xticklabels=['V8.1','V8.2','V8.3'],ylabel='Official score (higher)')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False); ax.grid(alpha=.15)
    axes[0].legend(fontsize=8)
    finish(fig,'table_seed_variability','Full benchmark: individual seeds, mean and range',
        'Expanded vertical axes. Dots/triangles are individual seeds; horizontal ticks are their means and vertical segments their ranges. Two-seed ranges are not confidence intervals.')
    groups=value['costs']['groups']; keys=list(groups)
    fig,ax=plt.subplots(figsize=(11,6)); hours=[groups[key]['allocated_gpu_seconds']/3600 for key in keys]
    bars=ax.barh([key.replace('_',' ') for key in keys],hours,color='#1769aa')
    ax.bar_label(bars,labels=[f'{number:.2f}' for number in hours],padding=4); ax.set_xlabel('Terminal allocated GPU-hours')
    if hours:ax.set_xlim(0,max(hours)*1.17)
    ax.grid(axis='x',alpha=.15)
    finish(fig,'allocation_costs','Measured campaign GPU allocation costs',
        'Failed attempts retained; shared preparation/calibration/controls counted once. GPU allocation hours are not elapsed wall time. CPU scoring is excluded and reported separately.')
    manifest=dict(source_aggregate_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),files=files,
        visual_inspection='pending',measured_values_only=True,no_uncertainty_intervals_fabricated=True)
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(figures=4,exports=len(files),visual_inspection='pending')))


if __name__=='__main__':
    main()
