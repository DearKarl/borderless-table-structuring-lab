"""Create six public-safe configurations sharing four verified controller files."""
import argparse
import json
import platform
import shutil
from pathlib import Path

import numpy as np

from .direct_controller import DirectScaleModel
from .features import FEATURE_NAMES
from .io_utils import atomic_json,digest,read_json,utc

CONFIGS={'GBDT_V6.1.1':('gbdt',('table',),'G-T'),'GBDT_V6.1.2':('gbdt',('equation',),'G-F'),
         'GBDT_V6.1.3':('gbdt',('table','equation'),'G-TF'),'MLP_V6.2.1':('mlp',('table',),'N-T'),
         'MLP_V6.2.2':('mlp',('equation',),'N-F'),'MLP_V6.2.3':('mlp',('table','equation'),'N-TF')}


def package(models,dataset,output,events):
    models=Path(models);dataset=Path(dataset);out=Path(output)
    fitted=read_json(models/'TRAINING_RECEIPT.json');frozen=read_json(dataset/'DATASET_FROZEN.json')
    for split in ('train','validation','test'):
        if digest(dataset/(split+'.json'))!=frozen['split_sha256'][split]:raise ValueError('Frozen dataset changed')
    if fitted['train_sha256']!=frozen['split_sha256']['train'] or fitted['validation_sha256']!=frozen['split_sha256']['validation']:
        raise ValueError('Controllers were fitted on a different dataset')
    if len(fitted['models'])!=4:raise ValueError('Four actual fitted controllers are required')
    validation=read_json(dataset/'validation.json')['rows'];loaded={};receipts={}
    out.mkdir(parents=True,exist_ok=False)
    for key,metadata in fitted['models'].items():
        source=models/(key+'.json');model=DirectScaleModel(source,metadata['sha256'])
        if key!=model.model['family']+'_'+model.model['kind']:raise ValueError('Checkpoint task/family mismatch')
        destination=out/(key+'.json');shutil.copyfile(source,destination)
        deployed=DirectScaleModel(destination,metadata['sha256'])
        rows=[r['features'] for r in validation if r['kind']==model.model['kind']]
        before=np.asarray(model.predict(rows));after=np.asarray(deployed.predict(rows))
        if not np.isfinite(after).all() or not np.array_equal(before,after):raise ValueError('Packaged load/prediction differs')
        loaded[key]={'file':destination.name,'sha256':digest(destination),'family':model.model['family'],'kind':model.model['kind']}
        receipts[key]={'checkpoint_sha256':digest(destination),'validation_rows_checked':len(rows),
                       'prediction_max_abs_difference':float(np.max(np.abs(before-after))),
                       'loaded_through':'versions.v5_native.direct_controller.DirectScaleModel','device':'cpu'}
    source_root=Path(__file__).resolve().parent
    source_files={p.name:digest(p) for p in sorted(source_root.iterdir()) if p.is_file() and p.suffix in ('.py','.json')}
    for version,(family,kinds,alias) in CONFIGS.items():
        config={'version':version,'internal_alias':alias,'family':family,'kinds':list(kinds),'contract':'direct_log_scale_v1',
                'controllers':{kind:loaded[family+'_'+kind] for kind in kinds},'features':FEATURE_NAMES,
                'dataset_manifest_sha256':digest(dataset/'DATASET_FROZEN.json'),'split_sha256':frozen['split_sha256'],
                'source_code_sha256':source_files,'source_revision_note':'Working-tree V6 implementation identified by exact per-file hashes; not a claimed published revision.',
                'native_input':{'render_dpi':200,'native_long_edge_cap':3500,'source':'Actual default TeleOCR read_fn/do_parse rendered image',
                                'original_rgb_bypass':False,'formula_200_over_72':False},
                'controller':{'scale_interval':[.5,3.],'regresses':'log(scale)','runtime_search':False,'gain_head':False,
                              'rereads_per_region':1,'failure':'Retain native on infeasible or invalid reread'},
                'dependencies':{'python_for_controller':'>=3.10','numpy':'>=2.2','fit_numpy':fitted['numpy_version'],
                                'fit_scikit_learn':fitted['sklearn_version'],'fit_python':platform.python_version(),
                                'ocr':'Separately supplied pinned TeleOCR/vLLM assets; see repository protocol'},
                'invocation':f'python -m versions.v5_native.predict --package artifacts/models/v6/{version}.json --kind {kinds[0]} --features native_features.json --width 100 --height 40',
                'loaded_validation':{kind:receipts[family+'_'+kind] for kind in kinds},
                'benchmark_results':'Pending; packaging is a technical load check, not efficacy evidence'}
        atomic_json(out/(version+'.json'),config)
    summary={'at':utc(),'event':'package_complete','versions':list(CONFIGS),'unique_checkpoint_count':4,
             'checkpoints':loaded,'load_validation':receipts,'dataset_manifest_sha256':digest(dataset/'DATASET_FROZEN.json'),
             'training_receipt_sha256':digest(models/'TRAINING_RECEIPT.json'),'split_counts':frozen['splits'],
             'public_safe':True,'ocr_weights_trained':False,'device':'cpu'}
    atomic_json(out/'PACKAGE_MANIFEST.json',summary)
    with open(events,'a',encoding='utf-8') as stream:stream.write(json.dumps(summary)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--output',required=True);p.add_argument('--phase-events',required=True);a=p.parse_args()
    print(json.dumps(package(a.models,a.dataset,a.output,a.phase_events),indent=2))
