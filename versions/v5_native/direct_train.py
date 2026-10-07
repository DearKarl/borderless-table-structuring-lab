"""Matched task-specific direct log-scale training; source-page regions only."""
import argparse
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from .direct_controller import DirectScaleModel
from .features import FEATURE_NAMES
from .io_utils import atomic_json,digest,utc
from .matched_train import GBDT,MLP,fit_mlp,forward,preprocessing,transform,trees


def bundle(path,split):
    d=json.loads(Path(path).read_text(encoding='utf-8'))
    if d['provenance']['scope']!='external_only' or d['provenance']['unit_contract']!='source_page_native_predicted_region_v1' or d['split']!=split:
        raise ValueError('Direct training requires an isolated external source-page region bundle')
    return d


def arrays(rows):
    if not rows or len({r['region_id'] for r in rows})!=len(rows):
        raise ValueError('Require exactly one target row per distinct native region')
    x=np.asarray([r['features'] for r in rows],dtype=np.float64)
    y=np.asarray([r['target_log_scale'] for r in rows],dtype=np.float64)
    if x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Invalid features or direct target')
    if np.any(y<math.log(.5)-1e-12) or np.any(y>math.log(3)+1e-12):raise ValueError('Scale target out of bounds')
    return x,y,np.ones(len(rows),dtype=np.float64)


def fit(train_path,validation_path,output,events):
    train=bundle(train_path,'train');val=bundle(validation_path,'validation')
    if {r['source_group'] for r in train['rows']} & {r['source_group'] for r in val['rows']}:
        raise ValueError('Document groups overlap training and validation')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    receipt={'at':utc(),'device':'cpu','contract':'direct_log_scale_v1','models':{},
             'numpy_version':np.__version__,'sklearn_version':sklearn.__version__,
             'train_sha256':digest(train_path),'validation_sha256':digest(validation_path),
             'auxiliary_gain_head':False,'runtime_search':False}
    def emit(event,**extra):
        with open(events,'a',encoding='utf-8') as f:f.write(json.dumps({'at':utc(),'event':event,'device':'cpu',**extra})+'\n')
    for kind in ('table','equation'):
        rows=[r for r in train['rows'] if r['kind']==kind and r['target_valid']]
        vrows=[r for r in val['rows'] if r['kind']==kind and r['target_valid']]
        raw,y,w=arrays(rows);vraw,vy,vw=arrays(vrows)
        pre=preprocessing(raw,w);x=transform(raw,pre);vx=transform(vraw,pre)
        mean=float(np.mean(y));scale=max(float(np.std(y)),1e-12)
        identity={'contract':'direct_log_scale_v1','features':FEATURE_NAMES,'kind':kind,'preprocessor':pre,
                  'training_bundle_sha256':digest(train_path),'validation_bundle_sha256':digest(validation_path),
                  'region_weight':1,'rows':len(rows),'source_groups':len({r['source_group'] for r in rows}),
                  'row_order_sha256':hashlib.sha256(json.dumps([r['region_id'] for r in rows]).encode()).hexdigest()}
        for family in ('gbdt','mlp'):
            started=time.monotonic();emit('controller_fit_started',family=family,kind=kind,rows=len(rows),contract='direct_log_scale_v1')
            def hard_stop():
                emit('controller_fit_failed',family=family,kind=kind,error='Hard two-hour fit ceiling');os._exit(124)
            timer=threading.Timer(7200,hard_stop);timer.daemon=True;timer.start()
            try:
                if family=='gbdt':
                    model=HistGradientBoostingRegressor(**GBDT).fit(x,y,sample_weight=w)
                    checkpoint={**identity,'family':family,'parameters':GBDT,'baseline':float(model._baseline_prediction[0,0]),'trees':trees(model)}
                    expected=model.predict(vx)
                    log={'train_log_scale_mse':float(np.mean((model.predict(x)-y)**2)),
                         'validation_log_scale_mse':float(np.mean((expected-vy)**2))}
                else:
                    net,log=fit_mlp(x,(y-mean)/scale,w,vx,(vy-mean)/scale,vw,started+7200)
                    checkpoint={**identity,'family':family,'parameters':MLP,'target_mean':mean,'target_scale':scale,
                                'layers':[{'weight':net[i].tolist(),'bias':net[i+1].tolist()} for i in range(0,len(net),2)]}
                    expected=forward(net,vx)[0]*scale+mean
                    log['validation_log_scale_mse']=float(np.mean((expected-vy)**2))
                path=output/(family+'_'+kind+'.json');atomic_json(path,checkpoint)
                deployed=DirectScaleModel(path,digest(path));observed=np.asarray(deployed.predict(vraw.tolist()))
                difference=float(np.max(np.abs(observed-expected)))
                if difference>1e-10:raise RuntimeError('Saved controller prediction differs from fit')
                atomic_json(output/(family+'_'+kind+'_LOG.json'),log)
                result={**identity,'sha256':digest(path),'elapsed_seconds':time.monotonic()-started,
                        'deployment_max_abs_error':difference,'validation_log_scale_mse':log['validation_log_scale_mse']}
                receipt['models'][family+'_'+kind]=result
                emit('controller_fit_completed',family=family,kind=kind,checkpoint_sha256=digest(path),elapsed_seconds=result['elapsed_seconds'])
            except BaseException as exc:
                emit('controller_fit_failed',family=family,kind=kind,error=type(exc).__name__+': '+str(exc));raise
            finally:timer.cancel()
    atomic_json(output/'TRAINING_RECEIPT.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--validation',required=True)
    p.add_argument('--output',required=True);p.add_argument('--phase-events',required=True);a=p.parse_args()
    print(json.dumps(fit(a.train,a.validation,a.output,a.phase_events),indent=2))
