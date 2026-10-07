"""One direct log-scale prediction per native region; no runtime scale search."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


class DirectScaleModel:
    def __init__(self, checkpoint, expected_sha256):
        raw=Path(checkpoint).read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest()
        if self.sha256!=expected_sha256:raise RuntimeError('Frozen controller checkpoint differs')
        self.model=json.loads(raw)
        if self.model['contract']!='direct_log_scale_v1' or self.model['features']!=FEATURE_NAMES:
            raise ValueError('Not a direct-scale checkpoint with native-only features')
        self.calls=0;self.rows_predicted=0

    def predict(self,rows):
        self.calls+=1;self.rows_predicted+=len(rows)
        x=np.asarray(rows,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():
            raise ValueError('Invalid native-region feature vector')
        pre=self.model['preprocessor'];x=(x-np.asarray(pre['mean']))/np.asarray(pre['scale'])
        if self.model['family']=='mlp':
            for i,layer in enumerate(self.model['layers']):
                x=x@np.asarray(layer['weight'])+np.asarray(layer['bias'])
                if i+1<len(self.model['layers']):x=np.maximum(x,0)
            return (x[:,0]*self.model['target_scale']+self.model['target_mean']).tolist()
        if self.model['family']!='gbdt':raise ValueError('Unknown controller family')
        out=[]
        for row in x:
            score=self.model['baseline']
            for tree in self.model['trees']:
                i=0
                while not tree[i]['is_leaf']:
                    node=tree[i];i=node['left'] if row[node['feature_idx']]<=node['num_threshold'] else node['right']
                score+=tree[i]['value']
            out.append(float(score))
        return out

    def choose(self,features,width,height):
        if len(features)!=len(FEATURE_NAMES) or not all(math.isfinite(v) for v in features):
            return {'action':'no_change','reason':'invalid_features','scale':None,'controller_calls':0}
        raw=self.predict([list(features)])[0]
        if not math.isfinite(raw):
            return {'action':'no_change','reason':'invalid_prediction','scale':None,'controller_calls':1}
        bounded=min(math.log(3.),max(math.log(.5),raw));scale=math.exp(bounded)
        scale=min(3.,max(.5,scale))
        size=[max(1,round(width*scale)),max(1,round(height*scale))]
        feasible=size[0]*size[1]<=64000000
        return {'action':'reread' if feasible else 'no_change','reason':'direct_scale' if feasible else 'pixel_limit',
                'raw_log_scale':raw,'bounded_log_scale':bounded,'scale':scale,'clipped':raw!=bounded,
                'requested_dimensions':size,'controller_calls':1,'scale_head_calls':1,'gain_head_calls':0,
                'runtime_search':False,'model_sha256':self.sha256}


def direct_target(native_quality,candidates):
    """One region/one target from saved external measurements, including no benefit."""
    valid=[r for r in candidates if r.get('accepted') and r.get('quality_valid')
           and math.isfinite(r.get('quality',float('nan')))]
    if not valid:return {'target_valid':False,'reason':'no_valid_measured_reread'}
    best=max(r['quality'] for r in valid)
    tied=[r for r in valid if best-r['quality']<=1e-6]
    chosen=min(tied,key=lambda r:(abs(math.log(r['scale'])),r['scale']))
    native_valid=native_quality is not None and math.isfinite(native_quality)
    return {'target_valid':True,'target_log_scale':math.log(chosen['scale']),'target_scale':chosen['scale'],
            'target_candidate_index':chosen['index'],'best_measured_quality':best,
            'chosen_measured_quality':chosen['quality'],'best_gain_over_native':best-native_quality if native_valid else None,
            'non_beneficial':best<=native_quality+1e-6 if native_valid else None,'tied_candidate_indices':[r['index'] for r in tied],
            'distinct_tied_inputs':len({r['realized']['identity'] for r in tied}),
            'ties_span_log_scale':max(math.log(r['scale']) for r in tied)-min(math.log(r['scale']) for r in tied),
            'optimum_scope':'finite empirical measurements; not certified global or continuous optimum'}
