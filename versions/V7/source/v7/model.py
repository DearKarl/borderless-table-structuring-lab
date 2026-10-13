"""Portable frozen net-TEDS-gain tree inference and strict acceptance margin."""
import hashlib,json,math
from pathlib import Path
from .features import FEATURE_NAMES


class NetGainModel:
    def __init__(self,path,expected_sha256):
        raw=Path(path).read_bytes();self.sha256=hashlib.sha256(raw).hexdigest()
        if self.sha256!=expected_sha256:raise ValueError('checkpoint_hash_mismatch')
        self.model=json.loads(raw)
        if self.model['contract']!='v7_fixed_table_net_gain_v1' or self.model['features']!=FEATURE_NAMES:raise ValueError('feature_contract_mismatch')
        self.calls=0;self.rows_predicted=0

    def predict(self,rows):
        self.calls+=1;self.rows_predicted+=len(rows);out=[]
        for row in rows:
            if len(row)!=len(FEATURE_NAMES) or not all(math.isfinite(x) for x in row):raise ValueError('invalid_features')
            score=self.model['baseline']
            for tree in self.model['trees']:
                i=0
                while not tree[i]['is_leaf']:
                    n=tree[i];i=n['left'] if row[n['feature_idx']]<=n['num_threshold'] else n['right']
                score+=tree[i]['value']
            out.append(float(score))
        return out

    def decide(self,features):
        if not features['common_valid'] or not features['required_features_known']:
            return {'accepted':False,'predicted_gain':None,'reason':'common_invalid_or_unknown_features','model_calls':0,'model_rows':0}
        prediction=self.predict([features['values']])[0]
        margin=self.model['margin'];threshold=math.inf if margin=='infinity' else float(margin)
        return {'accepted':bool(math.isfinite(prediction) and prediction>threshold),
            'predicted_gain':prediction,'margin':margin,'reason':'strict_predicted_gain_margin',
            'model_calls':1,'model_rows':1}
