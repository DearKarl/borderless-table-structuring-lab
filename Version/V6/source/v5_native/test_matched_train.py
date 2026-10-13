"""Numerical checks on synthetic fixtures; no research fit or real labels."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from .controller import ResponseModel
from .features import FEATURE_NAMES
from .io_utils import digest
from .matched_train import (MLP, arrays, fit_mlp, forward, initialize,
                            load_bundle, loss_gradient, preprocessing, transform)


class MatchedRegressionContracts(unittest.TestCase):
    def test_weighted_gradient_matches_finite_difference(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(7, 18)); y = rng.normal(size=7); w = rng.uniform(.1, 1, size=7)
        net = initialize(18, rng)
        _, gradients = loss_gradient(net, x, y, w)
        for layer in range(len(net)):
            index = (0, 0) if net[layer].ndim == 2 else (0,)
            original = net[layer][index]
            epsilon = 1e-6
            net[layer][index] = original + epsilon
            plus = loss_gradient(net, x, y, w)[0]
            net[layer][index] = original - epsilon
            minus = loss_gradient(net, x, y, w)[0]
            net[layer][index] = original
            self.assertAlmostEqual((plus-minus)/(2*epsilon), gradients[layer][index], places=6)

    def test_each_region_has_unit_weight_and_aliases_are_rejected(self):
        rows = [{"kind":"equation","region_id":r,"realized_identity":str(i),
                 "features":[0]*len(FEATURE_NAMES),"scale":1.,"gain":0.}
                for r,count in [('a',2),('b',5)] for i in range(count)]
        _, _, w = arrays(rows)
        self.assertAlmostEqual(sum(w[:2]),1.)
        self.assertAlmostEqual(sum(w[2:]),1.)
        with self.assertRaises(ValueError):
            arrays(rows+[rows[0]])

    def test_mlp_checkpoint_load_matches_with_training_only_preprocessing(self):
        rng = np.random.default_rng(5)
        x=rng.normal(size=(32,18));x[0,2]=np.nan
        vx=rng.normal(size=(12,18));vx[:,0]+=4
        w=np.ones(32);vw=np.ones(12)
        pre=preprocessing(x,w)
        clean=transform(x,pre);vclean=transform(vx,pre)
        y=clean[:,0];vy=vclean[:,0]
        params={**MLP,'max_epochs':5,'patience':3}
        net,log=fit_mlp(clean,y,w,vclean,vy,vw,time.monotonic()+20,params)
        self.assertGreaterEqual(log['selected_epoch'],1)
        self.assertLessEqual(log['selected_epoch'],log['epochs_completed'])
        checkpoint={'family':'mlp','features':FEATURE_NAMES+['log_scale'],'preprocessor':pre,
                    'target_mean':.3,'target_scale':2.,
                    'layers':[{'weight':net[i].tolist(),'bias':net[i+1].tolist()} for i in range(0,6,2)]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            p=Path(d)/'synthetic.json';p.write_text(json.dumps(checkpoint),encoding='utf-8')
            loaded=ResponseModel(p,digest(p))
            np.testing.assert_allclose(loaded.predict(vx.tolist()),forward(net,vclean)[0]*2+.3,atol=1e-12)
            with self.assertRaises(ValueError):loaded.predict([[0]*19])

    def test_label_unit_and_split_isolation(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            p=Path(d)/'invalid.json'
            p.write_text(json.dumps({'provenance':{'scope':'external_only'},'split':'train'}),encoding='utf-8')
            with self.assertRaises(ValueError):load_bundle(p,'train')
            p.write_text(json.dumps({'provenance':{'scope':'external_only','unit_contract':'native_predicted_region_v1'},'split':'test'}),encoding='utf-8')
            with self.assertRaises(ValueError):load_bundle(p,'train')


if __name__=='__main__':unittest.main()
