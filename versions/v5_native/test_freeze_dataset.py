"""Prevent split leakage, test-driven controls, and quality-based row deletion."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from .direct_controller import direct_target
from .features import FEATURE_NAMES
from .freeze_dataset import freeze
from .io_utils import atomic_json,read_json


class DatasetContracts(unittest.TestCase):
    def fixture(self,root,test_quality=1.):
        pool={'GT_access':False,'rows':[]};scores={'scope':'external_only','completed_at':'fixture','rows':[]}
        for kind in ('table','equation'):
            for split in ('train','validation','test'):
                rid=kind+'-'+split
                sample={'region_id':rid,'kind':kind,'split':split,'source_group':rid,'features':[1.]*len(FEATURE_NAMES),
                        'unit_contract':'source_page_native_predicted_region_v1','input_sha256':rid,
                        'native_pixel_identity':{'pixel_sha256':rid},'native_capture_pixel_sha256':rid+'page'}
                pool['rows'].append(sample)
                quality=.4 if split=='train' else (.9 if split=='validation' else test_quality)
                candidates=[{'index':0,'scale':1.,'accepted':True,'quality_valid':True,'quality':.4,'realized':{'identity':'a'}},
                            {'index':1,'scale':1.25,'accepted':True,'quality_valid':True,'quality':quality,'realized':{'identity':'b'}}]
                scores['rows'].append({k:sample[k] for k in ('region_id','kind','split','source_group','features')}|
                                     {'baseline_metrics':{'quality_valid':True,'quality':.5},'candidates':candidates,
                                      'measurement_probe_sha256':rid}|direct_target(.5,candidates))
        atomic_json(root/'pool.json',pool);atomic_json(root/'scores.json',scores)
        return pool,scores

    def run_freeze(self,root,name):
        return freeze([root/'pool.json'],[root/'scores.json'],root/name,root/'events.jsonl')

    def test_non_beneficial_training_examples_remain(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            root=Path(d);self.fixture(root);result=self.run_freeze(root,'frozen')
            self.assertEqual(result['valid_targets'],6);self.assertEqual(result['non_beneficial_retained'],2)
            self.assertEqual(len(read_json(root/'frozen/train.json')['rows']),2)
            self.assertEqual(result['split_sha256'].keys(),{'train','validation','test'})

    def test_fixed_control_is_independent_of_test_scores(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            root=Path(d);self.fixture(root,1.);self.run_freeze(root,'first')
            self.fixture(root,0.);self.run_freeze(root,'second')
            a=read_json(root/'first/FIXED_CONTROLS.json');b=read_json(root/'second/FIXED_CONTROLS.json')
            self.assertEqual(a,b);self.assertEqual(a['scales'],{'table':1.25,'equation':1.25})

    def test_document_crossing_split_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            root=Path(d);pool,scores=self.fixture(root)
            pool['rows'][2]['source_group']=pool['rows'][0]['source_group']
            scores['rows'][2]['source_group']=pool['rows'][0]['source_group']
            atomic_json(root/'pool.json',pool);atomic_json(root/'scores.json',scores)
            with self.assertRaisesRegex(ValueError,'crosses split'):self.run_freeze(root,'frozen')


if __name__=='__main__':unittest.main()
