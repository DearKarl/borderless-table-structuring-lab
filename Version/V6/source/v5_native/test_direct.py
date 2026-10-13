"""Direct-scale contracts checked without research data or OCR."""
import json,math,pathlib,tempfile,unittest
from .direct_controller import DirectScaleModel,direct_target
from .features import FEATURE_NAMES
from .io_utils import digest


class DirectContracts(unittest.TestCase):
    def test_ties_choose_actual_closest_scale_and_keep_no_benefit(self):
        candidates=[{'accepted':True,'quality_valid':True,'quality':.4,'scale':s,'index':i,
                     'realized':{'identity':str(i)}} for i,s in enumerate((.5,2.,1.13))]
        target=direct_target(.8,candidates)
        self.assertEqual(target['target_scale'],1.13)
        self.assertTrue(target['non_beneficial'])
        self.assertEqual(target['tied_candidate_indices'],[0,1,2])
        self.assertFalse(direct_target(.5,[])['target_valid'])

    def test_one_prediction_no_grid_or_nearest_scale(self):
        checkpoint={'contract':'direct_log_scale_v1','features':FEATURE_NAMES,'family':'gbdt',
                    'baseline':math.log(1.173),'trees':[],
                    'preprocessor':{'mean':[0]*len(FEATURE_NAMES),'scale':[1]*len(FEATURE_NAMES)}}
        with tempfile.TemporaryDirectory(dir=pathlib.Path(__file__).resolve().parent) as d:
            p=pathlib.Path(d)/'synthetic.json';p.write_text(json.dumps(checkpoint),encoding='utf-8')
            model=DirectScaleModel(p,digest(p));action=model.choose([0]*len(FEATURE_NAMES),100,40)
            self.assertAlmostEqual(action['scale'],1.173)
            self.assertEqual(model.calls,1);self.assertEqual(model.rows_predicted,1)
            self.assertFalse(action['runtime_search']);self.assertFalse(hasattr(model,'numeric_scales'))
            self.assertEqual(action['requested_dimensions'],[117,47])
            self.assertEqual(model.choose([float('nan')]*len(FEATURE_NAMES),100,40)['action'],'no_change')
            self.assertEqual(model.calls,1)


if __name__=='__main__':unittest.main()
