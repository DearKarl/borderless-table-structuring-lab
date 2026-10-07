"""Meaningful actor routing, physical reuse and interrupted-output contracts."""
import hashlib,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

from .actor import DirectActors
from .features import FEATURE_NAMES
from .io_utils import atomic_json,commit_stage,digest,finalize_outputs,read_json


class Observer:
    def __init__(self):self.calls=0
    def reread(self,view,kind,preview):
        self.calls+=1
        return {'output':'x+1','physical_requests':1,'ocr_seconds':.01,'actual_tensor_identity_verified':True,'request_ids':['mock']}


def preview(client,image,kind):
    identity=hashlib.sha256((kind+str(image.size)).encode()+image.tobytes()).hexdigest()
    return {'identity':identity,'tensors':{},'crop':{'size':list(image.size)}}


class ActorContracts(unittest.TestCase):
    def bindings(self,root):
        result={};n=len(FEATURE_NAMES)
        for family in ('gbdt','mlp'):
            for kind in ('table','equation'):
                m={'contract':'direct_log_scale_v1','features':FEATURE_NAMES,'family':family,'kind':kind,
                   'preprocessor':{'mean':[0]*n,'scale':[1]*n}}
                if family=='gbdt':m.update(baseline=math.log(1.173),trees=[])
                else:m.update(layers=[{'weight':[[0] for _ in range(n)],'bias':[math.log(1.173)]}],target_mean=0,target_scale=1)
                p=root/(family+'_'+kind+'.json');atomic_json(p,m);result[family+'_'+kind]={'path':str(p),'sha256':digest(p)}
        return result

    def test_matched_actors_predict_once_and_share_one_physical_reread(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d,patch('versions.v5_native.actor.realized_view',side_effect=preview):
            observer=Observer();actors=DirectActors(None,observer,self.bindings(Path(d)))
            crop=Image.new('RGB',(100,40),'white');prepared=actors.prepare(crop,'x','equation')
            a=actors.apply(crop,'x','equation',prepared,'gbdt',0)
            b=actors.apply(crop,'x','equation',prepared,'mlp',0)
            self.assertEqual(observer.calls,1)
            self.assertEqual(a['physical_requests'],1);self.assertEqual(b['physical_requests'],0);self.assertTrue(b['reused'])
            for row in (a,b):
                self.assertEqual(row['predict_api_calls'],1);self.assertEqual(row['predicted_rows'],1)
                self.assertAlmostEqual(row['decision']['scale'],1.173)
                self.assertTrue(row['input_changed_vs_1x']);self.assertTrue(row['canonical_output_changed'])
            self.assertEqual(a['output'],b['output'])

    def test_family_task_binding_and_invalid_features_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d,patch('versions.v5_native.actor.realized_view',side_effect=preview):
            bindings=self.bindings(Path(d));observer=Observer();actors=DirectActors(None,observer,bindings)
            crop=Image.new('RGB',(100,40),'white');prepared=actors.prepare(crop,'x','equation');prepared['features'][0]=float('nan')
            row=actors.apply(crop,'x','equation',prepared,'gbdt',0)
            self.assertEqual(row['predict_api_calls'],0);self.assertEqual(observer.calls,0)
            bindings['gbdt_table']=bindings['mlp_table']
            with self.assertRaises(ValueError):DirectActors(None,observer,bindings)

    def test_hard_stop_keeps_each_versions_last_atomic_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d:
            root=Path(d);folder=root/'pages'/'p';folder.mkdir(parents=True)
            commit_stage(folder,'native',{'value':'native'},'native')
            for version in ('D0','GBDT_V6.1.1','MLP_V6.2.1'):
                commit_stage(folder/'arms'/version,'native',{'value':'native'},'native')
            commit_stage(folder/'arms'/'GBDT_V6.1.1','direct_actor',{'value':'changed'},'changed')
            record={'status':'hard_timeout','worker_id':'test'}
            finalize_outputs(folder,root,'p',record,('D0','GBDT_V6.1.1','MLP_V6.2.1'))
            self.assertEqual((root/'arms/GBDT_V6.1.1/markdown/p.md').read_text(),'changed')
            self.assertEqual((root/'arms/MLP_V6.2.1/markdown/p.md').read_text(),'native')
            self.assertEqual(len(read_json(root/'receipts/p.json')['arm_receipts']),3)

    def test_failed_ocr_preserves_prediction_and_submission_evidence(self):
        class FailedObserver(Observer):
            def __init__(self):
                super().__init__();self.events=[];self.request_ids=[];self.submitted_request_ids=[]
            def emit(self,row):self.events.append(row)
            def reread(self,view,kind,preview):
                self.request_ids.append('rejected-prompt')
                raise ValueError('Context length rejected before model submission')
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as d,patch('versions.v5_native.actor.realized_view',side_effect=preview):
            observer=FailedObserver();actors=DirectActors(None,observer,self.bindings(Path(d)))
            crop=Image.new('RGB',(100,40),'white');prepared=actors.prepare(crop,'x','equation')
            with self.assertRaises(ValueError):actors.apply(crop,'x','equation',prepared,'gbdt',0)
            predicted=[x for x in observer.events if x['event']=='controller_evaluated']
            failed=[x for x in observer.events if x['event']=='regional_ocr_failed']
            self.assertEqual(predicted[0]['predicted_rows'],1)
            self.assertEqual(failed[0]['attempted_requests'],1)
            self.assertEqual(failed[0]['submitted_requests'],0)
            self.assertEqual(actors.bank,{})


if __name__=='__main__':unittest.main()
