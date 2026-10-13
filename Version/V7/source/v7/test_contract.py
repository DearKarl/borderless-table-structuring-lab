"""Scientific contract edge cases without fitting research models."""
import json,unittest
import numpy as np
from .features import extract,simple_rule,structure,FEATURE_NAMES
from .train import choose_margin,group_folds

class Contract(unittest.TestCase):
    def test_spans(self):
        s=structure('<table><tr><td rowspan="2">a</td><td colspan="2">b</td></tr><tr><td>c</td><td>d</td></tr></table>')
        self.assertEqual((s['rows'],s['columns'],s['cells']),(2,3,4));self.assertTrue(s['valid'])
    def test_unknown_does_not_pass(self):
        f=extract([1.]*17,'<table><tr><td>x','<table><tr><td>x</td></tr></table>',True)
        self.assertEqual(len(f['values']),34);self.assertFalse(f['required_features_known']);self.assertFalse(simple_rule(f))
    def test_empty_duplicate_rule(self):
        n='<table><tr><td>a</td><td>b</td></tr></table>';c=n.replace('>b<','>a<')
        self.assertFalse(simple_rule(extract([1.]*17,n,c,True)))
        self.assertTrue(simple_rule(extract([1.]*17,n,n,True)))
    def test_group_folds(self):
        rows=[{'source_group':str(i//2)} for i in range(20)];folds=group_folds(rows)
        self.assertTrue(all(folds[i]==folds[i+1] for i in range(0,20,2)));self.assertEqual(set(folds),set(range(5)))
    def test_margin_all_keep_and_ties(self):
        rows=[dict(required_features_known=True,common_valid=True,raw_output_changed=True,native_quality=1.,candidate_quality=.9)]
        selected,_=choose_margin(rows,[.1]);self.assertEqual(selected['margin'],'infinity')
        rows[0]['candidate_quality']=1.;selected,_=choose_margin(rows,[.1]);self.assertEqual(selected['margin'],'infinity')
    def test_numpy_margin_receipt_is_serializable(self):
        rows=[dict(required_features_known=True,common_valid=True,raw_output_changed=True,native_quality=.5,candidate_quality=.7)]
        selected,choices=choose_margin(rows,np.asarray([.1]));json.dumps({'selected':selected,'choices':choices},allow_nan=False)

if __name__=='__main__':unittest.main()
