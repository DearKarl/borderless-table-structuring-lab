"""Paired quality does not turn ambiguity/missing metrics into successes."""
import unittest
from .paired_quality import canonical,outcome,paired_samples,sample_key


class PairedQualityContracts(unittest.TestCase):
    def test_metric_direction_ties_and_unknowns(self):
        self.assertEqual(outcome(.2,.1,-1),'improved')
        self.assertEqual(outcome(.9,.8,1),'worsened')
        self.assertEqual(outcome(.5,.5000001,1),'tied')
        self.assertEqual(outcome(None,.9,1),'unmatchable')

    def test_duplicate_ground_truth_correspondence_is_unmatchable(self):
        row={'image_name':'page.png','gt_idx':[1],'gt':'x','metric':{'CDM':1.,'Edit_dist':0.}}
        key=sample_key(row)
        result=paired_samples(([row,row],{key:[row,row]},{}),([row],{key:[row]},{}),'equation')
        self.assertEqual(result['CDM']['outcomes']['unmatchable'],1)
        self.assertEqual(result['CDM']['outcomes']['tied'],0)
        self.assertIsNone(result['CDM']['mean_raw_delta_on_paired_units'])

    def test_canonicalization_preserves_internal_text(self):
        self.assertEqual(canonical('$$ x + y $$','equation'),'x + y')
        self.assertEqual(canonical('<html><body><table>\n<tr><td>a b</td></tr></table></body></html>','table'),'<table><tr><td>a b</td></tr></table>')
        self.assertNotEqual(canonical(r'\text{a b}','equation'),canonical(r'\text{ab}','equation'))


if __name__=='__main__':unittest.main()
