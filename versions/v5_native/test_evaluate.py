"""Scoring completeness and denominator guards, independent of benchmark values."""
import unittest
from .evaluate import metrics,METRIC_PATHS


class MetricContracts(unittest.TestCase):
    def fixture(self):
        raw={'match_debug':{'page_count':1651}}
        for name,path in METRIC_PATHS.items():
            node=raw
            for key in path[:-1]:node=node.setdefault(key,{})
            node[path[-1]]=.2 if name.endswith('_edit') else .9
        return raw

    def test_all_requested_metrics_and_official_overall(self):
        result=metrics(self.fixture())
        self.assertEqual(len(result['values']),8)
        self.assertAlmostEqual(result['values']['overall'],100*(.8+.9+.9)/3)
        self.assertEqual(result['missing_reasons'],{})
        self.assertEqual(result['direction']['formula_edit'],'lower')

    def test_absent_component_is_unknown_and_never_zero(self):
        raw=self.fixture();del raw['table']['page']['TEDS']
        result=metrics(raw)
        self.assertIsNone(result['values']['table_teds']);self.assertIsNone(result['values']['overall'])
        self.assertIn('table_teds',result['missing_reasons'])
        self.assertIsNotNone(result['values']['table_edit'])

    def test_success_only_denominator_is_rejected(self):
        raw=self.fixture();raw['match_debug']['page_count']=1650
        with self.assertRaises(ValueError):metrics(raw)


if __name__=='__main__':unittest.main()
