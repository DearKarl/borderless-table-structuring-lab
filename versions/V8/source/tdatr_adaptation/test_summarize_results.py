"""Check reporting denominators, missing metrics, and complete scalar coverage."""
import math
import unittest
from .summarize_results import finite_metrics,paired_confirmation,training_statistics


class ResultReductionTests(unittest.TestCase):
    def test_absent_metrics_stay_absent_and_nonfinite_values_fail(self):
        values=finite_metrics({'table_teds':.8})
        self.assertEqual(values['table_teds'],.8);self.assertIsNone(values['overall'])
        with self.assertRaises(ValueError):finite_metrics({'table_teds':float('nan')})
        with self.assertRaises(ValueError):finite_metrics({'table_teds':1.01})

    def test_paired_confirmation_preserves_all_missing_and_failed_scores(self):
        baseline={str(i):dict(teds=1.,exact_table=True) for i in range(512)}
        candidate={str(i):dict(teds=1. if i else 0.,exact_table=bool(i)) for i in range(512)}
        value=paired_confirmation(candidate,baseline)
        self.assertEqual(value['source_denominator'],512)
        self.assertEqual(value['mean_teds_delta'],-1/512)
        self.assertEqual((value['improved'],value['worsened'],value['tied']),(0,1,511))
        self.assertEqual(value['harmed_exact_tables'],1)
        del candidate['0']
        with self.assertRaises(ValueError):paired_confirmation(candidate,baseline)

    def test_training_statistics_detects_a_dropped_update(self):
        training=dict(status='complete',optimizer_updates=7500,exposures=120000,task_exposures={},diagnostic_seconds=31.,elapsed_seconds=8000.)
        scalars=[dict(step=i,exposures=16*i,stage='detail' if i<=3750 else 'fusion',epoch=((i-1)%3750)//1250+1,
            update_seconds=1.,target_tokens=10,generation_nll_numerator=5.,peak_gpu_allocated_bytes=100,losses={'total':.5}) for i in range(1,7501)]
        diagnostics=[dict(step=i,partition=partition) for i in range(0,7501,250) for partition in ['train','dev']]
        value=training_statistics(training,scalars,diagnostics)
        self.assertEqual(value['stages']['fusion']['exposures'],60000)
        self.assertEqual(value['stages']['detail']['token_weighted_generation_nll'],.5)
        with self.assertRaises(ValueError):training_statistics(training,scalars[:-1],diagnostics)
        with self.assertRaises(ValueError):training_statistics(training,scalars,diagnostics[:-1])


if __name__=='__main__':unittest.main()
