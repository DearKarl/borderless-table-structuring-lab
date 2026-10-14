"""A complete denominator and independent per-version selection are mandatory."""
import unittest
from .select import select

class SelectionTests(unittest.TestCase):
    def endpoints(self):
        rows=[]
        for version,values in [('V8.4',[.6,.8,.8]),('V8.5',[.9,.85,.8])]:
            for epoch,value in enumerate(values,1):
                rows.append(dict(version=version,checkpoint_sha256=version+str(epoch),checkpoint_path='unused',score_sha256='score',
                    score=dict(status='complete',cohort='dev',denominator=256,epoch=epoch,rows=[dict(id=str(i)) for i in range(256)],
                        aggregate=dict(teds=value),inference_status='complete')))
        return rows
    def test_each_version_uses_own_best_and_earliest_tie(self):
        chosen=select(self.endpoints())
        self.assertEqual(chosen['V8.4']['epoch'],2);self.assertEqual(chosen['V8.5']['epoch'],1)
    def test_missing_endpoint_or_source_is_not_dropped(self):
        rows=self.endpoints()
        with self.assertRaises(ValueError):select(rows[:-1])
        rows[0]['score']['rows'].pop()
        with self.assertRaises(ValueError):select(rows)

if __name__=='__main__':unittest.main()
