"""Reporting guards for retained failures; fixture values are not results."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from .large_six_summary_pdf_v2 import qualify, validate_inherited


def fixture():
    selection = {'models': []}
    prior = []
    manifest = dict(selection=selection, execution_correction=dict(
        selection_sha256='selected', fault_evidence_sha256='prior'),
        attempted_failure_evidence=dict(sha256='prior', manifests=prior), model_inference={})
    selection['sha256'] = 'selected'
    manifest['execution_correction']['sha256'] = 'correction'
    value = dict(versions={}, comparisons={}, seed_variability={}, limits=[])
    for recipe in (1, 2, 3):
        public = f'V8.{recipe}'
        value['seed_variability'][public] = {'metric': {'mean': 0.5}}
        for seed in (0, 1):
            run = f'V7.3.{recipe-1}-seed{seed}'
            name = f'{public}/seed{seed}'
            model = dict(run_id=run, public_recipe=public, seed=seed, checkpoint_sha256='weights')
            selection['models'].append(model)
            rows = []
            if recipe in (1, 2) and seed == 0:
                for i in range(161):
                    rows.append(dict(page_id=f'fixture-{i}', original_page_id=f'source-{i}', input_sha256='input',
                        model_calls=0, status='failed', inherited_attempted_failure=True, inference_replayed=False,
                        outputs={'candidate': {'empty': True}}))
                prior.append(dict(manifest=dict(model_run_id=run, selection_sha256='selected',
                    checkpoint_sha256='weights', rows=deepcopy(rows), attempted_page_ids=[r['page_id'] for r in rows])))
            manifest['model_inference'][name] = dict(status='failed' if rows else 'complete', rows=rows)
            value['versions'][name] = {'values': {'metric': 0.5}}
            value['comparisons'][name+' - base'] = {'metrics': {'metric': {'raw_left_minus_right': 0.1}}}
    return value, manifest


class PdfReportingTests(unittest.TestCase):
    def test_marks_affected_contrasts_and_seed_summaries_without_changing_values(self):
        value, manifest = fixture()
        original = deepcopy(value)
        with patch('versions.generator_pilot.large_six_summary_pdf_v2.bound', side_effect=lambda item: item):
            actual = qualify(value, manifest)
        self.assertEqual(actual['execution_qualification']['inherited_PDF_failures'], 322)
        for name in actual['versions']:
            self.assertEqual(actual['versions'][name]['values'], original['versions'][name]['values'])
        for name in actual['comparisons']:
            self.assertEqual(actual['comparisons'][name]['metrics'], original['comparisons'][name]['metrics'])
        self.assertTrue(actual['comparisons']['V8.1/seed0 - base']['affected_by_retained_PDF_failures'])
        self.assertFalse(actual['comparisons']['V8.1/seed1 - base']['affected_by_retained_PDF_failures'])
        self.assertTrue(actual['seed_variability']['V8.2']['metric']['affected_by_retained_PDF_failures'])
        self.assertFalse(actual['seed_variability']['V8.3']['metric']['affected_by_retained_PDF_failures'])

    def test_replayed_or_nonempty_prior_failure_is_rejected(self):
        _, manifest = fixture()
        rows = manifest['model_inference']['V8.1/seed0']['rows']
        expected = {r['page_id']: deepcopy(r) for r in rows}
        rows[0]['inference_replayed'] = True
        with self.assertRaisesRegex(ValueError, 'replayed'): validate_inherited(rows, expected)
        rows[0]['inference_replayed'] = False
        rows[0]['outputs']['candidate']['empty'] = False
        with self.assertRaisesRegex(ValueError, 'empty disposition'): validate_inherited(rows, expected)

    def test_missing_or_unexpected_inherited_pages_are_rejected(self):
        _, manifest = fixture()
        rows = manifest['model_inference']['V8.1/seed0']['rows']
        expected = {r['page_id']: deepcopy(r) for r in rows}
        with self.assertRaisesRegex(ValueError, 'membership'): validate_inherited(rows[:-1], expected)
        with self.assertRaisesRegex(ValueError, 'membership'): validate_inherited(rows, {})


if __name__ == '__main__':
    unittest.main()
