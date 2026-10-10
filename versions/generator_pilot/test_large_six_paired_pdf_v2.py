"""Preserve measured paired counts while qualifying inherited PDF failures."""
import copy
import unittest

from .large_six_paired_pdf_v2 import qualify


def inputs():
    models = [f'V8.{recipe}/seed{seed}' for recipe in (1, 2, 3) for seed in (0, 1)]
    pairs = [(model, control) for model in models for control in ('base', 'legacy_simple_rule')]
    pairs += [(f'{left}/seed{seed}', f'{right}/seed{seed}')
              for left, right in [('V8.2', 'V8.1'), ('V8.3', 'V8.2')] for seed in (0, 1)]
    names = [left + ' - ' + right for left, right in pairs]
    affected = {'V8.1/seed0', 'V8.2/seed0'}
    aggregate = dict(distinct_pages=1651,
        versions={name: {} for name in models + ['base', 'legacy_simple_rule']},
        comparisons={name: dict(affected_by_retained_PDF_failures=any(x in affected for x in name.split(' - '))) for name in names},
        execution_qualification=dict(inherited_PDF_failures=322, affected_models=sorted(affected),
                                     metric_values_changed=False, page_denominators_changed=False))
    paired = dict(schema_version=1, page_denominator=1651,
        raw_GT_or_prediction_bodies_included=False, descriptive_only=True,
        comparisons={name: dict(page_outcomes={'table_TEDS': {'outcomes': dict(improved=11, worsened=23, tied=19, unmatchable=1598)}})
                     for name in names})
    return paired, aggregate


class QualificationTests(unittest.TestCase):
    def test_qualifies_affected_comparisons_without_changing_counts_or_input(self):
        paired, aggregate = inputs()
        original = copy.deepcopy(paired)
        value = qualify(paired, aggregate)
        self.assertEqual(paired, original)
        self.assertEqual(sum(row['affected_by_retained_PDF_failures'] for row in value['comparisons'].values()), 6)
        for name, row in value['comparisons'].items():
            self.assertEqual(row['page_outcomes'], paired['comparisons'][name]['page_outcomes'])
        self.assertFalse(value['paired_outcome_values_changed'])
        self.assertFalse(value['sample_correspondence_changed'])

    def test_rejects_denominator_membership_or_qualification_mismatch(self):
        for mutation in ('denominator', 'membership', 'qualification', 'retained'):
            paired, aggregate = inputs()
            if mutation == 'denominator': paired['page_denominator'] = 1490
            elif mutation == 'membership': paired['comparisons'].pop(next(iter(paired['comparisons'])))
            elif mutation == 'qualification':
                aggregate['comparisons']['V8.3/seed0 - V8.2/seed0']['affected_by_retained_PDF_failures'] = False
            else: aggregate['execution_qualification']['inherited_PDF_failures'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualify(paired, aggregate)


if __name__ == '__main__':
    unittest.main()
