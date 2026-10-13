"""Qualify saved paired outcomes without changing correspondence or metrics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .large_six_summary_pdf_v2 import WARNING


def qualify(paired, aggregate):
    qualification = aggregate['execution_qualification']
    affected = {'V8.1/seed0', 'V8.2/seed0'}
    if (aggregate['distinct_pages'] != 1651
            or qualification['inherited_PDF_failures'] != 322
            or set(qualification['affected_models']) != affected
            or qualification['metric_values_changed'] is not False
            or qualification['page_denominators_changed'] is not False):
        raise ValueError('Verified full-denominator execution qualification required')
    if (paired['page_denominator'] != 1651
            or paired['raw_GT_or_prediction_bodies_included'] is not False
            or paired['descriptive_only'] is not True
            or set(paired['comparisons']) != set(aggregate['comparisons'])
            or len(paired['comparisons']) != 16):
        raise ValueError('Saved paired outcomes and qualified aggregate differ')
    value = copy.deepcopy(paired)
    for name, comparison in value['comparisons'].items():
        members = name.split(' - ')
        if len(members) != 2 or any(member not in aggregate['versions'] for member in members):
            raise ValueError('Unknown paired comparison member')
        affected_pair = any(member in affected for member in members)
        if aggregate['comparisons'][name]['affected_by_retained_PDF_failures'] != affected_pair:
            raise ValueError('Aggregate comparison qualification differs')
        comparison['affected_by_retained_PDF_failures'] = affected_pair
        if affected_pair:
            comparison['execution_interpretation'] = WARNING
    value['schema_version'] = 2
    value['execution_qualification'] = copy.deepcopy(qualification)
    value['paired_outcome_values_changed'] = False
    value['sample_correspondence_changed'] = False
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired', required=True)
    parser.add_argument('--aggregate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    raw_paired = Path(args.paired).read_bytes()
    raw_aggregate = Path(args.aggregate).read_bytes()
    value = qualify(json.loads(raw_paired), json.loads(raw_aggregate))
    value['unqualified_paired_source_sha256'] = hashlib.sha256(raw_paired).hexdigest()
    value['qualification_source_aggregate_sha256'] = hashlib.sha256(raw_aggregate).hexdigest()
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(comparisons=16, retained_PDF_failures=322,
                          paired_outcome_values_changed=False)))


if __name__ == '__main__':
    main()
