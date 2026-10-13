"""Add verified execution qualifications to the unchanged six-model reducer."""
import argparse
import csv
import json
from pathlib import Path

from .large_six_summary import bound, public_identity, read, summarize, write_outputs


WARNING = (
    'V8.1/seed0 and V8.2/seed0 each retain 161 empty technical failures from '
    'the PDF assembly defect. Their full-denominator scores, contrasts and '
    'two-seed means/ranges describe execution outcomes; they are not clean '
    'estimates of model quality or training effects.'
)


def validate_inherited(rows, expected):
    """Require exact failed membership and explicit evidence of no replay."""
    retained = [row for row in rows if row.get('inherited_attempted_failure')]
    if len(retained) != len(expected) or {row['page_id'] for row in retained} != set(expected):
        raise ValueError('Inherited technical-failure membership differs')
    for row in retained:
        output = row['outputs']['candidate']
        if (row['status'] == 'complete' or row.get('inference_replayed') is not False
                or output['empty'] is not True):
            raise ValueError('Prior failed page was replayed or lost its empty disposition')
        prior = expected[row['page_id']]
        for key in ('original_page_id', 'input_sha256', 'model_calls'):
            if row.get(key) != prior.get(key):
                raise ValueError('Prior failed source or call evidence differs')
    return len(retained)


def qualify(value, manifest):
    correction = bound(manifest['execution_correction'])
    evidence = bound(manifest['attempted_failure_evidence'])
    selection = bound(manifest['selection'])
    if (correction['selection_sha256'] != manifest['selection']['sha256']
            or correction['fault_evidence_sha256'] != manifest['attempted_failure_evidence']['sha256']):
        raise ValueError('Correction and frozen attempted-failure evidence differ')
    prior_by_run = {item['manifest']['model_run_id']: item['manifest'] for item in evidence['manifests']}
    if len(evidence['manifests']) != 2 or set(prior_by_run) != {'V7.3.0-seed0', 'V7.3.1-seed0'}:
        raise ValueError('Unexpected corrected model set')
    qualifications = {}
    for model in selection['models']:
        name = public_identity(model)
        raw = bound(manifest['model_inference'][name])
        prior = prior_by_run.get(model['run_id'])
        expected = {}
        if prior is not None:
            if (prior['selection_sha256'] != manifest['selection']['sha256']
                    or prior['checkpoint_sha256'] != model['checkpoint_sha256']):
                raise ValueError('Prior attempt was a different selected model')
            expected = {row['page_id']: row for row in prior['rows']}
            if (len(expected) != 161 or len(prior['rows']) != 161
                    or set(expected) != set(prior['attempted_page_ids'])):
                raise ValueError('Frozen prior failure denominator differs')
        retained = validate_inherited(raw['rows'], expected)
        total_failed = sum(row['status'] != 'complete' for row in raw['rows'])
        qualifications[name] = dict(
            inference_status=raw['status'], inherited_PDF_failures=retained,
            all_incomplete_pages=total_failed, contains_retained_PDF_failures=bool(retained),
            automatic_inference_retry=False,
            interpretation=WARNING if retained else 'No retained failure from the PDF assembly defect; inspect all other execution dispositions separately.')
        value['versions'][name]['execution_qualification'] = qualifications[name]
    if sum(item['inherited_PDF_failures'] for item in qualifications.values()) != 322:
        raise ValueError('Retained PDF failure total differs')
    affected = sorted(name for name, item in qualifications.items() if item['contains_retained_PDF_failures'])
    for pair, comparison in value['comparisons'].items():
        members = pair.split(' - ')
        comparison['affected_by_retained_PDF_failures'] = any(name in affected for name in members)
    for recipe, metrics in value['seed_variability'].items():
        for record in metrics.values():
            record['affected_by_retained_PDF_failures'] = any(name.startswith(recipe + '/') for name in affected)
    value['schema_version'] = 2
    value['execution_qualification'] = dict(
        affected_models=affected, inherited_PDF_failures=322, models=qualifications,
        correction_sha256=manifest['execution_correction']['sha256'],
        prior_failure_evidence_sha256=manifest['attempted_failure_evidence']['sha256'],
        interpretation=WARNING, metric_values_changed=False, page_denominators_changed=False,
        confirmation_results_affected_by_this_PDF_defect=False)
    value['limits'].insert(0, WARNING)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    manifest = read(args.manifest)
    value = qualify(summarize(manifest), manifest)
    output = Path(args.output)
    write_outputs(value, output)
    markdown = output / 'METRICS.md'
    title, rest = markdown.read_text(encoding='utf-8').split('\n', 1)
    markdown.write_text(title + '\n\n> **Execution qualification:** ' + WARNING + '\n' + rest,
                        encoding='utf-8', newline='\n')
    csv_path = output / 'METRICS.csv'
    with csv_path.open(newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames)
        rows = list(reader)
    extra = ['inference_status', 'inherited_PDF_failures', 'all_incomplete_pages']
    with csv_path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields + extra)
        writer.writeheader()
        for row in rows:
            name = row['policy']
            if name in value['execution_qualification']['models']:
                q = value['execution_qualification']['models'][name]
                row.update({key: q[key] for key in extra})
            else:
                statuses = value['inference'][name]['page_status_counts']
                failed = sum(count for status, count in statuses.items() if status != 'complete')
                row.update(inference_status='failed' if failed else 'complete',
                           inherited_PDF_failures=0, all_incomplete_pages=failed)
            writer.writerow(row)
    print(json.dumps(dict(model_evaluations=6, control_policies=2, distinct_pages=1651,
        retained_PDF_failures=322, metric_values_changed=False)))


if __name__ == '__main__':
    main()
