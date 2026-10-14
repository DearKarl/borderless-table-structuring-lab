"""Bounded recovery of a verified failure before any model call.

This fixes the recording callback's keyword interface. It does not alter
processing, generation, checkpoints, selection, or page membership.
"""
import hashlib
import json
from pathlib import Path

ERROR = "main.<locals>.observed() missing 3 required positional arguments: 'images', 'prompts', and 'parameters'"
IDENTITIES = ('version', 'checkpoint_sha256', 'benchmark_inputs_sha256',
              'shared_selection_sha256', 'shared_cache_index_sha256')


def native_keyword_adapter(callback):
    """Accept the pinned native backend's actual keyword names unchanged."""
    def observed(image_objs, chat_prompts, sampling_params, **kwargs):
        return callback(image_objs, chat_prompts, sampling_params, **kwargs)
    return observed


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_prior(binding, pages):
    """Validate the diagnosis retained after user-authorized output deletion."""
    path = Path(binding['recovery_plan_path'])
    if digest(path) != binding['recovery_plan_sha256']:
        raise ValueError('Frozen recovery diagnosis changed')
    plan = json.loads(path.read_text())
    prior = plan['versions'][binding['version']]
    if prior['status'] != 'failed' or prior['model_calls'] != 0:
        raise ValueError('Recovery requires zero original model calls')
    if prior['full_denominator'] != 1651 or len(prior['pages']) != 1651:
        raise ValueError('Original denominator differs')
    if any(prior[key] != binding[key] for key in IDENTITIES):
        raise ValueError('Recovery changes a frozen identity')
    if prior['selection_sha256'] != binding['nomination_sha256']:
        raise ValueError('Recovery changes DEV nomination')
    if prior['benchmark_sha256'] != binding['recovery_from_benchmark_sha256']:
        raise ValueError('Original failed attempt identity differs')
    if prior['first_generation_receipt_exists'] or prior['calls_file_exists']:
        raise ValueError('Original model call evidence prohibits this recovery')
    if [row['page_id'] for row in prior['pages']] != [row['page_id'] for row in pages]:
        raise ValueError('Recovery changes page order or membership')
    successful = failed = 0
    by_page = {}
    for old, page in zip(prior['pages'], pages):
        if old['input_sha256'] != page['input_sha256'] or old['model_calls'] != 0:
            raise ValueError('Original page identity or call count differs')
        if old['status'] == 'complete':
            if old['predicted_tables'] != 0:
                raise ValueError('Successful model page must never be repeated')
            successful += 1
        elif old['status'] == 'failed' and old['error_type'] == 'TypeError' and old['reason'] == ERROR:
            failed += 1
        else:
            raise ValueError('Recovery includes an unregistered failure')
        by_page[old['page_id']] = old
    if (successful, failed) != (1189, 462):
        raise ValueError('Recovery cohort differs from diagnosed failure')
    return by_page
