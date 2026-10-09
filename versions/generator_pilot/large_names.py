"""Public V8 names for the six existing large-campaign fits.

Internal recipe keys and immutable evidence retain their historical names.
This module changes presentation and package identity only.
"""
import re

CANONICAL_NAMES = {'V7.3.0': 'V8.1', 'V7.3.1': 'V8.2', 'V7.3.2': 'V8.3'}
INTERNAL_NAMES = {value: key for key, value in CANONICAL_NAMES.items()}
METHODS = {'V8.1': 'Ordinary LoRA', 'V8.2': 'LoRA-GA',
           'V8.3': 'LoRA-GA with cell-location supervision'}


def internal_recipe(recipe):
    if recipe in CANONICAL_NAMES:
        return recipe
    if recipe in INTERNAL_NAMES:
        return INTERNAL_NAMES[recipe]
    raise ValueError('Unknown large-campaign recipe: ' + str(recipe))


def canonical_recipe(recipe):
    return CANONICAL_NAMES[internal_recipe(recipe)]


def identity(recipe, seed):
    if type(seed) is not int or seed not in (0, 1):
        raise ValueError('Only the two registered seeds are valid')
    internal = internal_recipe(recipe)
    canonical = CANONICAL_NAMES[internal]
    return dict(recipe=canonical, run_id=f'{canonical}/seed{seed}', seed=seed,
                historical_alias=internal, internal_run_id=f'{internal}-seed{seed}')


def package_recipe(manifest):
    """Resolve old/new packages and reject inconsistent provenance aliases."""
    recipe = internal_recipe(manifest['recipe'])
    if 'historical_alias' in manifest and manifest['historical_alias'] != recipe:
        raise ValueError('Package canonical name and historical alias disagree')
    if 'run_id' in manifest:
        expected = identity(recipe, manifest['seed'])
        if manifest['run_id'] != expected['run_id'] or manifest.get('internal_run_id') != expected['internal_run_id']:
            raise ValueError('Package run identities disagree')
    return recipe


def public_labels(value):
    """Return an aggregate-only presentation copy; never rewrite source evidence."""
    if isinstance(value, dict):
        return {public_labels(key): public_labels(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_labels(item) for item in value]
    if isinstance(value, str):
        for old, new in CANONICAL_NAMES.items():
            value = re.sub(r'(?<![A-Za-z0-9.])' + re.escape(old) + r'-seed([01])(?![0-9])',
                           lambda match: new + '/seed' + match.group(1), value)
            value = re.sub(r'(?<![A-Za-z0-9.])' + re.escape(old) + r'(?![A-Za-z0-9.])', new, value)
    return value


def mapping():
    return dict(scope='Naming only; the same six fits, not additional experiments',
                canonical_names=dict(CANONICAL_NAMES), methods=dict(METHODS),
                fits=[identity(recipe, seed) for recipe in CANONICAL_NAMES for seed in (0, 1)],
                immutable_runtime_paths_and_evidence_preserved=True)
