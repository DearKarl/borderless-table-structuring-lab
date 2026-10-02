"""Bind a complete tensor upper bound to this exact deployed runtime."""
from pathlib import Path
from .cache import checked, read


def identity(arm, artifact):
    assert type(artifact['bytes']) is int and artifact['bytes'] >= 0
    return arm, artifact['path'], artifact['sha256'], artifact['bytes']


def decision(row):
    assert row['disposition'] in ('included', 'excluded_non_weight')
    assert row['evidence_artifacts'], 'A coverage decision needs source evidence'
    for evidence in row['evidence_artifacts']:
        checked(evidence)
    return row['disposition'] == 'included'


def validate_capacity(freeze):
    inventory = read(checked(freeze['parameter_inventory']))
    assert inventory['complete'] is True and inventory['unknown_components'] == []
    assert inventory['deployment_coverage_proven'] is True
    assert set(inventory['runtime_bindings']) == {'mineru', 'paddle'}
    runtimes, expected_models = {}, set()
    for arm in ('mineru', 'paddle'):
        expected = freeze['native_bindings'][arm]['runtime_artifact']
        binding = inventory['runtime_bindings'][arm]
        assert binding['runtime_artifact'] == expected, 'Inventory belongs to another runtime'
        runtime = read(checked(expected)); assert runtime['arm'] == arm
        runtimes[arm] = runtime
        actual = {identity(arm, a) for a in runtime['models']}
        assert len(actual) == len(runtime['models'])
        bound = {identity(arm, a) for a in binding['model_artifacts']}
        assert len(bound) == len(binding['model_artifacts']) and bound == actual, 'Model binding incomplete or changed'
        expected_models.update(actual)
    covered_models, included = set(), set()
    for row in inventory['model_coverage']:
        item = identity(row['arm'], row['artifact'])
        assert item in expected_models and item not in covered_models
        covered_models.add(item)
        include = decision(row)
        if Path(row['artifact']['path']).suffix in ('.safetensors', '.pth', '.pt', '.onnx', '.pdiparams', '.pdparams', '.ftz', '.bin'):
            assert include, 'Known model weight may not be excluded from the upper bound'
        if include: included.add(item)
    assert covered_models == expected_models, 'Missing model or auxiliary artifact coverage'
    environment = freeze['environment_coverage_artifacts']
    assert set(environment) == {'mineru', 'paddle'}
    assert inventory['environment_coverage_artifacts'] == environment
    for arm, artifact in environment.items():
        proof = read(checked(artifact))
        assert proof['runtime_artifact'] == freeze['native_bindings'][arm]['runtime_artifact']
        assert proof['environment_freeze_sha256'] == runtimes[arm]['freeze_sha256']
        assert proof['complete'] is True
        discovery = read(checked(proof['discovery_artifact']))
        expected = {identity(arm, a) for a in discovery['files']}
        assert len(expected) == len(discovery['files'])
        seen = set()
        for row in proof['resources']:
            item = identity(arm, row['artifact'])
            assert item in expected and item not in seen and item not in expected_models
            seen.add(item)
            if decision(row): included.add(item)
        assert seen == expected, 'Environment resource lacks an include/exclude decision'
    components = inventory['unique_deployed_components']
    assert components and len({c['identity'] for c in components}) == len(components)
    counted = set()
    for component in components:
        assert type(component['tensor_numel_upper_bound']) is int and component['tensor_numel_upper_bound'] > 0
        # No cross-model sharing deductions: each artifact is conservatively counted.
        assert len(component['covered_artifacts']) == 1
        row = component['covered_artifacts'][0]
        item = identity(row['arm'], row['artifact'])
        assert item in included and item not in counted
        counted.add(item)
        assert component['evidence_artifacts']
        for evidence in component['evidence_artifacts']: checked(evidence)
    assert counted == included, 'Included weight missing from the capacity sum'
    upper = inventory['total_parameter_upper_bound']
    assert type(upper) is int and upper == sum(c['tensor_numel_upper_bound'] for c in components)
    assert upper <= 4_000_000_000
    exact = inventory['total_parameters_exact']
    assert type(inventory['classification_complete']) is bool
    if inventory['classification_complete']:
        assert type(exact) is int and 0 <= exact <= upper, 'Invalid exact classified parameter count'
    else:
        assert exact is None, 'Unclassified exact count must be null'
    return inventory
