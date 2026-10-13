"""Input-only candidates -> frozen policy -> one base Tele call, one deadline.
Normal feature unavailability is a ready=False row. Integrity faults propagate.
"""
from contextlib import contextmanager
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import threading
import time
from . import tele_bridge
from .input_features import bind_preflight_files, extract_features
from .inputs import CAP, image_fingerprint, pdf_geometry, transform_native_image
from .schema import SourceRecord
from .tele_bridge import page_input_hook
from .tele_worker import PageWorker, source_pdf
from .worker_contract import atomic_json, bound_file, digest, file_sha

_SERIAL = threading.Lock()
ROUNDING_FORMULA = 'min(200/72,3500/max(display_size));pdfium_ceil_each_axis'

@contextmanager
def count_calls(obj, name, *, forbidden=False):
    """Count actual calls, including forbidden preparation attempts."""
    original = getattr(obj, name)
    existed = name in vars(obj)
    previous = vars(obj).get(name)
    observed = dict(calls=0)
    def counted(*args, **kwargs):
        observed['calls'] += 1
        if forbidden:
            raise RuntimeError('Recognition attempted during candidate preparation')
        return original(*args, **kwargs)
    setattr(obj, name, counted)
    try:
        yield observed
    finally:
        if existed:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)

def rounding_proof(image_dict, source, page_size, evidence):
    dims, scale = pdf_geometry(page_size, 200)
    continuous = max(page_size) * scale
    if (not evidence or source.source_type != 'raster'
            or list(page_size) != list(source.oriented_size)
            or image_dict['img_pil'].size != dims or max(dims) != CAP + 1
            or image_dict.get('scale') != scale
            or not CAP < continuous <= math.nextafter(float(CAP), math.inf)):
        raise ValueError('Unproven native cap overflow')
    return dict(branch='evidence_bound_raster_single_ulp_ceil',
                evidence_sha256=digest(json.dumps(evidence, sort_keys=True).encode()),
                display_size=list(page_size), expected_size=list(dims),
                before_scale=scale, continuous_long_side=continuous)

def candidate_transform(image_dict, action, source, page_size, evidence=None):
    """Ordinary inputs delegate unchanged. Only proven ceil overflow is adapted."""
    image = image_dict['img_pil']
    proof = None
    if source.source_type == 'raster' and max(image.size) > CAP:
        proof = rounding_proof(image_dict, source, page_size, evidence)
    if proof is not None and action == 'B':
        from PIL import Image
        factor = min(1.5, CAP / max(image.size))
        dims = tuple(max(1, math.floor(x * factor + .5)) for x in image.size)
        result = dict(image_dict, img_pil=image.resize(dims, Image.Resampling.BICUBIC),
                      scale=image_dict['scale'] * factor)
        _, audit = transform_native_image(image_dict, 'A', source, page_size)
        audit.update(action='B', prepared=image_fingerprint(result['img_pil']),
                     scale=result['scale'], cap_noop=False,
                     page_affine_xy=[dims[i] / page_size[i] for i in (0, 1)])
    else:
        result, audit = transform_native_image(image_dict, action, source, page_size)
    audit['before_scale'] = image_dict['scale']
    audit['selector_rounding_proof'] = proof
    return result, audit

@contextmanager
def candidate_boundary(native):
    """Shared preparation/recognition adapter; reviewed loader evidence required.
    No vendor files are changed. Missing evidence leaves overflow fail-closed.
    """
    if not _SERIAL.acquire(blocking=False):
        raise RuntimeError('Concurrent selector boundary')
    original = tele_bridge.transform_native_image
    try:
        evidence = getattr(native, 'selector_rounding_evidence', None)
        if evidence is not None:
            loader = native.analyze.load_images_from_pdf
            path = inspect.getsourcefile(loader)
            if (evidence.get('schema') != 'v4_native_rounding_evidence_v1'
                    or evidence.get('formula') != ROUNDING_FORMULA or path is None
                    or file_sha(path) != evidence.get('loader_file_sha256')
                    or digest(inspect.getsource(loader).encode()) != evidence.get('loader_source_sha256')):
                raise ValueError('Native rounding source evidence mismatch')
            page_to_image = loader.__globals__['pdf_page_to_image'].__globals__['page_to_image']
            for name, function in (('page_to_image', page_to_image), ('pdfium_render', native.pdfium.PdfPage.render)):
                dependency = evidence.get('dependencies', {}).get(name, {})
                if (file_sha(inspect.getsourcefile(function)) != dependency.get('file_sha256')
                        or digest(inspect.getsource(function).encode()) != dependency.get('function_sha256')):
                    raise ValueError('Native rendering dependency evidence mismatch: ' + name)
        tele_bridge.transform_native_image = lambda *a: candidate_transform(*a, evidence=evidence)
        yield
    finally:
        tele_bridge.transform_native_image = original
        _SERIAL.release()

def action_features(native, path, item, *, deadline, clock=time.monotonic):
    started = clock()
    if started >= deadline:
        raise TimeoutError('No original page time remaining')
    source, selected, geometry = source_pdf(native, Path(path), item)
    if source.original_file_sha256 != item['input_sha256']:
        raise ValueError('Source and input hashes differ')
    binding = getattr(native, 'feature_binding', None)
    runtime = getattr(native, 'runtime_binding', None)
    if runtime is not None:
        runtime.assert_clean()
        if binding is None:
            binding = bind_preflight_files(runtime.model / 'preprocessor_config.json',
                runtime.site / 'transformers/models/qwen2_vl/image_processing_qwen2_vl.py',
                runtime.site / 'transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py')
            native.feature_binding = binding
    vectors, audit = {}, {}
    with count_calls(native.analyze, 'doc_analyze', forbidden=True) as observed_calls:
        with candidate_boundary(native):
            for action in source.available_actions():
                if clock() >= deadline:
                    raise TimeoutError('Selector consumed original page deadline')
                observed, images, doc = {}, [], None
                def collect(prepared, render):
                    observed['feature'] = extract_features(prepared, source, action, render, grid_binding=binding)
                try:
                    with page_input_hook(native.analyze, source, action,
                            deadline_seconds=deadline-clock(), clock=clock, on_prepared=collect) as render:
                        images, doc = native.analyze.load_images_from_pdf(selected)
                    row = observed['feature']
                    if set(row.get('reasons', [])) & {'unavailable_action', 'unbound_native_render_audit', 'unbound_upright_page_geometry'}:
                        raise ValueError('Feature extraction reported a source/render binding fault')
                    vectors[action] = row['vector'] if row['ready'] else None
                    audit[action] = dict(features=row, render=deepcopy(render))
                finally:
                    for value in images:
                        value['img_pil'].close()
                    if doc is not None:
                        doc.close()
    if runtime is not None:
        runtime.assert_clean()
    aliases = [[a, b] for a in audit for b in audit if a < b
               and audit[a]['render']['prepared'] == audit[b]['render']['prepared']]
    return vectors, dict(source=geometry, actions=audit, candidate_aliases=aliases,
                         preparation_seconds=clock()-started, recognition_calls=observed_calls['calls'])

def observed_predict(policy, features, *, clock=time.monotonic):
    started = clock()
    with count_calls(policy.estimator, 'predict') as observed:
        decision = policy.predict(features)
    return decision, dict(seconds=clock()-started, estimator_predict_calls=observed['calls'])

def verify_selected_render(expected, actual):
    if not isinstance(actual, dict) or any(actual.get(k) != expected.get(k)
            for k in ('action', 'original_source_type', 'prepared', 'scale', 'before', 'before_scale')):
        raise ValueError('Recognition pixels/action/scale differ from selected candidate')

def select_and_run(worker, item, input_root, output_root, policy, *, deadline, clock=time.monotonic):
    """Decision persists BEFORE one explicit base call; wrapper cannot recurse.
    Eligibility is checked by the selected driver. Core permits disabled-policy
    diagnostic replay, always labeled fixed_policy_default rather than adaptive.
    """
    source = SourceRecord(**item['source'])
    started = clock()
    features, preparation = action_features(worker.native, bound_file(input_root, item['file']),
                                             item, deadline=deadline, clock=clock)
    decision, prediction = observed_predict(policy, features, clock=clock)
    chosen = decision['action']
    if chosen not in source.available_actions():
        raise ValueError('Selector chose unavailable action')
    if clock() >= deadline:
        raise TimeoutError('Selection exhausted original page budget')
    selected = deepcopy(item)
    selected['action'] = chosen
    record = dict(schema='v4_selected_decision_v1', item_id=item['item_id'],
        page_id=item['page_id'], input_sha256=item['input_sha256'], policy_identity=policy.identity,
        policy_binding=getattr(policy, 'deployment_binding', None), margin=policy.config['margin'],
        learned_enabled=policy.config['learned_enabled'], candidate_features=features,
        selection=decision, fallback_reason=None if decision['reason']=='learned_gain' else decision['reason'],
        preparation=preparation, prediction=prediction, original_deadline=deadline)
    decision_path = bound_file(output_root, '_decisions/' + item['item_id'] + '.json')
    if decision_path.exists():
        raise FileExistsError('Decision already persisted; no implicit retry')
    decision_sha = atomic_json(decision_path, record)
    if clock() >= deadline:
        raise TimeoutError('Decision persistence exhausted original deadline')
    before = clock()
    with candidate_boundary(worker.native):
        with count_calls(worker.native.analyze, 'doc_analyze') as calls:
            result = PageWorker.run(worker, selected, input_root, output_root, deadline)
    recognition_seconds = clock()-before
    audit_path = bound_file(output_root, result['audit_file'])
    if file_sha(audit_path) != result['audit_sha256']:
        raise ValueError('Recognition audit hash mismatch')
    audit = json.loads(audit_path.read_bytes())
    if audit.get('render') is not None:
        verify_selected_render(preparation['actions'][chosen]['render'], audit['render'])
    if result['status'] == 'completed' and (calls['calls'] != 1 or audit.get('render') is None):
        raise ValueError('Completed page without one observed native call and matching pixels')
    if calls['calls'] > 1:
        raise ValueError('Repeated native page recognition')
    output = dict(schema='v4_selected_result_v1', result=result, selection=decision,
        decision_file=decision_path.relative_to(Path(output_root).resolve()).as_posix(), decision_sha256=decision_sha,
        preparation=preparation, prediction=prediction, recognition_seconds=recognition_seconds,
        total_seconds=clock()-started, original_deadline=deadline, native_page_pipeline_calls=calls['calls'])
    output['receipt_sha256'] = atomic_json(bound_file(output_root, '_selected/' + item['item_id'] + '.json'), output)
    return output
