"""Bind the reviewed native consumers; importing this module uses only stdlib.

This is a guard for the frozen call chain, not a sandbox for hostile native code.
The process owns these import/loader guards for its lifetime and exits on failure.
"""
import ctypes
import functools
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import site
import sys
import uuid

from .worker_contract import bound_file, file_sha, hash_value

SCHEMA = 'v4_runtime_binding_v1'
AUXILIARIES = {
    'onnxruntime.InferenceSession': 'magika/models/standard_v3_3/model.onnx',
    'fasttext.load_model': 'fast_langdetect/ft_detect/resources/lid.176.ftz',
}
MODULE_FILES = {
    'onnxruntime': 'onnxruntime/__init__.py',
    'onnxruntime.capi.onnxruntime_inference_collection': 'onnxruntime/capi/onnxruntime_inference_collection.py',
    'fasttext': 'fasttext/__init__.py',
    'fasttext.FastText': 'fasttext/FastText.py',
    'magika': 'magika/__init__.py',
    'magika.magika': 'magika/magika.py',
    'fast_langdetect': 'fast_langdetect/__init__.py',
    'fast_langdetect.ft_detect': 'fast_langdetect/ft_detect/__init__.py',
    'fast_langdetect.ft_detect.infer': 'fast_langdetect/ft_detect/infer.py',
    'transformers.models.qwen2_5_vl.processing_qwen2_5_vl': 'transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py',
    'transformers.models.qwen2_vl.image_processing_qwen2_vl': 'transformers/models/qwen2_vl/image_processing_qwen2_vl.py',
    'transformers.models.qwen2_vl.image_processing_qwen2_vl_fast': 'transformers/models/qwen2_vl/image_processing_qwen2_vl_fast.py',
    'pypdfium2': 'pypdfium2/__init__.py',
    'pypdfium2._helpers': 'pypdfium2/_helpers/__init__.py',
    'pypdfium2._helpers.document': 'pypdfium2/_helpers/document.py',
    'pypdfium2._helpers.page': 'pypdfium2/_helpers/page.py',
    'pypdfium2_raw': 'pypdfium2_raw/__init__.py',
    'pypdfium2_raw.bindings': 'pypdfium2_raw/bindings.py',
}
PDFIUM_LIBRARY = 'pypdfium2_raw/libpdfium.so'
REQUIRED_SITE = set(MODULE_FILES.values()) | set(AUXILIARIES.values()) | {
    PDFIUM_LIBRARY, 'magika/config/content_types_kb.min.json',
    'magika/models/standard_v3_3/config.min.json',
}
PACKAGES = {'torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors',
            'accelerate', 'pypdfium2', 'pillow', 'numpy', 'opencv-python-headless',
            'onnxruntime', 'magika', 'fast-langdetect', 'fasttext-predict', 'loguru'}
SITE_MODULES = {'torch', 'torchvision', 'transformers', 'tokenizers', 'safetensors',
                'accelerate', 'pypdfium2', 'pypdfium2_raw', 'PIL', 'numpy', 'cv2',
                'onnxruntime', 'magika', 'fast_langdetect', 'fasttext', 'fasttext_pybind', 'loguru'}
LEARNED_SUFFIXES = {'.onnx', '.ftz', '.bin', '.pt', '.pth', '.safetensors',
                    '.ckpt', '.gguf', '.tflite', '.h5', '.pb', '.pkl', '.pickle', '.joblib'}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exact_keys(value, keys, message):
    require(isinstance(value, dict) and set(value) == set(keys), message)


def absolute(value):
    return isinstance(value, str) and Path(value).is_absolute() and '..' not in Path(value).parts


def validate_binding(c):
    """No I/O: historical v1 can still parse, but cannot initialize a real worker."""
    require(c.get('schema') == 'v4_tele_worker_v2', 'Real native worker requires bound v2 contract')
    b = c.get('runtime_binding')
    exact_keys(b, ('schema', 'interpreter', 'site_files', 'device', 'external_image', 'weight_receipt'),
               'Missing or malformed runtime binding')
    require(b['schema'] == SCHEMA, 'Unknown runtime binding schema')
    i = b['interpreter']
    exact_keys(i, ('executable', 'prefix', 'version', 'site'), 'Malformed interpreter binding')
    require(all(absolute(i[k]) for k in ('executable', 'prefix', 'site')) and
            isinstance(i['version'], str) and re.fullmatch(r'\d+\.\d+\.\d+', i['version']), 'Invalid interpreter identity')
    files = b['site_files']
    require(isinstance(files, dict) and REQUIRED_SITE <= set(files), 'Missing named consumer source/library hashes')
    for rel, sha in files.items():
        require(isinstance(rel, str) and rel and not Path(rel).is_absolute() and
                '..' not in Path(rel).parts and hash_value(sha), 'Invalid site file binding')
    for rel in AUXILIARIES.values():
        require(files[rel] == c['aux_files'].get(rel), 'Packaged consumer and staged auxiliary hashes differ')
    versions = {k.lower(): v for k, v in c['runtime_versions'].items()}
    require(len(versions) == len(c['runtime_versions']) and PACKAGES <= set(versions), 'Incomplete/duplicate runtime versions')
    require('modeling_naviocr.py' in c['model_files'], 'Missing trusted local model source')
    d = b['device']
    exact_keys(d, ('uuid', 'driver_path', 'driver_sha256'), 'Missing actual-device binding')
    require(isinstance(d['uuid'], str) and re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', d['uuid']) and
            absolute(d['driver_path']) and hash_value(d['driver_sha256']), 'Invalid CUDA binding')
    uuid.UUID(d['uuid'][4:])
    e = b['external_image']
    exact_keys(e, ('image', 'receipt'), 'Malformed external image binding')
    require(isinstance(e['image'], str) and e['image'].startswith('sha256:') and hash_value(e['image'][7:]), 'Unfrozen image reference')
    for receipt in (e['receipt'], b['weight_receipt']):
        exact_keys(receipt, ('path', 'sha256'), 'Missing named external receipt')
        require(absolute(receipt['path']) and hash_value(receipt['sha256']), 'Unbound external receipt')
    return b


def read_receipt(binding):
    path = Path(binding['path'])
    require(path.stat().st_size <= 1024 * 1024, 'Oversized runtime receipt')
    require(file_sha(path) == binding['sha256'], 'Runtime receipt SHA mismatch')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def observe_cuda_uuid(expected, lib):
    """Same four CUDA Driver API calls as the accepted V31 implementation."""
    class Uuid(ctypes.Structure):
        _fields_ = [('bytes', ctypes.c_ubyte * 16)]
    for name, args in (
        ('cuInit', [ctypes.c_uint]), ('cuDeviceGetCount', [ctypes.POINTER(ctypes.c_int)]),
        ('cuDeviceGet', [ctypes.POINTER(ctypes.c_int), ctypes.c_int]),
        ('cuDeviceGetUuid', [ctypes.POINTER(Uuid), ctypes.c_int]),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, ctypes.c_int
    require(lib.cuInit(0) == 0, 'CUDA initialization failed')
    count = ctypes.c_int()
    require(lib.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value == 1, 'Exactly one visible CUDA device required')
    device = ctypes.c_int()
    require(lib.cuDeviceGet(ctypes.byref(device), 0) == 0, 'CUDA device unavailable')
    value = Uuid()
    require(lib.cuDeviceGetUuid(ctypes.byref(value), device) == 0, 'CUDA UUID query failed')
    actual = 'GPU-' + str(uuid.UUID(bytes=bytes(value.bytes)))
    require(actual.lower() == expected.lower(), 'Observed CUDA UUID mismatch')
    return actual


def mapped_library(text, prefix, expected):
    paths = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and Path(fields[5]).name.startswith(prefix):
            require(not fields[5].endswith(' (deleted)'), 'Deleted native library mapping')
            paths.add(str(Path(fields[5]).resolve()))
    require(paths == {str(Path(expected).resolve())}, 'Loaded native library mapping differs: ' + prefix)


class ImportOrigins:
    """Check sources before execution, including dynamic model cache copies."""
    def __init__(self, binding):
        self.binding = binding

    def find_spec(self, fullname, path=None, target=None):
        top = fullname.partition('.')[0]
        if top not in SITE_MODULES | {'TeleOCR', 'transformers_modules'}:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        # Preserve ordinary optional-import / find_spec absence semantics.
        # Actual critical consumer modules are separately required and observed.
        if spec is None:
            return None
        self.binding.check_spec(fullname, spec)
        return spec


class RuntimeBinding:
    def __init__(self, contract, vendor_root, model_root, aux_root):
        self.contract = contract
        self.profile = validate_binding(contract)
        self.vendor, self.model, self.aux = (Path(p).resolve() for p in (vendor_root, model_root, aux_root))
        self.site = Path(self.profile['interpreter']['site']).resolve()
        self.cache = None
        self.verified = []
        self.events = []
        self.violations = []
        self.origins = {}
        self.objects = {}
        self.patches = []
        self.report = dict(schema=SCHEMA, verification_reads=self.verified, loader_events=self.events,
                           violations=self.violations, import_sources=self.origins, objects=self.objects,
                           fasttext_capacity='conservatively included even without a loader event')
        self.allowed_aux = {api: bound_file(self.site, rel).resolve() for api, rel in AUXILIARIES.items()}

    def fail(self, message):
        self.violations.append(message)
        raise RuntimeError(message)

    def verify(self, path, expected, purpose):
        actual = file_sha(path)
        if actual != expected:
            self.fail('File SHA mismatch: ' + str(path))
        self.verified.append(dict(path=str(Path(path).resolve()), sha256=actual, purpose=purpose,
                                  is_model_load=False))
        return actual

    def preflight(self):
        i = self.profile['interpreter']
        observed = dict(executable=os.path.abspath(sys.executable), prefix=os.path.abspath(sys.prefix),
                        version='.'.join(map(str, sys.version_info[:3])), site=str(self.site))
        require(observed == i, 'Observed interpreter/prefix differs')
        require(self.site.is_relative_to(Path(sys.prefix).resolve()), 'Site outside interpreter prefix')
        observed_sites = [str(Path(p).resolve()) for p in site.getsitepackages()]
        require(str(self.site) in observed_sites and str(self.site) in
                [str(Path(p).resolve()) for p in sys.path if p], 'Expected site is not an active interpreter package path')
        self.report['interpreter_observed'] = observed
        self.report['interpreter_site_paths_observed'] = observed_sites
        versions = {k: importlib.metadata.version(k) for k in self.contract['runtime_versions']}
        require(versions == self.contract['runtime_versions'], 'Frozen runtime package versions differ')
        distributions = {k: str(Path(importlib.metadata.distribution(k).locate_file('')).resolve()) for k in versions}
        require(set(distributions.values()) == {str(self.site)}, 'Package metadata from a different site root')
        self.report['versions_observed'] = versions
        self.report['distribution_roots_observed'] = distributions
        for rel, sha in self.profile['site_files'].items():
            self.verify(bound_file(self.site, rel), sha, 'named site file verification')
        e = self.profile['external_image']
        receipt = read_receipt(e['receipt'])
        require(receipt.get('schema') == 'v4_external_image_v1' and receipt.get('job_id') == self.contract['job_id']
                and receipt.get('image') == e['image'] and isinstance(receipt.get('container_id'), str)
                and re.fullmatch('[0-9a-f]{64}', receipt['container_id'])
                and isinstance(receipt.get('inspected_at_utc'), str), 'External image receipt does not bind this job')
        self.report['image_external'] = dict(expected_reference=e['image'], receipt=receipt,
                                           receipt_sha256=e['receipt']['sha256'], observed_by_worker=False)
        weight = read_receipt(self.profile['weight_receipt'])
        require(weight.get('schema') == 'v4_weight_observation_v1' and
                weight.get('sha256') == self.contract['model_files']['model.safetensors'] and
                type(weight.get('bytes')) is int and weight['bytes'] == (self.model / 'model.safetensors').stat().st_size
                and absolute(weight.get('path')) and Path(weight['path']).name == 'model.safetensors'
                and isinstance(weight.get('observed_at_utc'), str), 'Missing current weight observation')
        self.report['weight_external'] = weight
        self.report['staged_verification'] = dict(status='pending', is_model_load=False)
        return versions

    def install(self, output_root):
        sensitive = SITE_MODULES | {'TeleOCR', 'transformers_modules'}
        require(not any(name.partition('.')[0] in sensitive for name in sys.modules), 'Native module imported before consumer guards')
        output = Path(output_root).resolve()
        self.cache = output / 'runtime_hf_modules'
        self.cache.mkdir(parents=False, exist_ok=False)
        os.environ['HF_MODULES_CACHE'] = str(self.cache)
        self.finder = ImportOrigins(self)
        sys.meta_path.insert(0, self.finder)
        sys.addaudithook(self.audit_open)
        ort = importlib.import_module('onnxruntime')
        ft = importlib.import_module('fasttext')
        self.install_loaders(ort, ft)

    def check_spec(self, name, spec):
        top = name.partition('.')[0]
        root = self.vendor if top == 'TeleOCR' else self.cache if top == 'transformers_modules' else self.site
        if spec.origin is None:
            require(spec.submodule_search_locations and all(Path(p).resolve().is_relative_to(root)
                    for p in spec.submodule_search_locations), 'Unbound namespace: ' + name)
            return
        path = Path(spec.origin).resolve()
        if not path.is_relative_to(root):
            self.fail('Module origin outside frozen root: ' + name)
        rel = path.relative_to(root).as_posix()
        if top == 'TeleOCR':
            sha = self.contract['vendor_files'].get(rel)
            require(sha is not None and path.suffix == '.py', 'Unfrozen Tele module: ' + name)
        elif top == 'transformers_modules':
            if path.name == '__init__.py' and path.stat().st_size == 0:
                sha = file_sha(path)
            else:
                sha = self.contract['model_files'].get(path.name) if path.suffix == '.py' else None
                require(sha is not None, 'Unfrozen dynamic model source: ' + name)
        else:
            expected = MODULE_FILES.get(name)
            if expected is not None and rel != expected:
                self.fail('Critical module relative path differs: ' + name)
            sha = self.profile['site_files'].get(rel)
        if sha is not None:
            self.verify(path, sha, 'module source before import')
        self.origins[name] = dict(path=str(path), sha256=sha, source_hash_verified=sha is not None)

    def audit_open(self, event, args):
        # Native C/C++ opens are covered at the two proven auxiliary loader APIs.
        # Python hash reads are only file opens, never counted as model loads.
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0]))
        if path.suffix.lower() not in LEARNED_SUFFIXES:
            return
        allowed = set(self.allowed_aux.values())
        allowed.update(bound_file(self.model, n).resolve() for n in self.contract['model_files'])
        if path.resolve() not in allowed:
            self.fail('Unknown learned file open: ' + str(path))

    def checked_load(self, api, path, call, object_after=None):
        event = dict(api=api, status='attempted', path=None, object_id=None)
        self.events.append(event)
        try:
            if not isinstance(path, (str, os.PathLike)):
                self.fail('Only bound file paths accepted by ' + api)
            supplied = Path(path)
            event['path'] = str(supplied)
            expected = self.allowed_aux[api]
            # A same-hash copy/alias elsewhere is not the verified package path.
            if not supplied.is_absolute() or supplied.absolute() != expected or supplied.resolve() != expected:
                self.fail('Unapproved actual loader path: ' + api)
            rel = AUXILIARIES[api]
            self.verify(expected, self.contract['aux_files'][rel], 'actual auxiliary loader boundary')
            result = call()
            event.update(status='loaded', sha256=self.contract['aux_files'][rel],
                         object_id=id(object_after if object_after is not None else result))
            return result
        except BaseException as exc:
            event.update(status='failed', error=repr(exc))
            if not self.violations:
                self.violations.append('Auxiliary loader failed: ' + api)
            raise

    def install_loaders(self, ort, ft):
        cls = ort.InferenceSession
        defining = sys.modules['onnxruntime.capi.onnxruntime_inference_collection']
        ft_defining = sys.modules['fasttext.FastText']
        require(cls is defining.InferenceSession and ft.load_model is ft_defining.load_model,
                'Auxiliary loader export differs from frozen defining module')
        init, load = cls.__init__, ft.load_model
        require(init.__module__ == defining.__name__ and load.__module__ == ft_defining.__name__,
                'Auxiliary callable origin differs')
        init_sig, load_sig = inspect.signature(init), inspect.signature(load)
        @functools.wraps(init)
        def guarded_init(instance, *args, **kwargs):
            path = init_sig.bind(instance, *args, **kwargs).arguments['path_or_bytes']
            return self.checked_load('onnxruntime.InferenceSession', path,
                                     lambda: init(instance, *args, **kwargs), instance)
        @functools.wraps(load)
        def guarded_load(*args, **kwargs):
            path = load_sig.bind(*args, **kwargs).arguments['path']
            return self.checked_load('fasttext.load_model', path, lambda: load(*args, **kwargs))
        cls.__init__ = guarded_init
        ft.load_model = ft_defining.load_model = guarded_load
        self.patches = [(ort, 'InferenceSession', cls), (defining, 'InferenceSession', cls),
                        (cls, '__init__', guarded_init), (ft, 'load_model', guarded_load),
                        (ft_defining, 'load_model', guarded_load)]

    def assert_clean(self):
        if self.violations:
            raise RuntimeError('Runtime binding violation: ' + self.violations[-1])
        require(all(getattr(obj, attr) is wrapper for obj, attr, wrapper in self.patches), 'Loader guard was replaced')

    def observe_device(self):
        d = self.profile['device']
        path = Path(d['driver_path']).resolve()
        self.verify(path, d['driver_sha256'], 'CUDA driver file verification')
        lib = ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL)
        require(Path('/proc/self/maps').stat().st_size <= 8 * 1024 * 1024, 'Oversized process maps')
        with Path('/proc/self/maps').open() as stream:
            maps = stream.read(8 * 1024 * 1024 + 1)
        require(len(maps) <= 8 * 1024 * 1024, 'Oversized process maps')
        mapped_library(maps, 'libcuda.so', path)
        actual = observe_cuda_uuid(d['uuid'], lib)
        self.report['device_observed'] = dict(uuid=actual, visible_count=1, driver_path=str(path),
                                               driver_sha256=d['driver_sha256'], source='CUDA Driver API')

    def observe_class(self, label, cls, module_name, class_name):
        require(cls.__name__ == class_name and cls.__module__ == module_name, 'Actual object class differs: ' + label)
        module = sys.modules[module_name]
        require(getattr(module, class_name) is cls, 'Object class export differs: ' + label)
        path = Path(inspect.getfile(cls)).resolve()
        require(path == Path(module.__file__).resolve(), 'Object source differs from loaded module')
        self.check_spec(module_name, module.__spec__)
        require(self.origins[module_name]['source_hash_verified'], 'Actual object source has no frozen hash')
        self.objects[label] = dict(class_name=class_name, module=module_name,
                                   source_path=str(path), source_sha256=file_sha(path))

    def observe_pdfium(self, raw):
        expected = bound_file(self.site, PDFIUM_LIBRARY).resolve()
        info_path = Path(raw._libs_info['pdfium']['path']).resolve()
        loaded_path = Path(raw._libs['pdfium']._name).resolve()
        require(info_path == loaded_path == expected, 'Actual PDFium loaded library path differs')
        sha = self.verify(expected, self.profile['site_files'][PDFIUM_LIBRARY], 'actual PDFium library')
        self.report['renderer_observed'] = dict(path=str(loaded_path), sha256=sha,
                                               source='bindings._libs_info and CDLL._name')

    def observe_native(self, backend, pdfium):
        model_cls = type(backend.model)
        require(model_cls.__module__.startswith('transformers_modules.') and self.cache is not None,
                'Expected locally cached trusted model class')
        self.observe_class('model', model_cls, model_cls.__module__, self.contract['expected']['model_class'])
        require(self.objects['model']['source_sha256'] == self.contract['model_files']['modeling_naviocr.py'],
                'Actual dynamic model code differs')
        self.observe_class('processor', type(backend.processor),
                           'transformers.models.qwen2_5_vl.processing_qwen2_5_vl', 'Qwen2_5_VLProcessor')
        self.observe_class('image_processor', type(backend.processor.image_processor),
                           'transformers.models.qwen2_vl.image_processing_qwen2_vl_fast', 'Qwen2VLImageProcessorFast')
        self.observe_class('pdf_document', pdfium.PdfDocument, 'pypdfium2._helpers.document', 'PdfDocument')
        self.observe_class('pdf_page', pdfium.PdfPage, 'pypdfium2._helpers.page', 'PdfPage')
        self.observe_pdfium(sys.modules['pypdfium2_raw.bindings'])
        require(str(backend.model.device) in ('cuda', 'cuda:0'), 'Native model is not on the verified CUDA device')
        self.report['model_device_observed'] = str(backend.model.device)
        magika = sys.modules['TeleOCR.tools.guess_suffix_or_lang'].magika
        require(any(e['api'] == 'onnxruntime.InferenceSession' and e['status'] == 'loaded'
                    and e['object_id'] == id(magika._onnx_session) for e in self.events),
                'No observed load for the actual native Magika session')
        self.report['magika_session_object_id'] = id(magika._onnx_session)
        self.assert_clean()
