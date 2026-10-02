"""Verify complete native caches before admitting a page to hybrid_v1."""
import hashlib
import json
from pathlib import Path

CSV_CODE = '3817c4787fcd9c699d862a5a1f6997e53bf5bc0a3350b875a575dc64829cc9a0'
PATH_CODE = '5186bb18bb5ebd70ea7260fc107f7302cac645f4fc91dc1f588b3b7874722d4e'
OLD_CODE = '168a79fb2af517500e03895f5703f4de000824c81c282423221cd660c85db6cb'
PROBE = '3a7690eea552ac77213bcf9e09df2b076ccfd5111dd4722a514dd910323a14c2'
# Exact unchanged native files from the approved old code lock.
NATIVE_FILES = {'capture_utils.py': 'd91b56f2d343fda64688c9b44d3934ba1cdc417026abb5822e73782e5605605d', 'complementarity_ovis_smoke.py': 'fa3c44c2d30e2e91981204c94e00ef763ad56f946d117ba6fb4661fc990259b5', 'core.py': '8522f06abb6b8efb0e14df576e6a89d4e8c7770487a14525748f1ff9e1caa5c0', 'device_proof.py': '6a99da0279499ecb094ab6f42835eee6eba735aab7ba8ed06ddfb49d57b9db34', 'image_only_audit.py': 'b0bf4133520d68989cea890764b5fc6c3bf14c8e027330135c0bfd1640851162', 'MODEL_MANIFEST.json': 'd062d5366c4607965466769048308a721d3491cdfd52c2101f792a7bf7c72562', 'native_full.py': '5073d0b74acd881200e82fc3885f19d339e338f4d15282853c9ace0f9baf5ef7', 'native_schema.py': '6746a70eeee3d1d53bbd186fb81f4966685d39437217018ec3230e87b317c8e2', 'native_stop_audit.py': '417084a1613fc2fa6e23794ebffb0c91baf4a78dca22a8f369736f839d92e561', 'paddle_full.py': 'aa2bbb21f46714bffec5a1049626b43f67c9fcdeb7ac8a04e8a3bd80f70d83d7', 'PADDLE_PARAMETERS_TEMPLATE.json': '9b5316d04ddf067d07597d2fd7eb970799798261c96ac9a8880a058314fa69d3'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked(artifact):
    path = Path(artifact['path'])
    assert path.is_file() and not path.is_symlink(), 'Missing or linked artifact'
    assert path.stat().st_size == artifact['bytes'] and sha(path) == artifact['sha256'], 'Artifact changed'
    return path


class NativeCache:
    def __init__(self, lock_path, terminal_root, pages, arm, expected_lock_sha,
                 expected_runtime_key):
        assert arm in ('mineru', 'paddle')
        self.arm = arm
        self.lock_path = Path(lock_path)
        assert sha(self.lock_path) == expected_lock_sha
        self.lock = read(self.lock_path)
        self.root = Path(terminal_root)
        self.runtime = read(self.root.parent / 'RUNTIME.json')
        assert key(self.runtime) == expected_runtime_key == self.lock['runtime_key']
        assert self.runtime['arm'] == arm
        assert self.runtime['full_code_lock_sha256'] in (OLD_CODE, CSV_CODE, PATH_CODE)
        if self.runtime['full_code_lock_sha256'] == PATH_CODE:
            assert arm == 'mineru', 'Path-v3 compatibility is MinerU-only'
        assert self.lock['complete'] is True and self.lock['arm'] == ('B' if arm == 'mineru' else arm)
        assert len(pages) == len(self.lock['pages']) == 1651
        self.pages = {p['page_id']: p for p in pages}
        self.rows = {p['page_id']: p for p in self.lock['pages']}
        assert len(self.pages) == len(self.rows) == 1651 and self.pages.keys() == self.rows.keys()
        self.primary = Path(self.lock['prediction_directory'])
        self.expected_runtime_key = expected_runtime_key
        self.compat_checked = set()
        for pid, row in self.rows.items():
            assert row['input_sha256'] == self.pages[pid]['input_sha256']
        self.path_compatibility = None
        if self.runtime['full_code_lock_sha256'] == PATH_CODE:
            from .cache_path_v3 import verify_run
            self.path_compatibility, self.path_source_runtime = verify_run(self.root.parent, self.runtime, self.pages)

    def lineage(self, row, runtime=None, depth=0, seen=None):
        runtime = self.runtime if runtime is None else runtime
        assert depth <= 2, 'Unexpected compatibility chain depth'
        assert row['runtime_key'] == key(runtime)
        seen = set() if seen is None else seen
        if 'original_runtime_key' not in row:
            assert not row.get('source_terminal') and not row.get('compatibility_receipt')
            return
        source_path = checked(row['source_terminal'])
        identity = (str(source_path.resolve()), row['source_terminal']['sha256'])
        assert identity not in seen, 'Cyclic native provenance'
        seen = seen | {identity}
        original = read(source_path)
        compatibility_path = checked(row['compatibility_receipt'])
        compatibility = read(compatibility_path)
        path_transition = runtime['full_code_lock_sha256'] == PATH_CODE
        if path_transition:
            from .cache_path_v3 import OLD_RUN, PATH_RUN
            assert depth == 0 and self.arm == 'mineru'
            assert source_path.resolve() == (OLD_RUN/'terminal'/(row['page_id']+'.json')).resolve()
            assert compatibility_path.resolve() == (PATH_RUN/'COMPATIBILITY_RECEIPT.json').resolve()
            assert self.path_compatibility is not None and compatibility == self.path_compatibility
            old = self.path_source_runtime
        else:
            assert runtime['full_code_lock_sha256'] == CSV_CODE
            assert compatibility['kind'] == 'FULL-TRANSPORT-CSV-v2'
            assert compatibility['old_code_lock_sha256'] == OLD_CODE
            assert compatibility['new_code_lock_sha256'] == CSV_CODE
            old = read(checked(compatibility['old_runtime_artifact']))
        assert compatibility['new_runtime_key'] == row['runtime_key']
        assert compatibility['old_runtime_key'] == row['original_runtime_key'] == original['runtime_key']
        for field in ('page_id', 'input_sha256', 'status', 'prediction_sha256', 'primary_bytes'):
            assert original[field] == row[field], 'Adoption changed native result'
        assert row['original_prediction_sha256'] == row['prediction_sha256']
        for field in ('actual_started', 'native_artifacts_verified', 'native_stop_audit_complete',
                      'schema_present', 'schema_proof', 'stops'):
            assert original.get(field) == row.get(field), 'Adoption changed native audit evidence'
        originals = {a['path']: a for a in original['native_artifacts']}
        derived = {a['path']: a for a in row['native_artifacts']}
        assert len(originals) == len(original['native_artifacts'])
        assert all(derived.get(p) == a for p, a in originals.items())
        for artifact in originals.values():
            checked(artifact)
        if not path_transition and str(compatibility_path) not in self.compat_checked:
            new = read(checked(compatibility['new_runtime_artifact']))
            assert key(old) == compatibility['old_runtime_key'] and key(new) == row['runtime_key']
            assert new == runtime
            assert {k: v for k, v in old.items() if k != 'full_code_lock_sha256'} == {
                k: v for k, v in new.items() if k != 'full_code_lock_sha256'}
            assert compatibility['native_inference_signature'] == key({
                'runtime_inputs': {k: v for k, v in old.items() if k != 'full_code_lock_sha256'},
                'unchanged_native_files': NATIVE_FILES})
            assert compatibility['CSV_probe_artifact']['sha256'] == PROBE
            checked(compatibility['CSV_probe_artifact'])
            assert compatibility['dispatcher_exit_proof']['original_exited'] is True
            assert all(not item['state']['Running'] for item in compatibility['owned_containers_stopped'])
            self.compat_checked.add(str(compatibility_path))
        expected_old_code = CSV_CODE if path_transition else OLD_CODE
        assert old['full_code_lock_sha256'] == expected_old_code
        assert old['arm'] == self.arm and key(old) == original['runtime_key']
        self.lineage(original, old, depth+1, seen)

    def file_name(self, pid, suffix=''):
        return pid + suffix

    def page(self, pid):
        assert pid and pid not in ('.', '..') and '/' not in pid and '\\' not in pid
        terminal_path = self.root / self.file_name(pid, '.json')
        row = read(terminal_path)
        locked = self.rows[pid]
        assert row['runtime_key'] == self.expected_runtime_key
        for field in ('page_id', 'input_sha256', 'status', 'prediction_sha256', 'native_artifacts'):
            assert row[field] == locked[field]
        if 'source_runtime_key' in locked:
            assert locked['source_runtime_key'] == row.get('original_runtime_key', row['runtime_key'])
            assert locked.get('source_terminal') == row.get('source_terminal')
            assert locked.get('compatibility_receipt') == row.get('compatibility_receipt')
        assert row['actual_started'] is True and row['native_artifacts_verified'] is True
        assert row['status'] in ('success', 'failed', 'truncated')
        self.lineage(row)
        artifacts = {a['path']: a for a in row['native_artifacts']}
        assert len(artifacts) == len(row['native_artifacts'])
        for artifact in artifacts.values():
            checked(artifact)
        primary = self.primary / self.file_name(pid, '.md')
        assert not primary.is_symlink() and primary.stat().st_size == row['primary_bytes']
        assert sha(primary) == row['prediction_sha256']
        raw = primary.read_bytes()
        if row['status'] != 'success':
            assert raw == b''
            return dict(raw=raw, native=None, status=row['status'], terminal=row,
                        assets=[], terminal_sha256=sha(terminal_path),
                        native_page_directory_name=self.file_name(pid))
        assert row['native_stop_audit_complete'] and row['schema_present']
        proof = row['schema_proof']
        assert proof['valid'] and proof['arm'] == self.arm
        # Pick from this terminal's locked page folder, never a directory scan.
        name = 'structured_content.json' if self.arm == 'mineru' else 'native.json'
        candidates = [Path(p) for p in artifacts if Path(p).name == name and Path(p).parent.name == self.file_name(pid)]
        assert len(candidates) == 1, 'Native artifact absent or ambiguous'
        native_path = candidates[0]
        receipt_path = native_path.parent / 'receipt.json'
        assert str(receipt_path) in artifacts
        receipt = read(receipt_path)
        assert receipt['page_id'] == pid and receipt['input_sha256'] == self.pages[pid]['input_sha256']
        assert receipt['status'] == 'returned'
        native = read(native_path)
        if self.arm == 'mineru':
            assert proof['page_id'] == pid and proof['input_sha256'] == receipt['input_sha256']
            saved = native_path.parent / 'markdown.md'
            assert str(saved) in artifacts and sha(saved) == row['prediction_sha256'] == proof['saved_markdown_sha256']
        else:
            assert sha(native_path) == proof['native_sha256']
            assert (native['width'], native['height']) == (self.pages[pid]['width'], self.pages[pid]['height'])
            assert native['input_path'] == '/images/' + Path(self.pages[pid]['image_path']).name
        for asset in proof['assets']:
            assert asset['path'] in artifacts
            assert asset['sha256'] == artifacts[asset['path']]['sha256']
            checked(asset)
        return dict(raw=raw, native=native, status=row['status'], terminal=row,
                    assets=proof['assets'], terminal_sha256=sha(terminal_path),
                    native_page_directory_name=self.file_name(pid))
