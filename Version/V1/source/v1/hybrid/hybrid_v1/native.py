"""Real M/P container adapters using the approved native lifecycle and workers."""
import importlib.util
import sys
import time
from pathlib import Path

from .assembly import save
from .cache import CSV_CODE, NativeCache, checked, key, read, sha
from .fresh_contract import FRESH_CODE, PATH_POLICY_SHA, verify_code, make_runtime
from .page_paths import component_name, mapping_for_pages


def verify_started_terminals(root,shard,runtime):
    for event in (shard/'events').glob('*.started.json'):
        pid=read(event)['page_id']
        assert event.name==component_name(pid,'.started.json')
        terminal=root/'terminal'/component_name(pid,'.json')
        assert terminal.exists(),'Uncollected real attempt'
        row=read(terminal)
        assert row['page_id']==pid and row['runtime_key']==key(runtime)
        assert not any(name in row for name in ('original_runtime_key','source_terminal','compatibility_receipt'))


class NativeResults(NativeCache):
    """The same page verifier for new results, without importing an external cache."""
    def __init__(self, root, pages, rows, arm, runtime):
        self.arm = arm
        self.root = Path(root) / 'terminal'
        self.primary = Path(root) / 'primary'
        self.pages = {p['page_id']: p for p in pages}
        self.rows = {r['page_id']: r for r in rows}
        self.runtime = runtime
        self.expected_runtime_key = key(runtime)
        self.compat_checked = set()
        assert len(self.pages) == len(pages) and len(self.rows) == len(rows)
        for pid, row in self.rows.items():
            assert row['input_sha256'] == self.pages[pid]['input_sha256']
            assert row['runtime_key'] == self.expected_runtime_key
            assert not any(name in row for name in ('original_runtime_key','source_terminal','compatibility_receipt')), 'Fresh run may not adopt external results'

    def file_name(self,pid,suffix=''):
        if self.runtime.get('full_code_lock_sha256')==FRESH_CODE:
            assert self.runtime['path_policy_sha256']==PATH_POLICY_SHA
            return component_name(pid,suffix)
        return super().file_name(pid,suffix)


class RealNativeAdapters:
    def __init__(self, freeze, output, pages, manifest_sha256):
        assert sys.platform == 'linux', 'Native inference runs in the frozen Linux environments'
        self.freeze = freeze
        self.output = Path(output)
        self.pages = pages
        self.manifest_sha256 = manifest_sha256
        controller = checked(freeze['native_controller'])
        code = controller.parent
        verify_code(code,freeze)
        path_map=mapping_for_pages(pages,require_original_primary=True)
        sys.path.insert(0, str(code))
        spec = importlib.util.spec_from_file_location('hybrid_v1_native_controller', controller)
        self.controller = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.controller)
        assert self.output.resolve().is_relative_to(
            (self.controller.ROOT / 'artifacts/official-full-20260921-v1/hybrid_v1').resolve())
        assert code.resolve().is_relative_to((self.controller.ROOT / 'inference/hybrid-v1').resolve())
        image_root = Path(freeze['input_root']).resolve()
        assert image_root == self.controller.ROOT / 'inference/official-full-20260921-v1/images' or image_root.is_relative_to(
            (self.controller.ROOT / 'inference/hybrid-v1').resolve())
        self.code = code
        self.runtime = {}
        self.error_counts = {}
        for arm in ('mineru', 'paddle'):
            binding = freeze['native_bindings'][arm]
            original = read(checked(binding['runtime_artifact']))
            assert original['arm'] == arm and original['full_code_lock_sha256'] == CSV_CODE
            assert sys.executable == original['controller_interpreter']
            assert sha(Path(sys.executable).resolve()) == original['controller_interpreter_sha256']
            import PIL
            assert PIL.__version__ == original['controller_pillow_version']
            assert sha(PIL.__file__) == original['controller_pillow_source_sha256']
            for model in original['models']:
                checked({**model, 'path': str(self.controller.ROOT / model['path'])})
            environment = self.controller.ROOT / 'receipts/REMOTE_HYBRID_BENCHMARK_v1' / (
                'environment-' + self.controller.ENVS[arm])
            assert sha(environment / 'EXIT.json') == original['environment_receipt_sha256']
            assert read(environment / 'EXIT.json')['installation_complete']
            assert sha(environment / '5.log') == original['freeze_sha256']
            driver = self.controller.ROOT / 'receipts/GPU_BIND_v1/attempt-v2/DRIVER_BUNDLE.json'
            assert sha(driver) == original['driver_sha256']
            for item in read(driver)['files']:
                if item['present']:
                    assert sha(item['target']) == item['sha256']
            for item in binding['environment_and_driver_artifacts']:
                checked(item)
            image = self.controller.query(['docker', 'image', 'inspect', original['image']])
            import json
            assert json.loads(image)[0]['Id'] == original['image']
            runtime = make_runtime(original,arm,freeze,manifest_sha256)
            self.runtime[arm] = runtime
            self.error_counts[arm] = {}
            root = self.output / 'native' / arm
            if root.exists():
                assert read(root / 'RUNTIME.json') == runtime, 'Cannot resume another freeze'
                assert read(root / 'PAGE_FILE_MAP.json') == path_map
                for batch in root.glob('BATCH_*.json'):
                    assert read(batch)['paused'] is False, 'Paused native failure requires owner review'
                for shard in (root / 'shards').iterdir():
                    assert (shard / 'EXIT.json').exists(), 'Unclosed native lifecycle requires explicit recovery'
                    start = read(shard / 'START.json')
                    state = json.loads(self.controller.query(['docker', 'inspect', start['cid']]))[0]
                    assert state['Id'] == start['cid'] and not state['State']['Running']
                    assert state['Image'] == runtime['image']
                    verify_started_terminals(root,shard,runtime)
                for terminal in (root / 'terminal').glob('*.json'):
                    error = read(terminal).get('error')
                    if error:
                        self.error_counts[arm][error] = self.error_counts[arm].get(error, 0) + 1
                for terminal in (root / 'terminal').glob('*.json'):
                    row=read(terminal)
                    assert self.controller.failure_action(row,self.controller.verify_terminal_timeout(row))!='stop', 'Unclassified native failure requires owner review'
            else:
                root.mkdir(parents=True, exist_ok=False)
                for name in ('terminal', 'primary', 'shards'):
                    (root / name).mkdir()
                save(root / 'RUNTIME.json', runtime)
                save(root / 'PAGE_FILE_MAP.json',path_map)

    def batch(self, arm, pages, index):
        root = self.output / 'native' / arm
        prior = [read(root / 'terminal' / component_name(p['page_id'],'.json')) for p in pages
                 if (root / 'terminal' / component_name(p['page_id'],'.json')).exists()]
        verifier = NativeResults(root, pages, prior, arm, self.runtime[arm])
        reused = {r['page_id']: {**verifier.page(r['page_id']), 'resumed_within_run': True} for r in prior}
        pending = [p for p in pages if p['page_id'] not in reused]
        if not pending:
            return reused, [], False
        index = max([int(p.name) for p in (root / 'shards').iterdir()] + [-1]) + 1
        context = dict(arm=arm, run=root, run_id=self.freeze['run_id'], code=self.code,
                       image=self.runtime[arm]['image'], key=key(self.runtime[arm]),
                       pred=root / 'primary', terminal=root / 'terminal',
                       input_root=self.freeze['input_root'])
        gpu = None
        deadline = time.monotonic() + 60
        while gpu is None and time.monotonic() < deadline:
            gpu = self.controller.acquire_gpu()
            if gpu is None:
                time.sleep(2)
        assert gpu is not None, 'No allowlisted idle leased GPU; no work dispatched'
        try:
            rows, remaining, pause = self.controller.run_shard(pending, index, context, gpu)
        finally:
            gpu[2].close()
        for row in rows:
            if row.get('error'):
                error = row['error']
                self.error_counts[arm][error] = self.error_counts[arm].get(error, 0) + 1
                # Cumulative error histogram is retained; proof, not string equality, governs continuation.
            if self.controller.failure_action(row,self.controller.verify_terminal_timeout(row)) == 'stop':
                pause = True
        # An unstarted page is not synthesized into a terminal or silently omitted.
        save(root / ('BATCH_%06d.json' % index), dict(terminal=len(rows),
             unstarted=[p['page_id'] for p in remaining], paused=pause))
        verifier = NativeResults(root, pages, rows, arm, self.runtime[arm])
        results = {row['page_id']: {**verifier.page(row['page_id']), 'resumed_within_run': False} for row in rows}
        results.update(reused)
        if remaining and not pause:
            continuation, remaining, pause = self.batch(arm, remaining, index + 1)
            assert not results.keys() & continuation.keys()
            results.update(continuation)
        return results, remaining, pause
