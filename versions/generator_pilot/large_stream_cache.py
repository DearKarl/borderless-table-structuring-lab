"""Consume only immutable terminal shared shards, then verify the sealed index.

This changes availability timing only. Page order, cache bytes and inference
functions remain those of the original frozen six-model implementation.
"""
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


class StreamingCache:
    def __init__(self, root, binding, rows, wait_callback=None):
        self.root, self.binding = Path(root), binding
        path = self.root/'STREAM_DEPENDENCIES.json'
        if sha(path) != binding['stream_dependencies_sha256']:
            raise ValueError('Streaming dependency manifest changed')
        self.manifest = read(path)
        if self.manifest['selection_sha256'] != binding['selection_sha256']:
            raise ValueError('Streaming selection differs')
        if self.manifest['full_denominator'] != 1651 or len(rows) != 1651:
            raise ValueError('Streaming requires the complete1651-page manifest')
        self.sources = {r['page_id']: r for r in rows}
        self.shards = {}
        project = self.root.parent
        for shard in self.manifest['shards']:
            folder = Path(shard['root'])
            if folder.parent != project or not folder.name.startswith('generator-pilot-20261008-large-lora-benchmark-shared-'):
                raise ValueError('Shared dependency is outside this campaign')
            for page in shard['page_ids']:
                if page in self.shards or page not in self.sources:
                    raise ValueError('Duplicate or unexpected shared membership')
                self.shards[page] = shard
        if set(self.shards) != set(self.sources):
            raise ValueError('Shared dependencies omit pages')
        self.loaded, self.used = {}, {}
        self.wait_callback = wait_callback
        self.record_consumption = True

    def wait(self, reason):
        if self.wait_callback:
            self.wait_callback(reason)
        time.sleep(15)

    def terminal_shard(self, shard):
        folder = Path(shard['root'])
        result_path, execution_path = folder/'BENCHMARK.json', folder/'EXECUTION.json'
        key = str(folder)
        if key in self.loaded:
            value = self.loaded[key]
            if sha(result_path) != value['result_sha256'] or sha(execution_path) != value['execution_sha256']:
                raise ValueError('Previously consumed terminal shared receipt changed')
            return value['rows']
        while True:
            try:
                execution = read(execution_path)
                raw = result_path.read_bytes()
                report = json.loads(raw)
            except (FileNotFoundError, json.JSONDecodeError):
                self.wait('shared_shard_not_terminal:'+folder.name)
                continue
            if (not execution.get('worker_exited') or not execution.get('lease_released')
                    or report.get('status') not in ('complete','failed')
                    or len(report.get('rows',[])) != len(shard['page_ids'])):
                self.wait('shared_shard_awaiting_terminal_reconciliation:'+folder.name)
                continue
            break
        upstream = read(folder/'STAGE_BINDING.json')
        if (upstream['worker'] != 'large_benchmark_shared.py'
                or upstream['selection_sha256'] != self.binding['selection_sha256']
                or upstream['benchmark_inputs_sha256'] != shard['input_manifest_sha256']
                or sha(folder/'BENCHMARK_INPUTS.json') != shard['input_manifest_sha256']
                or report['selection_sha256'] != self.binding['selection_sha256']):
            raise ValueError('Terminal shared provenance differs')
        if upstream['code_hashes'] != shard['code_hashes']:
            raise ValueError('Shared execution source binding differs')
        for name, expected in shard['code_hashes'].items():
            member = Path(name)
            if member.is_absolute() or '..' in member.parts or sha(folder/member) != expected:
                raise ValueError('Shared source bytes changed')
        mapped = {r['page_id']: r for r in report['rows']}
        if len(mapped) != len(report['rows']) or set(mapped) != set(shard['page_ids']):
            raise ValueError('Terminal shard membership differs')
        for page, row in mapped.items():
            source = self.sources[page]
            if any(row[k] != source[k] for k in ('original_page_id','input_sha256')):
                raise ValueError('Terminal shared input identity differs')
            if row.get('cache'):
                cache = Path(row['cache']['path'])
                if cache != folder/'pages'/page/'native-cache.pkl' or sha(cache) != row['cache']['sha256']:
                    raise ValueError('Terminal shared cache path or bytes differ')
        value = dict(result_sha256=hashlib.sha256(raw).hexdigest(), execution_sha256=sha(execution_path), rows=mapped)
        self.loaded[key] = value
        receipt = dict(shard=folder.name, result_sha256=value['result_sha256'], execution_sha256=value['execution_sha256'], page_count=len(mapped))
        if self.record_consumption:
            with (self.root/'CONSUMED_SHARED_SHARDS.jsonl').open('a') as stream:
                stream.write(json.dumps(receipt)+'\n')
        return mapped

    def __getitem__(self, page):
        if page in self.used:
            raise ValueError('A model page cannot be executed twice')
        row = self.terminal_shard(self.shards[page])[page]
        value = {k:row[k] for k in ('page_id','original_page_id','input_sha256','status','reason','cache') if k in row}
        self.used[page] = value
        return value

    def finalize(self):
        path, seal = self.root/'SHARED_CACHE_INDEX.json', self.root/'CACHE_SEAL.json'
        while not path.exists() or not seal.exists():
            self.wait('awaiting_complete_shared_cache_index_seal')
        identity = read(seal)
        if identity['selection_sha256'] != self.binding['selection_sha256'] or sha(path) != identity['shared_cache_index_sha256']:
            raise ValueError('Final cache-index seal differs')
        index = read(path)
        mapped = {row['page_id']:row for row in index['rows']}
        if (index['selection_sha256'] != self.binding['selection_sha256'] or index['denominator'] != 1651
                or len(index['rows']) != 1651 or set(mapped) != set(self.sources)):
            raise ValueError('Final sealed cache membership differs')
        if any(mapped[page] != row for page,row in self.used.items()):
            raise ValueError('Consumed inputs differ from the final shared index')
        for key, value in self.loaded.items():
            folder = Path(key)
            if (sha(folder/'BENCHMARK.json') != value['result_sha256']
                    or index['shard_receipt_hashes'][folder.name.split('large-lora-',1)[1]] != value['result_sha256']):
                raise ValueError('Final shared receipt differs from consumed shard')
        return identity['shared_cache_index_sha256']
