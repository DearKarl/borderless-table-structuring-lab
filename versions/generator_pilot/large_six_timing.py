"""Report measured elapsed windows separately from summed GPU allocation time.

Consumes a terminal cost snapshot and its hash-bound saved execution receipts.
No runtime, quality evaluation or model selection is performed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .large_six_summary import cost_group, costs


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if parsed.tzinfo is None:
        raise ValueError('Execution timestamps must declare their time zone')
    return parsed.timestamp()


def window(intervals):
    if not intervals:
        return dict(timed_allocations=0, first_start_utc=None, last_finish_utc=None,
                    elapsed_window_seconds=None, occupied_union_seconds=None)
    intervals = sorted(intervals)
    if any(end < start for start, end in intervals):
        raise ValueError('Execution finished before it started')
    left, right = intervals[0]
    occupied = 0.0
    for start, end in intervals[1:]:
        if start > right:
            occupied += right - left
            left, right = start, end
        else:
            right = max(right, end)
    occupied += right - left
    first, last = intervals[0][0], max(end for _, end in intervals)
    return dict(timed_allocations=len(intervals),
        first_start_utc=datetime.fromtimestamp(first, timezone.utc).isoformat(),
        last_finish_utc=datetime.fromtimestamp(last, timezone.utc).isoformat(),
        elapsed_window_seconds=last-first, occupied_union_seconds=occupied)


def summarize(ledger, receipt_root):
    cost_summary = costs(ledger)
    groups = defaultdict(list)
    runs = defaultdict(lambda: defaultdict(list))
    all_intervals = []
    absent = defaultdict(int)
    source_hashes = []
    cached_path = cached_hash = cached_receipt = None
    for row in ledger['completed_stages']:
        group = cost_group(row['stage'])
        if 'observation_file' not in row:
            absent[group] += 1
            continue
        relative = Path(row['observation_file'].replace('\\', '/'))
        path = (receipt_root / relative).resolve()
        if receipt_root.resolve() not in path.parents:
            raise ValueError('Timing receipt is outside the supplied receipt root')
        if path != cached_path:
            raw = path.read_bytes()
            cached_path, cached_hash, cached_receipt = path, hashlib.sha256(raw).hexdigest(), json.loads(raw)
        if cached_hash != row['observation_sha256']:
            raise ValueError('Timing receipt path or bytes differ from cost ledger')
        receipt = cached_receipt
        if 'execution' in receipt:
            execution = receipt['execution']
        elif 'stages' in receipt and row['stage'] in receipt['stages']:
            execution = receipt['stages'][row['stage']]['execution']
        else:
            raise ValueError('Bound receipt does not identify this stage execution')
        for key in ('status', 'gpu_seconds', 'worker_exited', 'lease_released'):
            if execution[key] != row[key]:
                raise ValueError('Execution receipt and terminal allocation ledger differ')
        interval = timestamp(execution['started_at_utc']), timestamp(execution['finished_at_utc'])
        groups[group].append(interval)
        all_intervals.append(interval)
        source_hashes.append(row['observation_sha256'])
        for alias, recipe in [('v730','V8.1'), ('v731','V8.2'), ('v732','V8.3')]:
            for seed in (0,1):
                if alias+'-seed'+str(seed) in row['stage']:
                    runs[recipe+'/seed'+str(seed)][group].append(interval)
    return dict(schema_version=1, cost_ledger_allocations=cost_summary['completed_allocations'],
        source_receipt_hashes=sorted(source_hashes), measured_gpu_execution_window=window(all_intervals),
        groups={group: dict(**window(groups[group]), allocations_missing_timestamps=absent[group],
            allocated_gpu_seconds=cost_summary['groups'][group]['allocated_gpu_seconds'])
            for group in cost_summary['groups']},
        models={name: {group: window(items) for group,items in parts.items()} for name,parts in runs.items()},
        notes=['Elapsed window spans first start to last finish and includes gaps; it is not summed device time.',
               'Occupied union counts overlapping allocations once and excludes gaps within the group.',
               'Missing timestamp receipts are explicit; their GPU allocation seconds remain in costs.',
               'These are GPU execution windows. Preparation before first launch, CPU scoring, packaging, review and publication are not a full task wall-time measurement.',
               'CPU evaluator elapsed times are reported separately for each policy in the benchmark aggregate.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ledger', required=True)
    parser.add_argument('--receipt-root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    ledger_bytes = Path(args.ledger).read_bytes()
    value = summarize(json.loads(ledger_bytes), Path(args.receipt_root))
    value['cost_ledger_sha256'] = hashlib.sha256(ledger_bytes).hexdigest()
    with Path(args.output).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(timed_allocations=value['measured_gpu_execution_window']['timed_allocations'],
        cost_ledger_allocations=value['cost_ledger_allocations'], public_runtime_bindings_included=False)))


if __name__ == '__main__':
    main()
