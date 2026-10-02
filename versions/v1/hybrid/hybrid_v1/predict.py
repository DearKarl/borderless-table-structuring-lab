"""Unified original-image entry; fresh and verified-cache development modes."""
import argparse
import json
import os
from pathlib import Path
import re

from .assembly import assemble_page, artifact, save
from .cache import NativeCache, checked, key, read, sha
from .capacity import validate_capacity
from .page_paths import component_name, mapping_for_pages


def call_evidence(prediction, fresh):
    row = prediction['terminal']
    pid = row['page_id']
    native_directory = prediction.get('native_page_directory_name',pid)
    artifacts = row['native_artifacts']
    def candidates(name):
        return [Path(a['path']) for a in artifacts if Path(a['path']).name == name
                and Path(a['path']).parent.name == native_directory]
    native_receipts = candidates('receipt.json')
    assert len(native_receipts) <= 1
    returned = bool(native_receipts and read(native_receipts[0])['status'] == 'returned')
    calls = candidates('generation-calls.json')
    assert len(calls) <= 1
    capture = read(calls[0]) if calls else None
    assert capture is None or isinstance(capture, list)
    return dict(actual_native_started=row['actual_started'], native_returned=returned,
                terminal_status=row['status'], native_invocations_this_run=int(fresh and row['actual_started']),
                generation_calls_captured=len(capture) if capture is not None else None,
                capture_present=capture is not None,
                generation_capture_complete=None,
                generation_calls_captured_semantics='persisted returned-generation records; not all attempted calls',
                generation_attempt_completeness='unknown: native workers do not journal every attempted call before invocation',
                generation_calls_are_not_native_invocation_count=True)


def validate_pages(pages, input_root):
    from PIL import Image
    assert pages and len({p['page_id'] for p in pages}) == len(pages)
    assert len({Path(p['image_path']).name for p in pages}) == len(pages), 'Ambiguous Docker image basenames'
    root = Path(input_root).resolve()
    for page in pages:
        assert set(page) <= {'page_id', 'image_path', 'input_sha256', 'width', 'height', 'bytes', 'relative_path'}
        pid = page['page_id']
        assert isinstance(pid, str) and pid and pid not in ('.', '..') and '/' not in pid and '\\' not in pid
        source = Path(page['image_path'])
        assert all(type(page[k]) is int and page[k] > 0 for k in ('width', 'height'))
        assert source.resolve().is_relative_to(root) and not source.is_symlink()
        assert sha(source) == page['input_sha256']
        with Image.open(source) as image:
            assert image.format in ('PNG', 'JPEG') and image.size == (page['width'], page['height'])
    return mapping_for_pages(pages,require_original_primary=True)


def run(pages, output, freeze, mode, manifest_sha256):
    assert mode in ('fresh', 'reuse') and freeze['product'] == 'hybrid_v1'
    assert re.fullmatch('[a-z0-9-]{1,24}', freeze['run_id'])
    assert freeze['execution_policy'] == {'M_workers': 1, 'P_workers': 1,
                                          'arm_order': ['mineru', 'paddle'], 'shard_size': 8}
    assert freeze['input_manifest_sha256'] == manifest_sha256
    if freeze.get('final_acceptance'):
        assert mode == 'fresh' and len(pages) == 1651
        validate_capacity(freeze)
        from .fresh_contract import verify_capacity_binding
        verify_capacity_binding(freeze)
    for item in freeze['code_artifacts']:
        checked(item)
    from hybrid.formula_v0 import core as formula_core
    required_code = {p.resolve() for p in Path(__file__).parent.glob('*.py')}
    required_code.add(Path(formula_core.__file__).resolve())
    assert required_code <= {Path(a['path']).resolve() for a in freeze['code_artifacts']}
    path_map=validate_pages(pages, freeze['input_root'])
    root = Path(output)
    start = dict(product='hybrid_v1', mode=mode, system_freeze_sha256=key(freeze),
                 input_manifest_sha256=manifest_sha256, page_count=len(pages))
    if root.exists():
        assert read(root / 'SYSTEM_FREEZE.json') == freeze
        assert read(root / 'INPUT_MANIFEST.json') == {'pages': pages, 'GT_free': True}
        assert read(root / 'RUN_START.json') == start
        assert not (root / 'PAUSED.json').exists(), 'A paused run requires explicit owner recovery'
    else:
        root.mkdir(parents=True, exist_ok=False)
        save(root / 'SYSTEM_FREEZE.json', freeze)
        save(root / 'INPUT_MANIFEST.json', {'pages': pages, 'GT_free': True})
        save(root / 'RUN_START.json', start)
    if (root/'PAGE_FILE_MAP.json').exists():
        assert read(root/'PAGE_FILE_MAP.json')==path_map
    else:
        save(root/'PAGE_FILE_MAP.json',path_map)
    name_max=os.pathconf(root,'PC_NAME_MAX');assert name_max>=255
    name_proof={**path_map['summary'],'passed':True,'filesystem_name_max':name_max,
                'mapping':artifact(root/'PAGE_FILE_MAP.json'),'before_native_dispatch':True}
    if (root/'NAME_PREFLIGHT.json').exists():
        assert read(root/'NAME_PREFLIGHT.json')==name_proof
    else:
        save(root/'NAME_PREFLIGHT.json',name_proof)
    import fcntl
    leader = (root / 'controller.lease').open('a')
    fcntl.flock(leader, fcntl.LOCK_EX | fcntl.LOCK_NB)
    caches = {}
    if mode == 'reuse':
        for arm in ('mineru', 'paddle'):
            binding = freeze['cache_bindings'][arm]
            caches[arm] = NativeCache(binding['lock_path'], binding['terminal_root'], pages, arm,
                                     binding['lock_sha256'], binding['runtime_key'])
            expected_native = read(checked(freeze['native_bindings'][arm]['runtime_artifact']))
            assert key(expected_native) == caches[arm].expected_runtime_key
        native = None
    else:
        from .native import RealNativeAdapters
        native = RealNativeAdapters(freeze, root, pages, manifest_sha256)
    rows = []
    baseline_rows = []
    receipts = []
    for index, offset in enumerate(range(0, len(pages), 8)):
        group = pages[offset:offset + 8]
        predictions = {}
        for arm in ('mineru', 'paddle'):
            if mode == 'reuse':
                predictions[arm] = {p['page_id']: caches[arm].page(p['page_id']) for p in group}
            else:
                predictions[arm], remaining, pause = native.batch(arm, group, index)
                if remaining or pause:
                    save(root / 'PAUSED.json', dict(arm=arm, shard=index, unstarted=remaining,
                                                  native_paused=pause, complete=False))
                    return {'complete': False, 'paused': True, 'assembled': len(rows)}
        for page in group:
            pid = page['page_id']
            m, p = predictions['mineru'][pid], predictions['paddle'][pid]
            provenance = dict(mode=mode, cache_policy='disabled' if mode == 'fresh' else 'verified_external_native',
                              cache_hits={'M': mode == 'reuse', 'P': mode == 'reuse'},
                              native_calls_this_run={'M': int(mode == 'fresh'), 'P': int(mode == 'fresh')},
                              native_terminal_sha256={'M': m['terminal_sha256'], 'P': p['terminal_sha256']},
                              native_runtime_key={'M': m['terminal']['runtime_key'], 'P': p['terminal']['runtime_key']},
                              native_call_evidence={'M': call_evidence(m, mode == 'fresh'),
                                                    'P': call_evidence(p, mode == 'fresh')},
                              resumed_within_run={'M': m.get('resumed_within_run', False),
                                                  'P': p.get('resumed_within_run', False)},
                              native_calls_this_invocation={
                                  'M': int(mode == 'fresh' and not m.get('resumed_within_run', False)),
                                  'P': int(mode == 'fresh' and not p.get('resumed_within_run', False))},
                              system_freeze_sha256=key(freeze))
            folder = root / 'pages' / component_name(pid)
            if folder.exists():
                receipt, row = read(folder / 'receipt.json'), read(folder / 'terminal.json')
                assert receipt['native_provenance']['system_freeze_sha256'] == key(freeze)
                assert receipt['native_provenance']['native_terminal_sha256'] == provenance['native_terminal_sha256']
                assert row['input_sha256'] == page['input_sha256'] and row['page_id'] == pid
                assert sha(root / 'primary' / component_name(pid,'.md')) == row['prediction_sha256']
                for item in row['native_artifacts']:
                    checked(item)
            else:
                receipt, row = assemble_page(page, m, p, root, provenance)
            receipts.append(receipt); rows.append(row)
            baseline_rows.append({field: m['terminal'][field] for field in
                                  ('page_id', 'input_sha256', 'status', 'prediction_sha256', 'native_artifacts')})
    summary = dict(product='hybrid_v1', complete=True, mode=mode, pages=len(rows),
                   system_freeze_sha256=key(freeze), input_manifest_sha256=manifest_sha256,
                   native_invocations={'M': len(rows) if mode == 'fresh' else 0,
                                       'P': len(rows) if mode == 'fresh' else 0},
                   cache_hits_per_arm=len(rows) if mode == 'reuse' else 0,
                   modified_pages=sum(r['replacement_count'] > 0 for r in receipts),
                   formula_replacements=sum(r['replacement_count'] for r in receipts),
                   all_byte_invariants_verified=all(r['nonreplacement_bytes_verified'] for r in receipts),
                   GT_used=False, quality_gain_claimed=False)
    if (root / 'ASSEMBLY_RECEIPT.json').exists():
        assert read(root / 'ASSEMBLY_RECEIPT.json') == summary
    else:
        save(root / 'ASSEMBLY_RECEIPT.json', summary)
    if len(rows) == 1651:
        final = dict(arm='hybrid_v1', complete=True, runtime_key=key(freeze),
             prediction_directory=str(root / 'primary'), pages=rows,
             assembly_receipt=artifact(root / 'ASSEMBLY_RECEIPT.json'), mode=mode)
        if (root / 'COMPLETE_INPUT_LOCK.json').exists():
            assert read(root / 'COMPLETE_INPUT_LOCK.json') == final
        else:
            save(root / 'COMPLETE_INPUT_LOCK.json', final)
        baseline = dict(arm='B', complete=True, runtime_key=key(freeze),
                        prediction_directory=str(root / 'native/mineru/primary') if mode == 'fresh'
                        else str(caches['mineru'].primary), pages=baseline_rows,
                        system_freeze_sha256=key(freeze), mode=mode)
        if (root / 'B_COMPLETE_INPUT_LOCK.json').exists():
            assert read(root / 'B_COMPLETE_INPUT_LOCK.json') == baseline
        else:
            save(root / 'B_COMPLETE_INPUT_LOCK.json', baseline)
    return summary


def main():
    parser = argparse.ArgumentParser(prog='hybrid_v1')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('predict-page', 'predict-manifest'):
        command = commands.add_parser(name)
        command.add_argument('--image' if name == 'predict-page' else '--manifest', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--freeze', type=Path, required=True)
        command.add_argument('--freeze-sha256', required=True)
        command.add_argument('--mode', choices=('fresh', 'reuse'), default='fresh')
    args = parser.parse_args()
    assert sha(args.freeze) == args.freeze_sha256
    freeze = read(args.freeze)
    if args.command == 'predict-page':
        from PIL import Image
        source = args.image.resolve()
        with Image.open(source) as image:
            width, height = image.size
        page = dict(page_id=source.stem, image_path=str(source), input_sha256=sha(source), width=width, height=height)
        pages = [page]
        manifest_sha256 = key({'pages': pages, 'GT_free': True})
    else:
        document = read(args.manifest)
        assert set(document) <= {'revision', 'pages', 'GT_free'} and document['GT_free'] is True
        pages = document['pages']; manifest_sha256 = sha(args.manifest)
    result = run(pages, args.output, freeze, args.mode, manifest_sha256)
    print(json.dumps(result), flush=True)
    if not result['complete']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
