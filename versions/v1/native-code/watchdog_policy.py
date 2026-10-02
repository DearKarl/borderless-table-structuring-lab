"""Pure fail-closed classification; caller binds JSON bytes to native artifacts."""
import math
import hashlib
import json
from pathlib import Path

POLICY = 'HYBRID-WATCHDOG-POLICY-v1'
PAGE_BUDGET_SECONDS = 900


def verified_page_timeout(page, event, decision, exit_record, receipt_exists):
    try:
        assert receipt_exists is False
        assert event['page_id'] == page['page_id'] == decision['page_id']
        assert decision['policy'] == POLICY and decision['phase'] == 'page'
        assert decision['input_sha256'] == page['input_sha256']
        assert decision['page_budget_seconds'] == PAGE_BUDGET_SECONDS
        elapsed = decision['phase_elapsed_seconds']
        assert type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > PAGE_BUDGET_SECONDS
        assert decision['loaded'] is True and decision['running_before_stop'] is True
        assert decision['started_event_sha256'] and decision['runtime_key']
        assert exit_record['cid'] == decision['cid']
        assert exit_record['watchdog_runtime_key'] == decision['runtime_key']
        assert exit_record['timeout'] == 'page' and exit_record['watchdog_kill_succeeded'] is True
        assert exit_record['all_owned_container_processes_stopped'] is True
        state = exit_record['state']
        assert state['Running'] is False and state['OOMKilled'] is False and state['ExitCode'] == 137
    except (AssertionError, KeyError, TypeError, ValueError, OverflowError):
        return False
    return True


def failure_action(row, verified_watchdog=False):
    """Counts are maintained separately; a proof is not inferred from the error string."""
    if (verified_watchdog is True and row.get('status') == 'failed'
            and row.get('error') == 'owned_container_page' and row.get('primary_bytes') == 0):
        return 'continue_unstarted'
    if verify_terminal_empty(row):
        return 'continue_unstarted'
    if row.get('status') == 'failed' or row.get('error'):
        return 'stop'
    return 'unchanged'


def verify_terminal_timeout(row):
    """Recheck proof bytes on both fresh collection and same-run inspection."""
    try:
        if row.get('error') != 'owned_container_page' or row.get('status') != 'failed' or row.get('primary_bytes') != 0:
            return False
        items = row['native_artifacts']
        exits = [Path(x['path']) for x in items if Path(x['path']).name == 'EXIT.json']
        assert len(exits) == 1
        folder = exits[0].parent
        def load(path):
            matches = [a for a in items if Path(a['path']) == path]
            assert len(matches) == 1 and path.is_file() and not path.is_symlink()
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == matches[0]['sha256']
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        ex, _ = load(folder / 'EXIT.json')
        decision, decision_sha = load(folder / 'WATCHDOG.json')
        start, _ = load(folder / 'START.json')
        assigned, assigned_sha = load(folder / 'ASSIGNED.json')
        _, loaded_sha = load(folder / 'loaded.json')
        pid = row['page_id']
        assert Path(pid).name == pid and pid not in ('.', '..')
        event, event_sha = load(folder / 'events' / (pid + '.started.json'))
        pages = [p for p in assigned['pages'] if p['page_id'] == pid]
        assert len(pages) == 1 and pages[0]['input_sha256'] == row['input_sha256']
        assert start['cid'] == decision['cid'] == ex['cid']
        assert start['page_timeout'] == 900 and start['load_timeout'] == 600
        assert row['runtime_key'] == start['runtime_key'] == assigned['runtime_key'] == decision['runtime_key']
        assert decision['assigned_sha256'] == assigned_sha and decision['loaded_sha256'] == loaded_sha
        assert decision['started_event_sha256'] == event_sha and ex['watchdog_sha256'] == decision_sha
        return verified_page_timeout(pages[0], event, decision, ex, (folder / pid / 'receipt.json').exists())
    except (AssertionError, KeyError, TypeError, ValueError, OSError):
        return False


"""Appended to the pinned watchdog policy; validates empty-return proof bytes."""


def verify_terminal_empty(row):
    try:
        assert row['status'] == 'failed' and row['primary_bytes'] == 0 and row.get('error') is None
        assert row['schema_present'] is True and row['schema_proof']['valid'] is True
        assert row['native_stop_audit_complete'] is False and row['stops'] == []
        items = row['native_artifacts']
        proofs = [x for x in items if Path(x['path']).name == 'EMPTY_RETURN_AUDIT.json']
        assert len(proofs) == 1
        proof_path = Path(proofs[0]['path'])
        assert proof_path.is_file() and not proof_path.is_symlink()
        raw = proof_path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == proofs[0]['sha256']
        proof = json.loads(raw)
        assert proof == row['empty_return_proof']
        assert proof['policy'] == 'HYBRID-NATIVE-EMPTY-v1' and proof['kind'] == 'verified_native_empty_return'
        assert proof['page_id'] == row['page_id'] and proof['input_sha256'] == row['input_sha256']
        assert proof['runtime_key'] == row['runtime_key'] and proof['status_remains'] == 'failed'
        assert proof['quality_correctness_claimed'] is False and proof['GT_used'] is False
        assert proof['generation_stop_audit_complete'] is False
        folder = proof_path.parent
        required = {str(folder / n) for n in ('native.json','receipt.json','prediction.md','generation-calls.json')}
        required |= {str(folder.parent / n) for n in ('ASSIGNED.json','START.json','EXIT.json','parameters.json',
                     'effective-pipeline.yaml','loaded-model-details.json','model-generation-config.json','loaded.json')}
        required.add(str(folder.parent / 'events' / (row['page_id'] + '.started.json')))
        assert required <= set(proof['artifact_pins']) and row['actual_started'] is True
        for name, digest in proof['artifact_pins'].items():
            matches = [x for x in items if x['path'] == name]
            assert len(matches) == 1 and matches[0]['sha256'] == digest
            p = Path(name)
            assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
        native = json.loads((folder / 'native.json').read_bytes())
        assert native['parsing_res_list'] == [] and native['layout_det_res']['boxes'] == []
        assert (folder / 'prediction.md').read_bytes() == b''
        return True
    except (AssertionError, KeyError, TypeError, ValueError, OSError):
        return False
