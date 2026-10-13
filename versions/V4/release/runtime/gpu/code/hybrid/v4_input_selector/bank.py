"""Host-only frozen S1/S2 bank. Never send master/roster/labels to inference.

The native container provider is explicit; this module does not start Docker,
SSH, a scheduler, a model or a training process on import.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib.parse import quote
import uuid

from .schema import FEATURES, SourceRecord, numeric_features
from .input_features import DEFINITION_ID,GRID_HASHES
from .worker_contract import (ROOT_KEYS, PAGE_SECONDS, CLEANUP_SECONDS, atomic_json, bound_file, digest, file_sha,
                              hash_value, safe_id, validate_contract)
from .supervisor import run_queue, work_alarm

OWNER = 'public-owner-requires-binding'
PLAN_SHA = 'a2a67eb7dbda8a8d32b4b42a03f66ff1c173f6de14e1fee327918d4d2aebce68'
PUBLIC_PLAN_SHA = '96424e60b86c28cbc491fbf29738e445f069906700d8bfeb49c47b7f5ea43b3e'
PUBLIC_SPLIT = dict(fit=32, calibration=8, held_out=8)
PUBLIC_STARTS = 144
WALL_SECONDS = GPU_SECONDS = 43200
STARTS = 180
SMOKE_STARTS = 6
INNER_CLEANUP_SECONDS = 25
OUTER_CLEANUP_SECONDS = 30
PERSIST_SECONDS = 5
READINESS = {'runtime_native', 'parameter_inventory', 'actions_metrics', 'feature_boundary', 'launch_adapter'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def value_sha(value):
    return digest(canonical(value))


def keys(value, names, message):
    require(isinstance(value, dict) and set(value) == set(names), message)


def checked_json(binding, max_bytes=16 * 1024 * 1024):
    keys(binding, ('path', 'sha256'), 'Exact path/hash descriptor required')
    p = Path(binding['path'])
    require(p.is_absolute() and hash_value(binding['sha256']), 'Unbound host artifact')
    with p.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    require(len(raw) <= max_bytes and digest(raw) == binding['sha256'], 'Host artifact size/hash differs')
    return json.loads(raw)


def no_unknowns(value):
    if isinstance(value, dict):
        return all(no_unknowns(v) for v in value.values())
    if isinstance(value, list):
        return all(no_unknowns(v) for v in value)
    return value is not None


class FrozenBank:
    """Read-only host authority. Construction refuses an unaccepted roster."""
    def __init__(self, path, expected_sha):
        self.master = m = checked_json(dict(path=str(Path(path).resolve()), sha256=expected_sha))
        keys(m, ('schema', 'bank_id', 'bank_root', 'plan_sha256', 'roster', 'roster_acceptance',
                 'worker_template', 'readiness', 'lease_directory', 'gpu_uuid', 'adapter_sources', 'input_root'), 'Malformed bank master')
        self.public = m['schema'] == 'nju_public_bank_master_v1'
        self.max_starts = PUBLIC_STARTS if self.public else STARTS
        self.smoke_seconds = 3600 if self.public else WALL_SECONDS
        self.max_allocations = 2 if self.public else None
        require(m['schema'] in ('v4_bank_master_v1','nju_public_bank_master_v1') and safe_id(m['bank_id'])
                and m['plan_sha256'] == (PUBLIC_PLAN_SHA if self.public else PLAN_SHA),
                'Unknown bank/plan identity')
        require(Path(m['bank_root']).is_absolute() and Path(m['lease_directory']).is_absolute(), 'Canonical host paths required')
        self.root = Path(m['bank_root']).resolve()
        require(str(self.root) == m['bank_root'], 'Bank root must be canonical; no alternate master location')
        self.sha = expected_sha
        self.master_path = Path(path).resolve()
        require(Path(m['input_root']).is_absolute(), 'Dedicated projected input root required')
        self.input_root = Path(m['input_root']).resolve()
        require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', m['gpu_uuid']) is not None, 'Exact GPU UUID required')
        uuid.UUID(m['gpu_uuid'][4:])
        roster = checked_json(m['roster'])
        keys(roster, ('schema', 'pages'), 'Malformed frozen roster')
        require(roster['schema'] == ('nju_public_roster_v1' if self.public else 'v4_frozen_roster_v1')
                and len(roster['pages']) == (48 if self.public else 60), 'Frozen profile page count/schema differs')
        acceptance = checked_json(m['roster_acceptance'])
        if self.public:
            require(acceptance.get('schema') == 'nju_public_roster_acceptance_v1' and acceptance.get('owner') == OWNER
                    and acceptance.get('roster_sha256') == m['roster']['sha256']
                    and acceptance.get('source_join_reviewed') is True and acceptance.get('source_separation') is True
                    and acceptance.get('public_silver_reference_acknowledged') is True
                    and acceptance.get('protocol_addendum_sha256') and hash_value(acceptance['protocol_addendum_sha256']),
                    'Root public-source/split/protocol acceptance missing')
        else:
            require(acceptance.get('schema') == 'v4_roster_acceptance_v1' and acceptance.get('owner') == OWNER
                and acceptance.get('roster_sha256') == m['roster']['sha256'] and acceptance.get('independent_review') is True
                and acceptance.get('provenance_legal') is True and acceptance.get('source_separation') is True,
                'Missing independent frozen roster acceptance')
        families, page_ids, originals, refs = {}, set(), set(), {}
        self.items, self.host_pages, self.staging = {}, {}, {}
        fit_pages = []
        split_pages = dict(fit=0, calibration=0, held_out=0)
        for n, row in enumerate(roster['pages']):
            keys(row, ('page_id', 'source_family', 'split', 'input', 'reference'), 'Unexpected host roster fields')
            require(safe_id(row['page_id']) and row['page_id'] not in page_ids and safe_id(row['source_family']), 'Repeated/malformed roster identity')
            page_ids.add(row['page_id'])
            family, split = row['source_family'], row['split']
            require(split in split_pages and families.setdefault(family, split) == split, 'Source family crosses splits')
            split_pages[split] += 1
            inp, ref = row['input'], row['reference']
            keys(inp, ('path', 'sha256', 'source'), 'Malformed original input binding')
            keys(ref, ('path','sha256','score_version') if self.public else ('path', 'sha256', 'states'), 'Malformed reference binding')
            if self.public:
                require(ref['score_version'] == 'PublicTranscriptEdit_v1', 'Public score version differs')
            else:
                require(set(ref['states']) == {'text', 'table', 'formula'} and
                    all(v in ('present', 'verified_absent') for v in ref['states'].values()), 'Unlabeled page is not bank eligible')
            require(Path(inp['path']).is_absolute() and Path(ref['path']).is_absolute(), 'Host source/reference paths must be absolute')
            require(hash_value(inp['sha256']) and hash_value(ref['sha256']) and
                    file_sha(ref['path']) == ref['sha256'], 'Frozen reference changed')
            source = SourceRecord(**inp['source'])
            require(source.original_file_sha256 == inp['sha256'], 'Original source/hash differs')
            if self.public:
                require(source.source_type == 'original_pdf' and source.original_page_ordinal == 0,
                        'Public supplied single-page PDF must use worker index zero')
            original = (inp['sha256'], source.original_page_ordinal)
            require(original not in originals, 'Repeated original page in bank')
            originals.add(original)
            opaque = 'p' + str(n).zfill(3)
            suffix = Path(inp['path']).suffix.lower()
            require(suffix in ('.pdf', '.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp'), 'Unsupported staged input suffix')
            file = 'pages/' + opaque + suffix
            self.staging[file] = dict(path=inp['path'], sha256=inp['sha256'])
            self.host_pages[opaque] = deepcopy(row)
            refs[row['page_id']] = ref['sha256']
            if split == 'fit':
                fit_pages.append(opaque)
            for action in source.available_actions():
                slot = opaque + '_' + action
                self.items[slot] = dict(item_id=slot, page_id=opaque, file=file, input_sha256=inp['sha256'],
                                        action=action, source=deepcopy(inp['source']))
        if self.public:
            require(split_pages == PUBLIC_SPLIT and len(families) >= 24 and
                    max(sum(row['source_family']==g for row in self.host_pages.values()) for g in families) <= 2,
                    'Public bank requires 48 pages, 32/8/8, >=24 groups and <=2 pages/group')
        else:
            require({s: list(families.values()).count(s) for s in split_pages} == dict(fit=12, calibration=4, held_out=4)
                and split_pages['held_out'] <= 12, 'Current frozen 20-family 12/4/4 allocation required')
        require(acceptance.get('reference_hashes') == refs and set(acceptance.get('reviewed_page_ids', [])) == page_ids,
                'Root reference integrity review does not cover the frozen roster')
        self.smoke = {k for k, item in self.items.items() if item['page_id'] in fit_pages[:2]}
        require(len(fit_pages) >= 2 and len(self.smoke) <= SMOKE_STARTS and len(self.items) <= self.max_starts, 'Frozen slot ceiling exceeded')
        self.template = checked_json(m['worker_template'])
        keys(self.template, (ROOT_KEYS | {'runtime_binding'}) - {'job_id', 'wall_seconds', 'items'}, 'Inference template must exclude job/queue fields')
        require(self.template['schema'] == 'v4_tele_worker_v2', 'Bound native worker template required')
        profile = deepcopy(self.template['runtime_binding'])
        require(profile['external_image']['receipt'] is None, 'External launch receipt must be supplied by live owned inspection')
        del profile['external_image']['receipt']
        require(no_unknowns(profile) and profile['device']['uuid'] == m['gpu_uuid'], 'Unresolved runtime source/weight/device facts')
        readiness = checked_json(m['readiness'])
        core = {k: v for k, v in m.items() if k != 'readiness'}
        require(readiness.get('schema') == 'v4_bank_readiness_v1' and readiness.get('owner') == OWNER and
                readiness.get('master_core_sha256') == value_sha(core) and readiness.get('accepted') is True and
                set(readiness.get('evidence', {})) == READINESS, 'Root launch readiness is incomplete')
        for binding in readiness['evidence'].values():
            checked_json(binding)
        self.adapter_sources = checked_json(m['adapter_sources'])
        require(isinstance(self.adapter_sources, dict) and self.adapter_sources, 'Missing adapter source freeze')
        for name, h in self.adapter_sources.items():
            require(Path(name).is_absolute() and hash_value(h) and file_sha(name) == h, 'Adapter source differs')
        self.private_paths = [self.master_path, self.root / 'bank.sqlite'] + [Path(m[k]['path']).resolve()
            for k in ('roster', 'roster_acceptance', 'readiness', 'worker_template', 'adapter_sources')]
        self.private_paths += [Path(row['reference']['path']).resolve() for row in self.host_pages.values()]
        require(not any(p == self.input_root or p.is_relative_to(self.input_root) for p in self.private_paths),
                'GT/master files overlap the inference input mount')

    def verify_projected_inputs(self, slots):
        for file in {self.items[s]['file'] for s in slots}:
            require(file_sha(bound_file(self.input_root, file)) == self.staging[file]['sha256'], 'Projected input bytes differ')

    def projection(self, slots, image_receipt, remaining):
        require(slots and len(set(slots)) == len(slots) and all(s in self.items for s in slots), 'Unfrozen/duplicate bank slot')
        require(math.isfinite(remaining) and 600 <= remaining <= WALL_SECONDS, 'Insufficient bounded initialization/page/cleanup time')
        c = deepcopy(self.template)
        c.update(job_id=self.master['bank_id'], wall_seconds=math.floor(remaining), items=[deepcopy(self.items[s]) for s in slots])
        c['runtime_binding']['external_image']['receipt'] = deepcopy(image_receipt)
        # Exact worker schema forbids roster, reference, family, split and labels.
        return validate_contract(c)


def host_clock():
    return dict(wall=time.time(), mono=time.monotonic(), boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


class BankLedger:
    def __init__(self, bank, *, create=False, clock=host_clock):
        self.bank, self.clock = bank, clock
        self.path = bank.root / 'bank.sqlite'
        if create:
            bank.root.mkdir(parents=False, exist_ok=False)
            db = sqlite3.connect(self.path)
            try:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL)')
                db.execute('CREATE TABLE events (seq INTEGER PRIMARY KEY, body TEXT NOT NULL, sha TEXT NOT NULL)')
                state = dict(master_sha256=bank.sha, phase='S1', started=None, last_clock=None,
                    gpu_seconds=0., active=None, allocations=[], gate=None, reserved_count=0,
                    slots={s: dict(phase='S1' if s in bank.smoke else 'S2', status='pending', reservation=None, result=None) for s in bank.items})
                db.execute('INSERT INTO state VALUES(1,?)', (canonical(state).decode(),))
                event = dict(seq=1, kind='created', previous=bank.sha, state_sha256=value_sha(state), detail={})
                db.execute('INSERT INTO events VALUES(?,?,?)', (1, canonical(event).decode(), value_sha(event)))
                db.commit()
            finally:
                db.close()
        require(self.path.is_file(), 'No existing canonical bank ledger; never recreate on resume')
        self.snapshot()

    @contextmanager
    def connection(self):
        uri = 'file:' + quote(self.path.as_posix(), safe='/:') + '?mode=rw'
        db = sqlite3.connect(uri, uri=True, timeout=5)
        db.execute('PRAGMA synchronous=FULL')
        try:
            yield db
        finally:
            db.close()

    def load(self, db):
        state = json.loads(db.execute('SELECT body FROM state WHERE id=1').fetchone()[0])
        require(state['master_sha256'] == self.bank.sha and set(state['slots']) == set(self.bank.items), 'Canonical master/slot identity differs')
        previous = self.bank.sha
        for seq, body, h in db.execute('SELECT seq,body,sha FROM events ORDER BY seq'):
            event = json.loads(body)
            require(event['seq'] == seq and event['previous'] == previous and value_sha(event) == h, 'Bank journal chain differs')
            previous = h
        require(event['state_sha256'] == value_sha(state), 'Bank state differs from committed event')
        require(state['reserved_count'] == sum(r['reservation'] is not None for r in state['slots'].values()), 'Global reservation count differs')
        return state, seq, previous

    def snapshot(self):
        with self.connection() as db:
            return self.load(db)[0]

    def timing(self, state, now):
        require(all(type(now[k]) in (int, float) and math.isfinite(now[k]) for k in ('wall', 'mono')) and now['boot'], 'Invalid host clock')
        old = state['last_clock']
        if old is not None:
            require(now['boot'] == old['boot'] and now['mono'] >= old['mono'] and now['wall'] >= old['wall'], 'Host reboot/clock rollback; no budget reset')
        spent = state['gpu_seconds']
        if state['active'] is not None:
            start = state['active']['started']
            spent += max(now['wall'] - start['wall'], now['mono'] - start['mono'])
        if state['started'] is None:
            remaining = min(WALL_SECONDS, getattr(self.bank,'smoke_seconds',WALL_SECONDS))
            return dict(remaining=remaining, global_remaining=WALL_SECONDS, gpu_seconds=spent, deadline_mono=now['mono'] + remaining)
        start = state['started']
        wall_remaining = min(start['wall'] + WALL_SECONDS - now['wall'], start['mono'] + WALL_SECONDS - now['mono'])
        remaining = min(wall_remaining, GPU_SECONDS - spent)
        global_remaining = remaining
        if state['phase']=='S1':
            cap=getattr(self.bank,'smoke_seconds',WALL_SECONDS)
            ended=state.get('smoke_ended') or now
            remaining=min(remaining, cap-max(ended['wall']-start['wall'],ended['mono']-start['mono']), cap-spent)
        return dict(remaining=remaining, global_remaining=global_remaining, gpu_seconds=spent, deadline_mono=now['mono'] + remaining)

    def remaining(self):
        return self.timing(self.snapshot(), self.clock())

    def change(self, kind, operation):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                state, seq, previous = self.load(db)
                now = self.clock()
                timing = self.timing(state, now)
                detail = operation(state, now, timing)
                state['last_clock'] = now
                event = dict(seq=seq + 1, kind=kind, previous=previous, state_sha256=value_sha(state), detail=detail)
                db.execute('UPDATE state SET body=? WHERE id=1', (canonical(state).decode(),))
                db.execute('INSERT INTO events VALUES(?,?,?)', (seq + 1, canonical(event).decode(), value_sha(event)))
                db.commit()
                return detail
            except BaseException:
                db.rollback()
                raise

    def begin_allocation(self, *, resume_unstarted=False):
        def mutate(s, now, t):
            require(s['active'] is None, 'Prior allocation is unresolved; cannot allocate again')
            require(not s['allocations'] or resume_unstarted, 'Another allocation requires explicit resume-unstarted')
            cap=getattr(self.bank,'max_allocations',None)
            require(cap is None or len(s['allocations']) < cap, 'Approved allocation ceiling exhausted; never reset')
            require(t['remaining'] >= 600, 'Global wall/GPU budget exhausted')
            if s['started'] is None:
                s['started'] = now
            allocation = dict(id=uuid.uuid4().hex, started=now, gpu_uuid=self.bank.master['gpu_uuid'], count=1,
                              status='reserved_before_acquisition')
            s['active'] = allocation
            s['allocations'].append(deepcopy(allocation))
            return allocation
        return self.change('allocation_reserved', mutate)

    def require_allocation(self, state, allocation):
        require(state['active'] is not None and state['active']['id'] == allocation, 'Wrong/unresolved allocation owner')

    def reserve(self, allocation, slot):
        def mutate(s, now, t):
            self.require_allocation(s, allocation)
            require(t['remaining'] > 60, 'Global deadline before dispatch')
            require(slot in s['slots'], 'Unknown slot')
            row = s['slots'][slot]
            require(row['phase'] == s['phase'], 'Explicit S1 gate required before S2')
            require(row['status'] == 'pending' and row['reservation'] is None, 'Never replay or refund a reserved slot')
            require(s['reserved_count'] < getattr(self.bank,'max_starts',STARTS) and (s['phase'] != 'S1' or sum(r['reservation'] is not None for r in s['slots'].values() if r['phase'] == 'S1') < SMOKE_STARTS), 'Global start ceiling')
            row.update(status='uncertain', reservation=dict(allocation=allocation, clock=now))
            s['reserved_count'] += 1
            return dict(slot=slot, charge=1, allocation=allocation)
        return self.change('slot_reserved_before_send', mutate)

    def result(self, allocation, slot, result, output_root):
        audit_path = bound_file(output_root, result['audit_file'])
        require(file_sha(audit_path) == result['audit_sha256'], 'Unbound validated result audit')
        audit = json.loads(audit_path.read_bytes())
        if result['status'] == 'completed':
            features = audit.get('features', {})
            row = features.get('row')
            source = self.bank.items[slot]['source']
            source_binding = {k: source[k] for k in ('original_file_sha256', 'original_page_ordinal', 'oriented_size')}
            require(features.get('status') in ('ready', 'unavailable') and isinstance(row, dict)
                    and set(row.get('features', {})) == set(FEATURES)
                    and row.get('definition_id') == DEFINITION_ID
                    and row.get('action') == self.bank.items[slot]['action']
                    and row.get('audit', {}).get('recognition_used') is False
                    and row['audit'].get('pdf_text_access') is False
                    and row['audit'].get('source_binding') == source_binding
                    and row['audit'].get('prepared') == audit.get('render', {}).get('prepared'), 'Missing pre-recognition feature observation')
            require(row.get('ready') is (features['status'] == 'ready'), 'Feature readiness disagrees')
            if row['ready']:
                require(row.get('vector') == numeric_features(row['features']) and
                        row['audit'].get('preflight_binding_hashes') == list(GRID_HASHES), 'Unbound numeric feature vector')
        def mutate(s, now, t):
            self.require_allocation(s, allocation)
            row = s['slots'][slot]
            require(row['status'] == 'uncertain' and row['reservation']['allocation'] == allocation, 'Unreserved/repeated result')
            require(result['status'] in ('completed', 'failed', 'timeout'), 'Unknown inference terminal result')
            row.update(status=result['status'], result=dict(payload=deepcopy(result), output_root=str(Path(output_root).resolve())))
            return dict(slot=slot, status=result['status'], audit_sha256=result['audit_sha256'])
        return self.change('slot_result', mutate)

    def failure(self, allocation, slot, error):
        def mutate(s, now, t):
            self.require_allocation(s, allocation)
            if slot in s['slots'] and s['slots'][slot]['reservation'] is not None:
                s['slots'][slot]['supervisor_error'] = error
                if s['slots'][slot]['status'] == 'completed':
                    s['slots'][slot]['status'] = 'uncertain'
            return dict(slot=slot, error=error, refunds=0)
        return self.change('queue_failure', mutate)

    def finish_allocation(self, allocation, release):
        require(release.get('allocation') == allocation and release.get('container_absent') is True and
                release.get('lease_released') is True and not release.get('error'), 'Missing positive owned allocation release')
        def mutate(s, now, t):
            self.require_allocation(s, allocation)
            s['gpu_seconds'] = t['gpu_seconds']
            s['allocations'][-1].update(status='released', ended=now, release=deepcopy(release))
            s['active'] = None
            if s['phase']=='S1' and all(s['slots'][k]['status'] in ('completed','failed','timeout') for k in self.bank.smoke):
                s['smoke_ended']=deepcopy(now)
            return dict(allocation=allocation, gpu_seconds=s['gpu_seconds'], release=release)
        return self.change('allocation_released', mutate)

    def s1_digest(self, state=None):
        s = state or self.snapshot()
        return value_sha({slot: s['slots'][slot] for slot in sorted(self.bank.smoke)})

    def accept_s1(self, receipt):
        gate = checked_json(receipt)
        def mutate(s, now, t):
            require(s['phase'] == 'S1' and s['gate'] is None and (not getattr(self.bank,'public',False) or s['active'] is None)
                    and t['global_remaining'] > 60 and t['remaining'] >= 0, 'S1 gate cannot reset phase/budget or bypass release/smoke ceiling')
            require(all(s['slots'][slot]['status'] in ('completed', 'failed', 'timeout') for slot in self.bank.smoke), 'Unstarted/uncertain smoke slots block gate')
            require(gate.get('schema') == 'v4_s1_gate_v1' and gate.get('owner') == OWNER and
                    gate.get('master_sha256') == self.bank.sha and gate.get('s1_result_sha256') == self.s1_digest(s) and
                    gate.get('checks') == dict(parsing_measurements=True, actions=True, data_separation=True, wiring_metrics_resolved=True),
                    'Explicit root S1 acceptance for these exact results required')
            s.update(phase='S2', gate=deepcopy(receipt))
            return dict(phase='S2', gate=receipt, global_starts_unchanged=s['reserved_count'])
        return self.change('s1_gate_accepted', mutate)

    def pending(self, phase):
        state = self.snapshot()
        require(phase == state['phase'], 'Requested phase is not authorized')
        return [slot for slot in self.bank.items if state['slots'][slot]['phase'] == phase and state['slots'][slot]['status'] == 'pending']


class ExistingRuntimeAdapter:
    """Narrow host wrapper around a separately frozen existing launch harness.

Callbacks normalize the existing shared lease/live GPU probe/owned inspection/
transport/cleanup evidence. They are not called anywhere during local preparation.
"""
    def __init__(self, bank, *, lease_factory, fresh_probe, inspect_owned, transport_factory, release_owned):
        self.bank = bank
        self.lease_factory, self.fresh_probe = lease_factory, fresh_probe
        self.inspect_owned, self.transport_factory, self.release_owned = inspect_owned, transport_factory, release_owned
        self.lease = None
        self.entered = False
        for fn in (lease_factory, fresh_probe, inspect_owned, transport_factory, release_owned):
            source = inspect.getsourcefile(fn)
            require(source is not None and self.bank.adapter_sources.get(str(Path(source).resolve())) == file_sha(source),
                    'Unfrozen launch adapter callback')

    def acquire(self, allocation, deadline):
        uid = self.bank.master['gpu_uuid']
        self.lease = self.lease_factory(self.bank.master['lease_directory'], [uid])
        require(list(self.lease.keys) == [uid] and Path(self.lease.directory).resolve() == Path(self.bank.master['lease_directory']).resolve(),
                'Shared lease key/directory differs')
        self.lease.__enter__()
        self.entered = True
        before = time.time()
        observed = self.fresh_probe(uid, deadline)
        require(observed.get('uuid') == uid and observed.get('memory_used_mib') == 0 and observed.get('compute_pids') == []
                and before <= observed.get('observed_at', -1) <= time.time(), 'Fresh exact UUID idle observation required after lease')
        inspected_after = time.time()
        image = self.inspect_owned(allocation, deadline)
        require(image.get('job_id') == self.bank.master['bank_id'] and image.get('image') == self.bank.template['runtime_binding']['external_image']['image']
                and image.get('gpu_uuid') == uid and image.get('owner_token') == allocation
                and inspected_after <= image.get('observed_at', -1) <= time.time()
                and isinstance(image.get('container_id'), str) and re.fullmatch('[0-9a-f]{64}', image['container_id']),
                'Owned image/device/mount inspection differs')
        run = self.bank.root / 'runs' / allocation
        required_mounts = {str(self.bank.input_root): (str(self.bank.input_root), False),
                           str(run / 'control'): (str(run / 'control'), False),
                           str(run / 'output'): (str(run / 'output'), True)}
        observed_mounts = {}
        for mount in image.get('mounts', []):
            keys(mount, ('source', 'target', 'writable'), 'Malformed actual mount inspection')
            source = Path(mount['source']).resolve()
            require(source.is_absolute() and mount['target'] not in observed_mounts and type(mount['writable']) is bool, 'Ambiguous mount')
            require(not any(p == source or p.is_relative_to(source) for p in self.bank.private_paths), 'Host master/GT would be mounted into inference')
            require(not mount['writable'] or (str(source), mount['target']) == (str(run / 'output'), str(run / 'output')), 'Unexpected writable inference mount')
            observed_mounts[mount['target']] = (str(source), mount['writable'])
        require(all(observed_mounts.get(target) == values for target, values in required_mounts.items()), 'Required narrow same-path inference mounts absent')
        self.observed = image
        receipt = dict(schema='v4_external_image_v1', job_id=image['job_id'], image=image['image'],
                       container_id=image['container_id'], inspected_at_utc=image['inspected_at_utc'])
        return receipt

    def transport(self, owner, logs, projection_path, projection_sha, output_root, deadline):
        require(self.lease is not None and self.entered, 'No held shared lease')
        return self.transport_factory(owner=owner, logs=str(logs), contract=str(projection_path),
            contract_sha256=projection_sha, output_root=str(output_root), deadline=deadline,
            input_root=str(self.bank.input_root), container_id=self.observed['container_id'], gpu_uuid=self.bank.master['gpu_uuid'])

    def release(self, allocation, deadline):
        release = self.release_owned(allocation, deadline)
        require(release.get('allocation') == allocation and release.get('container_absent') is True and not release.get('error')
                and (not hasattr(self, 'observed') or release.get('container_id') == self.observed['container_id']),
                'Owned runtime release unresolved; preserve allocation charge')
        require(self.lease is not None, 'Cannot claim release of an unheld lease')
        self.lease.__exit__(None, None, None)
        self.lease = None
        self.entered = False
        return dict(release, lease_released=True)


class BankBudget:
    """Only this host object sees the global ledger; worker receives no master."""
    def __init__(self, ledger, allocation, *, finalizer=None, startup_hard_deadline=None):
        self.ledger, self.allocation = ledger, allocation
        self.cleanup = None
        self.finalizer, self.failures = finalizer, []
        self.deadline = ledger.remaining()['deadline_mono']
        self.hard_deadline = min(self.deadline, startup_hard_deadline if startup_hard_deadline is not None else self.deadline)

    def absolute_deadline(self):
        self.deadline = min(self.deadline, self.ledger.remaining()['deadline_mono'])
        return self.deadline

    def inner_cleanup_deadline(self, hard_deadline, now):
        return min(hard_deadline - OUTER_CLEANUP_SECONDS - PERSIST_SECONDS, now + INNER_CLEANUP_SECONDS)

    def before_start(self):
        self.ledger.require_allocation(self.ledger.snapshot(), self.allocation)
        require(self.ledger.remaining()['remaining'] >= 60, 'Global initialization deadline exhausted')

    def reserve(self, item):
        require(item == self.ledger.bank.items[item['item_id']], 'Dispatch differs from frozen bank projection')
        self.ledger.reserve(self.allocation, item['item_id'])

    def result(self, item, result, output_root):
        self.ledger.result(self.allocation, item['item_id'], result, output_root)

    def failure(self, current, error):
        # Reserve the cleanup tail for processes, not journal I/O. Charges were
        # already committed before dispatch; defer failure details until release.
        self.failures.append((current['item_id'] if current else None, repr(error)))

    def finish(self, cleanup, hard_deadline, clock, alarm):
        self.cleanup = cleanup
        self.hard_deadline = hard_deadline
        if self.finalizer is not None:
            return self.finalizer(hard_deadline, clock, alarm, self.failures)
        with alarm(hard_deadline):
            for slot, error in self.failures:
                self.ledger.failure(self.allocation, slot, error)
        return dict(container_absent=False, lease_released=False, error='No owned allocation finalizer; charge retained')


class BankSession:
    """One allocation/queue, finalized synchronously before run_phase returns.

Run in the durable host job, not an SSH foreground session. No polling/retry loop
is provided. Enter and run immediately; acquisition plus queue preparation/startup
share a bounded 600 second envelope. A phase gate runs after release; its next
session requires explicit resume-unstarted and retains the original bank clock.
A crash leaves the durable allocation/slot reservations unresolved.
"""
    def __init__(self, ledger, adapter, *, resume_unstarted=False, alarm=work_alarm):
        self.ledger, self.adapter = ledger, adapter
        self.alarm = alarm
        self.clock = lambda: ledger.clock()['mono']
        ledger.bank.verify_projected_inputs(ledger.pending(ledger.snapshot()['phase']))
        # Capture before allocation so setup cannot silently renew a deadline.
        self.deadline = ledger.remaining()['deadline_mono']
        self.hard_deadline = min(self.deadline, self.clock() + PAGE_SECONDS)
        self.allocation = ledger.begin_allocation(resume_unstarted=resume_unstarted)['id']
        self.jobs = []
        self.finalization = None
        self.run = ledger.bank.root / 'runs' / self.allocation
        # Already charged before the live shared-lease probe or initialization.
        work_deadline = self.hard_deadline - CLEANUP_SECONDS
        try:
            with alarm(work_deadline):
                (self.run / 'control').mkdir(parents=True, exist_ok=False)
                (self.run / 'output').mkdir(exist_ok=False)
                receipt = adapter.acquire(self.allocation, work_deadline)
                path = self.run / 'control' / 'IMAGE_RECEIPT.json'
                self.image_receipt = dict(path=str(path), sha256=atomic_json(path, receipt))
                if self.clock() >= work_deadline:
                    raise TimeoutError('Acquisition/receipt work deadline')
        except BaseException as exc:
            result = self._finalize(self.hard_deadline, self.clock, alarm, [(None, repr(exc))])
            if result.get('error') or result.get('persistence_error'):
                exc.add_note('Unresolved acquisition finalization: ' + repr(result))
            raise

    def run_phase(self, phase, *, clock=time.monotonic, alarm=None, resume_unstarted=False):
        require(self.finalization is None and not self.jobs, 'Session already used; create an explicit resume-unstarted session')
        guard = alarm or self.alarm
        budget = None
        try:
            with guard(self.hard_deadline - CLEANUP_SECONDS):
                slots = self.ledger.pending(phase)
                require(slots, 'No unstarted slots in this phase; no replay')
                bank = self.ledger.bank
                bank.verify_projected_inputs(slots)
                queue = bank.projection(slots, self.image_receipt, self.ledger.remaining()['remaining'])
                job_id = uuid.uuid4().hex
                contract_path = self.run / 'control' / (job_id + '.json')
                contract_sha = atomic_json(contract_path, queue)
                output = self.run / 'output' / job_id
                budget = BankBudget(self.ledger, self.allocation, finalizer=self._finalize, startup_hard_deadline=self.hard_deadline)
                self.jobs.append(dict(phase=phase, output=str(output), contract_sha256=contract_sha, budget=budget))
            factory = lambda owner, logs: self.adapter.transport(owner, logs, contract_path, contract_sha, output,
                min(budget.absolute_deadline(), budget.hard_deadline) - CLEANUP_SECONDS)
            return run_queue(queue, contract_sha, output, factory, clock=clock, alarm=guard, budget=budget)
        except BaseException as exc:
            if self.finalization is None:
                hard_deadline = budget.hard_deadline if budget is not None else self.hard_deadline
                self._finalize(hard_deadline, clock, guard, [(None, repr(exc))])
            raise
        finally:
            if self.finalization is None:
                hard_deadline = budget.hard_deadline if budget is not None else self.hard_deadline
                self._finalize(hard_deadline, clock, guard)

    def _finalize(self, hard_deadline, clock, alarm, failures=()):
        if self.finalization is not None:
            return self.finalization
        hard_deadline = min(hard_deadline, self.deadline)
        deadline = min(hard_deadline - PERSIST_SECONDS, clock() + OUTER_CLEANUP_SECONDS)
        release = dict(allocation=self.allocation, container_absent=False, lease_released=False, error=None,
                       outer_deadline=deadline, hard_deadline=hard_deadline)
        # Exactly one attempt, including when close/__exit__ follows run_phase.
        self.finalization = release
        try:
            with alarm(deadline):
                release.update(self.adapter.release(self.allocation, deadline))
                if clock() > deadline:
                    raise TimeoutError('Owned finalizer exceeded its allotted tail')
                require(release.get('allocation') == self.allocation and release.get('container_absent') is True
                        and release.get('lease_released') is True and not release.get('error'), 'Owned release unresolved')
        except BaseException as exc:
            release['error'] = repr(exc)
        release.update(outer_deadline=deadline, hard_deadline=hard_deadline)
        release['queue_cleanups'] = [job['budget'].cleanup for job in self.jobs]
        release['completed_at'] = clock()
        try:
            with alarm(hard_deadline):
                for slot, error in failures:
                    self.ledger.failure(self.allocation, slot, error)
                if release['error']:
                    self.ledger.failure(self.allocation, None, 'Unresolved owned finalization: ' + release['error'])
                else:
                    self.ledger.finish_allocation(self.allocation, release)
                if clock() > hard_deadline:
                    raise TimeoutError('Allocation persistence exceeded original hard deadline')
        except BaseException as exc:
            release['persistence_error'] = repr(exc)
        return release

    def close(self):
        release = self._finalize(self.hard_deadline, self.clock, self.alarm)
        if release.get('error') or release.get('persistence_error'):
            raise RuntimeError('Owned finalization unresolved; no automatic retry: ' + repr(release))
        return release

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
