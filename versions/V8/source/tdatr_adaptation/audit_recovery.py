"""Explicit recovery from a verified copy of an interrupted preparation audit."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def recover(binding, rows, root):
    specification=binding['recovery']
    old=Path(specification['directory'])
    previous_binding=json.loads(Path(specification['binding']).read_text())
    current={k:v for k,v in binding.items() if k not in ['recovery','output_root']}
    previous={k:v for k,v in previous_binding.items() if k!='output_root'}
    if current!=previous:raise ValueError('Recovery may not alter the frozen input binding')
    for path,expected in specification['file_hashes'].items():
        if sha(path)!=expected:raise ValueError('Preserved recovery input changed: '+path)
    report=json.loads((old/'AUDIT.json').read_text())
    if report['status']!='running' or report['formal_optimizer_updates']!=0 or report['GPU_allocations']!=0:
        raise ValueError('Only interrupted CPU preparation can be recovered')
    prefix=rows[:report['completed']]
    expected={r['id']:r for r in prefix if r['status']=='input_eligible'}
    if not expected or report['native_cache_exact']!=len(expected) or report['full_cell_ranges_verified']!=len(expected):
        raise ValueError('Saved audit denominators do not identify an exact verified prefix')
    # Work only on a fresh copy. SQLite may roll back a hot journal here; the
    # original interrupted database and its sidecars stay byte-for-byte intact.
    for path in old.glob('DETAIL_INPUTS.sqlite*'):
        if str(path) not in specification['file_hashes']:raise ValueError('Unbound database sidecar')
        shutil.copyfile(path,root/path.name)
    db=sqlite3.connect(root/'DETAIL_INPUTS.sqlite')
    try:
        if db.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise ValueError('Recovery database integrity failed')
        actual={r[0] for r in db.execute('SELECT id FROM examples')}
        if not set(expected)<=actual:raise ValueError('A previously verified example is absent')
        extra=actual-set(expected)
        for identifier in extra:
            db.execute('DELETE FROM tasks WHERE id=?',(identifier,))
            db.execute('DELETE FROM examples WHERE id=?',(identifier,))
        if db.execute('SELECT COUNT(*) FROM tasks WHERE id NOT IN (SELECT id FROM examples)').fetchone()[0]:
            raise ValueError('Orphaned task rows in recovered database')
        counts=Counter();lengths=defaultdict(list)
        for identifier,partition,record,ranges,spans,eligible in db.execute('SELECT * FROM examples'):
            source=expected[identifier]
            if json.loads(record)!=source or partition!=source['partition']:raise ValueError('Recovered example identity changed')
            cell_ranges=json.loads(ranges);logical=json.loads(spans)
            if len(cell_ranges)!=len(source['normalized_boxes']) or len(logical)!=len(cell_ranges):
                raise ValueError('Recovered cell correspondence changed')
            task_rows=list(db.execute('SELECT task,length(input_ids),target_start,eos_position,sequence_tokens,target_tokens FROM tasks WHERE id=?',(identifier,)))
            if sorted(r[0] for r in task_rows)!=sorted(json.loads(eligible)):raise ValueError('Recovered task eligibility changed')
            for task,byte_count,start,end,tokens,target_tokens in task_rows:
                if byte_count!=tokens*4 or not 0<start<=end<tokens or target_tokens!=end-start:
                    raise ValueError('Recovered token metadata inconsistent')
                key=partition+':'+task;counts[key]+=1;lengths[key].append(tokens)
        if dict(counts)!=report['eligible']:raise ValueError('Recovered task counts differ from the saved audit')
        maximum={k:max(v) for k,v in lengths.items()}
        if maximum!=report['maximum_tokens']:raise ValueError('Recovered task lengths differ from the saved audit')
        for path,expected_hash in report['source_manifest_hashes'].items():
            if sha(path)!=expected_hash:raise ValueError('Source label manifest changed after interruption')
        db.commit()
        report['eligible']=counts;report['unavailable']=Counter(report['unavailable'])
        report['task_sequence_lengths']=lengths
        report['recovery']=dict(source_directory=str(old),preserved_file_hashes=specification['file_hashes'],
            recovered_prefix=len(prefix),verified_examples=len(expected),discarded_unreported_suffix=len(extra),
            previous_elapsed_seconds=report['elapsed_seconds'],automatic_retry=False,reason='Explicit correction of an erroneous stale-stop action')
        return db,report
    except BaseException:
        db.close()
        raise
