"""Pre-outcome deterministic raster calibration roster. No model calls.

freeze selects metadata only. materialize is a separate bounded root step after
recipe/sample hashes are reviewed. Missing/invalid selected pages never refill.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time
from hybrid.v4_input_selector.scale_policy import SEED, canonical
from hybrid.v4_input_selector.worker_contract import file_sha
from .contracts import DATASET, REVISION, MARGINS, MODEL_SHA, ORIGINAL_MANIFEST_SHA, source_files

RECIPE = dict(id='olmocr_pdf_to_raster_72dpi_001', dpi=72, cap=3500,
    scale='min(72/72,3500/max(pdf_display_size))', rotation=0,
    source_intrinsic_rotation='PDFium applies intrinsic rotation',
    renderer='pypdfium2', pypdfium2_version='5.13.0', pillow_version='12.1.0',
    mode='RGB', output='PNG', optimize=False, compress_level=6,
    rationale='72 DPI preserves displayed PDF point dimensions as raster pixels; fixed before outcomes')


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    return file_sha(path)


def rank(kind, value):
    return hashlib.sha256(f'{SEED}|v4-raster-cal-001|{kind}|{value}'.encode()).hexdigest()


def select_pages(rows, exclusions):
    blocked = {key:set(exclusions[key]) for key in
               ('group_id', 'pdf_sha256', 'reference_sha256', 'source_pdf_token', 'url')}
    groups = defaultdict(list)
    for row in rows:
        if (not row['acquired'] or not row['source_complete'] or row['errors']
                or row['is_rotation_valid'] is not True or row['rotation_correction'] != 0
                or row['identical_transcription_rows'] != 1 or row['reference_chars'] < 32
                or row['pdf_bytes'] > 64*1024**2
                or any(row.get(k) in values for k, values in blocked.items())):
            continue
        groups[row['group_id']].append(row)
    if len(groups) < 8:
        raise ValueError('Fewer than eight eligible distinct groups; no expansion')
    chosen = []
    for group in sorted(groups, key=lambda g: (rank('group', g), g))[:8]:
        row = min(groups[group], key=lambda r: (rank('page', r['member']), r['member']))
        chosen.append(dict(row, page_id='r_'+rank('identity', row['archive']+':'+row['member'])[:20]))
    if len({r['pdf_sha256'] for r in chosen}) != 8 or len({r['reference_sha256'] for r in chosen}) != 8:
        raise ValueError('Selected cross-group aliases; no automatic refill')
    return chosen


def freeze(joined, old_provenance, overlap, output, *, code_root, rounding_evidence=None):
    rows = [json.loads(x) for x in Path(joined).read_text(encoding='utf-8-sig').splitlines()]
    old = json.loads(Path(old_provenance).read_bytes())
    known = json.loads(Path(overlap).read_bytes())
    if known.get('schema') != 'v4_known_overlap_exclusions_v1':
        raise ValueError('Explicit known-overlap registry required')
    exclusions = {k:set(known['exclusions'][k]) for k in
                  ('group_id','pdf_sha256','reference_sha256','source_pdf_token','url')}
    for row in old['pages']:
        for key in exclusions:
            if row.get(key): exclusions[key].add(row[key])
    chosen = select_pages(rows, exclusions)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    plan = dict(schema='v4_frozen_raster_calibration_plan_v1', task='V4-ACTUAL-GBDT-001',
        dataset=DATASET, revision=REVISION, split='train', seed=SEED, pages=chosen,
        recipe=RECIPE, replacement_policy='none; retain invalid/unrun outcomes and fail gate',
        model_sha256=MODEL_SHA, original_manifest_sha256=ORIGINAL_MANIFEST_SHA, fixed_action='B',
        margins=list(MARGINS), gate=dict(valid_paired_pages=8, gain='group mean gain > group mean(2/reference_symbols)',
            added_terminal_failures=0, total_cost_ratio_max=1.10),
        budgets=dict(gpu_allocation_seconds=14400, wall_seconds=21600, cpu_seconds=1800,
            cpu_threads=1, item_work_seconds=540, item_cleanup_seconds=60, recognition_starts_max=16),
        activation=False, known_overlap_reviewed=known.get('review_complete') is True,
        rounding_evidence_sha256=file_sha(rounding_evidence) if rounding_evidence else None,
        input_bindings={str(p):file_sha(p) for p in (joined,old_provenance,overlap)},
        code_files={p.relative_to(code_root).as_posix():file_sha(p) for p in source_files(code_root)},
        exclusions={k:sorted(v) for k,v in exclusions.items()},
        archive_sha256='aaa9c5afbd375ea8928d8ea7a0f33b41b339ce560fb5002d96c05af8db7dc200',
        parquet_sha256='5e47fa4c8151123fab20277594c41dd2680ef6399cdd8019d6f2fd78acf551e4',
        no_omnidoc_tuning=True, held_history_sealed=True, measurements_started=0)
    return write_new(output/'PLAN.json', plan)


def materialize(plan_path, plan_sha, archive, parquet, output):
    from .contracts import checked_json
    plan = checked_json(plan_path, plan_sha)
    if plan['recipe'] != RECIPE or plan['known_overlap_reviewed'] is not True:
        raise ValueError('Recipe or known overlap review not frozen')
    if file_sha(archive) != plan['archive_sha256'] or file_sha(parquet) != plan['parquet_sha256']:
        raise ValueError('Acquired train assets changed')
    import importlib.metadata
    import pypdfium2 as pdfium
    import pyarrow.parquet as pq
    from PIL import Image
    from hybrid.v4_input_selector.public_score import canonicalize, VERSION
    if importlib.metadata.version('pypdfium2') != RECIPE['pypdfium2_version'] or Image.__version__ != RECIPE['pillow_version']:
        raise ValueError('Raster recipe runtime differs')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    write_new(out/'PRE_RENDER_FREEZE.json', dict(plan_sha256=plan_sha, recipe=RECIPE,
        selected_pdf_hashes={r['member']:r['pdf_sha256'] for r in plan['pages']}, model_calls=0))
    deadline = time.monotonic()+600
    selected = {r['member']:r for r in plan['pages']}; raw_pages = {}
    with tarfile.open(archive, 'r|gz') as tar:
        for entry in tar:
            if time.monotonic() >= deadline: raise TimeoutError('Bounded materialization expired')
            if entry.name not in selected: continue
            if not entry.isfile() or entry.issparse() or entry.size > 64*1024**2:
                raise ValueError('Unsafe selected archive member')
            with tar.extractfile(entry) as f: raw = f.read(64*1024**2+1)
            if hashlib.sha256(raw).hexdigest() != selected[entry.name]['pdf_sha256']:
                raise ValueError('Selected PDF hash mismatch')
            raw_pages[entry.name] = raw
            if len(raw_pages)==8: break
    if len(raw_pages)!=8: raise ValueError('Missing selected source; no refill')
    refs = {}; by_row={r['row']:r for r in plan['pages']}; offset=0
    for batch in pq.ParquetFile(parquet).iter_batches(batch_size=1024,columns=['pdf_relpath','natural_text']):
        if time.monotonic() >= deadline: raise TimeoutError('Materialization deadline')
        for j,value in enumerate(batch.to_pylist()):
            if offset+j in by_row:
                row=by_row[offset+j]
                if value['pdf_relpath'] != row['archive']+':'+row['member']:
                    raise ValueError('Reference provenance mismatch')
                refs[row['page_id']]=value['natural_text']
        offset+=batch.num_rows
        if offset>max(by_row):break
    pages=[]; items=[]
    (out/'inputs').mkdir(); (out/'references').mkdir()
    for row in plan['pages']:
        pid=row['page_id']; doc=pdfium.PdfDocument(raw_pages[row['member']])
        try:
            if len(doc)!=1:raise ValueError('Expected one-page dataset member')
            page=doc[0]
            try:
                size=list(page.get_size()); scale=min(1.,3500/max(size))
                bitmap=page.render(scale=scale,rotation=0)
                try:
                    image=bitmap.to_pil().convert('RGB')
                    try:
                        image.save(out/'inputs'/(pid+'.png'),format='PNG',optimize=False,compress_level=6)
                        dims=list(image.size)
                    finally:image.close()
                finally:bitmap.close()
            finally:page.close()
        finally:doc.close()
        ih=file_sha(out/'inputs'/(pid+'.png'))
        source=dict(original_file_sha256=ih, source_type='raster', original_page_ordinal=0,
            oriented_size=dims,pdf_cropbox=None,pdf_rotation=None,pdf_content_type='unknown')
        text=refs[pid]; canonical=canonicalize(text,deadline=deadline)
        if not 32 <= canonical.non_whitespace or len(canonical.symbols)>65536:
            raise ValueError('Selected reference invalid; no replacement')
        ref=dict(schema='nju_public_reference_v1',page_id=pid,source_group=row['group_id'],
            split='calibration',input_sha256=ih,score_version=VERSION,natural_text=text,
            canonical_symbols=len(canonical.symbols),reference_provenance='olmOCR model-assisted silver, not human truth')
        refsha=write_new(out/'references'/(pid+'.json'),ref)
        pages.append(dict(page_id=pid,source_group=row['group_id'],input_sha256=ih,
            source_pdf_sha256=row['pdf_sha256'],source_member=row['member'],file='inputs/'+pid+'.png',source=source,
            reference_file='references/'+pid+'.json',reference_sha256=refsha))
        for action in ('A','B'):
            items.append(dict(item_id=pid+'_'+action,page_id=pid,file='inputs/'+pid+'.png',
                              input_sha256=ih,source=source,action=action))
    return write_new(out/'RASTER_ROSTER.json',dict(schema='v4_raster_roster_v1',plan_sha256=plan_sha,pages=pages,
        items=items,recipe=RECIPE,model_calls=0,activation=False))


def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest='command',required=True)
    f=s.add_parser('freeze')
    for k in ('joined','old-provenance','overlap','output','code-root'):f.add_argument('--'+k,required=True)
    f.add_argument('--rounding-evidence')
    m=s.add_parser('materialize')
    for k in ('plan','plan-sha','archive','parquet','output'):m.add_argument('--'+k,required=True)
    a=p.parse_args()
    if a.command=='freeze':result=freeze(a.joined,a.old_provenance,a.overlap,a.output,code_root=Path(a.code_root),rounding_evidence=a.rounding_evidence)
    else:result=materialize(a.plan,a.plan_sha,a.archive,a.parquet,a.output)
    print(result)

if __name__=='__main__':main()
