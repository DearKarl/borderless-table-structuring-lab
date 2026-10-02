"""Product receipts around the unchanged formula_v0 byte-splicing core."""
import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from hybrid.formula_v0.core import assemble
from .cache import checked, sha
from .page_paths import component_name


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def artifact(path):
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def assemble_page(page, m, p, output, provenance):
    pid = page['page_id']
    assert pid and pid not in ('.', '..') and '/' not in pid and '\\' not in pid
    result, receipt = assemble(m['raw'], m['native'], p['native'], page['width'],
                               page['height'], m['status'], p['status'])
    folder = Path(output) / 'pages' / component_name(pid)
    folder.mkdir(parents=True, exist_ok=False)
    primary_root = Path(output) / 'primary'
    primary_root.mkdir(exist_ok=True)
    name=component_name(pid,'.md')
    assert name==pid+'.md','Official primary must preserve original page_id'
    primary = primary_root / name
    with primary.open('xb') as stream:
        stream.write(result)
    with (folder / 'prediction.md').open('xb') as stream:
        stream.write(result)
    copied = {}
    # Keep M materialized image references and bytes; P contributes formula text only.
    for item in m['assets'] if m['status'] == 'success' else []:
        reference = item['reference']
        assert reference and not urlsplit(reference).scheme and '\\' not in reference
        relative = Path(unquote(reference))
        assert not relative.is_absolute() and '..' not in relative.parts
        target = folder / relative
        assert target.resolve().is_relative_to(folder.resolve())
        source = checked(item)
        if str(target) in copied:
            assert copied[str(target)]['sha256'] == item['sha256']
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
        assert sha(target) == item['sha256']
        copied[str(target)] = artifact(target)
    receipt.update(product='hybrid_v1', rule_provenance=receipt['version'], page_id=pid,
                   original_image_sha256=page['input_sha256'], native_provenance=provenance,
                   M_assets=list(copied.values()), native_calls_this_run=provenance['native_calls_this_run'])
    receipt_path = folder / 'receipt.json'
    save(receipt_path, receipt)
    terminal = dict(page_id=pid, input_sha256=page['input_sha256'], status=m['status'],
                    prediction_sha256=sha(primary), native_artifacts=[artifact(receipt_path),
                    artifact(folder / 'prediction.md'), *copied.values()])
    save(folder / 'terminal.json', terminal)
    return receipt, terminal
