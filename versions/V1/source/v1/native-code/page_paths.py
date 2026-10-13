"""UTF-8 component limits without changing document identity or image input."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

PREFIX = '__page_sha256_'
SUFFIXES = ('', '.started.json', '.json', '.md')


def component_name(page_id, suffix=''):
    assert isinstance(page_id, str) and page_id not in ('', '.', '..')
    assert not any(c in page_id for c in ('/', '\\', '\x00'))
    assert suffix in SUFFIXES
    name = page_id + suffix
    if len(name.encode('utf-8')) > 255 or page_id.startswith(PREFIX):
        name = PREFIX + hashlib.sha256(page_id.encode('utf-8')).hexdigest() + suffix
    assert len(name.encode('utf-8')) <= 255
    return name


def mapping_for_pages(pages, require_original_primary=False):
    rows = []
    seen = {suffix: set() for suffix in SUFFIXES}
    for page in pages:
        pid = page['page_id']
        names = {suffix: component_name(pid, suffix) for suffix in SUFFIXES}
        for suffix, name in names.items():
            assert name not in seen[suffix], 'Duplicate identity or filename collision'
            seen[suffix].add(name)
        if require_original_primary:
            assert names['.md'] == pid + '.md', 'Official evaluator requires original page_id.md'
        rows.append({'page_id': pid, 'input_sha256': page['input_sha256'],
                     'directory': names[''], 'event': names['.started.json'],
                     'terminal': names['.json'], 'primary': names['.md']})
    return {'schema': 'page-paths-v3', 'encoding': 'utf-8', 'component_limit': 255,
            'original_page_id_preserved': True, 'pages': rows,
            'summary': {'pages': len(rows),
                        'original_primary_names_preserved': all(r['primary']==r['page_id']+'.md' for r in rows),
                        'max_original_primary_utf8_bytes': max((len((r['page_id']+'.md').encode('utf-8')) for r in rows),default=0),
                        'changed_component_counts': {field:sum(r[field]!=r['page_id']+suffix for r in rows)
                            for field,suffix in [('directory',''),('event','.started.json'),('terminal','.json'),('primary','.md')]}}}


def atomic_write_json(path, value):
    path = Path(path)
    assert len(path.name.encode('utf-8')) <= 255
    payload = json.dumps(value, ensure_ascii=False, indent=2,
                         default=lambda x: sorted(x) if isinstance(x, set) else str(x))
    # A short, unique sibling avoids both NAME_MAX growth and shared temporary names.
    fd, name = tempfile.mkstemp(prefix='.json-', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
