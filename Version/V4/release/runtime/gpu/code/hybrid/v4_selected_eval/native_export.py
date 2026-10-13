"""Native converter only. No OCR, scoring, or alternate text renderer."""
import importlib
import json
from pathlib import Path
from hybrid.v4_input_selector.worker_contract import atomic_json, file_sha

def export_middle(out,item_id):
    out=Path(out);audit=json.loads((out/'audit.json').read_bytes())
    if file_sha(out/'native_middle.json')!=audit['native_middle_sha256']:raise ValueError('Native middle changed')
    middle=json.loads((out/'native_middle.json').read_bytes())
    exporter=importlib.import_module('TeleOCR.src.vlm_middle_json_mkcontent')
    markdown=exporter.union_make(middle['pdf_info'],'images/'+item_id)
    if not isinstance(markdown,str):raise TypeError('Native union_make must return Markdown text')
    with (out/'native.md').open('xb') as stream:stream.write(markdown.encode('utf-8'));stream.flush()
    atomic_json(out/'export.json',dict(item_id=item_id,markdown_sha256=file_sha(out/'native.md'),
        converter_sha256=file_sha(Path(exporter.__file__)),native_middle_sha256=audit['native_middle_sha256']))
