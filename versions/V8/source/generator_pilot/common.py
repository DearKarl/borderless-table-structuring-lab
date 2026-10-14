"""Paths and durable receipts for the local preparation ticket."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'artifacts/generator_pilot'

def now():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

def event(kind, **detail):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'EVENTS.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(at_utc=now(), kind=kind, **detail)) + '\n')
