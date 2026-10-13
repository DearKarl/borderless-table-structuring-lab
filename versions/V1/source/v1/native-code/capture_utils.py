"""Persist per-page diagnostics before clearing resident-worker buffers."""
import json
from pathlib import Path

def persist_buffers(folder,buffers):
    names=['generation-calls.json','effective-generation-configs.json']
    assert 1<=len(buffers)<=2
    for name,value in zip(names,buffers):
        encoded=json.dumps(value,ensure_ascii=False,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x))
        path=Path(folder)/name
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))==json.loads(encoded),'Existing diagnostic capture differs'
        else:
            with path.open('x',encoding='utf-8') as f:f.write(encoded)
