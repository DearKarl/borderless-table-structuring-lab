"""Pure deterministic full-run planning and terminal receipt rules."""
import hashlib
import json

TERMINAL={'success','failed','truncated'}

def runtime_key(document):
    return hashlib.sha256(json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def pending_pages(pages,completed,key):
    assert len({p['page_id'] for p in pages})==len(pages)
    mapping={p['page_id']:p for p in pages}
    for pid,receipt in completed.items():
        assert receipt['page_id']==pid
        assert pid in mapping and receipt['runtime_key']==key,'Refuse cross-runtime cached receipt'
        assert receipt['status'] in TERMINAL and receipt['input_sha256']==mapping[pid]['input_sha256']
    return [p for p in sorted(pages,key=lambda r:(r['input_sha256'],r['page_id'])) if p['page_id'] not in completed]

def shard(pages,size=8):
    assert type(size) is int and 1<=size<=32
    return [pages[i:i+size] for i in range(0,len(pages),size)]

def primary_bytes(status,raw):
    assert status in TERMINAL and isinstance(raw,bytes)
    return raw if status=='success' else b''

def timeout_partition(assigned,completed,started):
    """A timed-out started page is terminal; never retry it or earlier completions."""
    ids=[p['page_id'] for p in assigned];assert len(ids)==len(set(ids))
    assert set(completed)<=set(ids)
    if started is not None:assert started in ids and started not in completed
    remaining=[p for p in assigned if p['page_id'] not in completed and p['page_id']!=started]
    return started,remaining

def complete_lock(pages,receipts,key,arm,prediction_directory):
    assert len(pages)==1651 and len(receipts)==1651
    assert not pending_pages(pages,receipts,key)
    rows=[]
    for page in pages:
        receipt=receipts[page['page_id']]
        assert receipt['page_id']==page['page_id']
        assert receipt['native_artifacts_verified'] is True
        assert receipt['status']=='success' or receipt['primary_bytes']==0
        assert receipt['prediction_sha256'] and isinstance(receipt['native_artifacts'],list)
        rows.append({k:receipt[k] for k in ['page_id','input_sha256','status','prediction_sha256','native_artifacts']})
    return {'arm':arm,'complete':True,'runtime_key':key,'prediction_directory':prediction_directory,'pages':rows}
