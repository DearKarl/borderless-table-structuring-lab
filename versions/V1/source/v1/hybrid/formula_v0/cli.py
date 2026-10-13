"""Path-only, GT-free assembly CLI. Cache assembly is explicitly not inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .core import assemble, sha

FIELDS={'page_id','width','height','M_raw','M_structured','P_native','M_status','P_status'}


def code_receipt():
    return [{'name':p.name,'sha256':sha(p.read_bytes())} for p in sorted(Path(__file__).parent.glob('*.py'))]


def load_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def assemble_record(row,out):
    assert set(row)<=FIELDS and {'page_id','width','height','M_raw','M_status','P_status'}<=set(row),'Only prediction paths/dimensions/status are admitted; no GT/metrics'
    pid=row['page_id'];assert isinstance(pid,str) and pid and pid not in ['.','..'] and '/' not in pid and '\\' not in pid
    assert row['M_status'] in ['success','failed','truncated','missing'] and row['P_status'] in ['success','failed','truncated','missing']
    raw=Path(row['M_raw']).read_bytes() if row['M_status']=='success' else b''
    def optional(name):
        p=row.get(name)
        return load_json(p) if p and Path(p).is_file() else None
    m=optional('M_structured');p=optional('P_native')
    result,receipt=assemble(raw,m,p,row['width'],row['height'],row['M_status'],row['P_status'])
    folder=Path(out)/pid;folder.mkdir(parents=True,exist_ok=False)
    (folder/'prediction.md').write_bytes(result)
    receipt.update(page_id=pid,mode='cached_predictions_assembly_not_end_to_end_inference',code=code_receipt(),
        input_files={k:{'path':row[k],'sha256':sha(Path(row[k]).read_bytes())} for k in ['M_raw','M_structured','P_native'] if row.get(k) and Path(row[k]).is_file()})
    save(folder/'receipt.json',receipt)
    return receipt


def manifest(path,out):
    doc=load_json(path);assert set(doc)=={'pages'} and isinstance(doc['pages'],list)
    assert len({r['page_id'] for r in doc['pages']})==len(doc['pages'])
    root=Path(out);root.mkdir(parents=True,exist_ok=False)
    rows=[assemble_record(row,root) for row in doc['pages']]
    reasons=Counter(r.get('page_abstention','') for r in rows if r.get('page_abstention'))
    for r in rows:
        reasons.update(x['reason'] for x in r['pairs'] if not x['selected'])
        reasons.update(x['reason'] for x in r['abstentions'])
    receipt={'mode':'cached_predictions_assembly_not_end_to_end_inference','input_manifest_sha256':sha(Path(path).read_bytes()),'pages':len(rows),
        'modified_pages':sum(r['replacement_count']>0 for r in rows),'replacements':sum(r['replacement_count'] for r in rows),
        'abstention_reasons':dict(reasons),'code':code_receipt(),'GT_or_score_used':False,'all_byte_invariants_verified':all(r['nonreplacement_bytes_verified'] for r in rows),
        'outputs':[{'page_id':r['page_id'],'sha256':r['output_sha256']} for r in rows]}
    save(root/'ASSEMBLY_RECEIPT.json',receipt);return receipt


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='mode',required=True)
    for name in ['assemble-page','assemble-manifest']:
        p=sub.add_parser(name);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    a=parser.parse_args()
    if a.mode=='assemble-manifest':result=manifest(a.manifest,a.output)
    else:
        row=load_json(a.manifest);result=assemble_record(row,a.output)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
