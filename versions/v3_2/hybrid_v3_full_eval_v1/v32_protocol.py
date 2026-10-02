"""V32 identities are fatal; only named content rejections admit one native fallback."""
import hashlib
import re
from pathlib import Path
from .core import ContractError, canonical, read

CONFIG_SHA='3ffdddf16298c59b5a7cf6c95337db88f7f467b2a4a81595ff6611c0276db88d'
IDENTITY=('run_id','page_id','slot_id','request_id','input_sha256','crop','png_sha256','model_sha256','config_sha256','source_sha256')
class ContentRejected(ContractError):pass

def manifest():
    value=read(Path(__file__).with_name('v32_manifest.json'))
    if canonical(value)!=CONFIG_SHA:raise ContractError('Frozen Ovis config changed')
    return value

def validate(request,response):
    if not isinstance(response,dict) or any(request.get(k)!=response.get(k) for k in IDENTITY):raise ContractError('Ovis response identity mismatch')
    ev=response.get('engine_evidence',{})
    if (ev.get('identity')!={k:request[k] for k in IDENTITY} or ev.get('submitted') is not True
        or ev.get('output_request_id')!=ev.get('external_req_id') or not ev.get('request_id')
        or canonical(ev.get('payload'))!=ev.get('payload_sha256')):raise ContractError('Ovis actual engine evidence missing/changed')
    payload=ev['payload'];ids=payload.get('prompt_token_ids');features=payload.get('mm_features')
    if (not isinstance(ids,list) or not ids or len(ids)+16384>32768 or ev.get('prompt_length')!=len(ids)
        or ev.get('prompt_ids_sha256')!=canonical(ids) or not features):raise ContractError('Ovis observed input budget/evidence differs')
    for f in features:
        if f.get('modality')!='image' or set(f.get('tensors',{}))!={'pixel_values','image_grid_thw'}:raise ContractError('Ovis tensor evidence missing')
    g=response.get('generation',{});tokens=g.get('token_ids');raw=response.get('raw_text')
    if (not isinstance(raw,str) or len(raw)>1024*1024 or g.get('finished') is not True
        or g.get('config_sha256')!=CONFIG_SHA or g.get('eos_token_id')!=248044
        or not isinstance(tokens,list) or any(type(x) is not int or x<0 for x in tokens) or len(tokens)>16384):
        raise ContractError('Ovis completion schema/config invalid')
    sampling=payload.get('sampling_params')
    if (not isinstance(sampling,dict)
        or type(sampling.get('_eos_token_id')) is not int or sampling['_eos_token_id']!=248046
        or type(sampling.get('stop_token_ids')) is not list or sampling['stop_token_ids']!=[248044]
        or any(type(t) is not int for t in sampling['stop_token_ids'])
        or type(sampling.get('_all_stop_token_ids')) is not list or sampling['_all_stop_token_ids']!=[248044,248046]
        or any(type(t) is not int for t in sampling['_all_stop_token_ids'])
        or sampling.get('ignore_eos') is not False or sampling.get('stop') not in (None,[])
        or type(sampling.get('min_tokens')) is not int or sampling['min_tokens']!=0
        or type(sampling.get('n')) is not int or sampling['n']!=1
        or type(sampling.get('max_tokens')) is not int or sampling['max_tokens']!=16384
        or type(sampling.get('temperature')) not in (int,float) or sampling['temperature']!=0):
        raise ContractError('Ovis actual stopping payload differs')
    primary=sampling['_eos_token_id'];additional=sampling['stop_token_ids'][0]
    effective_stops={primary,additional}
    if g.get('finish_reason')=='length':
        if any(t in effective_stops for t in tokens):raise ContractError('Contradictory Ovis length/stop')
        raise ContentRejected('length_truncation')
    reason=g.get('stop_reason')
    if (g.get('finish_reason')!='stop' or not tokens or any(t in effective_stops for t in tokens[:-1])
        or not ((tokens[-1]==primary and reason is None)
                or (tokens[-1]==additional and type(reason) is int and reason==additional))):
        raise ContractError('Ovis raw stop contract incompatible')
    if not raw.strip():raise ContentRejected('empty_text')
    if re.search(r'<\s*/?\s*(?:table|thead|tbody|tr|td|th|img|image|svg|html|body)\b',raw,re.I) or re.search(r'!\[[^\]]*\]\s*\(',raw):
        raise ContentRejected('unexpected_table_or_image_wrapper')
    return raw.strip()
