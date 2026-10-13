"""Authoritative selection at the pre-post-process raw boundary, never Markdown."""
import copy
from pathlib import Path
from .core import ContractError, read

def build(primary, final, events, requests):
    prim={r['slot_id']:r for r in primary}
    req={r['request_id']:r for r in requests}
    if len(req)!=len(requests):raise ContractError('Duplicate request identity')
    for p in primary:
        if p.get('request_id'):
            bound=req.get(p['request_id'])
            if (bound is None or bound['kind']!='content' or bound['slot_id']!=p['slot_id']
                or bound['stage']!='completed' or bound['raw_content']!=p['raw_content']):
                raise ContractError('Immutable native primary differs from bound completion')
    result=[]
    for index,item in enumerate(final):
        slot=item['slot_id'];p=prim.get(slot)
        history=[e for e in events if e.get('slot_id')==slot]
        committed=[e for e in history if e.get('reason')=='committed']
        layout=committed[-1] if committed else None
        tele_id=layout['fresh_request_id'] if layout else p.get('request_id') if p else None
        if p is None and layout is None:raise ContractError('Final new slot lacks committed layout')
        source='tele_fresh' if layout else 'native'
        reason='committed' if layout else 'native retained'
        formulas=[e for e in history if 'native_request_id' in e and e['native_request_id']==tele_id]
        formula=formulas[-1] if formulas else None
        expert_id=None
        if formula:
            reason=formula['reason']
            if formula.get('actual_content_source')=='paddle':
                source='paddle';expert_id=formula['expert_response']['request_id']
        if source=='paddle':expected=formula['selected']
        elif tele_id:
            binding=req.get(tele_id)
            if binding is None or binding['slot_id']!=slot or binding['stage']!='completed':
                raise ContractError('Selected Tele request missing or mismatched')
            expected=binding['raw_content']
        else:expected=p['raw_content']
        if item['block'].get('content')!=expected:
            raise ContractError('Final raw differs from selected source')
        result.append({'slot_id':slot,'final_index':index,'boundary':'final_pre_post_raw',
            'selected_content':copy.deepcopy(expected),'source':source,'selection_reason':reason,
            'native_request_id':p.get('request_id') if p else None,
            'tele_request_id':tele_id,'fresh_request_id':tele_id if layout else None,
            'layout_request_id':layout['layout_request_id'] if layout else None,
            'expert_request_id':expert_id,'primary_content':copy.deepcopy(p['raw_content']) if p else None,
            'decisions':copy.deepcopy(history)})
    if len(result)!=len({r['slot_id'] for r in result}):raise ContractError('Duplicate final stable slot')
    return result

def verify(folder):
    folder=Path(folder)
    primary=read(folder/'PRIMARY_CONTENT.json');final=read(folder/'V31_FINAL_RAW.json')
    events=read(folder/'V31_TRANSACTIONS.json');requests=read(folder/'REQUEST_BINDINGS.json')
    if len(primary)!=len({p['slot_id'] for p in primary}):raise ContractError('Duplicate native primary slot')
    expected=build(primary,final,events,requests)
    if read(folder/'CONTENT.json')!=expected:
        raise ContractError('Authoritative final selection contradicts raw output or source linkage')
    return True
