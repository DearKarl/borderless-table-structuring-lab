import json
def read(p):return json.loads(p.read_text())

def audit_stops(folder, receipt):
    path = folder/'generation-calls.json'
    if path.exists():
        calls = read(path)
        stops, complete = [], bool(calls)
        if 'generation_calls' in receipt:
            complete &= receipt['generation_calls'] == len(calls)
        for call in calls:
            if 'responses' in call:
                responses = call['responses']
                complete &= bool(responses)
                for response in responses:
                    stop = response.get('finish_reason')
                    complete &= stop in ('stop','length') and isinstance(response.get('token_ids'),list)
                    stops.append(stop)
            else:
                tokens = call.get('token_ids',call.get('tokens',[]))
                reasons = call.get('stop_reasons',[])
                complete &= bool(tokens) and len(tokens)==len(reasons)
                for token_row,stop in zip(tokens,reasons):
                    complete &= stop in ('eos','length')
                    eos = call.get('effective_eos_token_id',[])
                    cap = call.get('effective_output_token_cap')
                    if stop == 'eos':
                        complete &= any(t in eos for t in token_row)
                    elif stop == 'length':
                        complete &= cap is not None and len(token_row)>=cap
                    stops.append(stop)
        return any(s == 'length' for s in stops), bool(complete), stops
    token_path = folder/'generated-tokens.json'
    if not token_path.exists():
        return False,False,[]
    tokens = read(token_path)
    reason = receipt.get('stop_reason')
    complete = len(tokens)==receipt.get('output_tokens')
    if reason == 'eos':
        pos = receipt.get('first_stop_position')
        complete &= isinstance(pos,int) and 0<=pos<len(tokens) and tokens[pos]==receipt.get('stop_token')
    else:
        complete &= reason == 'token_cap' and receipt.get('truncated') is True
    return bool(receipt.get('truncated')),bool(complete),[reason]
