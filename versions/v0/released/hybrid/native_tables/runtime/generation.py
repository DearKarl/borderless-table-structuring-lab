"""Unmodified strict generation/loader checks from the evaluated pipeline."""
import hashlib
import json
CONFIG = {"system": "You are a helpful assistant.", "max_new_tokens": 4096}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def generation_record(prompt_ids, sequence, eos_ids):
    if sequence[:len(prompt_ids)] != prompt_ids:
        raise ValueError('Generation changed its prompt prefix')
    tail = sequence[len(prompt_ids):]
    if not tail or len(tail) > CONFIG['max_new_tokens']:
        raise ValueError('Invalid generation length')
    eos = eos_ids if isinstance(eos_ids, (list, tuple)) else [eos_ids]
    ended = tail[-1] in eos
    return {'input_token_ids': prompt_ids, 'raw_token_ids': tail,
            'raw_token_ids_sha256': digest(tail), 'generated_token_count': len(tail),
            'terminal_eos_present': ended, 'max_tokens_reached': len(tail) == 4096,
            'truncated': len(tail) == 4096 and not ended,
            'termination': 'EOS' if ended else ('TOKEN_BUDGET' if len(tail) == 4096 else 'OTHER'),
            'complete_eos_candidate': ended}


def check_loading_info(info):
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        if key not in info or info[key]:
            raise ValueError('Exact loading failed: ' + key + '=' + repr(info.get(key)))
