"""Observation-only native generation hooks; never edits arguments or return values."""
import hashlib
import inspect
import json
import math
import time
from .region_protocol import RunBlocked, termination


def audit_snapshot(value):
    """JSON-safe copies, with small tensor values and bounded large tensor metadata."""
    if value is None or isinstance(value,(str,int,float,bool)): return value
    if isinstance(value,dict): return {str(k):audit_snapshot(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [audit_snapshot(v) for v in value]
    if hasattr(value,'shape') and hasattr(value,'tolist'):
        tensor=value.detach().cpu() if hasattr(value,'detach') else value
        count=math.prod(value.shape)
        if count<=256:
            return {'kind':'tensor','shape':list(value.shape),'dtype':str(value.dtype),
                    'values':audit_snapshot(tensor.tolist())}
        return {'kind':'tensor',**tensor_binding({'value':value})['value']}
    if hasattr(value,'item') and type(value).__module__.startswith('numpy'):
        return audit_snapshot(value.item())
    if type(value).__name__ == 'AddedToken' and type(value).__module__.startswith('tokenizers'):
        return {'kind':'AddedToken', **{k:getattr(value,k) for k in
            ('content','single_word','lstrip','rstrip','normalized','special')}}
    if type(value).__name__ == 'dtype' and type(value).__module__ == 'torch':
        return {'kind':'torch.dtype','value':str(value)}
    raise TypeError('Unsupported audit value: '+type(value).__module__+'.'+type(value).__name__)


def scalar_snapshot(value):
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, dict): return {str(k): scalar_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [scalar_snapshot(x) for x in value]
    if hasattr(value, 'shape'):
        return {'class': type(value).__name__, 'shape': list(value.shape), 'dtype': str(value.dtype)}
    return {'class': type(value).__name__}


def tensor_binding(inputs):
    """Copy for hashing only. The exact original input objects continue into generate."""
    result = {}
    for key, value in inputs.items():
        if hasattr(value, 'shape'):
            tensor = value.detach().cpu() if hasattr(value, 'detach') else value
            try:
                array = tensor.numpy()
                raw = array.tobytes()
                encoding = 'native_numpy_bytes'
            except (TypeError, RuntimeError):
                raw = json.dumps(tensor.tolist(), separators=(',', ':'), allow_nan=False).encode()
                encoding = 'numeric_json_exact_values'
            result[key] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                           'sha256': hashlib.sha256(raw).hexdigest(), 'hash_encoding': encoding}
    return result


def install_tele_audit(model, emit):
    names = ('generate', '_prepare_generation_config', '_prepare_generated_length')
    originals = {name: getattr(model, name) for name in names}
    current = {}
    def check_config(resolved):
        if resolved.get('max_length') != 128000 or resolved.get('max_new_tokens') is not None:
            raise RunBlocked('Tele actual length priority differs from contract')
        for key, expected in [('do_sample', False), ('repetition_penalty', 1.0), ('no_repeat_ngram_size', 100)]:
            if resolved.get(key) != expected: raise RunBlocked('Tele decode setting differs: ' + key)
    def config_hook(*args, **kwargs):
        result = originals['_prepare_generation_config'](*args, **kwargs)
        current['merged_generation_config'] = audit_snapshot(result[0].to_dict())
        return result
    def length_hook(*args, **kwargs):
        result = originals['_prepare_generated_length'](*args, **kwargs)
        current['effective_generation_config'] = audit_snapshot(result.to_dict())
        # Validate resolved settings before the native generation loop starts.
        check_config(current['effective_generation_config'])
        return result
    def generate(*args, **kwargs):
        current.clear()
        started = time.monotonic()
        inputs = kwargs.get('input_ids')
        if inputs is None and args: inputs = args[0]
        if inputs is None or len(inputs.shape) != 2 or inputs.shape[0] != 1:
            raise RunBlocked('Expected explicit batch1 Tele input IDs')
        input_ids = inputs.cpu().tolist()[0]
        call = {'engine': 'tele', 'input_length': len(input_ids), 'input_token_ids': input_ids,
                'raw_kwargs': scalar_snapshot(kwargs), 'input_tensors': tensor_binding(kwargs)}
        try:
            result = originals['generate'](*args, **kwargs)
            resolved = current.get('effective_generation_config')
            if resolved is None: raise RunBlocked('Tele effective generation config was not observed')
            check_config(resolved)
            sequences = result.sequences if hasattr(result, 'sequences') else result
            ids = sequences.cpu().tolist()[0]
            if ids[:len(input_ids)] != input_ids: raise RunBlocked('Tele prompt prefix changed')
            eos = resolved['eos_token_id']; eos = eos if isinstance(eos, list) else [eos]
            call['termination'] = termination(ids[len(input_ids):], eos, 128000-len(input_ids))
            call.update(current, elapsed_seconds=time.monotonic()-started, returned=True)
            emit(call)
            return result
        except BaseException as exc:
            call.update(current, elapsed_seconds=time.monotonic()-started, returned=False,
                        error_type=type(exc).__name__, error=str(exc))
            emit(call); raise
    model.generate, model._prepare_generation_config, model._prepare_generated_length = generate, config_hook, length_hook
    def restore():
        for name, original in originals.items(): setattr(model, name, original)
    return restore


def install_paddle_audit(model, emit):
    original_generate, original_greedy = model.generate, model.greedy_search
    signature = inspect.signature(original_greedy)
    current = {}
    def greedy(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs); arguments.apply_defaults()
        values = arguments.arguments
        if values.get('trunc_input') is not True: raise RunBlocked('Paddle is not returning new tokens only')
        inputs = values['input_ids']
        if inputs.shape[0] != 1: raise RunBlocked('Paddle batch size changed')
        eos = values['eos_token_id']; eos = eos if isinstance(eos, list) else [eos]
        if eos != [2] or values['pad_token_id'] != 0:
            raise RunBlocked('Paddle EOS/PAD differs from pinned model')
        if values['max_length'] - inputs.shape[-1] != 4096:
            raise RunBlocked('Paddle effective budget differs from contract')
        current.update(decode_strategy='greedy_search', effective_max_length=values['max_length'],
                       prompt_length=inputs.shape[-1], eos_ids=eos, pad_token_id=values['pad_token_id'], trunc_input=True)
        return original_greedy(*args, **kwargs)
    def generate(inputs, **kwargs):
        current.clear()
        started = time.monotonic()
        audit = {'engine': 'paddle', 'raw_kwargs': scalar_snapshot(kwargs), 'input_tensors': tensor_binding(inputs)}
        try:
            returned = original_generate(inputs, **kwargs)
            if current.get('decode_strategy') != 'greedy_search':
                raise RunBlocked('Paddle native greedy path was not observed')
            if not isinstance(returned, tuple) or len(returned) < 1:
                raise RunBlocked('Paddle native return signature changed')
            token_rows = returned[0].tolist()
            if len(token_rows) != 1: raise RunBlocked('Paddle token result arity changed')
            audit.update(current, **termination(token_rows[0], current['eos_ids'], 4096),
                         elapsed_seconds=time.monotonic()-started, returned=True)
            emit(audit)
            return returned
        except BaseException as exc:
            audit.update(current, elapsed_seconds=time.monotonic()-started, returned=False,
                         error_type=type(exc).__name__, error=str(exc))
            emit(audit); raise
    model.generate, model.greedy_search = generate, greedy
    def restore(): model.generate, model.greedy_search = original_generate, original_greedy
    return restore
