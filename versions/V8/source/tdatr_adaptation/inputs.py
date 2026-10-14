"""Text-only task changes on byte-verified native processor image caches."""
from array import array
import hashlib
import torch

from versions.generator_pilot.supervision import assistant_labels

SYSTEM_PROMPT = 'You are a helpful assistant.'
NATIVE_PROMPT = '\nThis is the image of a table. Please output the table in OTSL format.'


def encode_text(processor, image_tokens, prompt, target, maximum_tokens=16384):
    messages = [dict(role='system', content=SYSTEM_PROMPT),
        dict(role='user', content=[dict(type='image'), dict(type='text', text=prompt)])]
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    complete = processor.apply_chat_template(messages + [dict(role='assistant', content=target)],
        tokenize=False, add_generation_prompt=False)
    marker = processor.image_token
    if prefix.count(marker) != 1 or complete.count(marker) != 1:
        raise ValueError('Exactly one native image placeholder is required')
    # This is the native Qwen processor's image-token expansion. Every audited
    # table must reconstruct its original cached native IDs and labels exactly.
    prefix = prefix.replace(marker, marker * image_tokens)
    complete = complete.replace(marker, marker * image_tokens)
    tokenizer = processor.tokenizer
    prompt_ids = tokenizer(prefix, add_special_tokens=True)['input_ids']
    ids = tokenizer(complete, add_special_tokens=True)['input_ids']
    labels, counts = assistant_labels(prompt_ids, ids, [1] * len(ids),
        eos_id=tokenizer.eos_token_id, forbidden_target_ids=tokenizer.all_special_ids,
        maximum_tokens=maximum_tokens)
    encoded = tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    target_ids = ids[counts['target_start']:counts['eos_position']]
    if (target_ids != encoded['input_ids'] or
        tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target):
        raise ValueError('Task target tokenizer roundtrip or context changed')
    return dict(input_ids=ids, labels=labels, counts=counts, offsets=encoded['offset_mapping'],
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        target_sha256=hashlib.sha256(target.encode()).hexdigest())


def ids_blob(values):
    encoded = array('i', values)
    if encoded.itemsize != 4:
        raise ValueError('Expected 32-bit SQLite token storage')
    import sys
    if sys.byteorder != 'little': encoded.byteswap()
    return encoded.tobytes()


def task_batch(native_batch, task_row):
    """Keep the original image tensors while replacing only text and its mask."""
    ids = torch.frombuffer(bytearray(task_row['input_ids']), dtype=torch.int32).long()[None]
    labels = torch.full_like(ids, -100)
    start, end = task_row['target_start'], task_row['eos_position']
    labels[:, start:end + 1] = ids[:, start:end + 1]
    return {**{k:v for k,v in native_batch.items() if k not in ('input_ids','attention_mask','labels')},
        'input_ids':ids, 'attention_mask':torch.ones_like(ids), 'labels':labels}
