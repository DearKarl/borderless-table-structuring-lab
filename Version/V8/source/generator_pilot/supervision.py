"""Assistant-only labels for already image-expanded, untruncated token IDs."""
IGNORE_INDEX = -100


def gpu_uuid_key(value):
    """Normalize NVIDIA and torch spelling while rejecting malformed UUIDs."""
    from uuid import UUID
    return str(UUID(str(value).removeprefix('GPU-')))


def assistant_labels(prompt_ids, full_ids, attention_mask, *, eos_id,
                     forbidden_target_ids=(), maximum_tokens=16384):
    """Keep target through its first EOS; ignore prompt, pad and template suffix.

    ``prompt_ids`` contains no padding and includes the assistant role prefix.
    The caller must separately verify exact decoded target text and image tensors.
    Return unshifted labels: the model performs the causal shift internally.
    """
    if not prompt_ids or not full_ids or len(full_ids) != len(attention_mask):
        raise ValueError('Empty sequence or inconsistent attention-mask length')
    if any(x not in (0, 1) for x in attention_mask):
        raise ValueError('Attention mask must be binary')
    active = [i for i, attended in enumerate(attention_mask) if attended]
    if not active or active != list(range(active[0], active[-1] + 1)):
        raise ValueError('Attention must describe one contiguous unpadded sequence')
    ids = [full_ids[i] for i in active]
    if len(ids) > maximum_tokens:
        raise ValueError('Sequence exceeds the frozen token ceiling; truncation forbidden')
    if len(ids) <= len(prompt_ids) or ids[:len(prompt_ids)] != list(prompt_ids):
        raise ValueError('Processed prompt is not a strict prefix of the full sequence')
    start = len(prompt_ids)
    try:
        end = ids.index(eos_id, start)
    except ValueError as error:
        raise ValueError('Assistant EOS missing') from error
    if end == start:
        raise ValueError('Empty supervised target')
    if set(ids[start:end]) & set(forbidden_target_ids):
        raise ValueError('A reserved control/image token appears in the target')
    labels = [IGNORE_INDEX] * len(full_ids)
    for offset in range(start, end + 1):
        labels[active[offset]] = ids[offset]
    return labels, dict(prompt_tokens=start, target_tokens=end-start,
                        supervised_tokens=end-start+1, sequence_tokens=len(ids),
                        trailing_template_tokens=len(ids)-end-1,
                        padded_tokens=len(full_ids)-len(ids),
                        target_start=active[start], eos_position=active[end])


def expected_adapter_modules(layers=28, rank=8):
    """Code-derived candidates, subject to exact runtime named_modules checks."""
    dimensions = dict(q_proj=(1024, 2048), k_proj=(1024, 1024),
                      v_proj=(1024, 1024), o_proj=(2048, 1024))
    return [dict(name=f'model.language_model.layers.{layer}.self_attn.{name}',
                 in_features=dims[0], out_features=dims[1],
                 rank=rank, adapter_parameters=rank * sum(dims))
            for layer in range(layers) for name, dims in dimensions.items()]
