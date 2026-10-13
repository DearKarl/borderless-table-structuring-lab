"""Prepared offline CPU processor check; execute only under separate approval.

No model loading, inference, optimizer, network fetch or GPU allocation is used.
This validates the two approved native crops, not synthetic training eligibility.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .supervision import assistant_labels
except ImportError:
    from supervision import assistant_labels

PROMPT = '\nThis is the image of a table. Please output the table in OTSL format.'
SYSTEM_PROMPT = 'You are a helpful assistant.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_summary(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                sha256=hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest())


def validate(model_root, data_root, specification, *, return_batches=False, restrict_cuda=True, expected_records=2):
    # Set before package imports. No remote code is loaded by this processor check.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    if restrict_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    spec = json.loads(Path(specification).read_text(encoding='utf-8'))
    model_root, data_root = Path(model_root), Path(data_root)
    for name, expected in spec['checkpoint_processor_hashes'].items():
        if digest(model_root/name) != expected:
            raise ValueError('Checkpoint processor source hash mismatch')
    if not 1<=expected_records<=66 or len(spec['contents']) != expected_records:
        raise ValueError('Frozen TRAIN record count mismatch or exceeds approved 66 unique inputs')
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True,
                                              trust_remote_code=False)
    if processor.__class__.__name__ != 'Qwen2_5_VLProcessor':
        raise ValueError('Unexpected processor class')
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id != 151645:
        raise ValueError('Unexpected EOS identity')
    rows, batches = [], []
    for item in spec['contents']:
        for kind in ('image', 'label'):
            member = item[kind+'_member']
            if Path(member).name != member or digest(data_root/member) != item[kind+'_sha256']:
                raise ValueError('Sample member name or hash mismatch')
        label = json.loads((data_root/item['label_member']).read_text(encoding='utf-8'))
        if label['partition'] != 'train' or label['region_id'] != item['region_id']:
            raise ValueError('Frozen TRAIN identity mismatch')
        if label['image'] != item['image_member']:
            raise ValueError('Image-label binding mismatch')
        target = label['target_otsl']
        if not isinstance(target, str) or not target:
            raise ValueError('Empty supervised target')
        with Image.open(data_root/item['image_member']) as loaded:
            loaded.load()
            image = loaded.copy()
        w, h = image.size
        if image.mode != 'RGB' or min(w,h) < 28 or max(w,h)/min(w,h) > 50 or w*h > 64000000:
            raise ValueError('Native crop would trigger unvalidated TeleOCR preprocessing')
        messages = [dict(role='system', content=SYSTEM_PROMPT),
                    dict(role='user', content=[dict(type='image'), dict(type='text',text=PROMPT)])]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full = processor.apply_chat_template(messages+[dict(role='assistant', content=target)],
                                             tokenize=False, add_generation_prompt=False)
        inference = processor(text=[prompt], images=[image], padding=True, return_tensors='pt')
        training = processor(text=[full], images=[image], padding=True, return_tensors='pt', truncation=False)
        for key in ('pixel_values', 'image_grid_thw'):
            if key not in inference or not torch.equal(inference[key], training[key]):
                raise ValueError('Native inference/training image tensor mismatch')
        prompt_ids = inference['input_ids'][0][inference['attention_mask'][0].bool()].tolist()
        ids = training['input_ids'][0].tolist()
        mask = training['attention_mask'][0].tolist()
        labels, counts = assistant_labels(prompt_ids, ids, mask, eos_id=tokenizer.eos_token_id,
                                         forbidden_target_ids=tokenizer.all_special_ids)
        target_ids = ids[counts['target_start']:counts['eos_position']]
        if tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != target:
            raise ValueError('Target tokenization failed exact text round trip')
        grid = inference['image_grid_thw'][0].tolist()
        expected_image_tokens = int(grid[0]*grid[1]*grid[2]) // processor.image_processor.merge_size**2
        if prompt_ids.count(151655) != expected_image_tokens:
            raise ValueError('Expanded image placeholder/grid mismatch')
        if any(token != -100 for token in labels[:counts['target_start']]):
            raise ValueError('Prompt leaked into supervision')
        rows.append(dict(region_id=item['region_id'], partition='train', counts=counts,
                         native_crop_size=[w,h], image_tokens=expected_image_tokens,
                         exact_target_roundtrip=True, native_image_tensor_equivalence=True,
                         tensors={key:tensor_summary(inference[key]) for key in ('pixel_values','image_grid_thw')},
                         prompt_tensor_sha256=tensor_summary(inference['input_ids'])['sha256'],
                         labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest()))
        training['labels'] = torch.tensor([labels], dtype=torch.long)
        if return_batches:
            batches.append(training)
    report = dict(status='processor_and_supervision_checks_passed', rows=rows,
                model_imports=0, model_forward_calls=0, optimizer_updates=0,
                gpu_allocations=0, truncation=False, formal_finetuning_started=False,
                processor_class=processor.__class__.__name__,
                image_processor_class=processor.image_processor.__class__.__name__,
                limitations=['Default Transformers input contract; no vLLM runtime equivalence claim',
                             'No model import, loss, gradient, adapter or save/reload validation'])
    return (report, batches) if return_batches else report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--specification', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = validate(args.model_root, args.data_root, args.specification)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
