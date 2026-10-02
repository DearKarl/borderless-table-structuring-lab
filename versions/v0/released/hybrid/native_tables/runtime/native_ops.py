"""Unmodified scientific functions extracted from the evaluated pipeline."""
import hashlib
import math

REGION_PREPARATION = {'minimum_edge': 28, 'maximum_edge_ratio': 50,
                      'padding': 'centered RGB white on short axis only',
                      'enlargement': 'proportional ceil dimensions, PIL.BICUBIC',
                      'stage': 'recognition regions only, after rotation'}


PROMPTS = {
    'layout': '\nAnalyze the image layout.',
    'text': '\nPlease output the text content from the image.',
    'table': '\nThis is the image of a table. Please output the table in OTSL format.',
    'formula': '\nPlease write out the expression of the formula in the image using LaTeX format.',
    'code': '\nThe image contains a code snippet, please output the parsing result.',
    'seal': '\nSeal Recognition:',
    'char': '\nThis is a scientific figure. Please extract the table implied by this figure.',
}


def image_binding(image):
    return {'dimensions': list(image.size), 'mode': image.mode,
            'rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()}


def crop_block(image, block):
    values = block['bbox_1000']
    pixels = [value * image.size[axis % 2] // 1000 for axis, value in enumerate(values)]
    if not (0 <= pixels[0] < pixels[2] <= image.width
            and 0 <= pixels[1] < pixels[3] <= image.height):
        raise ValueError('INVALID_OR_SUBPIXEL_CROP')
    crop = image.crop(tuple(pixels))
    raw = image_binding(crop)
    if block['angle']:
        rotated = crop.rotate(block['angle'], expand=True)
        crop.close()
        crop = rotated
    return crop, {'source_pixel_xyxy': pixels, 'before_rotation': raw,
                  'angle': block['angle'], 'after_rotation': image_binding(crop)}


def prepare_recognition_image(crop):
    """Implement the public input recipe without changing the retained raw crop."""
    from PIL import Image
    if crop.mode != 'RGB' or min(crop.size) <= 0:
        raise ValueError('RECOGNITION_INPUT_MUST_BE_NONEMPTY_RGB')
    prepared = crop.copy()
    steps = []
    width, height = prepared.size
    if max(width, height) / min(width, height) > REGION_PREPARATION['maximum_edge_ratio']:
        minimum = math.ceil(max(width, height) / REGION_PREPARATION['maximum_edge_ratio'])
        target = (width, minimum) if width > height else (minimum, height)
        offset = ((target[0] - width) // 2, (target[1] - height) // 2)
        canvas = Image.new('RGB', target, (255, 255, 255))
        canvas.paste(prepared, offset)
        prepared.close()
        prepared = canvas
        steps.append({'type': 'center_white_pad', 'before': [width, height],
                      'after': list(target), 'offset': list(offset)})
    if min(prepared.size) < REGION_PREPARATION['minimum_edge']:
        width, height = prepared.size
        scale = REGION_PREPARATION['minimum_edge'] / min(width, height)
        target = (math.ceil(width * scale), math.ceil(height * scale))
        enlarged = prepared.resize(target, Image.Resampling.BICUBIC)
        prepared.close()
        prepared = enlarged
        steps.append({'type': 'minimum_edge_bicubic', 'before': [width, height],
                      'after': list(target), 'scale': scale})
    return prepared, {'raw_rotated_crop': image_binding(crop),
                      'model_image': image_binding(prepared), 'steps': steps}


def generate(torch, model, processor, image, task, helper):
    chat = processor.apply_chat_template([
        {'role': 'system', 'content': helper.CONFIG['system']},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': PROMPTS[task]}]},
    ], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[image], padding=True, return_tensors='pt')
    if 'labels' in inputs:
        raise ValueError('UNEXPECTED_MODEL_LABELS')
    inputs = inputs.to(device=model.device, dtype=model.dtype)
    prompt_ids = inputs.input_ids[0].tolist()
    with torch.inference_mode():
        outputs = model.generate(**inputs, use_cache=True, max_new_tokens=4096, do_sample=False)
    result = helper.generation_record(prompt_ids, outputs.cpu().tolist()[0],
                                      model.generation_config.eos_token_id)
    result['decoded_with_special_tokens'] = processor.batch_decode(
        [result['raw_token_ids']], skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    result['decoded_text'] = processor.batch_decode(
        [result['raw_token_ids']], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    result['processed_input_shapes'] = {k: list(v.shape) for k, v in inputs.items() if hasattr(v, 'shape')}
    result['task'] = task
    del outputs, inputs
    return result
