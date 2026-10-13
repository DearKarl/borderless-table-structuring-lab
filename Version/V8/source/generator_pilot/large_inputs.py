"""Exact default processor tensors, assistant masks and origin-cell anchors."""
import hashlib
import json
import re

try:
    from .supervision import assistant_labels
    from .validate_smoke_inputs import PROMPT, SYSTEM_PROMPT, tensor_summary
except ImportError:
    from supervision import assistant_labels
    from validate_smoke_inputs import PROMPT, SYSTEM_PROMPT, tensor_summary


def prepare_one(processor, image, canonical):
    import torch
    target=canonical['target_otsl']
    tokenizer=processor.tokenizer
    messages=[dict(role='system',content=SYSTEM_PROMPT),
        dict(role='user',content=[dict(type='image'),dict(type='text',text=PROMPT)])]
    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(messages+[dict(role='assistant',content=target)],tokenize=False,add_generation_prompt=False)
    inference=processor(text=[prompt],images=[image],padding=True,return_tensors='pt')
    training=processor(text=[full],images=[image],padding=True,return_tensors='pt',truncation=False)
    for key in ['pixel_values','image_grid_thw']:
        if not torch.equal(inference[key],training[key]):raise ValueError('Inference/training image tensor mismatch')
    prompt_ids=inference['input_ids'][0][inference['attention_mask'][0].bool()].tolist()
    ids=training['input_ids'][0].tolist()
    labels,counts=assistant_labels(prompt_ids,ids,training['attention_mask'][0].tolist(),
        eos_id=tokenizer.eos_token_id,forbidden_target_ids=tokenizer.all_special_ids,maximum_tokens=16384)
    target_ids=ids[counts['target_start']:counts['eos_position']]
    if tokenizer.decode(target_ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)!=target:
        raise ValueError('Target token roundtrip mismatch')
    encoded=tokenizer(target,add_special_tokens=False,return_offsets_mapping=True)
    if encoded['input_ids']!=target_ids:raise ValueError('Context changed target tokenization')
    offsets=encoded['offset_mapping']
    origins=list(re.finditer(r'<fcel>|<ecel>',target))
    cells=canonical['table']['cells']
    if len(origins)!=len(cells):raise ValueError('Cell origin count mismatch')
    anchors=[];failed=[]
    for cell,origin in zip(cells,origins):
        end=origin.end()
        # Natural BPE tokens may consume the closing '>' together with the
        # beginning of the cell text or the following structural tag. Anchor
        # after the first complete token consuming the marker; retain spill.
        choices=[i for i,(a,b) in enumerate(offsets) if a<end<=b]
        if not choices:
            failed.append(dict(row=cell['row'],col=cell['col'],origin_tag=origin.group(),character_end=end))
            continue
        i=choices[-1]
        if not any(a<end and b>origin.start() for a,b in offsets[:i+1]):
            raise ValueError('Anchor does not consume origin tag')
        position=counts['target_start']+i
        if labels[position]<0 or position>=counts['eos_position']:
            raise ValueError('Anchor is outside assistant target')
        anchors.append(dict(row=cell['row'],col=cell['col'],token_position=position,origin_tag=origin.group(),
            marker_character_end=end,anchor_token_character_span=list(offsets[i]),
            boundary_spill_characters=offsets[i][1]-end,
            empty=not cell['text'],rowspan=cell['rowspan'],colspan=cell['colspan']))
    positions=[a['token_position'] for a in anchors]
    if positions!=sorted(set(positions)):raise ValueError('Cell origins do not have unique monotonic anchors')
    grid=inference['image_grid_thw'][0].tolist()
    image_tokens=int(grid[0]*grid[1]*grid[2])//processor.image_processor.merge_size**2
    if prompt_ids.count(151655)!=image_tokens:raise ValueError('Image token/grid mismatch')
    training['labels']=torch.tensor([labels],dtype=torch.long)
    report=dict(counts=counts,native_crop_size=list(image.size),image_mode=image.mode,
        processor_grid_thw=grid,processor_patch_size=processor.image_processor.patch_size,
        image_tokens=image_tokens,tensors={k:tensor_summary(inference[k]) for k in ['pixel_values','image_grid_thw']},
        prompt_ids_sha256=tensor_summary(inference['input_ids'])['sha256'],
        labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest(),
        target_ids_sha256=hashlib.sha256(json.dumps(target_ids).encode()).hexdigest(),
        anchors=anchors,unrepresentable_anchors=failed,all_anchors_exact=not failed,
        anchor_schema='first-complete-origin-token-v2',
        token_roundtrip_exact=True,truncation=False)
    return training,report


def map_boxes(canonical, page_size, render_size, transform):
    """Map published PDF boxes through the verified actual native helper frame."""
    if transform['status']!='verified':raise ValueError('Unsupported crop transform')
    sx,sy=render_size[0]/page_size[0],render_size[1]/page_size[1]
    x0,y0,x1,y1=transform['raw_bbox_pixels']
    left,top=transform['padding_left_top']
    rx,ry=transform['resize_xy'];w,h=transform['output_size']
    boxes=[]
    for cell in canonical['cell_boxes']:
        px0,py0,px1,py1=cell['pdf_bbox']
        native=[px0*sx,py0*sy,px1*sx,py1*sy]
        if not (native[0]>=x0-2 and native[1]>=y0-2 and native[2]<=x1+2 and native[3]<=y1+2):
            raise ValueError('Cell not completely covered by predicted crop')
        mapped=[(native[0]-x0+left)*rx/w,(native[1]-y0+top)*ry/h,
            (native[2]-x0+left)*rx/w,(native[3]-y0+top)*ry/h]
        # Up to two native pixels of numeric/detection boundary tolerance were
        # prespecified. Clip only this audited boundary allowance to [0,1].
        clipped=[min(1.,max(0.,v)) for v in mapped]
        if not(clipped[0]<clipped[2] and clipped[1]<clipped[3]):raise ValueError('Degenerate normalized cell')
        boxes.append(dict(row=cell['row'],col=cell['col'],normalized_box=clipped,
            unclipped_box=mapped,boundary_clipped=clipped!=mapped))
    return boxes
