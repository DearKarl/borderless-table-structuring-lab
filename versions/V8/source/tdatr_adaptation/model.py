"""Shared model, objectives, and exact checkpoint handling for both TDATR fits."""
import torch

from versions.generator_pilot.large_adapters import attach, generation_loss, checkpoint_state, restore
from .backbone import NativeFeatureCapture, native_fusion_forward
from .sgcl import CellLocalizer, localization_losses


def attach_localizer(model, initial_state=None):
    if hasattr(model, 'tdatr_localizer'):
        raise ValueError('Localization head already attached')
    text=model.config.text_config
    vision=model.config.vision_config
    head=CellLocalizer(language_dim=text.hidden_size,decoder_layers=text.num_hidden_layers,
        vision_channels=[vision.hidden_size,vision.out_hidden_size])
    if initial_state is not None: head.load_state_dict(initial_state, strict=True)
    head=head.to(device=next(model.parameters()).device,dtype=torch.float32)
    model.add_module('tdatr_localizer',head)
    return NativeFeatureCapture(model)


def objective(model, batch, *, capture=None, cell_ranges=None, boxes=None, logical_spans=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if capture is None:
            output=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False)
            prediction=None
        else:
            output,prediction=native_fusion_forward(model,batch,cell_ranges,capture)
    ce,numerator,denominator=generation_loss(output.logits,batch['labels'])
    losses=dict(ce=ce)
    if prediction is not None:
        losses.update(localization_losses(prediction,
            torch.as_tensor(boxes,device=ce.device,dtype=torch.float32),
            torch.as_tensor(logical_spans,device=ce.device,dtype=torch.float32)))
    total=ce if prediction is None else ce+losses['weighted_localization']
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite objective')
    return total, losses, numerator, denominator


def trainable_parameters(model):
    return {name:p for name,p in model.named_parameters() if p.requires_grad}


def optimizer_for(model):
    parameters=trainable_parameters(model)
    ordinary=[p for name,p in parameters.items() if not name.startswith('model.visual.merger.')]
    merger=[p for name,p in parameters.items() if name.startswith('model.visual.merger.')]
    return torch.optim.AdamW([
        dict(params=ordinary,lr=1e-4,base_lr=1e-4),dict(params=merger,lr=1e-5,base_lr=1e-5)],
        betas=(.9,.999),eps=1e-8,weight_decay=.01)


def verify_gradients(model):
    totals={}
    for name,p in model.named_parameters():
        if not p.requires_grad:
            if p.grad is not None: raise ValueError('Frozen model parameter received a gradient')
            continue
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise FloatingPointError('Missing or nonfinite gradient: '+name)
        category='localizer' if name.startswith('tdatr_localizer.') else 'merger' if name.startswith('model.visual.merger.') else 'attention_adapter'
        norm=p.grad.detach().float().square().sum()
        totals[category]=totals.get(category,0)+norm
    return {key:float(value.sqrt()) for key,value in totals.items()}


def exact_save(model, path):
    from safetensors.torch import save_file, load_file
    state=checkpoint_state(model)
    save_file(state,str(path))
    loaded=load_file(str(path))
    if set(loaded)!=set(state) or any(not torch.equal(value,loaded[name]) for name,value in state.items()):
        raise ValueError('Checkpoint serialization changed a trainable tensor')
    return state
