"""Matched attention LoRA with explicit LoRA-GA initial-factor compensation.

Initialization follows Outsider565/LoRA-GA c4cd5372c75b290924214b348008891f744512ef,
ArB2r / stable / gamma16. The frozen residual is evaluated as a low-rank
subtraction instead of rounding W - scale B0 A0 into BF16 base weights.
This preserves the initial evaluation function exactly and keeps reconstruction
on the original checkpoint explicit. It is not an ordinary rank8 adapter package.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .supervision import expected_adapter_modules
except ImportError:
    from supervision import expected_adapter_modules

RANK=8
SCALE=2.
PROJECTION='model.visual.merger'


def gradient_factors(gradient):
    if gradient.ndim!=2 or min(gradient.shape)<2*RANK or not torch.isfinite(gradient).all():
        raise ValueError('Invalid initialization gradient')
    # Same SVD direction and stable scaling as the pinned author implementation.
    u,_,v=torch.svd_lowrank(gradient.float(),q=min(4*RANK,min(gradient.shape)),niter=4)
    amplitude=gradient.shape[0]**.25/math.sqrt(16)
    a=v.T[:RANK,:].contiguous()*amplitude
    b=u[:,RANK:2*RANK].contiguous()*amplitude
    return a,b


class LargeAttentionAdapter(nn.Module):
    def __init__(self,base,factors=None):
        super().__init__()
        self.base=base
        self.in_features=base.in_features;self.out_features=base.out_features
        device=base.weight.device
        self.lora_a=nn.Parameter(torch.empty(RANK,base.in_features,device=device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.empty(base.out_features,RANK,device=device,dtype=torch.float32))
        self.dropout=nn.Dropout(.05)
        if factors is None:
            nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5));nn.init.zeros_(self.lora_b)
            self.register_buffer('initial_a',None);self.register_buffer('initial_b',None)
        else:
            a,b=factors
            if tuple(a.shape)!=tuple(self.lora_a.shape) or tuple(b.shape)!=tuple(self.lora_b.shape):
                raise ValueError('Initialization factor shape mismatch')
            with torch.no_grad():self.lora_a.copy_(a);self.lora_b.copy_(b)
            self.register_buffer('initial_a',self.lora_a.detach().clone())
            self.register_buffer('initial_b',self.lora_b.detach().clone())

    def forward(self,x):
        original=self.base(x)
        with torch.autocast(device_type=x.device.type,enabled=False):
            value=x.float()
            delta=F.linear(F.linear(self.dropout(value),self.lora_a),self.lora_b)
            if self.initial_a is not None:
                delta=delta-F.linear(F.linear(value,self.initial_a),self.initial_b)
        return original+(delta*SCALE).to(original.dtype)

    def effective_weight(self,dtype=torch.float32):
        delta=self.lora_b.to(dtype)@self.lora_a.to(dtype)
        if self.initial_a is not None:delta=delta-self.initial_b.to(dtype)@self.initial_a.to(dtype)
        return self.base.weight.to(dtype)+SCALE*delta


def attach(model,*,factors=None,grounding=False):
    model.requires_grad_(False)
    rows=expected_adapter_modules()
    if factors is not None and set(factors)!={r['name'] for r in rows}:
        raise ValueError('Initialization module set differs')
    for row in rows:
        base=model.get_submodule(row['name'])
        if not isinstance(base,nn.Linear) or (base.in_features,base.out_features)!=(row['in_features'],row['out_features']):
            raise ValueError('Attention module identity differs')
        parent,name=row['name'].rsplit('.',1)
        setattr(model.get_submodule(parent),name,LargeAttentionAdapter(base,None if factors is None else factors[row['name']]))
    projection=model.get_submodule(PROJECTION)
    if type(projection).__name__!='Qwen2_5_VLPatchMerger':raise ValueError('Visual merger identity differs')
    if not isinstance(projection.mlp[0],nn.Linear) or not isinstance(projection.mlp[2],nn.Linear):
        raise ValueError('Visual merger projection structure differs')
    if projection.mlp[2].out_features!=1024:raise ValueError('Projection text dimension differs')
    # FP32 master parameters; all model forwards use common BF16 autocast.
    projection.float().requires_grad_(True)
    if grounding:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()] if torch.cuda.is_available() else []):
            torch.manual_seed(0)
            model.add_module('cell_box_head',nn.Linear(1024,4,device=next(model.parameters()).device,dtype=torch.float32))
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    attention={n:p for n,p in parameters.items() if n.endswith(('.lora_a','.lora_b'))}
    if len(attention)!=224 or sum(p.numel() for p in attention.values())!=2293760:
        raise ValueError('Attention parameter set differs')
    if any(not(n in attention or n.startswith(PROJECTION+'.') or n.startswith('cell_box_head.')) for n in parameters):
        raise ValueError('Unexpected trainable base parameter')
    return parameters


def generation_loss(logits,labels):
    """Assistant+EOS token mean within example; caller averages example means."""
    shifted=labels[:,1:]
    valid=shifted!=-100
    per_token=F.cross_entropy(logits[:,:-1].float().transpose(1,2),shifted,reduction='none',ignore_index=-100)
    numerators=per_token.sum(dim=1);denominators=valid.sum(dim=1)
    if (denominators==0).any():raise ValueError('Example without supervised tokens')
    return (numerators/denominators).mean(),numerators.sum(),denominators.sum()


def geometry_loss(head,hidden,anchor_positions,boxes):
    if hidden.shape[0]!=1 or len(anchor_positions)!=len(boxes) or not anchor_positions:
        raise ValueError('Geometry anchors/boxes mismatch')
    states=hidden[0,anchor_positions].float()
    with torch.autocast(device_type=hidden.device.type,enabled=False):
        predicted=torch.sigmoid(head(states))
    targets=torch.as_tensor(boxes,dtype=torch.float32,device=hidden.device)
    if targets.shape!=predicted.shape or not torch.isfinite(targets).all() or ((targets<0)|(targets>1)).any():
        raise ValueError('Invalid normalized box targets')
    return F.l1_loss(predicted,targets,reduction='mean'),predicted


def combined_objective(generation,geometry,coefficient):
    # A disabled auxiliary loss must preserve the exact generation graph,
    # rather than inserting a zero-valued mixed-precision backward branch.
    if coefficient==0:return generation
    return generation+coefficient*geometry


def checkpoint_state(model):
    names={n for n,p in model.named_parameters() if p.requires_grad}
    names.update(n for n,_ in model.named_buffers() if n.endswith(('.initial_a','.initial_b')))
    return {n:value.detach().cpu().contiguous() for n,value in model.state_dict().items() if n in names}


def restore(model,state):
    expected=checkpoint_state(model)
    if set(expected)!=set(state):raise ValueError('Checkpoint parameter/buffer set differs')
    destinations=dict(model.named_parameters());destinations.update(dict(model.named_buffers()))
    with torch.no_grad():
        for name,value in state.items():
            destination=destinations[name]
            if destination.shape!=value.shape or destination.dtype!=value.dtype:raise ValueError('Checkpoint type/shape differs')
            destination.copy_(value)
