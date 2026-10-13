#!/usr/bin/env python3
"""CPU tensor-level inference with the exact archived Explicit-v2 model.

This is not a PDF parser, correction executor, or benchmark evaluator.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from runtime.model import ExplicitV2LayoutTransformer
from runtime.safe_state import load_safe_state

WEIGHT_SHA = 'c1615ce37de058e82b0ec33a84fb06afb25d4f6d4dfa401ff1e67f9d547019f0'
FLOAT_KEYS = {'image', 'cell_features', 'cell_boxes', 'token_features', 'text_candidate_token_weights'}
BOOL_KEYS = {'cell_mask', 'token_mask', 'token_owner_candidate_mask', 'text_candidate_mask', 'text_candidate_is_raw'}
INPUT_KEYS = FLOAT_KEYS | BOOL_KEYS | {'grid_shape'}


def load_model(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('checkpoint must be a regular file')
    if hashlib.sha256(path.read_bytes()).hexdigest() != WEIGHT_SHA:
        raise ValueError('not the released Explicit-v2 Original checkpoint')
    model = ExplicitV2LayoutTransformer()
    result = model.load_state_dict(load_safe_state(path), strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    model.set_cross_modal_fusion_enabled(True)
    model.set_grounding_aware_decision_fusion_enabled(True)
    return model.eval().requires_grad_(False)


def fixture():
    """Project-authored, two-cell synthetic interface example; no benchmark data."""
    return {
        'image': torch.ones(1, 1, 32, 32),
        'cell_features': torch.zeros(1, 2, 16),
        'cell_boxes': torch.tensor([[[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]]),
        'cell_mask': torch.ones(1, 2, dtype=torch.bool),
        'grid_shape': torch.tensor([[1, 2]], dtype=torch.long),
        'token_features': torch.zeros(1, 2, 12),
        'token_mask': torch.ones(1, 2, dtype=torch.bool),
        'token_owner_candidate_mask': torch.tensor([[[True, False, True], [False, True, True]]]),
        'text_candidate_token_weights': torch.tensor([[[[0.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]]),
        'text_candidate_mask': torch.ones(1, 2, 2, dtype=torch.bool),
        'text_candidate_is_raw': torch.tensor([[[True, False], [True, False]]]),
    }


def read_inputs(path):
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) != INPUT_KEYS:
        raise ValueError('input must contain exactly the 11 documented tensor fields')
    return {name: torch.tensor(data, dtype=torch.float32 if name in FLOAT_KEYS else
                              torch.bool if name in BOOL_KEYS else torch.long)
            for name, data in value.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/training/model.safe-state'))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smoke', action='store_true', help='synthetic interface check; no quality score')
    group.add_argument('--input', type=Path, help='precomputed model tensor JSON, not a PDF or Raw HTML')
    parser.add_argument('--output', type=Path, help='new JSON output; never overwrite existing results')
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = load_model(args.checkpoint)
    inputs = fixture() if args.smoke else read_inputs(args.input)
    with torch.no_grad():
        outputs = model(**inputs)
    if not all(torch.isfinite(tensor).all().item() for tensor in outputs.values()):
        raise ValueError('non-finite model output')
    result = {
        'status': 'SYNTHETIC_CPU_INTERFACE_VERIFIED_NOT_A_QUALITY_SCORE' if args.smoke else 'TENSOR_INFERENCE_ONLY_NOT_COMMITTED_EDITS',
        'checkpoint_sha256': WEIGHT_SHA,
        'torch_version': torch.__version__,
        'state_tensors': len(model.state_dict()),
        'strict_load': True,
        'device': 'cpu',
        'output_shapes': {name: list(tensor.shape) for name, tensor in outputs.items()},
        'training': False,
        'benchmark_evaluation': False,
        'safe_executor_run': False,
    }
    if not args.smoke:
        result['outputs'] = {name: tensor.tolist() for name, tensor in outputs.items()}
    payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
