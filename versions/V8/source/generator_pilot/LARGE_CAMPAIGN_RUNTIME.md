# V8 reproduction source

V8.1 uses ordinary LoRA, V8.2 uses LoRA-GA, and V8.3 adds training-only cell
geometry supervision. [Method parameters](../../results/large-lora-001/EXECUTION_SPECIFICATION.json)
and [package identities](../../results/large-lora-001/MODEL_PACKAGES.json) describe
the available model outputs.

| Source | Role |
| --- | --- |
| [large_calibrate.py](large_calibrate.py) | TRAIN-only gradient initialization |
| [large_train.py](large_train.py) | Adapter/merger training and diagnostics |
| [large_generate.py](large_generate.py) | Default-input generation |
| [large_inputs.py](large_inputs.py) | Input tensors and assistant-token masks |
| [large_adapters.py](large_adapters.py) | Attention adapters and initial-factor compensation |
| [large_load.py](large_load.py) | Complete package restoration on its original pretrained model |

Workers require validated external stage bindings, original model assets and
input identities. They do not download benchmark data or launch a job on import.
LoRA-GA restoration subtracts saved initial factors and restores the visual
merger. Use the supplied loader rather than generic final-factor adapter loading.
Generation uses the native Transformers processor and greedy helper; the V8.3
cell head does not receive labels during inference.
