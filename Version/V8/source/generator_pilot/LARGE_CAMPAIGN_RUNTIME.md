# V8 runtime sources

The completed six-fit experiment uses the following workers. Scientific settings
are in the [execution specification](../../artifacts/results/v7_finetuning/large-lora-001/EXECUTION_SPECIFICATION.json),
with measured trajectories and DEV selections in the
[training analysis](../../artifacts/results/v7_finetuning/large-lora-001/TRAINING_ANALYSIS.md).
The source retains V7.3.0/V7.3.1/V7.3.2 as immutable internal recipe aliases for
V8.1/V8.2/V8.3; the [mapping](large_names.py) is explicit.

| Source | Responsibility |
|---|---|
| [large_calibrate.py](large_calibrate.py) | One shared 256-TRAIN mean-gradient initialization, SVD factors and initial-logit equality checks |
| [large_train.py](large_train.py) | Fixed 3,750-update fit, loss/resource logs, fixed teacher-forced panels and all three epoch checkpoints |
| [large_generate.py](large_generate.py) | Label-free default-input DEV/confirmation generation using frozen checkpoint and processor hashes |
| [large_load.py](large_load.py) | Reconstruction of a complete adapter/merger package on the exact original base |
| [large_adapters.py](large_adapters.py) | LoRA-GA initial-factor compensation, adapter attachment and loss definitions |
| [large_inputs.py](large_inputs.py) | Assistant-token masks and table-cell supervision interfaces |
| [large_plots.py](large_plots.py) | Sanitized numeric series and measured PNG/SVG plots |

Workers run from a prepared stage directory, with `STAGE_BINDING.json` describing
the locally installed runtime, base-model hashes, selected device and frozen input
artifacts. Training also requires its exact `EXECUTION_PROTOCOL.json` and common
input manifest; calibration uses `CALIBRATION_INPUTS.json`; generation uses
`INFERENCE_INPUTS.json`. A stage's `--preflight` checks input and runtime bindings
without making model calls. A separate supervisor supplies device isolation and
records allocation lifetime. These worker files are not a complete standalone
launcher or a public dataset download package.

Deployment bindings, source PDFs, native crops, processor caches, labels, raw
predictions and model weights are not part of this source publication. The public
specification is a reviewed projection of the executed protocol, not a replacement
protocol file whose hash can be passed to a historical run. It documents the later
six-model benchmark amendment separately from the unchanged training settings.

Executed-source SHA256 values refer to the exact worker bytes frozen for execution.
Git-source SHA256 values bind the published bytes. The repository's `.gitattributes`
preserves original source line endings, so the three executed workers retain their
exact execution hashes in Git. The workers' local execution bytes are unchanged.

LoRA-GA reconstruction subtracts the stored initial low-rank factors and restores
the trained visual merger. The package is therefore not a generic PEFT rank-8
adapter. `large_load.load(package, base, device=...)` checks the manifest, runtime,
original base assets, source and serialized tensors. Pair generation with
`torch.inference_mode()` and `large_load.generation_context(model)` because the
trained merger is FP32 and the base uses BF16. Selected package assembly and its
final verification remain part of the ongoing evaluation delivery.

The common inference backend is Transformers. The TeleOCR presence/frequency
penalties are not applied by this backend, so historical vLLM results are not
treated as matched controls. The geometry head is training-only; generation
receives no ground-truth cell locations.
