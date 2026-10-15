# TDATR-style native VLM adaptation

V8.4 uses ordinary attention LoRA. V8.5 uses compensated LoRA-GA with the same
detail and fusion supervision. Each version receives one fit with three detail
passes and three fusion passes over the original 20,000-table TRAIN cohort:
7,500 optimizer updates and 120,000 exposures. Older V8.1â€“V8.3 controls have half
as many exposures and are reused without refitting.

The [dataset and protocol](../../results/tdatr-v8-45-001/DATASET_AND_PROTOCOL.md)
and [training parameters and source identities](../../results/tdatr-v8-45-001/PROTOCOL.json)
describe task eligibility, inputs, optimization, selection, provenance and limits.
Results require the saved completed score receipts; source availability is not
evidence that the experiment completed or improved accuracy.

## Scientific implementation

| Source | Responsibility |
| --- | --- |
| `targets.py`, `audit_inputs.py`, `freeze_plan.py` | Detail targets, complete native tokenizer audit and shared realized training assignments |
| `cell_tokens.py`, `inputs.py` | Full-cell token intersections, assistant-only labels and unchanged native image tensors |
| `backbone.py`, `sgcl.py`, `model.py` | Native feature capture, language-layer aggregation, cell context, box refinement and localization losses |
| `calibrate.py`, `train.py` | Gradient-based initialization and one frozen detail-to-fusion fit |
| `inference.py`, `generate.py`, `benchmark_recovery.py` | Exact checkpoint restoration, native generation and annotation-free localization replay |
| `heldout_score.py`, `select.py` | Full source denominators and DEV-only epoch selection |
| `summarize_results.py`, `export_curves.py`, `plot_results.py`, `report_results.py` | Saved-result reduction, full numeric curves, figures and descriptive reporting |
| `portable_load.py`, `package_model.py`, `verify_packages.py` | Portable complete packages and byte/provenance verification |

Runtime launch bindings, server credentials, cohort payloads, private scheduling
guards and raw predictions stay outside the public artifact set. Scientific
modules are bound by their SHA-256 values in the frozen record. Delivery wrappers
were added after training freeze and do not change the fitted model or evaluator.

The adaptation follows [TDATR](https://arxiv.org/html/2603.22819v1), using the
existing VLM's pre-merger and merger maps in place of the paper's visual pyramid.
It uses table-only Markdown tasks, native OTSL fusion, full-cell mean pooling and
explicitly normalized localization losses. It is an adapted reproduction, not
an exact copy of the original architecture or training corpus.

## Complete model packages

Each selected package contains FP32 attention adapters, visual-merger parameters
and localization-head parameters. V8.5 also stores its initial factors so the
effective adapter remains `2 * (B A x - B0 A0 x)` over the original BF16 base.
Generic final-factor-only loading is not equivalent. No BF16 merged equivalence
is claimed.

Add a package's `code` directory to `PYTHONPATH`, then use:

```python
from versions.tdatr_adaptation.portable_load import load, generation_context

model, processor, capture, manifest = load(
    package_path, original_base_path, device="cuda:0"
)
# Use the preserved TeleOCR default table helper and native predicted crop.
# Pair torch.inference_mode() with generation_context(model) for generation.
capture.close()
```

The loader checks base, runtime, reconstruction-source and checkpoint identity.
`cuda:0` refers to the first device in the caller's permitted visible-device set.
Localization replays only the model's own OTSL in an additional forward; it takes
no geometry labels and does not rewrite official text or HTML. Invalid OTSL or
overlong replays remain unavailable localization outcomes.

No package or source module automatically launches a training job. CLI execution
requires explicit binding files. The original base remains responsible for
layout and non-table content. Default Transformers inference does not apply
presence/frequency penalties, so historical vLLM scores are not interchangeable.

## Published layout and execution history

The repository compatibility loader exposes this directory as
`versions.tdatr_adaptation`. The scored full benchmark used
`benchmark_recovery.py`: its native keyword adapter repairs a pre-call recording
error in the retained original `benchmark.py`. It keeps the frozen models,
inputs and evaluator. See the [completed numerical results](../../results/tdatr-v8-45-001/RESULTS.json).
`SOURCE_HASHES.json` distinguishes exact executed modules from delivery utilities
and explicit path-resolution changes in legacy helpers. Table parsing, scoring
and fitted model algorithms are unchanged. The original private controller and
GPU scheduling bindings are not published; workers require caller-supplied
bindings and original external assets. `TELEOCR_OTSL_DECODER` can identify the
preserved decoder source when using its optional loader.
