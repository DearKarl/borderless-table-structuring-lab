# Historical implementation inventory

Inspected 2026-10-07. All V0-V4 source archives listed below are present in the
local checkout and tracked by Git. Counts describe these archive directories,
not all related files elsewhere in the repository.

| Version | Source root | Tracked files | Python files | Reusable scope |
|---|---|---:|---:|---|
| V0 | [versions/v0](../versions/v0/README.md) | 41 | 29 | MinerU plus native-table integration and reproduction templates |
| V1 | [versions/v1](../versions/v1/README.md) | 34 | 29 | Formula integration, image input and controller logic |
| V2 | [versions/v2](../versions/v2/README.md) | 135 | 128 | TeleOCR/Paddle integration, assembly and failure handling |
| V3.1 | [versions/v3_1](../versions/v3_1/README.md) | 70 | 49 | Layout/formula expert profiles, coordinates, transactions and runtime contracts |
| V3.2 | [versions/v3_2](../versions/v3_2/README.md) | 92 | 67 | Ovis text dispatch, completion checks, native fallback and parameter locks |
| V4 | [release](../releases/v4-gbdt-evaluation-20261006/README.md) | 140 | 112 | Learned input selection, model, runtime and completed aggregate evaluation |

All 414 Python files in these roots parsed successfully with Python's AST
parser. This is a syntax check, not an import, dependency, GPU or completeness
test. One legacy invalid-escape warning occurred without a parse failure.

## V3.1 source versus reported run

The package contains cli.py, native.py, v31_providers.py, v31_rules.py,
v31_transactions.py, v31_coordinates.py, evaluation.py, preparation templates,
SOURCE_LOCK.json and a parent asset lock. These are actual implementation files.

The inspected [native.py](../versions/v3_1/hybrid_v3_full_eval_v1/native.py)
describes and calls the TeleOCR read_fn -> PDF -> do_parse route (lines 248-252).
The researcher-reported 97.6973 variant bypasses PDF transcoding/resampling.
The exact variant or input-path patch must therefore be identified before
claiming the archived entry reproduces that specific score. Source presence is
confirmed; this run-to-source correspondence is unresolved.

## V3.2 source versus reported run

The package additionally contains v32_dispatch.py, v32_provider.py,
v32_worker.py, v32_crop.py, v32_protocol.py, v32_parameters.py, model and parameter
locks, and the CLI integration. In [v32_dispatch.py](../versions/v3_2/hybrid_v3_full_eval_v1/v32_dispatch.py),
ON routes eligible text slots to Ovis and uses native recognition when the expert
response is rejected. Formula/table slots retain the native path, and geometry
is checked for preservation. Protocol acceptance is explicitly distinct from a
semantic quality decision.

The reported OFF/ON aggregate result is recorded in [RESULTS.md](RESULTS.md).
Its exact scored revision and full prediction manifest remain unbound here.
Formula/table aggregate changes do not by themselves demonstrate direct edits
to those components by the text expert.

## External assets and prediction availability

The public archives intentionally exclude pretrained weights, benchmark inputs,
GT and historical prediction caches, and require separately pinned TeleOCR
source. Their absence from these archive directories does not prove absence
from private local storage or a server. A complete search and identity check of
those runtime artifacts is a separate task; this inventory makes no claim that
they are missing everywhere. No historical code, models or scores were modified.
