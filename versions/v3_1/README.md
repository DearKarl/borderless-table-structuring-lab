# Hybrid V3.1

TeleOCR backbone with conservative PP-DocLayoutV3 layout proposals and PaddleOCR-VL1.6 formula recognition. Accepted geometry changes require fresh native recognition; ambiguous proposals fall back to the original result.

**Full benchmark: Pending.** The final historical V3.1 package completed bounded page/PDF and official-entry checks. Formula model loading was observed, but formula generation was not exercised in those checks. These checks do not establish a full-benchmark improvement. The public snapshot has publication-specific documentation and external-source changes and has not been rerun on GPU.

## Inspect the source

From `versions/v3_1`, using Python 3.12:

```sh
python -B -m hybrid_v3_full_eval_v1.cli --help
python -B -m hybrid_v3_full_eval_v1.cli inspect-assets
python -B -m hybrid_v3_full_eval_v1.cli package-check --root .
python -B -m hybrid_v3_full_eval_v1.cli environment-requirements --output requirements.json
```

These commands do not load models. CPU checks additionally require Pillow, NumPy, Pydantic v2 and pandas. Real inference requires a separately prepared Linux CUDA environment, immutable image identities, exact model/source hashes, parameter evidence and explicit GPU/resource budgets. Windows inspection does not establish Windows GPU support.

## Prepare and run

Start with `hybrid_v3_full_eval_v1/examples/preparation.template.json` and the [environment contract](hybrid_v3_full_eval_v1/ENVIRONMENT_PREPARATION.md). Set `asset_roots.tele_source` to your separately obtained pinned [TeleOCR](https://github.com/caipeng328/TeleOCR) checkout. All 55 required source members must match the recorded hashes; unrelated checkout metadata is excluded. No source download is performed.

```sh
python -B -m hybrid_v3_full_eval_v1.cli prepare-assets --configuration preparation.json --output new-plan
python -B -m hybrid_v3_full_eval_v1.cli bind-assets --plan new-plan/bind-plan.json --output new-runtime
python -B -m hybrid_v3_full_eval_v1.cli manifest --source input-images --output inputs.json
python -B -m hybrid_v3_full_eval_v1.cli preflight --profile v31-both --mode on --inputs inputs.json --runtime new-runtime/runtime.json --budget budget.json
python -B -m hybrid_v3_full_eval_v1.cli infer --profile v31-both --mode on --inputs inputs.json --runtime new-runtime/runtime.json --budget budget.json --output new-run
python -B -m hybrid_v3_full_eval_v1.cli collect --run new-run/RUN_MANIFEST.json --output new-collection
```

The preparation template starts with the V3.1 both-expert profile. For V3.2, supply `profile=v32-text`, the Ovis model root, separate Ovis environment, `v32` settings and the host-environment descriptor required by `v32_prepare.py`, `v32_binding.py` and `v32_cluster.py`; do not reuse a V3.1 environment as an Ovis deployment.

`off` preserves native stepping; `pass-through` exercises dispatch without expert changes; `on` enables the declared expert. Inputs retain their original image/PDF identity. Inference cannot read evaluation ground truth. Evaluation is a separate explicit CLI command with its own pinned evaluator and ground truth inputs.

Source and package hashes identify this archive. Historical runtime evidence does not transfer to a new environment. No models, datasets, predictions, private environment locks or host credentials are included. See [source provenance](hybrid_v3_full_eval_v1/PROVENANCE.md).
