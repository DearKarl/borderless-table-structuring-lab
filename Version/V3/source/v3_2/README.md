# Hybrid V3.2

TeleOCR backbone with an OvisOCR2 text-region expert. Only text slots are eligible; native formula and table paths remain intact. Strict completion, EOS and process-exit evidence control adoption and fallback.

**Reported full benchmark: OFF 93.4951, ON 93.7208 Overall (+0.2257 points).** The researcher reports 1,651 pages per arm using the same official evaluator and GT, with all failures retained. Table TEDS changes from 86.9556 to 87.3918 and formula CDM from 98.2000 to 98.4454; text quality reportedly regresses slightly, with no numeric text metric supplied. See the [full-set and supplementary results](../../research/RESULTS.md#v32-full-set-off-and-on-comparison). Raw scoring files and the exact scored revision remain to be added.

The archived implementation previously completed bounded crop and off/on/pass-through checks. Those checks are separate from the newly supplied full evaluation. Inference requires the Linux environment described below.

## Inspect the source

From `versions/v3_2`, using Python 3.12:

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
python -B -m hybrid_v3_full_eval_v1.cli preflight --profile v32-text --mode on --inputs inputs.json --runtime new-runtime/runtime.json --budget budget.json
python -B -m hybrid_v3_full_eval_v1.cli infer --profile v32-text --mode on --inputs inputs.json --runtime new-runtime/runtime.json --budget budget.json --output new-run
python -B -m hybrid_v3_full_eval_v1.cli collect --run new-run/RUN_MANIFEST.json --output new-collection
```

The preparation template starts with the V3.1 both-expert profile. For V3.2, supply `profile=v32-text`, the Ovis model root, separate Ovis environment, `v32` settings and the host-environment descriptor required by `v32_prepare.py`, `v32_binding.py` and `v32_cluster.py`; do not reuse a V3.1 environment as an Ovis deployment.

`off` preserves native stepping; `pass-through` exercises dispatch without expert changes; `on` enables the declared expert. Inputs retain their original image/PDF identity. Inference cannot read evaluation ground truth. Evaluation is a separate explicit CLI command with its own pinned evaluator and ground truth inputs.

Source and package hashes identify this archive. Historical runtime evidence does not transfer to a new environment. No models, datasets, predictions, private environment locks or host credentials are included. See [source provenance](hybrid_v3_full_eval_v1/PROVENANCE.md).
