# Baseline comparability audit

The saved local D0 result, **97.574874**, cannot be treated as a reproduction of the published TeleOCR **96.91** row or as a V7 improvement. Its runtime is reproducible from local saved bindings; row-specific upstream model, input, prediction and scoring bindings remain unavailable. This audit read aggregates and provenance only, with no benchmark example inspection, new OCR, rescoring or tuning.

## Reference identity and units

The [official frozen README](https://github.com/opendatalab/OmniDocBench/blob/f133a71e9e91c3621c7ce8994200a7b394a06eb3/README.md) labels the comparison `v1.6_full` and reports TeleOCR 1.2B; its changelog says the row was added 2026-09-11. Retrieval was 2026-10-07T20:25:25.452533+00:00. The row's actual execution date is unknown. The [model-author README](https://github.com/caipeng328/TeleOCR/blob/1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d/README.md) instead reports96.87; that separate row is not substituted here. Current documentation mentioning v1.7 does not establish that either row used that dataset.

Local CDM/TEDS values below are 0–1; multiply by 100 for the published percentage columns. Edit distances remain 0–1. Overall =100×((1−TextEdit)+FormulaCDM+TableTEDS)/3. Formula Edit, Table Edit, TEDS-S and reading order do not enter Overall.

| Metric | Local D0 | Official row, converted to local units |
|---|---:|---:|
| text_edit | 0.0270983427 | 0.0267 |
| formula_cdm | 0.9860259144 | 0.965895 |
| formula_edit | 0.0490054508 | Not published |
| table_teds | 0.9683186499 | 0.968183 |
| table_teds_structure | 0.9816733093 | 0.981806 |
| table_edit | 0.0358432633 | Not published |
| reading_order_edit | 0.1182972681 | 0.1184 |
| overall | 97.5748740567 | 96.91 |

## Arithmetic decomposition

Local minus displayed official Overall is **+0.664874057** points. The rounded official components imply **96.9126000**, so the component-explained gap is **+0.662274057**, plus **+0.0026000** from published rounding.

| Component | Contribution to Overall difference |
|---|---:|
| Text accuracy | -0.013278089 |
| Formula CDM | +0.671030481 |
| Table TEDS | +0.004521665 |

Formula CDM differs by **+2.0130914 percentage points** and explains almost all of the aggregate difference. This is arithmetic, not a diagnosis of the cause. The difference cannot be credited to V7, whose full comparison has not run.

## Provenance comparison

| Field | Saved local execution | Published reference | Assessment |
|---|---|---|---|
| Dataset | 1651 inputs; exact GT and input hashes frozen | v1.6_full label and1651 pages; row-specific GT/input hashes unavailable | Label/count agree; byte identity unknown |
| Aggregation | Official notebook page-macro fields; eight values retained | Published metric columns and Overall formula; raw report unavailable | Formula/units agree; per-component coverage unverified |
| Model | Fourteen asset hashes; inference weights pinned; requested snapshot unverified | TeleOCR 1.2B name; row-specific snapshot hashes unavailable | Asset equivalence unknown |
| Input | Default read_fn/do_parse;200DPI;native3500 cap; no original-RGB bypass | Exact row-producing input path/cap/rendering unavailable | Input equivalence unknown |
| Decoding | Prompts, helper options, bfloat16, seed0, vLLM0.11.0 and sampling pinned | Row-specific backend, prompts, sampling and concurrency unavailable | Runtime equivalence unknown |
| Matching | Pinned evaluator commit; quick_match;300s truncated/420s page timeout;8workers | Published row does not bind a scorer commit/config | Local implementation known; upstream-row equivalence unknown |
| Formula CDM | Python CDM;TeX Live 2025/ImageMagick 7.1.1-47/Ghostscript 9.55.0;313pages/2352units;zero recorded stage errors | Current documentation recommends same major tools; actual row environment/coverage unavailable | Environment recommendation is not execution provenance |
| Failures | 1651/1651 successful predictions;zeroempty;zero page/TEDS/CDM timeout/error records | Row-specific failure/retry/empty-prediction disposition unavailable | Failure-policy equivalence unknown |

## Formula scoring checks

The local report selects `display_formula.page.CDM.ALL`=0.986025914441, matching the pinned [official report code](https://github.com/opendatalab/OmniDocBench/blob/147cd5ac9472002f5751221d390bf00abdbc0d2f/src/runtime/eval_report.py). The sample-macro CDM is 0.978874574830; replacing the page mean with this value would change the estimand. The official row's raw report is unavailable, so its exact coverage cannot be checked.

All 1651 predictions enter evaluation; component-applicable denominators are text 1557, formula 313, table 458 and reading order 1638 pages. Formula scoring covers 2352 matched units. Saved CDM/TEDS/page-matching stage error and timeout counts are zero. The pinned [metric wrapper](https://github.com/opendatalab/OmniDocBench/blob/147cd5ac9472002f5751221d390bf00abdbc0d2f/src/metrics/cal_metric.py) strips math wrappers and may compare an alternate formula representation. The [CDM implementation](https://github.com/opendatalab/OmniDocBench/blob/147cd5ac9472002f5751221d390bf00abdbc0d2f/src/metrics/cdm/cdm.py) scores rendered glyph correspondence. These mechanisms make matching, normalization and rendering relevant, but aggregate scores cannot identify which mechanism explains this gap.

Six retrieved executable/configuration source files match the saved runtime SHA256 values exactly. The saved README hash differs from the README at the declared evaluator commit; full-tree identity is therefore not asserted. The source checks, version inventory and hashes are in [the machine-readable audit](BASELINE_COMPARABILITY.json).

Local scoring used Python 3.10.16,TeX Live 2025,ImageMagick 7.1.1-47,Ghostscript 9.55.0 and CJK gkai. These align with current upstream environment recommendations but do not prove the published row used that environment. Original formula-pair hashes, rendering traces and a row-specific environment would be needed for causal attribution. No external contact or extra scoring was performed.

## Frozen matched local comparator

The comparator binds the 1651 input identities and hashes, GT `a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`, scorer commit `147cd5ac9472002f5751221d390bf00abdbc0d2f`, config `56db52659910273376223a5646eabfaff21b25dc30f434e1c90aab1a233d335a`, model/processor file hashes, native input path, prompts, decoding and runtime. The local lock SHA256 is `1c1b64d330428b1757d9be9226dec1bb4cba7316da3ef8f8c24693e2e28f4331`. The requested upstream model snapshot was not verified; actual asset hashes are retained without claiming snapshot equivalence.

Before any gated V7 full run, revalidate those exact remote assets and retrieve the exact config bytes; the connection is currently unavailable. Use one shared native parse and one fixed table candidate, preserve every page/failure, and score all four outputs with the same frozen evaluator. V7.3−V7.1 is the primary local effect. The public row remains contextual; no official ranking or submission is claimed.
