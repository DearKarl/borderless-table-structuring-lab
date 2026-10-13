# V5: five TeleOCR system comparisons

These independent arms compare the pinned TeleOCR default image-to-PDF path,
direct original RGB input, table rereading, formula rereading, and a bounded
combination. Each arm processes every page afresh. Ground truth is mounted only
into separate CPU evaluation and post-evaluation analysis containers. No accuracy
gate selects which arms run.

| Arm | Ordered processing |
| --- | --- |
| V5.1 | Native `read_fn` / `do_parse` image-to-PDF route |
| V5.2 | Original RGB -> native two-step extraction -> MagicModel / union_make |
| V5.3 | V5.2 -> table crops at 1x |
| V5.4 | V5.2 -> display-formula crops at 1.25x |
| V5.5 | V5.2 -> formula125 -> anomaly guard -> table1x |

`protocol.json` defines the shared runtime, crop rules, decoding provenance,
limits and fallback policy. `regions.py` adapts the mechanisms reviewed in
[ki-OCR-v1](https://github.com/yushuosun/ki-OCR-v1), commit
`ac1ab4f5ded8b025f47a7683f58e65db4ac43a2b`, specifically the archived
`run_tele_img.py`, `stack_reread_v1.py`, `stack_guard_v1.py`, and
`dapp_assemble.py`. This comparison does not reproduce its full R3 system.

TeleOCR source is pinned to `9921cffe380efe4e2fa010258b3d0c3cb70bab2d`.
The weight SHA-256 is
`9817b18041bd96403f75e38a33b28ed0cd5fb2641f67eead673afabd3c408109`.
Runtime manifests additionally bind all tokenizer/processor files, source,
package versions, input hashes, driver, container image, prompts and decoding.
Model repository revision identity must be reported separately from file-hash
identity; matching the weights alone does not establish snapshot equivalence.
For this round, the requested model revision
`8706730e41382f4a8f2562616d47225fee8e2f7a` could not be queried. The reused
snapshot was checked file by file against the existing V4 pins; all 13 inference
file hashes also match the reviewed ki-OCR model pins. Their reference metadata
points to `e92585356c0d0b7b7a65938f3da035c6593cc9a6`. This establishes the
recorded file identities, not equivalence to the requested upstream revision.

The native PDF route renders at 200 dpi with a 3500-pixel long-edge ceiling.
The original-image route opens the original file as RGB. Regional rereads have
no added rotation or padding. Their crop conversion and validation preserve the
reference rules. Guard retries use nominal factors 1, 0.75 and 1.5, multiplied
by 200/72 for equations. The first valid output is accepted, with three attempts
at most. Empty text blocks are excluded from guard selection.

Intentional engineering differences from the reference batch scripts:

- One page per transaction, with a persistent engine and independent page shards.
- Audit history is separate from the block tree, preventing historical spans
  from entering later region selection.
- lxml validation is required. Invalid rereads preserve the original region.
- Each stage is committed atomically. A regional failure retains the last
  completed stage. A page without a valid native stage remains an empty file.
- Work stops at 590 seconds; an independent container guardian enforces the
  600-second page ceiling. Completed and failed pages are never rerun.
- CUDA graphs are enabled (`enforce_eager=false`). Attention kernels follow
  native capability selection rather than forcing an incompatible vision kernel.

`supervisor.py` launches the bounded GPU round from a frozen private manifest.
`guardian.py` enforces deadlines even if the host controller is lost.
`evaluate.py` creates all 1651 benchmark-named Markdown files for each arm and
runs the same pinned official evaluator, retaining empty failed predictions.
Metrics are text edit distance, formula CDM, TEDS, structure TEDS, reading-order
edit distance, and Overall. Overall is the mean of text accuracy, CDM and TEDS
on the 0–100 scale. New scores are not automatically comparable to historical
scores with different runtime or evaluator configurations.

`analyze.py` reads completed evaluator outputs and records descriptive paired
page changes on fixed ground-truth eligible page sets. It reports missing numeric
coverage rather than substituting each arm's successful subset. It also audits
native-stage byte agreement across the original-image arms: independent greedy
runs need not be byte identical, so end-to-end differences alone do not identify
the causal effect of a regional intervention. Invalid-to-valid region counts
measure validation rules, not accuracy gains. V5.5 versus V5.4 changes both the
guard and table stages and cannot isolate the guard's effect.

Run synthetic contract tests with:

```console
python -m unittest versions.v5.test_regions -v
python -m unittest versions.v5.test_analysis -v
```

Private manifests, inputs, predictions, intermediate blocks and execution logs
are excluded from Git. No model weights or benchmark payloads are bundled here.

<!-- V5_RESULTS_START -->
## Completed local results: 2026-10-07

All five independent arms completed 1,651 pages each: 8,255 predictions, zero
inference failures, zero empty predictions and zero infrastructure restarts in
the completed round. All five CPU scorers exited successfully. The pinned
evaluator recorded zero quick-match timeouts and zero page-matching timeouts.
All pages remained in the full evaluation; component metrics retain the
official component-specific aggregation rules.

Full precision values and protocol hashes are in
[AGGREGATE.json](../../artifacts/results/v5/AGGREGATE.json). Overall is on
0-100; every other metric below is on 0-1. Higher is better except edit distances.

| Arm | Overall | Text ED | Formula CDM | Table TEDS | TEDS-S | Reading-order ED |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V5.1 | 97.568231 | 0.027272 | 0.985886 | 0.968432 | 0.981673 | 0.118312 |
| V5.2 | 98.173146 | 0.025230 | 0.983813 | 0.986611 | 0.991897 | 0.118443 |
| V5.3 | 98.222117 | 0.025302 | 0.983782 | 0.988184 | 0.993662 | 0.118505 |
| V5.4 | 98.250602 | 0.025612 | 0.986514 | 0.986616 | 0.991897 | 0.118569 |
| V5.5 | 98.347514 | 0.025182 | 0.986513 | 0.989095 | 0.993738 | 0.118669 |

V5.5 has the highest observed Overall, 98.347514,
+0.174368 points relative to the original-image V5.2 baseline.
V5.4 has the highest CDM by a very small margin; V5.1 has the lowest
reading-order edit distance. These are local system measurements, with
no significance claim or established official leaderboard rank.

The fixed paired component populations contain 313 GT formula pages and
458 GT table pages. Improved/regressed counts below use the metric direction;
all remaining eligible pages are tied.

| Candidate minus baseline | Overall delta | Formula improved / regressed | Table improved / regressed |
| --- | ---: | ---: | ---: |
| V5.2 minus V5.1 | +0.604916 | 32 / 68 | 170 / 24 |
| V5.3 minus V5.2 | +0.048971 | 0 / 3 | 12 / 5 |
| V5.4 minus V5.2 | +0.077456 | 50 / 36 | 4 / 1 |
| V5.5 minus V5.2 | +0.174368 | 49 / 34 | 11 / 6 |
| V5.5 minus V5.4 | +0.096912 | 2 / 1 | 9 / 7 |

The all-page paired text and reading-order summaries have incomplete numeric
coverage: the scorer exports no per-page numeric value for 94 and 13 pages,
respectively. Their official aggregate metrics are reported above; no
successful-only subset replaces the declared 1,651-page paired population.

Every first layout request matches across the four original-image arms
(image pixels, dimensions, prompt hash and sampling record). Native Markdown
matches V5.2 byte-for-byte on 1,338/1,651 V5.3 pages, 1,354/1,651 V5.4 pages
and 1,371/1,651 V5.5 pages. The cause of the remaining differences is unproven.
These end-to-end comparisons do not isolate the causal effect of regional
patches. V5.5 versus V5.4 also changes both guard and table stages.

| Arm | GPUs | Inference wall (min) | Allocated GPU-hours | Median page (s) | Max page (s) | Sampled peak GPU memory (MiB) | CPU evaluation wall (min) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V5.1 | 1 | 126.66 | 2.111037 | 3.835 | 25.743 | 42285 | 31.76 |
| V5.2 | 1 | 103.63 | 1.727110 | 2.889 | 22.741 | 45393 | 30.52 |
| V5.3 | 2 | 63.17 | 2.045231 | 3.178 | 44.129 | 46775 | 29.74 |
| V5.4 | 1 | 106.03 | 1.767134 | 2.985 | 22.412 | 45393 | 30.33 |
| V5.5 | 2 | 65.57 | 2.108093 | 3.407 | 43.945 | 47011 | 30.52 |

Full inference used 9.758604 allocated GPU-hours.
Including 0.472764 GPU-hours for preparation and stopped startup attempts,
the recorded total is 10.231368 GPU-hours.
Allocation time includes engine startup and the exit observation interval.
Wall times for the two-GPU arms must be interpreted with their GPU counts.
All owned inference, evaluator and post-analysis containers exited; their
GPU allocations were released. No page exceeded the frozen time limit.

| Arm / stage | Pages with attempts | Region attempts | Accepted | Changed content | Rejected |
| --- | ---: | ---: | ---: | ---: | ---: |
| V5.3 / table1x | 463 | 671 | 670 | 17 | 1 |
| V5.4 / formula125 | 324 | 2376 | 2376 | 793 | 0 |
| V5.5 / formula125 | 324 | 2376 | 2376 | 793 | 0 |
| V5.5 / guard | 7 | 14 | 6 | 6 | 8 |
| V5.5 / table1x | 463 | 671 | 670 | 14 | 1 |

Stage coverage uses detected regions, so it differs from GT component page
counts. The guard found seven eligible regions on seven pages, made 14
bounded attempts and accepted six syntax-valid replacements. One region
remained unrepaired. A rejected reread retains the earlier region; it is
not a failed page. Syntax validity and changed content do not establish
accuracy improvements.

Verification included nine runtime/region/atomic contract tests, four
analysis tests, a fixed 16-page functional smoke per arm, real vLLM request
cancellation and renderer controls. All 8,255 final predictions were
matched to their scored copies by hash. The private evidence archive passed
verification of all 41,446 members. Benchmark payloads remain private.

Evaluator commit: `147cd5ac9472002f5751221d390bf00abdbc0d2f`.
Configuration SHA-256: `56db52659910273376223a5646eabfaff21b25dc30f434e1c90aab1a233d335a`.
The model-file identity and unresolved upstream-revision limit described
above remain applicable. Older local scores and the public TeleOCR reference
are separate evidence, not paired controls for this round.
<!-- V5_RESULTS_END -->
