# Integrated native-table system

The repository's main application is `btsl`, not a pair of manual model calls.
Its input is **source page images + frozen MinerU Raw Markdown**; its output is
final page Markdown with per-generation, per-page and whole-run hash receipts.
Training weights remain a separate research route.

## Observed result versus released software

The completed research system scored **98.75068667701048 full Table TEDS**
and **99.2913812392524 structure TEDS**, versus fixed Raw
**93.0862668980718 / 95.70961193860337**, on the 1,651-page local frozen
Official OmniDocBench protocol. Both arms have 458 GT-table pages and
665 matched samples, zero errors and zero timeouts.
[Byte-bound paired summary](../../artifacts/results/native-tables/PAIRED_TABLE_SUMMARY.json).

This is not public leaderboard acceptance. A new portable run is unscored
until evaluated. The old result used a preserved mixed-MPS temporal prefix;
changing hardware or execution epochs can change tokens. There has been
repeated aggregate benchmark exposure in this project; this is not a newly
unseen test set or a global SOTA claim.

## Component contracts

| Component | Contract |
|---|---|
| `btsl.inputs` | SHA-bound source images and matching Raw Markdown; no label/answer fields |
| `btsl.model` | Pinned 14-file upstream model, strict custom-class loading, BF16 greedy generation |
| `btsl.pipeline` | One fresh child per uncached call; ordered page execution, locks and resumable receipts |
| `runtime/native_ops.py` | Original layout/table prompts and crop/preparation/generation functions |
| `formats.py` | Byte-identical strict native layout and OTSL owner-grid parser |
| `assembly.py` + `consumer.py` | Evaluated assembler003 functions and frozen consumer-format closure |
| `btsl.evaluate` | Complete-page input freeze and isolated frozen Official evaluator |
| `model_manifest.json` + `runtime/EXTRACTION.json` | Exact weight/source hashes and extraction provenance |

### Fixed scientific policy

1. Every page receives native layout inference, even if MinerU found no table.
   Layout uses RGB resized to **1036×1036 with BICUBIC**.
2. All native `table` blocks are cropped from the **original-resolution image**.
   Coordinates are floored from the native 0–1000 scale; native orthogonal
   rotation is applied. Aspect ratio above 50 is white-padded; edges below
   28 pixels are proportionally enlarged using ceil/BICUBIC.
3. The pinned custom NaviOCR model generates OTSL with native processor limits
   3,136–12,845,056 pixels, BF16 SDPA, batch 1, greedy decoding, cache enabled,
   **4,096 maximum new tokens** and repetition penalty **1.05**. Raw Markdown
   and Gold are not supplied to that model.
4. Strict parsing reconstructs rectangular cell ownership and HTML. All
   detected table regions are attempted; one invalid completed region rejects
   the whole page, not just the unfavorable table.
5. Success removes every uniquely mapped Raw HTML table span and appends the
   **complete native collection** in layout order, separated by two newlines
   and terminated by one newline. The non-table byte complement is preserved.
   A successful zero-table layout removes Raw tables.
6. The frozen Official consumer's **format extraction only** must observe
   exactly the native collection. Extra Markdown/LaTeX table projections in
   the preserved complement cause whole-page Raw fallback. No reference answer,
   match result, metric, selection threshold or learned selector is consulted.

Invalid completed layouts/OTSL, invalid crop geometry, image decoding or
unprovable assembly trigger exact whole-page Raw fallback. OOM, missing calls,
loading errors, changed files and unknown runtime failures **stop without
fabricating a completed prediction**. Table structure may change; original
caption/table interleaving is not preserved. This is not an Overall-document
quality claim.

## Installation and input preparation

Run from the repository root with Python 3.10:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,test]'
btsl download --model-dir models/NaviDC-OCR
btsl verify-model --model-dir models/NaviDC-OCR
```

The downloader uses upstream revision
`710ea2e26d794fe89cbf3ece0402707c332a8671` (approximately 2.85 GB).
It does not substitute the subsequently renamed project's latest checkpoint
or generic Qwen weights. Read the pinned custom code and license before use.
A corrupted existing file is not overwritten. Completed downloaded files are
reused; inspect an interrupted `.partial` file before removing it and retrying.

Prepare a directory with `images/page-001.png` and `raw/page-001.md`, etc.
Use exact matching page stems. All files must remain within that directory.
Existing manifests cannot be overwritten.

```bash
btsl prepare --input-dir /path/to/input
btsl run --manifest /path/to/input/manifest.json \
  --model-dir models/NaviDC-OCR --device mps --output runs/my-run
btsl verify --run runs/my-run
```

Do not mix input versions. Raw must come from a separately pinned MinerU
deployment; this package deliberately does **not** regenerate an allegedly
identical baseline with a new backend. The historical Raw used
MinerU2.5-Pro-2604-1.2B/MLX. Benchmark payloads and its private frozen predictions
are not distributed. The old Hybrid asset Release contains MinerU weights
for provenance but does not recreate that complete baseline runtime.

For an invented one-page example, run
`python examples/create_demo.py --output runs/demo-input`, then prepare/run
that directory. This is a real model smoke test, not a benchmark score.

### Runtime and resumption

The exercised native runtime is macOS Apple Silicon, Python 3.10,
PyTorch 2.12.0, Transformers 4.57.6, Pillow 12.2.0.
The new portable environment pins SciPy 1.14.1: this host's 1.15.3 wheel failed
to load its Mach-O extension before model construction. This dependency
compatibility repair does not overwrite the historical research environment.
The original 256-query full-K/V vision/decoder adapters and Q1 unused-allocator
cache reclamation are included byte-for-byte. They check the pinned native
and Transformers sources, rather than patching unrelated models globally.
Implicit CPU fallback is rejected. Use sufficient free memory; no concurrent
model calls are launched.

`--device cuda` uses native CUDA BF16 SDPA without the MPS-only adapters.
It is an explicit **unvalidated backend**, not evidence of token/score identity.
Choose `CUDA_VISIBLE_DEVICES` yourself; the application does not seize GPUs or
stop other processes. Windows is not supported (the process lease uses POSIX).

Add `--resume` to the **identical** run command. Every committed page and
generation is verified and reused. If a child failed or was interrupted,
inspect `calls/*/attempt-*/console.log` and `FAILURE.json`, then explicitly
add `--retry-failed`. A new attempt preserves the old failure. Another worker
cannot open the same run while its parent or child owns the process lease.
Changing code, source inputs, model or runtime requires a new output directory.

Outputs:

```text
FREEZE.json                   Input, code, model and runtime bindings
calls/<key>/attempt-*/        Prepared image, raw token IDs, native text, loading audit, logs
page_receipts/<page>/         Native records, assembly receipt and page READY
pages/<page>.md               Final prediction consumed by evaluator
READY.json                   Published only after all pages and hashes close
```

A READY is software completion, not evidence of accuracy.

Current packaging checks include 77 model-free tests, byte-identical reassembly
of all 1,651 cached historical outputs without reading Gold, and a real MPS
layout/table inference on one invented page with zero additional calls on
resume. The GitHub workflow separately exercises the isolated evaluator with
an invented one-cell fixture; inspect its status for the release commit.

## Isolated Official evaluation

The exact 54-file scored evaluator source snapshot is included under
`evaluation/omnidocbench`; commit
`147cd5ac9472002f5751221d390bf00abdbc0d2f`.
Source SHA checks run before/after evaluation. The Docker runtime separates
inference dependencies from evaluator dependencies and has networking disabled
during scoring.

```bash
docker build --platform linux/amd64 -f evaluation/Dockerfile \
  -t btsl-evaluator:0.2.0 .
btsl evaluate --run runs/my-official-run \
  --gold /path/to/OmniDocBench.json \
  --baseline /path/to/fixed-raw-markdown \
  --output runs/my-official-evaluation
# The command above only prints a plan; review it, then add --execute.
```

Obtain benchmark images and annotations from the
[official OmniDocBench project](https://github.com/opendatalab/OmniDocBench)
under its terms. The default requires **1,651 complete pages** and annotation
SHA256
`a45cd84b04ad8b793e775089640e6b681209abea33ead54c1828ddca35fae496`.
The host only hashes the annotation file; the scoring container parses it.
A different version requires `--dataset custom --expected-pages N` and must
not be labeled the same frozen protocol.

The adapter freezes full-page `quick_match`, TEDS/Edit_dist, four matching
workers and four TEDS workers, 300/420-second timeouts, fallback span 10 and
order penalty 0.10. Full score is `table.page.TEDS.ALL × 100`;
structure is `table.page.TEDS_structure_only.ALL × 100`.
Input page completeness and processed count must match. Baseline is optional
but recommended for a paired comparison. Scoring output directories are
non-overwriting. Failures preserve logs and do not publish a success receipt.

The historical Docker image ID was
`sha256:6116ad72172e763b5c43e963d5efebf2093f2362b975f58156ce4f6c9142e617`.
A new build is identified by its actual image ID, not claimed as identical.
The Docker build has a table-only purpose; formula-CDM TeX dependencies are
not a supported evaluation mode of this wrapper.
Installing only the wheel requires `--evaluator-source` pointing to this
repository's frozen evaluator directory. A full Git clone/editable install
already locates it.

## Evidence and limitations

The release includes original aggregate result receipts but excludes all
private predictions, caches, Gold, images and VPN/host configuration.
Engineering tests and cached-output equivalence are not new TEDS results.
The [validation record](../../artifacts/integrated-system-validation-2026.09.18.json)
states exactly what was exercised, including unsupported/unverified paths.
No upstream-author score, fixture or successful software check is relabeled as
a new Official result.
