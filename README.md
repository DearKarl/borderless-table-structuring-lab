# Borderless Table Structuring Lab

Correcting a fixed MinerU parser through two separate routes: **Training** and **Hybrid**. On 2026-09-17, the native-table Hybrid completed the local frozen Official OmniDocBench protocol with **full Table TEDS 98.75068667701048**, meeting the project target of **strictly greater than 95.00**. This is not a verified public leaderboard listing, a global SOTA claim, or a production guarantee.

## Architecture and research entry points

```text
Fixed MinerU parser output + source page/table image
  |
  +-- Training: encoded cells / OCR candidates -> Explicit-v2 Transformer
  |             -> decision / topology / ownership / text heads
  |             -> historical guarded executor -> Original fallback
  |             Public package: exact weights + tensor-level inference
  |
  +-- Hybrid
        +-- Completed reference: OCR geometry + PP-OCRv5 + Tesseract
        |     -> anchored text consensus -> conservative HTML text patches
        |     Public package: table-image / Raw-HTML correction adapter
        |
        +-- Highest completed project system: pinned NaviDC-OCR
              -> native page layout -> every native table crop -> OTSL / HTML
              -> native table collection + MinerU non-table content
              -> whole-page Original fallback on defined execution failure
```

The **Data Engineering layer** supplies aligned images, Raw tables, cell/token
geometry, candidate text, and typed interfaces. The **model/decision layer**
either learns repair decisions (Training) or uses frozen OCR consensus / native
table collection replacement (Hybrid). Raw fallback and final committed output must remain distinct from
model proposals. Private data preparation pipelines are not distributed here.

New collaborators should choose one entry point:

1. **Study the trained architecture:** read [Training](training/README.md), download
   its exact checkpoint, run the CPU smoke command, then inspect
   `training/runtime/model.py`. Full PDF correction is not packaged for this route.
2. **Study the current mature-model combination:** read
   [native table architecture and handoff](hybrid/native_tables/README.md).
   This is the highest completed project system, separate from the historical sidecar.
3. **Try the completed reference's table adapter:** follow
   [Hybrid sidecar](hybrid/README.md) using your own table image and Raw HTML.
   Keep its known limitations visible.
4. **Understand evidence and contribute:** read [observed results](artifacts/observed-results.json)
   and [contribution rules](CONTRIBUTING.md). Start from an isolated, project-authored
   example; never treat interface tests as measured benchmark improvements.

The repository is a compact **model and inference handoff**, not the historical
experiment archive or a claim of fully reproducible end-to-end training.
Archived weights live in separate GitHub Releases; current NaviDC weights use a
pinned upstream download. Datasets, runs, historical contracts,
private logs, and temporary files are excluded. Third-party notices are required
provenance, not obsolete experiment files.

## Current results — completed 2026-09-17

These are local executions of the complete official protocol, not verified public leaderboard listings. Scores are full Table TEDS, not Overall or structure-only scores. The complete input contains 1,651 pages.

| Route / version | Full TEDS | Structure TEDS | Interpretation |
|---|---:|---:|---|
| Fixed MinerU Raw | 93.0862668980718 | 95.70961193860337 | Production fallback |
| **Training: Explicit-v2 Original** | **93.0862668980718** | 95.70961193860337 | Best verified trained checkpoint; zero adopted edits, equal to Raw |
| Hybrid: anchored PP-OCRv5 + Tesseract | 93.08808257262936 | 95.70961193860337 | Historical sidecar; tiny gain and known crop defect |
| Hybrid: complete-cell correction cycle 001 | 93.08512970738767 | 95.70961193860337 | Below Raw; preserved, not promoted |
| **Hybrid: native NaviDC table collection** | **98.75068667701048** | **99.2913812392524** | Highest completed project system; local scored target met |

The native-table Hybrid completed **1,651 pages and 2,296 successful canonical generation calls**. Against fixed Raw, its full-TEDS gain is **5.664419778938679 percentage points** and its structure-TEDS gain is **3.5817693006490288 points**; its structure-minus-full gap is **0.5406945622419244 points**. The paired evaluation used 458 GT-table pages and 665 matched samples for each system, with zero evaluation errors and zero timeouts for both. The historical OCR sidecar's gain remains only 0.0018156745575623745 points. Best completed **system** and best **trained checkpoint** are distinct: Explicit-v2 Original still equals Raw, with zero adopted edits.

[Aggregate result metadata and evidence hashes](artifacts/observed-results.json) identify the completed local result without distributing benchmark data. The score belongs to the frozen research pipeline, not the portable package, the upstream author's pipeline, or an unverified production deployment.

The [local result archive](artifacts/results/README.md) includes the completed
paired summaries/READY bindings, a failed-launch receipt and the dated native
progress snapshot as history. Below-baseline results are retained, not hidden.

## Training route

A learned Explicit model predicts structured repairs on top of MinerU outputs. The best evidenced checkpoint is **Explicit-v2 Original**, `checkpoint-003111-joint_low_lr`. Later evaluated trained variants scored lower.

**The exact best verified Training checkpoint has been recovered, hash-verified, and published separately.** Download it with `python3 training/download_model.py`. [Training documentation](training/README.md) provides CPU installation, strict loading, and a synthetic inference command. The archived model and safe-state reader are byte-identical to the research source. This is a runnable tensor-level model package, not a turnkey PDF correction pipeline or a newly reproduced benchmark score. No lower-scoring checkpoint or third-party MinerU weights are substituted.

## Hybrid route

### Highest completed system: mature native-table model

The completed system combines **fixed MinerU2.5-Pro-2604-1.2B non-table output**
with **NaviDC-OCR's complete native table collection**. NaviDC revision
`710ea2e26d794fe89cbf3ece0402707c332a8671` is pinned; the upstream project's
later TeleOCR name does not change the checkpoint evaluated here.

Every page gets native layout inference, including Raw pages with no detected
tables. All native table regions are recognized and strictly parsed from OTSL.
On success, all original table spans are removed and all native tables are
appended in layout order. Valid zero-table predictions remain zero. Defined
page execution/parse/assembly failures restore the entire Raw page. Missing
work or memory exhaustion never becomes a fabricated Raw-completed page.
Structure can change; original table/caption interleaving is not preserved.

This is output-level composition, **not weight averaging, a learned selector,
per-cell cherry-picking, or a new trained checkpoint**. Published author scores
do not establish this hybrid's score. Its own completed local full Table TEDS
is **98.75068667701048**. The earlier 633/1,651-page paused handoff remains a
dated historical snapshot, not current progress or a partial accuracy measure.

See [the native-table package](hybrid/native_tables/README.md) for the
architecture, pinned model/runtime details, available code and exact
reproducibility limits. Third-party weights are obtained from the pinned
upstream source, not silently included in the older Hybrid release archive.
The portable package is **not an end-to-end inference/evaluation pipeline**.
Its assembler exposes predecessor pure writeback logic; the completed
private run also checked the Official consumer's exact table extraction. That
consumer-format gate is not packaged, so a portable READY file is not an
Official-evaluation admission receipt.

### Historical completed reference: anchored OCR consensus

No new project model is trained. Fixed MinerU Raw outputs, image-only OCR geometry, PP-OCRv5 recognition, and Tesseract recognition are combined through a frozen, conservative ASCII cell-text policy. Exact table markup and cell count are preserved.

Read the [Hybrid guide](hybrid/README.md) for installation, input formats, token detection, correction commands, runtime versions, and limitations. The portable table adapter preserves the archived policy but is **not itself a newly scored full-benchmark run**. Its known continuation-line crop defect is disclosed, not silently repaired while retaining the old score.

## Download and run

```bash
git clone https://github.com/DearKarl/borderless-table-structuring-lab.git
cd borderless-table-structuring-lab
python3 scripts/download_release.py --route hybrid --output models/hybrid
# Independent project-trained model (4.75 MB):
python3 training/download_model.py --output models/training/model.safe-state
```

The downloader verifies archive parts and every extracted file and refuses an existing output directory. Mirrored large assets live in the `hybrid-2026.09.13.1` GitHub Release, not Git history. The converted PP-OCR recognizer is fetched from its **pinned official upstream revision**, with the recorded SHA256, instead of being mirrored or relicensed here. See [artifact inventory](artifacts/hybrid-2026.09.13.1.json).

Continue with [Hybrid setup and commands](hybrid/README.md). Locator and correction use separate pinned Python environments; install native Tesseract for your platform. MinerU base weights are included for provenance and optional upstream parsing. The historical Raw score used MLX on Apple Silicon, not an interchangeable generic backend.

Use your own matching table image and Raw HTML. No benchmark images, Gold, customer data, or historical execution contracts are distributed. Data-free integration checks are not TEDS gains.

## Layout

```text
training/       Trained model downloader, exact architecture, CPU inference and result
hybrid/         Historical OCR sidecar and portable inference tools
  native_tables/ Highest completed system's architecture and limited portable handoff
artifacts/      File inventory, pinned downloads, checksums
scripts/        Verified model downloader
notices/        Third-party licenses and provenance
tests/          Data-free package/interface checks
```

Earlier research files remain recoverable in Git history at commit `0f28d9512825101e61a0f9480e4815b9f1e75261`. Cleanup affects this compact release, not the original research archive or checkpoints.

## Evaluation integrity

Use unchanged OmniDocBench v1.6 full-page matching and page-macro full Table TEDS. Frozen evaluator commit: `147cd5ac9472002f5751221d390bf00abdbc0d2f`. Official inputs were evaluation-only; Gold stayed inside the evaluator; Customer50 was deferred. Repeated aggregate benchmark exposure exists.

Results belong to exact model, input, policy, runtime, and assembly versions. A different backend, portable adapter, upstream author score, or successful load does not reproduce the reported score by itself. Preserve Raw fallback and report known errors honestly.

The completed native run preserves mixed local MPS execution epochs: a committed
temporal prefix, full-K/V query-chunk resource adaptations, and a final Q1 path
with synchronization and unused allocator-cache release. Model weights, image
preparation and generation settings were unchanged by that final resource step.
This is not a claim of bit-identical tokens, a quality benefit from cache release,
CUDA equivalence, or a uniform original runtime. Repeated aggregate exposure
remains disclosed; the local result does not establish global SOTA or production reliability.

## Licenses

Project-authored code has no blanket public redistribution license assigned; contact the maintainer for use beyond authorized project collaboration. Third-party components retain their own terms. MinerU model weights, MinerU software, PaddleOCR, PDF-Extract-Kit converted weights, and Tesseract do not share one uniform license. Read [third-party notices](notices/) before use.
