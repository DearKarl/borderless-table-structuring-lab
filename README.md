# Borderless Table Structuring Lab

Correcting a fixed MinerU parser through two separate routes: **Training** and **Hybrid**. Neither route has achieved our target: **Official OmniDocBench full Table TEDS strictly greater than 95.00**.

## Current results

These are local executions of the complete official protocol, not verified public leaderboard listings. Scores are full Table TEDS, not Overall or structure-only scores. The complete input contains 1,651 pages.

| Route / version | Full TEDS | Structure TEDS | Interpretation |
|---|---:|---:|---|
| Fixed MinerU Raw | 93.0862668980718 | 95.70961193860337 | Production fallback |
| **Training: Explicit-v2 Original** | **93.0862668980718** | 95.70961193860337 | Best verified trained checkpoint; zero adopted edits, equal to Raw |
| **Hybrid: anchored PP-OCRv5 + Tesseract** | **93.08808257262936** | 95.70961193860337 | Highest completed system aggregate; tiny gain and known crop defect |
| Hybrid: native NaviDC table collection | Pending | Pending | Current local full evaluation; not a released winner |

Hybrid's completed gain is only **0.0018156746 percentage points**. It is research-only, not a reliable production corrector. The ongoing NaviDC attempt is not promoted before its full result. This release consolidates existing work for collaborators; no further optimization is underway. [Aggregate result metadata and evidence hashes](artifacts/observed-results.json) identify the archived results without distributing benchmark data.

## Training route

A learned Explicit model predicts structured repairs on top of MinerU outputs. The best evidenced checkpoint is **Explicit-v2 Original**, `checkpoint-003111-joint_low_lr`. Later evaluated trained variants scored lower.

**Exact winning weights are currently missing from the local release workspace.** [Training documentation](training/README.md) and [model metadata](training/model_manifest.json) preserve the result and identity. No lower-scoring checkpoint or third-party MinerU weights are substituted. This is not yet a downloadable Training model release.

## Hybrid route

No new project model is trained. Fixed MinerU Raw outputs, image-only OCR geometry, PP-OCRv5 recognition, and Tesseract recognition are combined through a frozen, conservative ASCII cell-text policy. Exact table markup and cell count are preserved.

Read the [Hybrid guide](hybrid/README.md) for installation, input formats, token detection, correction commands, runtime versions, and limitations. The portable table adapter preserves the archived policy but is **not itself a newly scored full-benchmark run**. Its known continuation-line crop defect is disclosed, not silently repaired while retaining the old score.

## Download and run

```bash
git clone https://github.com/DearKarl/borderless-table-structuring-lab.git
cd borderless-table-structuring-lab
python3 scripts/download_release.py --route hybrid --output models/hybrid
```

The downloader verifies archive parts and every extracted file and refuses an existing output directory. Mirrored large assets live in the `hybrid-2026.09.13.1` GitHub Release, not Git history. The converted PP-OCR recognizer is fetched from its **pinned official upstream revision**, with the recorded SHA256, instead of being mirrored or relicensed here. See [artifact inventory](artifacts/hybrid-2026.09.13.1.json).

Continue with [Hybrid setup and commands](hybrid/README.md). Locator and correction use separate pinned Python environments; install native Tesseract for your platform. MinerU base weights are included for provenance and optional upstream parsing. The historical Raw score used MLX on Apple Silicon, not an interchangeable generic backend.

Use your own matching table image and Raw HTML. No benchmark images, Gold, customer data, or historical execution contracts are distributed. Data-free integration checks are not TEDS gains.

## Layout

```text
training/       Best trained checkpoint identity, result, restoration verifier
hybrid/         Best completed policy and portable inference tools
artifacts/      File inventory, pinned downloads, checksums
scripts/        Verified model downloader
notices/        Third-party licenses and provenance
tests/          Data-free package/interface checks
```

Earlier research files remain recoverable in Git history at commit `0f28d9512825101e61a0f9480e4815b9f1e75261`. Cleanup affects this compact release, not the original research archive or checkpoints.

## Evaluation integrity

Use unchanged OmniDocBench v1.6 full-page matching and page-macro full Table TEDS. Frozen evaluator commit: `147cd5ac9472002f5751221d390bf00abdbc0d2f`. Official inputs were evaluation-only; Gold stayed inside the evaluator; Customer50 was deferred. Repeated aggregate benchmark exposure exists.

Results belong to exact model, input, policy, runtime, and assembly versions. A different backend, portable adapter, upstream author score, or successful load does not reproduce the reported score by itself. Preserve Raw fallback and report known errors honestly.

## Licenses

Project-authored code has no blanket public redistribution license assigned; contact the maintainer for use beyond authorized project collaboration. Third-party components retain their own terms. MinerU model weights, MinerU software, PaddleOCR, PDF-Extract-Kit converted weights, and Tesseract do not share one uniform license. Read [third-party notices](notices/) before use.
