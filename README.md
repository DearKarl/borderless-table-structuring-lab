# Borderless Table Structuring Lab

An integrated, auditable **MinerU + NaviDC native-table system**, alongside a
separate **Training** route. The command-line application `btsl` coordinates
model acquisition, input validation, layout inference, table recognition,
whole-page assembly, recovery, and isolated evaluation.

**Highest completed project result:** full Table TEDS **98.75068667701048**,
structure TEDS **99.2913812392524**, over 1,651 pages in the **local frozen
Official OmniDocBench protocol**. This is **not** verified public leaderboard
acceptance, a global SOTA claim, or a score automatically inherited by a new
installation. [Original aggregate evidence](artifacts/results/native-tables/PAIRED_TABLE_SUMMARY.json).

## Two distinct research routes

| Route | Deliverable | Observed full Table TEDS |
|---|---|---:|
| **Integrated Hybrid** | Frozen MinerU non-table output + pretrained NaviDC native table collection; unified inference/evaluation application | **98.75068667701048** |
| **Training** | Exact Explicit-v2 Original weights and tensor-level inference | **93.0862668980718**, equal to Raw; zero adopted edits |
| Fixed MinerU Raw | Reference baseline and exact fallback | 93.0862668980718 |

The Hybrid improves full TEDS by **5.664419778938679 points** in the completed
paired run. Both arms used 458 GT-table pages and 665 matched samples, with no
evaluation errors/timeouts. Structure TEDS alone is not the objective.
[Complete results and limitations](artifacts/observed-results.json).

This is **modular software integration**, not weight fusion or a newly trained
joint model. Its integration includes a typed input boundary, pinned model
loader, crop geometry, strict OTSL parser, complete-page replacement policy,
consumer-format checks, atomic output receipts, resumable execution and an
evaluation adapter. It does not cherry-pick cells or consult reference answers.

## System architecture

```text
Source page images + frozen MinerU Raw Markdown
                 │
       source-only manifest + SHA checks
                 │
       NaviDC layout on every page
                 │
       original-resolution table crops
                 │
       native OTSL → strict owner-grid → HTML
                 │
       complete table collection assembly
       + byte-preserved MinerU non-table complement
       + frozen consumer-format validation
                 │
       final Markdown + per-page/run receipts
                 │
       separate frozen Official evaluator ← Gold (evaluation only)
```

A defined page parse/crop/assembly failure restores the **whole Raw page**.
Missing work, OOM or a model/runtime failure stops execution; it is never
silently counted as a completed Raw page. Valid zero-table predictions are
valid outputs. Table structure can change. Caption/table interleaving is not
preserved; no Overall-document-quality improvement is claimed.

## Install and run

macOS Apple Silicon is the exercised inference backend. Linux/CUDA BF16 is an
explicit alternative implementation path, **not yet hardware-validated or
claimed equivalent**. Use Python **3.10** and a fresh environment. Allow about
3 GB for pinned NaviDC assets plus runtime packages and per-call caches.
The observed MPS host has 24 GiB unified memory; the model allocator cap is
12 GiB. Runtime and image-size requirements are not silently reduced.

```bash
git clone https://github.com/DearKarl/borderless-table-structuring-lab.git
cd borderless-table-structuring-lab
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,test]'
btsl download --model-dir models/NaviDC-OCR
btsl verify-model --model-dir models/NaviDC-OCR
```

The downloader pins upstream revision
`710ea2e26d794fe89cbf3ece0402707c332a8671` and checks all 14 files by
SHA256 and size. Weights are downloaded from their upstream source, **not
embedded in Git or the older OCR-sidecar Release**. Custom model code is
hash-verified before import. Keep its license and model card.

To run an invented example through the real model:

```bash
python examples/create_demo.py --output runs/demo-input
btsl prepare --input-dir runs/demo-input
btsl run --manifest runs/demo-input/manifest.json \
  --model-dir models/NaviDC-OCR --device mps --output runs/demo
btsl verify --run runs/demo
```

For your data, place matching filenames under `input/images/` and
`input/raw/`, for example `page-001.png` and `page-001.md`.
Then substitute that directory in `prepare`. The explicit upstream boundary
is **original page images plus fixed MinerU Raw Markdown**. This application
does not silently regenerate MinerU using a different backend. Obtain Raw
with your separately versioned MinerU deployment. The historical fixed Raw
used MinerU2.5-Pro-2604-1.2B on MLX; generic upstream reruns are new baselines.

Final outputs are `runs/demo/pages/*.md`. Resume with the identical command
plus `--resume`; verified completed generations/pages are not repeated.
After inspecting an incomplete call, an explicit `--resume --retry-failed`
creates a new attempt while retaining the failed one. Do not edit code, model
files or input data inside a frozen run.

## Evaluate

[Detailed inference and evaluation guide](hybrid/native_tables/README.md)
contains the complete commands, data boundary, failure semantics and
reproduction limits. The frozen evaluator source and Docker build recipe
are included. Scoring requires legally obtained benchmark annotations and a
complete prediction directory; annotations never enter model inference.

```bash
docker build --platform linux/amd64 -f evaluation/Dockerfile -t btsl-evaluator:0.2.0 .
btsl evaluate --run runs/official --gold /path/to/OmniDocBench.json \
  --baseline /path/to/fixed-raw-markdown --output runs/official-eval
# Review the plan, then repeat with --execute to invoke the evaluator.
```

The default checks the exact historical 1,651-page Official annotation hash.
A different dataset must explicitly use `--dataset custom --expected-pages N`;
it must not be reported as the same Official protocol. Evaluation runs in a
separate network-disabled container, never in the model process. Each scoring
output is non-overwriting. A fresh Docker build is recorded as a new runtime,
not falsely identified as the historical image.

The new portable wrapper is **not separately full-benchmark-scored**.
The historical run included mixed MPS resource epochs; no cross-device
token/score identity is promised. The assembly extraction and runtime wrappers
are distinguishable in [provenance](hybrid/native_tables/provenance.json).
[Release validation](artifacts/integrated-system-validation-2026.09.18.json)
documents engineering checks, not new accuracy evidence.

## Training route and historical references

The best verified project-trained checkpoint remains **Explicit-v2 Original**,
`checkpoint-003111-joint_low_lr`. Download its exact weights separately:

```bash
python training/download_model.py --output models/training/model.safe-state
```

[Training setup](training/README.md) provides architecture, strict loading and
CPU tensor-level inference. It is not a turnkey PDF correction pipeline.
The older PP-OCRv5/Tesseract sidecar is preserved under
[hybrid/README.md](hybrid/README.md); its tiny gain and known crop defect are
not hidden. Its older weight Release is not the new NaviDC system.

## Repository map

```text
btsl/                         Unified application, inputs, model, pipeline, evaluator adapter
hybrid/native_tables/         Scientific parser, assembly, pinned model and runtime recipes
  runtime/                    Extracted original inference functions and memory adapters
  vendor/omnidocbench/         Frozen Gold-free consumer-format functions and notices
evaluation/                   Isolated evaluator recipe and exact scored source snapshot
examples/                     Invented image/Raw example; no benchmark data
training/                     Separate trained model and downloader
tests/                        Data-free software checks
artifacts/                    Model manifests and aggregate-only evidence
notices/                      Third-party provenance and license texts
```

No private Handoff, VPN configuration, credentials, benchmark images/Gold,
customer documents, private per-page results or historical execution contracts
are published. See [contribution rules](CONTRIBUTING.md) and
[third-party notices](notices/THIRD_PARTY.md). A new clone does not include the
private benchmark prediction cache or guarantee a public leaderboard listing.
