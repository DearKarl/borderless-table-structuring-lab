# Native-table hybrid — unscored research architecture

This is the current **non-training** experiment, separate from the previously
scored OCR-consensus system in `../README.md`. It combines fixed MinerU output
with an entire table collection predicted by pretrained NaviDC-OCR. It does not
average model weights or ask a newly trained editor to select text actions.

**Snapshot:** 633 / 1,651 local evaluation pages were committed when the user
paused the job on 2026-09-15. The full Table TEDS result is **PENDING**, not zero,
not above 95, and not a public leaderboard acceptance. These counts describe the
research run, not a benchmark included in this repository. Restarting it is a
separate operation; downloading this package does not resume anything.

## Architecture and intended benefit

```text
User-owned document/page image
  ├─ Fixed MinerU2.5-Pro-2604-1.2B → Raw Markdown + exact table spans
  │                                    └─ preserve non-table byte chunks
  └─ Pinned NaviDC-OCR → 1036×1036 native layout
                         └─ every native table label, in native order
                              └─ original-image crop + native preparation
                                   └─ greedy native OTSL generation
                                        └─ strict owner-grid → HTML
              ↓
Complete valid page: all native tables + original non-table complement
Invalid completed page: whole-page exact Raw, never a favorable subset
Missing work / OOM / global error: stop or pause, never fill unattempted pages
```

The earlier conservative OCR system barely changed the baseline. This challenger
allows both text and topology to change: it can replace wrong rows, spans or
missing tables rather than only swapping a few cell strings. This wider action
space may also lose a column or associate text with the wrong row. There is no
learned quality selector and no count/text-disagreement threshold that chooses
favorable predictions after looking at a benchmark.

The fixed baseline is full Table TEDS **93.0862668980718**. The published
NaviDC-OCR **97.05** motivates this experiment, but belongs to the authors'
system and protocol. It is **not** this hybrid's measured score. Independent
public-image inspection found actual bracket repairs and heading-span changes,
but also a lost column and row/content errors. That evidence supports testing,
not a guaranteed gain. Only a completed frozen Official-protocol evaluation
can establish this candidate's result; a public listing needs a separate real
submission/acceptance receipt.

## What this directory actually provides

| File | Purpose |
|---|---|
| `architecture.json` | Precise model, generation, layout, assembly and observed runtime recipe |
| `model_manifest.json` | Pinned upstream revision and SHA256/size for all 14 model files |
| `verify_model.py` | Streaming local model-file verification; no model loading |
| `formats.py` | Byte-identical project-authored strict layout/OTSL parser |
| `assembly.py` | Unchanged pure assembly functions from predecessor assembler001; not the complete current assembler003 |
| `assemble.py` | Portable CLI for one user-owned page and its completed native outputs |
| `example.py` | Synthetic, model-free example; no benchmark or private document |
| `provenance.json` | Source hashes and clear extraction/integration boundaries |

**This release is not an end-to-end inference runner.** The long-running local
controller, its private page/cache inventory and the Official evaluator are not
shipped. The portable CLI validates/assembles outputs; it does not recreate the
633-page cache, load NaviDC, run MinerU or claim token-equivalent reproduction.
The model's upstream architecture is obtained at the pinned revision below.
The observed MPS query-chunk adapters and process scheduler are documented,
not reimplemented here or silently substituted with a different runtime.

The current research assembler003 also checks the **Official consumer's table
projection** against the native collection. That additional consumer-format
closure is **not included** in this portable predecessor001 subset. For example,
a leftover Raw Markdown pipe table or LaTeX table in the preserved complement
may be interpreted as an additional table by the Official consumer even though
the HTML-only portable check accepts it. Portable receipts therefore declare
`official_consumer_format_verified: false`. A portable `READY.json` is **not**
current Official evaluation admission or equivalence to assembler003. Apply the
unchanged current consumer closure in a separately bound evaluator integration
before treating these outputs as the exact research candidate.

## Retrieve the exact upstream model

Run these commands from the repository root, using a separate environment and
a model directory outside Git. Review upstream custom code before importing it.
The pinned card declares Apache-2.0. Upstream code and weights are **not** copied
into this repository; review their notices and applicable terms yourself.

```bash
python -m pip install 'huggingface_hub==0.36.2'
hf download StarDoc-AI/NaviDC-OCR \
  --revision 710ea2e26d794fe89cbf3ece0402707c332a8671 \
  --include README.md added_tokens.json chat_template.jinja config.json \
    generation_config.json merges.txt model.safetensors modeling_naviocr.py \
    preprocessor_config.json special_tokens_map.json tokenizer.json \
    tokenizer_config.json video_preprocessor_config.json vocab.json \
  --local-dir /your/model-directory/NaviDC-OCR
python -m hybrid.native_tables.verify_model \
  --model-dir /your/model-directory/NaviDC-OCR
```

Download size is approximately 2.85 GB. The weight SHA256 is
`9817b18041bd96403f75e38a33b28ed0cd5fb2641f67eead673afabd3c408109`;
the custom source SHA256 is
`7adac1cc17009f9f1b8e0c58284d7c0b9989a590f017b9e5964cc42270207922`.
The 14-file verifier checks all metadata, tokenizer files and weights, not just
the main tensor file. Download and full model inference were not rerun as part
of this packaging change.

The upstream project was subsequently renamed TeleOCR. Do not switch to its
latest branch or a generic Qwen checkpoint when reproducing this frozen recipe.
Use the revision above and the corresponding custom NaviOCR class. A compatible
integration must require empty missing/unexpected/mismatched/error key lists;
ignored loading errors or random initialization are not this experiment.

## Inference integration contract

The adapter supplying JSON to this package must implement the following recipe:

1. Decode the original image to RGB. Run native layout on a **1036×1036 BICUBIC**
   version. Preserve native block order and all exact `table` labels, including
   pages where MinerU did not detect a table. Unsupported polygon layout is a
   page-format failure in this first axis-aligned implementation.
2. Convert 0–1000 layout coordinates by integer floor against the **original**
   dimensions, then apply the native orthogonal rotation. Do not crop the resized
   layout image. If aspect ratio exceeds 50, center-pad the short axis white;
   if the short edge is below 28, proportionally enlarge using ceil/BICUBIC.
3. Use the pinned processor (native pixel limits 3,136–12,845,056) and exact
   system/task strings in `architecture.json`, including leading newlines.
   Use BF16, SDPA, batch 1, greedy decoding, cache enabled, maximum **4,096 new
   tokens**, and pinned generation-config repetition penalty **1.05**.
4. Preserve every raw generation and its actual terminal-EOS/truncation metadata.
   For each detected table, provide exactly one generation in layout order.
   No extra caption generation, arbitrary HTML pass-through or cell selection.
5. A resource shortage pauses unfinished work. Unknown model/API/input-binding
   failures stop it. Do not disguise missing work as valid Raw abstention.

The observed local runtime is Python 3.10.20 / PyTorch 2.12.0 /
Transformers 4.57.6 on an M4 Pro with 24 GiB unified memory. It uses a **12 GiB
MPS allocator cap**, one fresh child per uncached generation and no implicit
CPU fallback. Vision full-attention blocks 7/15/23/31 use 256-query chunks with
all original keys/values. A later decoder resource recovery also uses 256-query
chunks, full K/V and explicit absolute causal offsets. This is a memory change,
not reduced-resolution/window attention or a promise of bit-identical tokens.
The preserved temporal prefix spans more than one local OS/resource recipe;
final evaluation must disclose that provenance. A different machine/runtime
needs its own recorded execution binding, not a false identity claim.

## Model-free quickstart

Python 3.10 or newer is enough. This path needs no Torch, network, model weights
or dataset and is covered by data-free tests.

```bash
python -m hybrid.native_tables.example --output-dir /tmp/native-table-example
python -m hybrid.native_tables.assemble \
  --raw /tmp/native-table-example/raw.md \
  --image /tmp/native-table-example/page.ppm \
  --native /tmp/native-table-example/native.json \
  --output-dir /tmp/native-table-assembled
python -m pytest -q tests/test_native_tables_release.py
```

Use new, nonexistent output directories. `page.md`, `receipt.json`, and a final
`READY.json` are written only after input validation; an interrupted directory
without READY is not a completed output. The example uses invented table text,
not an OCR observation or a performance result.

For your own page, `native.json` has this interface (hash values abbreviated
here only; the example writes real SHA256 values):

```json
{
  "schema": "native-table-page-input/1",
  "status": "GENERATIONS_COMPLETE",
  "raw_markdown_sha256": "64 lowercase hexadecimal characters",
  "source_image_sha256": "64 lowercase hexadecimal characters",
  "layout": {
    "decoded_text": "<box:0 0 1000 1000><label:table><up>",
    "terminal_eos_present": true,
    "truncated": false
  },
  "tables": [{
    "layout_index": 0,
    "decoded_text": "<fcel>Item<fcel>Total<nl><fcel>A<fcel>10<nl>",
    "terminal_eos_present": true,
    "truncated": false
  }]
}
```

The flags must be taken from actual completed token generation, not invented by
the caller. This CLI verifies syntax and bindings, not whether the model really
saw the image or whether its answer is correct. The producer must retain its
own source/generation evidence. A valid layout with non-table blocks and no
`table` labels uses `tables: []`; successful zero-table detection removes all
Raw tables. A missing generation is an error, not a zero-table prediction.

## Exact writeback and failure semantics

`fcel` opens a nonempty cell; `ecel` opens an empty cell. `lcel`, `ucel`, and
`xcel` extend the same owner right, down, or in both dimensions. `nl` ends a row.
The parser requires a rectangular owner grid, legal nonoverlapping spans and
complete rows. It never invents padding, drops malformed rows or accepts model
HTML as trusted output. Cell text is escaped before HTML serialization.

Original table literals must map to unique, exact, nonoverlapping spans. The
assembler removes the whole original collection, keeps the byte-exact ordered
non-table complement, and appends all new tables in native layout order. It
**does not preserve original caption/table interleaving or global reading order**.
Its scope is Table TEDS, not an Overall document-score claim. An ambiguous Raw
mapping or any invalid completed native table returns the entire exact Raw page.
One good table from a failed multi-table page is never cherry-picked.

## Evidence, validity and references

No benchmark images, annotations, page IDs, per-page scores, private paths,
training histories or governance contracts are included. The local research run
uses Official images/Raw outputs **evaluation-only after freeze**; Gold enters
only the evaluator, not inference or selection. Customer50 remains deferred.
Aggregate benchmark exposure has occurred in earlier experiments. Do not use
Official per-record answers to tune this candidate or label an adaptive result
as an untouched holdout. Always retain below-target results and exact artifacts.

- [NaviDC-OCR technical report, pinned v1](https://arxiv.org/html/2608.12898v1):
  pretrained content/structure-decoupled document parsing and author results.
- [Pinned model card](https://huggingface.co/StarDoc-AI/NaviDC-OCR/blob/710ea2e26d794fe89cbf3ece0402707c332a8671/README.md):
  model identity, native interface, OTSL markers and declared license.
- [MinerU2.5 technical report](https://arxiv.org/html/2509.22186v1):
  decoupled global layout and high-resolution local recognition. MinerU already
  uses a structured table representation; OTSL is not our new invention.
- [Actual baseline model](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B):
  fixed Pro revision family, not generic MinerU or the paper's score.
- [Official OmniDocBench evaluator](https://github.com/opendatalab/OmniDocBench):
  a complete pinned protocol, not a substitute syntax, fixture or action score.

The code tests in this directory protect serialization and scope. They do not
establish model quality, reproduce the author's 97.05, or show that the strict
project target **full Table TEDS > 95.00** has been achieved.
