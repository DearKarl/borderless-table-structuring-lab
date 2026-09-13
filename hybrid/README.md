# Hybrid route: frozen OCR consensus sidecar

This is the **best completed, scored hybrid system**, not the trained-model
route. Its historical full-page Official OmniDocBench protocol result was
**93.08808257262936 full Table TEDS**, versus fixed MinerU raw
**93.0862668980718**. The difference is only **+0.00181567455756 points**.
It has **not reached 95**, is not a verified public leaderboard listing, and is
not evidence of general improvement over MinerU. See the aggregate evidence in
the repository's result records. Keep exact MinerU raw as the production fallback.

The newer **MinerU + NaviDC native-table collection** experiment is a different
challenger. Its current local run is not completed/scored at publication time.
Do not attribute this sidecar's score to NaviDC or call the challenger the winner.

## What is runnable here

`correct_table.py` accepts your own matching raw HTML table and table-crop image.
`locate_tokens.py` can first produce image-only OCR tokens. Together they expose
the historical recognition and writeback policy without project paths, a
benchmark dataset, an optimizer, or reference answers. Neither portable adapter
has received a complete benchmark evaluation; **the adapters themselves are
UNSCORED**. A different operating system or native OCR binary can change outputs.
The historical score belongs to the complete frozen full-page pipeline,
including its original MinerU output, PaddleOCR locator, recognizers, assembly,
runtime, and official matcher—not to arbitrary user tables or this new wrapper.

### Exact policy retained

1. Parse one explicit HTML table. Reject merged/nested cells, irregular rows,
   blank cells in a proposed row, formula-like content, and multiline text.
2. Group detected text tokens into geometrically unambiguous rows. Require the
   same token/DOM-cell count, at least three columns, exactly one differing cell,
   and at least two exact anchors unique across the table.
3. Crop the **detected text-token box** with floor/ceil integer coordinates and
   add the historical 3-pixel white border. Run PP-OCRv5 recognition on CPU and
   native Tesseract English with `--psm 7` on that same crop.
4. Commit only when both fresh strings equal the locator string after outer
   trimming. The locator is not a third independent vote. Preserve surrounding
   whitespace and all structural markup bytes. Count an edit only after the
   final structural SHA check.

**Known limitation deliberately preserved:** a detected token box is not a
complete-cell crop. It can omit context or part of the cell. Two OCR engines
can agree on that incomplete crop and still make the table worse. This release
does not fix that historical bug, expand the action scope, or claim the score
would survive such a fix. It is a research-review artifact, not an automatic
production replacement. No merge/split/topology or multilingual/formula recovery
is enabled in this policy.

## Install and download

Use two separate environments because the original locator and correction
runtimes have incompatible NumPy/Pillow versions. The commands below assume
Python **3.12.2** and **3.10.20** are installed. Exact package versions are checked
before model construction. These versions describe the original runtimes; a
fresh cross-platform end-to-end installation has not been validated here.

```bash
python scripts/download_release.py --route hybrid --output models/hybrid

python3.12 -m venv .venv-locator
.venv-locator/bin/python -m pip install -r hybrid/requirements-locator.txt

python3.10 -m venv .venv-correction
.venv-correction/bin/python -m pip install -r hybrid/requirements-correction.txt
```

Install native Tesseract separately and make `tesseract` available on `PATH`
(for example, your operating system's package manager). The English traineddata
is supplied in the downloaded bundle. The wrapper records the actual Tesseract
version and executable SHA; it does **not** require or claim equivalence to the
historical macOS binary. `7` is the page-segmentation mode, not the Tesseract
version. The wrapper never downloads models at inference time.

Downloaded layout:

```text
models/hybrid/
  mineru/                         # Pinned MinerU raw checkpoint, separate upstream runtime
  ocr/
    ch_PP-OCRv5_rec_server_infer.pth
    ppocrv5_dict.txt
    eng.traineddata
  locator/
    PP-OCRv5_server_det/
    PP-LCNet_x1_0_textline_ori/
    PP-OCRv5_server_rec/
```

Read the bundled license notices before redistributing or commercial use.
MinerU 3.1.15 has terms beyond a blanket Apache claim. Upstream model, code and
native-tool licenses remain applicable independently of this repository.

## Run on one table

Provide an original table crop and the **exact matching** raw HTML table from
your upstream MinerU run. Do not resize the image after obtaining token boxes.
Do not supply benchmark labels, reference text, or answer-derived boxes.

```bash
.venv-locator/bin/python hybrid/locate_tokens.py \
  --table-image /absolute/path/table.png \
  --model-dir models/hybrid --output runs/my-locator

.venv-correction/bin/python hybrid/correct_table.py \
  --html /absolute/path/raw-table.html \
  --table-image /absolute/path/table.png \
  --ocr-tokens runs/my-locator/tokens.json \
  --model-dir models/hybrid --output runs/my-correction
```

The output directory must not already exist. Read `READY.json` before adopting
`final.html`; an interrupted/failed directory is not an admitted result.
`raw.html` retains exact input bytes. `RECEIPT.json` records grounding rejections,
actual OCR observations, crop hashes, committed cell spans and structural hashes.
`INPUT_FREEZE.json` records local input/model/source/runtime hashes. Runtime,
missing-model, hash and recognition failures stop with nonzero exit status and
`FAILURE.json`; they are not silently counted as successful raw fallbacks.
Unsupported HTML is explicitly reported as raw-preserved, not an edit.

The optional locator retains the original textline-orientation model and
thresholds. Images strictly larger than **3,000,000 pixels** use the frozen
`text_det_limit_type=max, text_det_limit_side_len=640` prediction override.
Returned token boxes remain in original table-image coordinates. There is no
hand-selected sample list or confidence tuning. The parser drops/counts empty
text or invalid geometry as in its historical source.

For an entirely project-authored format demonstration:

```bash
.venv-correction/bin/python hybrid/examples/create_example.py --output runs/toy-input
# Use runs/toy-input/table.png and raw.html with the two commands above.
```

The toy generator is not a performance test, does not guarantee an edit, and
does not create OCR observations. Only actual image inference supplies them.

### Token envelope

The locator writes `schema="hybrid-image-only-tokens/v1"`, `status="ok"`,
`input_image_sha256`, `image_width`, `image_height`, and
`coordinate_space="absolute_xyxy_in_table_crop"`. Its `ocr_tokens` list uses
unique nonnegative integer `token_index`, string `text`, and finite positive-area
`bbox=[x0,y0,x1,y1]` inside the image. Optional original parser fields are
`source_detection_index`, `confidence`, `polygon`, and `alternatives`; confidence
and alternatives are not correction votes. The wrapper verifies the image SHA
and dimensions. The user remains responsible for generating tokens from that
image alone; a hash cannot prove that external annotations were never consulted.

## Verification and provenance

Model-free tests use only project-authored strings/geometry and fake recognizers:

```bash
python -m pytest tests/test_hybrid_release.py
python hybrid/correct_table.py --help
python hybrid/locate_tokens.py --help
```

The three policy files and locator producer in `vendor/` are byte-identical
copies of the original sources. Their SHA256 bindings are in `common.py` and
verified before execution. Only bounded functions of the locator producer are
imported; never invoke the vendored historical CLI, which contains old workspace
defaults. No historical execution contracts or benchmark inputs are required.
The portable wrappers change paths and explicit I/O packaging, not thresholds,
recognition crop preparation, parser or writeback policy. Model-free checks and
`--help` are not a score or proof of full-runtime reproducibility.
