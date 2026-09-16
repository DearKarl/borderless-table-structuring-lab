# Third-party models and software

This release keeps original pretrained weights unchanged. They are not models
trained by this project. The project Training route is documented separately.
The Hybrid sidecar is the best completed local official-protocol system so far,
not a verified public leaderboard submission or a result above 95.

## Mirrored pretrained artifacts

| Path in the downloaded model directory | Source | Applicable notice |
|---|---|---|
| `mineru/` | [OpenDataLab MinerU2.5-Pro-2604-1.2B](https://modelscope.cn/models/OpenDataLab/MinerU2.5-Pro-2604-1.2B) | The original model card declares Apache License 2.0. See `MINERU_MODEL_CARD.md` and `APACHE-2.0.txt`. |
| `locator/PP-OCRv5_server_det/` | [PaddlePaddle PP-OCRv5_server_det](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det) | Apache 2.0; see the original model card and `PADDLEOCR_LICENSE.txt`. |
| `locator/PP-OCRv5_server_rec/` | [PaddlePaddle PP-OCRv5_server_rec](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_rec) | Apache 2.0; see the original model card and `PADDLEOCR_LICENSE.txt`. |
| `locator/PP-LCNet_x1_0_textline_ori/` | [PaddleOCR model list](https://www.paddleocr.ai/main/en/version3.x/module_usage/textline_orientation_classification.html) | Apache 2.0; see `PADDLEOCR_LICENSE.txt`. |
| `ocr/eng.traineddata` | [Tesseract tessdata](https://github.com/tesseract-ocr/tessdata) | Apache 2.0; see `TESSERACT_LICENSE.txt` and `TESSDATA_LICENSE.txt`. No native Tesseract executable is distributed. |
| `ocr/ppocrv5_dict.txt` | MinerU 3.1.15 resource derived from PaddleOCR | Preserve PaddleOCR attribution and the installed MinerU license described below. |

The inventory pins each file by SHA256 and byte count. Upstream model-card
scores are the authors' claims, not this project's reproduced scores.

## Current native-table experiment (separate upstream download)

The pending MinerU + NaviDC architecture is documented in
[`hybrid/native_tables/README.md`](../hybrid/native_tables/README.md).
It uses unchanged `StarDoc-AI/NaviDC-OCR` weights at the exact revision recorded in
that package's model manifest. These weights are **not** in the older Hybrid
release archives above. The documented upstream download plus local verifier
checks the pinned files against their SHA256 values. The
[pinned upstream model card](https://huggingface.co/StarDoc-AI/NaviDC-OCR/blob/710ea2e26d794fe89cbf3ece0402707c332a8671/README.md)
declares Apache-2.0 and is itself hash-bound in that manifest. Upstream claims
do not establish a project benchmark result. This experiment still has no
completed full Table TEDS score.

## Required separately downloaded file

`ocr/ch_PP-OCRv5_rec_server_infer.pth` is **not mirrored in the GitHub assets**.
The downloader obtains the exact file directly from the official
[PDF-Extract-Kit-1.0 repository](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0)
at revision `ed6b654c018d742e65a17671e379c5e6ecc87ec9`. Its SHA256 is
`4767ddc90c1532ec01d881a980dae0a0b92679f4f82f88c4e9f92563de69e740`.

That model repository explicitly states **AGPL-3.0**, not Apache 2.0. The exact
upstream model card is preserved as `PDF_EXTRACT_KIT_MODEL_CARD.md`, together
with `AGPL-3.0.txt`. Upstream corresponding software is available at
[PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit) and
[MinerU](https://github.com/opendatalab/MinerU). Downloading or using the file
does not remove its license obligations. This project does not claim that a
source link alone clears every downstream redistribution or service use.

The GitHub archive alone is therefore not a complete offline installation.
The downloader verifies both mirrored assets and this required upstream file
before publishing the final model directory. For an offline installation,
place this exact file at `ocr/ch_PP-OCRv5_rec_server_infer.pth` under the
directory passed to `--asset-dir`, alongside the split archive parts.

## Installed MinerU runtime license

`MINERU_3.1.15_LICENSE.md` is an unchanged copy from the installed 3.1.15
distribution. It declares Apache 2.0 **with additional terms**, including
commercial thresholds and online-service attribution. Do not describe the
complete runtime as unconditionally plain Apache 2.0. Review the full text
before commercial or hosted use. The upstream model card and runtime package
are distinct artifacts and their notices are preserved separately.

No dataset, private page, crop, benchmark reference answer, native executable,
historical execution contract, or project-trained checkpoint is included in
these Hybrid weight archives. This notice is an attribution inventory, not
legal advice or an additional license grant.
