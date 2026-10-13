# V4 reproduction source and outputs

[V4 method and numerical results](../README.md)

`model/` contains the GBDT and its configuration/identity metadata.
`runtime/` contains the GPU, CPU and supporting source used by this route.
`dependencies/` identifies the external assets and environment requirements.
`results/RESULTS.json` and `results/METRICS.json` retain original numerical values,
component denominators and failure accounting. No new score is produced here.

From this directory, run `python -B verify_delivery.py` to check the retained
source, model, numerical outputs and current delivery inventory. This is a
file-integrity check without model loading, network or inference.

Deployment placeholders must be replaced with separately validated bindings.
Input pages, ground truth and third-party weights are not distributed here.
Original executed identities remain in `provenance/`; `DELIVERY_MANIFEST.json`
describes the retained repository files after user-requested document cleanup.
