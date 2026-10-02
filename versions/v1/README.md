# Hybrid V1: MinerU with a formula expert

MinerU supplies the page backbone. PaddleOCR-VL1.6 supplies formula recognition; strict one-to-one geometry and syntax checks control replacement while preserving non-formula content. The internal `hybrid/formula_v0` name denotes the assembly prototype, not measured Hybrid V0.

Historical fresh-run Overall: **93.2174987411406**. The verified-cache development mode is a distinct diagnostic mode and must not be described as fresh inference.

From this directory:

```sh
python -B scripts/hybrid_v1.py --help
python -B scripts/hybrid_formula_v0.py --help
```

Use `predict-page` or `predict-manifest` with an original PNG/JPEG image, input manifest, explicit system freeze and freeze hash. Fresh mode requires the locked native controller and its source compatibility records. Runtime model assets, image identities and exact parameter capacity evidence are external prerequisites. The historical cache adapter uses `/srv/hybrid-research` as an example root and still requires the original immutable evidence; this is not an automatic cache migration.

Publication preserves inference and acceptance logic. Private caches, original deployment records and images are excluded; new source locks do not replace the historical runtime identity checks. No GPU inference was rerun for publication.

The final measured source comes from the frozen delivery archive identified in `ORIGIN.json`, including `native-code/` and its watchdog policy. Set `HYBRID_GPU_ALLOWLIST` to an explicit comma-separated allowlist when preparing a new deployment. Historical code/freeze constants are retained for provenance; the sanitized archive does not satisfy old private compatibility receipts automatically. Rebinding and validating a new deployment is separate work, not a completed result of this archive.
