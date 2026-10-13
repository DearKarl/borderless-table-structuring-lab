# Source provenance

This is the public source archive for Hybrid V3.1. The inference framework and the nine files under `assets/vendor/native_code/` are project-owned. Original member hashes and the historical parent freeze identity remain recorded in `assets/vendor/PROVENANCE.json` and `PARENT_ASSET_LOCK.json`.

TeleOCR source is **external**, from [caipeng328/TeleOCR](https://github.com/caipeng328/TeleOCR) at revision `9921cffe380efe4e2fa010258b3d0c3cb70bab2d`. No TeleOCR source or model weights are redistributed in this directory. The exact 55-member inference subset remains hash-locked under `PARENT_ASSET_LOCK.json:model_files.tele_source`. Consult the upstream license terms when obtaining the dependency; the presence of an Apache license text here does not grant a license for external source.

The public preparation adapter requires an explicit external source root, validates the locked members and excludes unrelated checkout files. Documentation and status records were updated for publication. `SOURCE_LOCK.json` and `../PACKAGE_LOCK.json` identify these public bytes; they are not the historical GPU-tested source hashes. Model parameters, prompts, routing rules, inference member hashes and score aggregation are preserved. No GPU inference or scoring was rerun for publication.

The evaluator wrapper is project-owned and delegates to a separately provisioned pinned official evaluator. The historical framework adapts the original native client and layout coordinate logic; third-party evaluator code is not included here. See the repository's `notices/` directory for preserved existing attributions.
