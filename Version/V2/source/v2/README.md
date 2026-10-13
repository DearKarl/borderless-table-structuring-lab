# Hybrid V2: TeleOCR with a formula expert

TeleOCR provides the page backbone and PaddleOCR-VL1.6 supplies formula recognition. This is the Tele-base route, not the stopped Paddle-plus-NaviDC prototype.

Historical Overall: **97.06001875937447**, versus **97.4351088743502** for the paired TeleOCR raw arm. The hybrid regressed by **0.37509011497573 points**; this archive does not claim a gain.

Project-owned native loading, region protocol, expert process, assembly, full-run supervision, recovery, score admission and unit checks are under `hybrid_v2_tele_base/`. The full entry is `run_full.py`; `smoke.py` and `full_worker.py` implement native inference. These Linux controllers require explicit frozen deployment artifacts, an input manifest and external model/environment dependencies. They are archived with strict original evidence checks and are not a one-command installer.

Historical source stages are preserved separately under `stages/`: `parallel-v1` contains the independently scheduled hybrid arm, `hybrid-deadline-resume-v1` and `tele-deadline-resume-v1` contain the two bounded recovery stages, and `pair-score-v8` contains final paired score admission. Each has its own archive hash and original/public member hashes in `ORIGIN.json`. The top-level base comes from `FROZEN_CODE_v2.zip`; supplemental development modules are not presented as the exact historical measured snapshot. Stage launchers are archival code, not instructions to restart old runs. Their GPU identifiers are explicit placeholders requiring a new deployment binding.

The public example root is `/srv/hybrid-research`; set `HYBRID_GPU_UUID` for a new deployment. Container ownership labels use the `hybrid.*` namespace. These path/label publication changes require newly generated deployment evidence; old private containers and receipts must not be adopted using the public snapshot.

Obtain [TeleOCR](https://github.com/caipeng328/TeleOCR) source revision `9921cffe380efe4e2fa010258b3d0c3cb70bab2d` and model revision `8706730e41382f4a8f2562616d47225fee8e2f7a` separately under their upstream terms. No TeleOCR source, model weights or evaluation pages are included. Source-contract tests use the explicit `HYBRID_TELE_SOURCE` location. Chinese language and filename test strings are deliberately retained.

The original native BF16 configuration included mixed parameter dtypes; it was not an all-BF16 model. Tele native truncated outputs were retained for this paired evaluation, unlike the empty-output truncation policy of the MinerU/Paddle group. Compare routes with that limitation in mind. No GPU inference or scoring was rerun for publication.

Historical recovery test modules that read `EXTERNAL_GUARD.json`, `ORIGINAL_EXIT.json` or `BINDINGS.json` require private run fixtures and cannot run from this public archive alone. Data-free resource-resume, recovery-score and host-repair checks remain executable. The archive does not claim that every historical test is self-contained.
