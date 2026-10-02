# Hybrid V0: MinerU raw pages and NaviDC tables

The measured CUDA reproduction uses the released `btsl` 0.2.0 implementation at original commit `f059b62adb05b97ecc626712f6de7372cc2865c8`, frozen MinerU2.5-Pro-2605-1.2B raw Markdown, and fresh NaviDC table inference. Its historical Overall is **94.59925245273193** on the 1,651-page whole-page evaluation.

The exact released source snapshot is archived in `released/`, taken from `release-f059b62-v3.zip` with SHA256 `c437e646bebac82ef27374383759685328c9f4bb835f83e6a63a80bc0c2b143f`. Its member provenance is in `released/ORIGIN.json`. The repository root retains the earlier public application and full documentation. See [the native table guide](../../hybrid/native_tables/README.md) for the public inference entry. The `hybrid-v0-repro/` directory archives the bounded CUDA full/smoke/synthetic controllers and fixture generator. Files ending in `_template.py` require their explicit `__PINS__` and related deployment substitutions before execution; they are not standalone launch commands.

The templates reference a public example root `/srv/hybrid-research`. A new deployment must provide its own immutable dependency, model, input and resource locks. No private run receipts or cached outputs are distributed. Existing identity and capacity checks have not been bypassed.

The earlier MPS route's 98.75068667701048 Table TEDS used a different MinerU revision and older evaluator protocol. It is preserved as historical table-only evidence and must not be substituted for this V0 Overall score.
