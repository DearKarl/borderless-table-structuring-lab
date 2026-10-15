# Versions

[Project](../README.md)

| Version | Scientific outputs and reproduction source |
| --- | --- |
| [V0](V0/README.md) | Table replacement, earlier component results, source, Explicit-v2 inference and third-party notices |
| [V1](V1/README.md) | MinerU formula replacement and source |
| [V2](V2/README.md) | TeleOCR formula replacement and source |
| [V3](V3/README.md) | V3.1/V3.2 source and supplied measurements |
| [V4](V4/README.md) | Input-selector source, model and numerical results |
| [V5](V5/README.md) | Input/rereading source and complete available metrics |
| [V6](V6/README.md) | Scale-controller source, models, metrics and numeric statistics |
| [V7](V7/README.md) | Generator diagnostic metrics and legacy-selector source, models and results |
| [V8](V8/README.md) | LoRA/TDATR-style adaptation source, V8.1–V8.5 metrics, curves and model identities |

Only these nine version directories are published. `__init__.py` preserves old
Python imports such as `versions.v5` without duplicating source. V3.1 and V3.2
belong to V3; the older V7.3.x large-campaign aliases belong to V8.

Methods and measured outcomes remain in each version's README and result files.
TEDS/CDM units and component denominators are stated with the corresponding
measurements. Missing endpoints remain missing. Execution failures and negative
results remain part of the measurements. Old plans, progress reports and report
documents are removed; Git history retains the previous repository versions.

Run the shared application from the project root. Archived version-specific
source uses its documented environment and external assets. Frozen executed
configuration, model identities and source locks describe the measured setup;
they are not bindings to launch a new experiment.
