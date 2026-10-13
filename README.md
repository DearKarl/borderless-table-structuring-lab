# Borderless Table Structuring Lab

Research on document parsing and table structure, retaining measured scientific
outputs and the source needed to reproduce V0–V8.

[V0–V8](versions/README.md) · [Contributing](CONTRIBUTING.md)

| Version | Method |
| --- | --- |
| [V0](versions/V0/README.md) | MinerU + NaviDC table replacement and earlier table components |
| [V1](versions/V1/README.md) | MinerU + Paddle formula replacement |
| [V2](versions/V2/README.md) | TeleOCR + Paddle formula replacement |
| [V3](versions/V3/README.md) | Layout/formula integration and Ovis text replacement |
| [V4](versions/V4/README.md) | GBDT native-input selection |
| [V5](versions/V5/README.md) | Original RGB inputs and regional rereading |
| [V6](versions/V6/README.md) | GBDT/MLP regional scale controllers |
| [V7](versions/V7/README.md) | Generator pilots and separate legacy selector |
| [V8](versions/V8/README.md) | Large table-specialist LoRA comparisons |

Each directory contains its methods, available numerical results, limitations
and reproduction sources. Different models and protocols are not a controlled
cross-version ranking. Local scores do not establish an accepted leaderboard rank.

The shared application (`btsl`, `hybrid`), evaluator, download utilities and tests
remain at the root because the versions reuse them. Third-party licenses and
model notices are under [V0/notices](versions/V0/notices/THIRD_PARTY.md), with
additional original notices retained alongside their source packages.
