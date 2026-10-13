# V5: input paths and regional rereading

[Version index](../README.md) · [Project](../../README.md)

## Method

Five independent TeleOCR system runs compare input paths and bounded region rereading, with unchanged model weights. The regional mechanisms adapt the attributed ki-OCR-v1 reference; this is not a reproduction of its entire R3 system.

| Arm | Ordered intervention |
| --- | --- |
| V5.1 | Default native image-to-PDF route, 200 DPI |
| V5.2 | Direct original RGB input |
| V5.3 | V5.2 plus table rereading at 1× |
| V5.4 | V5.2 plus formula rereading at 1.25× |
| V5.5 | V5.2 plus formula rereading, bounded anomaly guard and table rereading |

## Complete available benchmark fields

Each arm completes all 1,651 pages with zero inference failures and zero empty predictions. Overall, Formula CDM, Table TEDS and Structure TEDS use 0–100; edit distances use 0–1. A dash means not supplied, not zero. Displayed values are rounded; linked records preserve available precision.

| Policy | Overall ↑ | Text ED ↓ | Formula CDM ↑ | Formula ED ↓ | Table TEDS ↑ | Structure TEDS ↑ | Table ED ↓ | Reading-order ED ↓ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V5.1 | 97.568231 | 0.027272 | 98.588639 | — | 96.843229 | 98.167331 | — | 0.118312 |
| V5.2 | 98.173146 | 0.025230 | 98.381257 | — | 98.661134 | 99.189715 | — | 0.118443 |
| V5.3 | 98.222117 | 0.025302 | 98.378168 | — | 98.818403 | 99.366227 | — | 0.118505 |
| V5.4 | 98.250602 | 0.025612 | 98.651424 | — | 98.661591 | 99.189715 | — | 0.118569 |
| V5.5 | 98.347514 | 0.025182 | 98.651273 | — | 98.909492 | 99.373837 | — | 0.118669 |

Formula ED and Table ED are not supplied in this preserved public aggregate. The [full-precision aggregate](results/AGGREGATE.json) includes every available metric, paired outcome, coverage and cost field.

## Interpretation and cost

Original-image V5.2 exceeds V5.1 by 0.604916 Overall points; V5.5 exceeds V5.2 by 0.174368. Native outputs nevertheless differed between arms despite identical first-request inputs/settings: V5.3/V5.4/V5.5 matched V5.2 Markdown on 1,338/1,354/1,371 of 1,651 pages. The cause is unproven, so these system differences do not isolate regional intervention effects. V5.5 versus V5.4 changes both guard and table stages.

Full inference cost was 9.758604 allocated GPU-hours; preparation and stopped startup attempts bring the total to 10.231368. Table rereads changed content on 17 V5.3 regions and 14 V5.5 regions; formula rereads changed 793 regions in each relevant arm. Changes or syntax validity do not by themselves establish repairs. All owned compute exited.

## Implementation and full evidence

- [Complete methods, paired repair/harm counts, input/output agreement, calls, timing and memory](source/v5/README.md).
- [Aggregate and all recorded fields](results/AGGREGATE.json).
- [Frozen protocol](source/v5/protocol.json) and [regional logic](source/v5/regions.py).

Model file hashes were checked, but the requested upstream model revision could not be independently queried. The detailed report preserves that limit. This is local official-protocol evaluation, not a leaderboard entry or a statistical significance claim.
