# Six-model official benchmark metrics

> **Execution qualification:** V8.1/seed0 and V8.2/seed0 each retain 161 empty technical failures from the PDF assembly defect. Their full-denominator scores, contrasts and two-seed means/ranges describe execution outcomes; they are not clean estimates of model quality or training effects.

Descriptive matched evaluation. Overall uses 0–100; all other fields use 0–1. Missing values remain missing.

| Policy | Status | overall | text_edit | formula_cdm | formula_edit | table_teds | table_teds_structure | table_edit | reading_order_edit |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | complete | 97.539151 | 0.029707 | 0.986258 | 0.049639 | 0.969623 | 0.982959 | 0.034118 | 0.118769 |
| legacy_simple_rule | complete | 97.419767 | 0.029707 | 0.986258 | 0.049639 | 0.966041 | 0.981646 | 0.037184 | 0.118769 |
| V8.1/seed0 | complete | 84.879353 | 0.129939 | 0.773333 | 0.255067 | 0.902986 | 0.932127 | 0.085605 | 0.211800 |
| V8.2/seed0 | complete | 85.094576 | 0.129939 | 0.773333 | 0.255067 | 0.909443 | 0.938920 | 0.081719 | 0.211855 |
| V8.3/seed0 | complete | 96.042049 | 0.029707 | 0.986258 | 0.049639 | 0.924711 | 0.954394 | 0.065634 | 0.118771 |
| V8.1/seed1 | complete | 96.316806 | 0.029707 | 0.986258 | 0.049639 | 0.932953 | 0.963210 | 0.060427 | 0.118769 |
| V8.2/seed1 | complete | 95.988052 | 0.029707 | 0.986258 | 0.049639 | 0.923091 | 0.954352 | 0.067085 | 0.118771 |
| V8.3/seed1 | complete | 96.108150 | 0.029707 | 0.986258 | 0.049639 | 0.926694 | 0.956492 | 0.064097 | 0.118771 |

Every policy retains 1,651 pages. Category details, failures, selected checkpoint hashes and descriptive contrasts are in AGGREGATE.json.
