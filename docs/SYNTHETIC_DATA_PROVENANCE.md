# Synthetic corpus provenance

This repository does not store rendered data, benchmark pages, model outputs, or
training payloads. The approved synthetic packages are shared separately.

## SynthFin v3.4

SynthFin v3.4 is a locally authored renderer for financial-report-style table
pages. It does not copy pixels or annotations from FinTabNet, PubTables,
OmniDocBench, CNInfo, or customer documents. Its vocabulary, company-like
names, financial values, page prose, table topology, and metadata are generated
from local templates and seeded random draws.

The renderer supports Chinese and English financial statements, borderless and
ruled tables, multi-row headers, row and column spans, dense pages, multiple
tables per page, narrative context, optional formulas, resolution ladders, and
light JPEG capture effects. The v3.4 correction avoids drawing ruled segments
through the interior of merged cells. The ground-truth structure is generated
from the same table state used by the renderer and is validated by round-trip
checks before use.

The shared package contains only synthetic images and its manifest. It does not
contain the upstream public datasets used as external references, model
weights, terminal benchmark content, or prediction files.

## invoice-synthetic-v1

The invoice package is fully self-generated. It uses local templates for
invoice-like layouts, random company names, dates, identifiers, addresses,
items, tax rates, totals, and Chinese amount-in-words strings. Each text draw is
recorded at generation time, so the visible-text annotations and rendered
images share one source of truth. The package does not use external invoice
images or scanned documents as backgrounds.

## Scope and limitations

These packages are research synthetic data, not official benchmark data and
not a claim of real-domain performance. They are intended for debugging,
controlled ablations, and reproducibility of rendering/label contracts. Users
should inspect the applicable license and sharing terms before redistributing
modified versions.
