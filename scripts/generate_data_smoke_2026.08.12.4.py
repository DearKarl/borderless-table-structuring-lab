from __future__ import annotations

import argparse
import json
from pathlib import Path

from borderless_table_structuring.synthetic_data import generate_smoke, verify_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate or verify the frozen 2026.08.12.4 synthetic-data smoke.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("configs/generation_parameters_2026.08.12.4.json"))
    parser.add_argument("--coverage", type=Path, default=Path("docs/corpus/COVERAGE_MATRIX_2026.08.12.4.csv"))
    parser.add_argument("--schema", type=Path, default=Path("schemas/synthetic_table_record_2026.08.12.4.json"))
    args = parser.parse_args()
    if args.verify_only:
        result = verify_smoke(args.output, args.schema)
    else:
        result = generate_smoke(args.output, args.config, args.coverage, args.schema)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
