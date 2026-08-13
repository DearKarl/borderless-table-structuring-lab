from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from borderless_table_structuring.shared_corpus import (
    build_shared_corpus,
    verify_shared_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify the frozen shared corpus 2026.08.12."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--expected-records", type=int, default=40000)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/shared_corpus_parameters_2026.08.12.json"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("docs/corpus/SHARED_CORPUS_COVERAGE_2026.08.12.csv"),
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("schemas/synthetic_table_record_2026.08.12.5.json"),
    )
    args = parser.parse_args()
    try:
        if args.verify_only:
            result = verify_shared_corpus(
                args.output, args.schema, expected_records=args.expected_records
            )
        else:
            result = build_shared_corpus(
                args.output, args.config, args.coverage, args.schema
            )
    except Exception as error:
        if args.output.exists() and (args.output / "reports").is_dir():
            failure = {
                "dataset_release": "shared-corpus-2026.08.12",
                "generator_release": "2026.08.12.5",
                "error_type": type(error).__name__,
                "error": str(error),
                "training": False,
                "terminal_inputs_used": False,
                "traceback": traceback.format_exc(),
            }
            (args.output / "reports" / "FAILURE_REPORT.json").write_text(
                json.dumps(failure, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
