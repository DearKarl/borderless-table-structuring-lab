from __future__ import annotations

import argparse
import json
from pathlib import Path

from borderless_table_structuring.corpus_release import (
    verify_latent_plan,
    write_latent_plan,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build or independently verify the 40,000-record latent plan."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/corpus_release_parameters_2026.08.12.6.json"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("docs/corpus/SHARED_CORPUS_COVERAGE_2026.08.12.csv"),
    )
    args = parser.parse_args()
    result = (
        verify_latent_plan(args.output, args.config, args.coverage)
        if args.verify_only
        else write_latent_plan(args.output, args.config, args.coverage)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
