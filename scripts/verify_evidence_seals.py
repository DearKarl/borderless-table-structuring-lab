"""Verify every sealed evidence artifact against its recorded SHA256.

Two seal formats are in use under ``docs/experiment-records``:

* ``<artifact>.sha256`` -- a ``sha256sum``-style line, ``<digest>  <filename>``;
* ``<artifact>.seal.json`` -- an object carrying ``artifact`` and
  ``artifact_sha256``.

The check reads the **working tree**, because that is what every other tool in
the repository loads. That makes it sensitive to line endings by design: a seal
that only verifies in the object database is not a seal a collaborator can
check. Measured on 5820ff3, before ``.gitattributes`` pinned ``eol=lf``, all
three sealed cards mismatched in a Windows working tree while their LF blobs
matched exactly.

A mismatch is reported, never repaired: re-sealing an artifact to match changed
bytes would defeat the seal. ``MISSING`` (artifact absent) and ``MISMATCH``
(bytes differ) are reported separately so a failure names its owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

RECORDS = Path(__file__).resolve().parent.parent / "docs" / "experiment-records"


def _iter_seals(root: Path) -> list[tuple[Path, str, str]]:
    """Yield ``(artifact_path, expected_digest, seal_name)`` for every seal."""
    seals: list[tuple[Path, str, str]] = []
    for path in sorted(root.glob("*.sha256")):
        line = path.read_text(encoding="utf-8").strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        seals.append((root / name.strip(), digest.strip(), path.name))
    for path in sorted(root.glob("*.seal.json")):
        seal = json.loads(path.read_text(encoding="utf-8"))
        artifact = seal.get("artifact")
        digest = seal.get("artifact_sha256")
        if artifact and digest:
            seals.append((root / artifact, digest, path.name))
    return seals


def verify(root: Path = RECORDS) -> dict[str, object]:
    checks = []
    for artifact, expected, seal_name in _iter_seals(root):
        if not artifact.is_file():
            checks.append({"seal": seal_name, "artifact": artifact.name,
                           "status": "MISSING"})
            continue
        raw = artifact.read_bytes()
        checks.append({
            "seal": seal_name,
            "artifact": artifact.name,
            "expected_sha256": expected,
            "actual_sha256": hashlib.sha256(raw).hexdigest(),
            "working_tree_has_crlf": b"\r\n" in raw,
            "status": "MATCH" if hashlib.sha256(raw).hexdigest() == expected
                      else "MISMATCH",
        })
    mismatched = [c for c in checks if c["status"] == "MISMATCH"]
    missing = [c for c in checks if c["status"] == "MISSING"]
    if mismatched:
        status = "BLOCKED_CONTENT_MISMATCH"
    elif missing:
        status = "BLOCKED_MISSING_ARTIFACT"
    else:
        status = "PASS"
    return {
        "schema_version": "borderless-table-structuring/evidence-seal-verification-v1",
        "records_root": str(root.name),
        "sealed_artifacts": len(checks),
        "checks": checks,
        "status": status,
        "hint": (
            "Working tree contains CRLF. Re-checkout with the repository "
            ".gitattributes in effect (git add --renormalize . && "
            "git checkout -- .) before concluding an artifact changed."
            if any(c.get("working_tree_has_crlf") for c in checks) else None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify sealed evidence cards.")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = verify()
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
