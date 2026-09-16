"""Stream-verify the 14 pinned upstream files; no download or Torch import."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(model_dir, manifest):
    for row in manifest["files"]:
        path = Path(model_dir) / row["path"]
        if path.stat().st_size != row["bytes"]:
            raise ValueError("MODEL_FILE_SIZE_MISMATCH: " + row["path"])
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != row["sha256"]:
            raise ValueError("MODEL_FILE_HASH_MISMATCH: " + row["path"])
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("model_manifest.json").read_text())
    print(json.dumps({"verified_files": verify(args.model_dir, manifest),
                      "revision": manifest["revision"], "model_executed": False}))


if __name__ == "__main__":
    main()
