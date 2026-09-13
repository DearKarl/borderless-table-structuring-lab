"""Read-only checkpoint identity check; never loads tensors or claims quality."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory, manifest):
    if not directory.is_dir():
        raise ValueError("The exact checkpoint directory is missing")
    verified = {}
    for name, expected in manifest["files"].items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Invalid checkpoint filename")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Missing regular checkpoint file: " + name)
        actual = sha256(path)
        if actual != expected:
            raise ValueError("Checkpoint identity mismatch: " + name)
        verified[name] = {"sha256": actual, "bytes": path.stat().st_size}
    return {"status": "EXACT_FILES_VERIFIED_NOT_INFERENCE_OR_SCORE", "files": verified}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(__file__).with_name("model_manifest.json").read_text())
    try:
        print(json.dumps(verify(args.checkpoint_dir, manifest), indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
