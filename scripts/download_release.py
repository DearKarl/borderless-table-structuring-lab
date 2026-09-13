#!/usr/bin/env python3
"""Download verified Hybrid weights. Standard library only; never runs a model."""

import argparse
import ctypes
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import urllib.request
import urllib.parse


RELEASE = "hybrid-2026.09.13.1"
BASE_URL = "https://github.com/DearKarl/borderless-table-structuring-lab/releases/download/" + RELEASE + "/"
EXTERNAL_URL = (
    "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/"
    "ed6b654c018d742e65a17671e379c5e6ecc87ec9/models/OCR/paddleocr_torch/"
    "ch_PP-OCRv5_rec_server_infer.pth"
)
CHUNK = 1024 * 1024


class IntegrityError(ValueError):
    """Artifact differs from the checked-in release inventory."""


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(value):
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise IntegrityError("invalid relative file name")
    path = PurePosixPath(value)
    if path.is_absolute() or any(p in (".", "..", "") for p in value.split("/")):
        raise IntegrityError("unsafe relative file name: " + value)
    if ":" in value or str(path) != value:
        raise IntegrityError("noncanonical file name: " + value)
    return value


def check_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise IntegrityError("invalid SHA256")


def check_record(record):
    check_sha(record["sha256"])
    if type(record["size"]) is not int or not 0 <= record["size"] <= 20 * 1024**3:
        raise IntegrityError("invalid artifact size")


def validate_manifest(manifest, route):
    if route != "hybrid" or manifest.get("route") != route:
        raise IntegrityError("only the Hybrid route is downloadable; Training checkpoint is not bundled")
    if manifest.get("schema") != "verified-model-release/v1" or manifest.get("release") != RELEASE:
        raise IntegrityError("unsupported release manifest")
    if manifest.get("archive", {}).get("format") != "tar-split-raw":
        raise IntegrityError("unsupported archive format")
    check_record(manifest["archive"])
    files = manifest.get("files", [])
    if not files or len(files) > 2000:
        raise IntegrityError("invalid inventory size")
    names = set()
    for record in files:
        name = safe_name(record["path"])
        if name in names or record["distribution"] not in ("github-release", "upstream-external"):
            raise IntegrityError("duplicate file or invalid distribution")
        names.add(name)
        check_record(record)
    for name in names:
        if any(str(p) in names for p in PurePosixPath(name).parents if str(p) != "."):
            raise IntegrityError("inventory file/directory collision")
    assets = manifest.get("assets", [])
    if not assets or len(assets) > 64:
        raise IntegrityError("invalid asset count")
    seen = set()
    for record in assets:
        name = safe_name(record["name"])
        if "/" in name or name in seen or record["url"] != BASE_URL + name:
            raise IntegrityError("asset must use the fixed GitHub release URL")
        seen.add(name)
        check_record(record)
        if record["size"] > 1024**3:
            raise IntegrityError("asset exceeds one GiB")
    if sum(a["size"] for a in assets) != manifest["archive"]["size"]:
        raise IntegrityError("archive/chunk size mismatch")
    external = manifest.get("external_files", [])
    ext_names = set()
    for record in external:
        name = safe_name(record["path"])
        check_record(record)
        if name in ext_names or name != "ocr/ch_PP-OCRv5_rec_server_infer.pth" or record["url"] != EXTERNAL_URL:
            raise IntegrityError("external artifact must use the pinned upstream URL")
        if record["size"] != 134640672 or record["sha256"] != "4767ddc90c1532ec01d881a980dae0a0b92679f4f82f88c4e9f92563de69e740":
            raise IntegrityError("external OCR identity mismatch")
        ext_names.add(name)
        expected = next((r for r in files if r["path"] == name), None)
        if not expected or any(record[k] != expected[k] for k in ("size", "sha256", "distribution")):
            raise IntegrityError("external inventory identity mismatch")
    if ext_names != {r["path"] for r in files if r["distribution"] == "upstream-external"}:
        raise IntegrityError("missing external file declaration")
    return manifest


def copy_verified(source, destination, record, archive_digest=None):
    digest = hashlib.sha256()
    count = 0
    with Path(destination).open("xb") as output:
        while True:
            chunk = source.read(CHUNK)
            if not chunk:
                break
            count += len(chunk)
            if count > record["size"]:
                raise IntegrityError("download exceeds declared size")
            digest.update(chunk)
            if archive_digest is not None:
                archive_digest.update(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if count != record["size"] or digest.hexdigest() != record["sha256"]:
        raise IntegrityError("size or SHA256 mismatch for " + record.get("name", record.get("path", "artifact")))


def obtain(record, destination, asset_dir=None, archive_digest=None):
    relative = record.get("name", record.get("path"))
    if asset_dir is not None:
        source_path = Path(asset_dir) / safe_name(relative)
        if source_path.is_symlink() or not source_path.is_file():
            raise IntegrityError("offline asset is not a regular file: " + str(source_path))
        with source_path.open("rb") as source:
            copy_verified(source, destination, record, archive_digest)
    else:
        request = urllib.request.Request(record["url"], headers={"User-Agent": "borderless-table-structuring-lab-release/1"})
        with urllib.request.urlopen(request, timeout=120) as source:
            if urllib.parse.urlsplit(source.geturl()).scheme != "https":
                raise IntegrityError("refusing non-HTTPS redirect")
            copy_verified(source, destination, record, archive_digest)


class JoinedReader(io.RawIOBase):
    """Sequential reader of already verified chunks, without a joined copy."""

    def __init__(self, paths):
        super().__init__()
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            raise IntegrityError("unbounded archive read refused")
        result = bytearray()
        while len(result) < size:
            if self.current is None:
                try:
                    self.current = next(self.paths).open("rb")
                except StopIteration:
                    break
            chunk = self.current.read(size - len(result))
            if chunk:
                result.extend(chunk)
            else:
                self.current.close()
                self.current = None
        return bytes(result)

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def extract_verified(paths, output, inventory):
    expected = {r["path"]: r for r in inventory if r["distribution"] == "github-release"}
    seen = set()
    with JoinedReader(paths) as reader, tarfile.open(fileobj=reader, mode="r|") as archive:
        for member in archive:
            name = safe_name(member.name)
            if not member.isfile() or member.pax_headers or member.sparse is not None:
                raise IntegrityError("only plain regular tar members are permitted")
            if name not in expected or name in seen or member.size != expected[name]["size"]:
                raise IntegrityError("extra, duplicate, or wrong-sized archive file: " + name)
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise IntegrityError("tar member has no content")
            with source:
                copy_verified(source, destination, expected[name])
            os.chmod(destination, 0o644)
            seen.add(name)
    if seen != set(expected):
        raise IntegrityError("archive is missing declared files")


def rename_no_replace(source, destination):
    """Atomic publication without replacing even an empty existing directory."""
    if sys.platform == "win32":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        operation = libc.renamex_np
        operation.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = [os.fsencode(source), os.fsencode(destination), 4]  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        operation = libc.renameat2
        operation.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = [-100, os.fsencode(source), -100, os.fsencode(destination), 1]  # RENAME_NOREPLACE
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace publication unsupported on this platform")
    operation.restype = ctypes.c_int
    if operation(*args) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def install(manifest_path, route, output, asset_dir=None):
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text(encoding="utf-8")), route)
    output = Path(output).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    if os.path.lexists(output):
        raise FileExistsError("output already exists; select a new directory: " + str(output))
    claim = output.parent / ("." + output.name + ".download-claim")
    descriptor = os.open(claim, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, (str(os.getpid()) + "\n").encode("ascii"))
        os.fsync(descriptor)
        with tempfile.TemporaryDirectory(prefix="." + output.name + ".download-", dir=output.parent) as temporary:
            root = Path(temporary)
            payload = root / "payload"
            payload.mkdir()
            paths = []
            archive_digest = hashlib.sha256()
            for index, record in enumerate(manifest["assets"]):
                path = root / record["name"]
                print("Verifying archive part %d/%d" % (index + 1, len(manifest["assets"])), flush=True)
                obtain(record, path, asset_dir, archive_digest)
                paths.append(path)
            if archive_digest.hexdigest() != manifest["archive"]["sha256"]:
                raise IntegrityError("combined archive SHA256 mismatch")
            extract_verified(paths, payload, manifest["files"])
            for record in manifest["external_files"]:
                destination = payload / record["path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                print("Verifying required upstream file: " + record["path"], flush=True)
                obtain(record, destination, asset_dir)
                os.chmod(destination, 0o644)
            actual = {p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file()}
            if actual != {r["path"] for r in manifest["files"]}:
                raise IntegrityError("final inventory differs")
            for record in manifest["files"]:
                path = payload / record["path"]
                if path.is_symlink() or path.stat().st_size != record["size"] or digest_file(path) != record["sha256"]:
                    raise IntegrityError("final file verification failed: " + record["path"])
            rename_no_replace(payload, output)
    finally:
        os.close(descriptor)
        claim.unlink()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, choices=["hybrid", "training"])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / (RELEASE + ".json"))
    parser.add_argument("--asset-dir", type=Path, help="Offline directory containing release chunks and external files at inventory-relative paths")
    args = parser.parse_args(argv)
    try:
        output = install(args.manifest, args.route, args.output, args.asset_dir)
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, "Artifact download failed safely: %s\n" % error)
    print("Verified model files saved to " + str(output))
    print("No inference or evaluation was run. Third-party licenses remain applicable.")


if __name__ == "__main__":
    main()
