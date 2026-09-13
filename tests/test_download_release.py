"""Model-free tests for checksummed downloads and hostile tar handling."""

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("release_download", Path(__file__).parents[1] / "scripts/download_release.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.assets = self.root / "assets"
        self.assets.mkdir()
        self.output = self.root / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, names=("mineru/config.json",), kind=None):
        body = b'{"fixture":true}\n'
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = len(body)
                if kind is not None:
                    member.type = kind
                    member.linkname = "../../outside"
                    member.size = 0
                    archive.addfile(member)
                else:
                    archive.addfile(member, io.BytesIO(body))
        raw = stream.getvalue()
        cut = len(raw) // 2
        chunks = [raw[:cut], raw[cut:]]
        assets = []
        for index, chunk in enumerate(chunks):
            name = "hybrid-models.tar.part%03d" % index
            (self.assets / name).write_bytes(chunk)
            assets.append({"name": name, "url": MODULE.BASE_URL + name, "size": len(chunk), "sha256": hashlib.sha256(chunk).hexdigest()})
        manifest = {
            "schema": "verified-model-release/v1", "route": "hybrid", "release": MODULE.RELEASE,
            "archive": {"format": "tar-split-raw", "size": len(raw), "sha256": hashlib.sha256(raw).hexdigest()},
            "assets": assets,
            "files": [{"path": "mineru/config.json", "size": len(body), "sha256": hashlib.sha256(body).hexdigest(), "distribution": "github-release"}],
            "external_files": [],
        }
        return manifest, body

    def run_install(self, manifest, route="hybrid"):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return MODULE.install(path, route, self.output, self.assets)

    def test_offline_round_trip(self):
        manifest, body = self.fixture()
        self.run_install(manifest)
        self.assertEqual((self.output / "mineru/config.json").read_bytes(), body)
        self.assertFalse((self.root / ".models.download-claim").exists())

    def test_checksum_mismatch_publishes_nothing(self):
        manifest, _ = self.fixture()
        (self.assets / manifest["assets"][0]["name"]).write_bytes(b"corrupt")
        with self.assertRaises(MODULE.IntegrityError):
            self.run_install(manifest)
        self.assertFalse(self.output.exists())

    def test_wrong_route_rejected(self):
        manifest, _ = self.fixture()
        with self.assertRaisesRegex(MODULE.IntegrityError, "Training"):
            self.run_install(manifest, "training")

    def test_traversal_rejected(self):
        manifest, _ = self.fixture(("../outside",))
        with self.assertRaises(MODULE.IntegrityError):
            self.run_install(manifest)
        self.assertFalse((self.root / "outside").exists())
        self.assertFalse(self.output.exists())

    def test_symlink_and_hardlink_rejected(self):
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
            manifest, _ = self.fixture(kind=kind)
            with self.assertRaises(MODULE.IntegrityError):
                self.run_install(manifest)
            self.assertFalse(self.output.exists())

    def test_extra_and_duplicate_files_rejected(self):
        for names in (("mineru/config.json", "extra"), ("mineru/config.json", "mineru/config.json")):
            manifest, _ = self.fixture(names)
            with self.assertRaises(MODULE.IntegrityError):
                self.run_install(manifest)

    def test_existing_empty_output_is_not_overwritten(self):
        manifest, _ = self.fixture()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_install(manifest)
        self.assertEqual(list(self.output.iterdir()), [])

    def test_publish_race_does_not_overwrite(self):
        source = self.root / "source"
        source.mkdir()
        (source / "owned").write_bytes(b"keep")
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            MODULE.rename_no_replace(source, self.output)
        self.assertTrue((source / "owned").exists())
        self.assertEqual(list(self.output.iterdir()), [])

    def test_wrong_url_rejected(self):
        manifest, _ = self.fixture()
        manifest["assets"][0]["url"] = "https://attacker.invalid/file"
        with self.assertRaises(MODULE.IntegrityError):
            self.run_install(manifest)

    def test_wrong_size_missing_file_and_wrong_file_sha(self):
        original, _ = self.fixture()
        for field, value in (("size", 999), ("path", "mineru/missing.json"), ("sha256", "0" * 64)):
            manifest = copy.deepcopy(original)
            manifest["files"][0][field] = value
            with self.assertRaises(MODULE.IntegrityError):
                self.run_install(manifest)
            self.assertFalse(self.output.exists())

    def test_inventory_path_collision_rejected(self):
        manifest, _ = self.fixture()
        extra = dict(manifest["files"][0], path="mineru")
        manifest["files"].append(extra)
        with self.assertRaises(MODULE.IntegrityError):
            self.run_install(manifest)

    def test_unpinned_external_url_rejected(self):
        manifest, _ = self.fixture()
        manifest["external_files"] = [{
            "path": "ocr/ch_PP-OCRv5_rec_server_infer.pth", "size": 134640672,
            "sha256": "4767ddc90c1532ec01d881a980dae0a0b92679f4f82f88c4e9f92563de69e740",
            "distribution": "upstream-external", "url": MODULE.EXTERNAL_URL.replace("ed6b654c018d742e65a17671e379c5e6ecc87ec9", "main"),
        }]
        with self.assertRaises(MODULE.IntegrityError):
            self.run_install(manifest)


if __name__ == "__main__":
    unittest.main()
