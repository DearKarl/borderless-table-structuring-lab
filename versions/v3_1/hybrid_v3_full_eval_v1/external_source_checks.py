"""Data-free external source subset and copy contract checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import asset_binder as binder
from .core import ContractError, read, sha


class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.tele = self.root / 'tele'
        self.tele.mkdir()
        original = read(binder.VENDOR / 'PARENT_ASSET_LOCK.json')
        names = original['model_files']['tele_source']
        self.assertEqual(len(names), 55)
        self.files = {}
        for name in names:
            p = self.tele / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('synthetic source: ' + name, encoding='utf-8')
            self.files[name] = sha(p)
        (self.tele / '.git').mkdir()
        (self.tele / '.git/config').write_text('unrelated metadata')
        self.parent = {'model_files': {'tele_source': self.files}}

    def test_exact_55_member_subset(self):
        self.assertEqual(binder.external_tele_files(self.tele, self.parent), self.files)
        self.assertNotIn('.git/config', self.files)

    def test_missing_source_rejected(self):
        (self.tele / next(iter(self.files))).unlink()
        with self.assertRaises(FileNotFoundError):
            binder.external_tele_files(self.tele, self.parent)

    def test_wrong_hash_rejected(self):
        (self.tele / next(iter(self.files))).write_text('changed')
        with self.assertRaises(ContractError):
            binder.external_tele_files(self.tele, self.parent)

    def test_bind_copies_only_locked_subset_before_later_validation(self):
        plan = {'schema': 1, 'parent_asset_lock_sha256': sha(binder.VENDOR / 'PARENT_ASSET_LOCK.json'),
                'assets': {'tele_source': {'path': str(self.tele), 'files': self.files, 'subset': True}},
                'environments': {}}
        plan_path = self.root / 'plan.json'
        plan_path.write_text(json.dumps(plan), encoding='utf-8')
        out = self.root / 'bound'
        # Later model validation must fail for this deliberately incomplete synthetic plan.
        with self.assertRaises((ContractError, KeyError)):
            binder.bind(plan_path, out)
        copied = {p.relative_to(out / 'assets/tele_source').as_posix()
                  for p in (out / 'assets/tele_source').rglob('*') if p.is_file()}
        self.assertEqual(copied, set(self.files))
        self.assertFalse((out / 'assets/tele_source/.git').exists())
        for name, expected in self.files.items():
            self.assertEqual(sha(out / 'assets/tele_source' / name), expected)


if __name__ == '__main__':
    unittest.main()
