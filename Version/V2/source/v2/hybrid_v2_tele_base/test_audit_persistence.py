import json
from pathlib import Path
import tempfile
import unittest
from .smoke import persist_page_audit
from .region_protocol import RunBlocked


class PersistenceTests(unittest.TestCase):
    def test_primary_exception_is_preserved_alongside_serialization_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);primary=ValueError('original inference failure')
            with self.assertRaises(ValueError) as caught:
                persist_page_audit(folder,[('GENERATION.json',object()),('REGIONS.json',[])],primary,'original trace')
            self.assertIs(caught.exception,primary)
            evidence=json.loads((folder/'PAGE_ERRORS.json').read_bytes())
            self.assertEqual(evidence['primary_error']['error'],'original inference failure')
            self.assertEqual(evidence['audit_errors'][0]['error_type'],'TypeError')
            self.assertFalse((folder/'GENERATION.json').exists())
            self.assertEqual(json.loads((folder/'REGIONS.json').read_bytes()),[])

    def test_audit_only_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            with self.assertRaises(RunBlocked):persist_page_audit(folder,[('GENERATION.json',object())])
            evidence=json.loads((folder/'PAGE_ERRORS.json').read_bytes())
            self.assertIsNone(evidence['primary_error'])
            self.assertEqual(len(evidence['audit_errors']),1)
