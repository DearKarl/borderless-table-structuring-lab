import unittest
from .lock_checks import digest
from .environment_probe import fasttext_header
import struct


class LockTests(unittest.TestCase):
    def test_reject_bad_digest_lengths_and_characters(self):
        self.assertEqual(digest('a'*64),'a'*64)
        for bad in ['a'*63,'a'*65,'g'*64,'A'*64,None,64]:
            with self.assertRaises(ValueError):digest(bad)

    def test_fasttext_header_bounds_and_version(self):
        raw=struct.pack('<2i12id3i2q',793712314,12,16,5,5,1000,5,1,1,3,2000000,2,4,100,.0001,7411,7235,176,563512702,42765)
        self.assertEqual(fasttext_header(raw)['dimension'],16)
        for bad in [raw[:90],bytes(92),raw[:4]+struct.pack('<i',13)+raw[8:]]:
            with self.assertRaises(ValueError):fasttext_header(bad)
