"""Recovery preserves source evidence and refuses inconsistent partial inputs."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from .audit_recovery import recover,sha

class RecoveryTests(unittest.TestCase):
    def fixture(self,root,corrupt=False):
        old=root/'old';old.mkdir();new=root/'new';new.mkdir()
        row=dict(id='one',partition='train',status='input_eligible',normalized_boxes=[[0,0,1,1]])
        db=sqlite3.connect(old/'DETAIL_INPUTS.sqlite')
        db.execute('CREATE TABLE examples(id TEXT PRIMARY KEY,partition TEXT,record TEXT,cell_ranges TEXT,logical_spans TEXT,eligible_tasks TEXT)')
        db.execute('CREATE TABLE tasks(id TEXT,task TEXT,input_ids BLOB,target_start INT,eos_position INT,sequence_tokens INT,target_tokens INT,prompt_sha256 TEXT,target_sha256 TEXT,PRIMARY KEY(id,task))')
        db.execute('INSERT INTO examples VALUES(?,?,?,?,?,?)',('one','train',json.dumps(row),'[[1,2]]','[[0,0,0,0]]','["html"]'))
        db.execute('INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)',('one','html',bytes(12),1,2,4 if corrupt else 3,1,'a','b'))
        db.commit();db.close()
        report=dict(status='running',completed=1,native_cache_exact=1,full_cell_ranges_verified=1,formal_optimizer_updates=0,GPU_allocations=0,
            eligible={'train:html':1},maximum_tokens={'train:html':3},unavailable={},source_manifest_hashes={},elapsed_seconds=1)
        (old/'AUDIT.json').write_text(json.dumps(report))
        original=dict(common_inputs='fixed',output_root=str(old))
        (old/'BINDING.json').write_text(json.dumps(original))
        hashes={str(p):sha(p) for p in old.iterdir()}
        binding=dict(common_inputs='fixed',output_root=str(new),recovery=dict(directory=str(old),binding=str(old/'BINDING.json'),file_hashes=hashes))
        return binding,[row],new,hashes

    def test_verified_copy_retains_original(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TDATR_TEST_TMP")) as folder:
            binding,rows,new,hashes=self.fixture(Path(folder))
            db,report=recover(binding,rows,new);db.close()
            self.assertEqual(report['recovery']['verified_examples'],1)
            self.assertEqual(report['task_sequence_lengths']['train:html'],[3])
            self.assertEqual(hashes,{p:sha(p) for p in hashes})

    def test_inconsistent_token_metadata_rejected(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TDATR_TEST_TMP")) as folder:
            binding,rows,new,_=self.fixture(Path(folder),corrupt=True)
            with self.assertRaisesRegex(ValueError,'token metadata'):recover(binding,rows,new)

if __name__=='__main__':unittest.main()
