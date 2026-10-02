import ast
import hashlib
from pathlib import Path
import tempfile
import unittest
from .full_state import exclusive_json,exclusive_bytes,sha,commit_page,pending_pages,phase_limit,validate_session_exit,validate_input_paths


class StateTests(unittest.TestCase):
    def test_original_mixed_jpg_png_filenames_and_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();rows=[]
            for i,name in enumerate(('source.a.jpg','source.b.png','source.c.jpeg')):
                path=root/name;path.write_bytes(('original-'+name).encode())
                rows.append({'page_id':'id-%d'%i,'image_path':str(path),'input_sha256':sha(path)})
            before={p.name:p.read_bytes() for p in root.iterdir()}
            validate_input_paths(rows,root)
            self.assertEqual(before,{p.name:p.read_bytes() for p in root.iterdir()})
            self.assertEqual([Path(r['image_path']).name for r in rows],list(before))

    def test_manifest_paths_reject_changed_duplicate_extra_and_outside_images(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();images=root/'images';images.mkdir();path=images/'original.jpg';path.write_bytes(b'original')
            row={'page_id':'id','image_path':str(path),'input_sha256':sha(path)}
            validate_input_paths([row],images)
            with self.assertRaises(ValueError):validate_input_paths([row,{**row,'page_id':'id2'}],images)
            with self.assertRaises(ValueError):validate_input_paths([{**row,'input_sha256':'a'*64}],images)
            with self.assertRaises(ValueError):validate_input_paths([{**row,'image_path':str(root/'original.jpg')}],images)
            (images/'unexpected.png').write_bytes(b'x')
            with self.assertRaises(ValueError):validate_input_paths([row],images)

    def setup_run(self,folder):
        root=Path(folder);session=root/'sessions/session-0001';page={'key':'p00000','page_id':'original 中文','input_sha256':'a'*64}
        for p in [root/'predictions',root/'terminals',session/'pages/p00000']:p.mkdir(parents=True)
        return root,session,page

    def result(self,session,page,data=b'original content',status='success'):
        folder=session/'pages'/page['key'];exclusive_bytes(folder/'prediction.md',data)
        exclusive_json(folder/'PAGE_RESULT.json',{**page,'status':status,'arm':'tele_raw','freeze_sha256':'b'*64,
                       'prediction_sha256':sha(folder/'prediction.md')})

    def test_commit_is_idempotent_and_pending_validates_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root,session,page=self.setup_run(folder);self.result(session,page)
            self.assertTrue(commit_page(root,session,page,'b'*64,'tele_raw'))
            self.assertFalse(commit_page(root,session,page,'b'*64,'tele_raw'))
            self.assertEqual(pending_pages(root,[page],'b'*64,'tele_raw'),[])
            (root/'predictions'/(page['page_id']+'.md')).write_bytes(b'changed')
            with self.assertRaises(ValueError):pending_pages(root,[page],'b'*64,'tele_raw')

    def test_orphan_primary_is_adopted_only_with_matching_original_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root,session,page=self.setup_run(folder);self.result(session,page)
            exclusive_bytes(root/'predictions'/(page['page_id']+'.md'),b'original content')
            self.assertEqual(pending_pages(root,[page],'b'*64,'tele_raw'),[page])
            self.assertTrue(commit_page(root,session,page,'b'*64,'tele_raw'))

    def test_failed_empty_and_native_truncated_content_are_preserved(self):
        for status,data in [('failed',b''),('truncated',b'partial native text')]:
            with self.subTest(status=status),tempfile.TemporaryDirectory() as folder:
                root,session,page=self.setup_run(folder);self.result(session,page,data,status)
                commit_page(root,session,page,'b'*64,'tele_raw')
                self.assertEqual((root/'predictions'/(page['page_id']+'.md')).read_bytes(),data)
                self.assertEqual(pending_pages(root,[page],'b'*64,'tele_raw'),[])

    def test_different_recipe_and_nonempty_failure_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root,session,page=self.setup_run(folder);self.result(session,page,b'invalid','failed')
            with self.assertRaises(ValueError):commit_page(root,session,page,'b'*64,'tele_raw')
            with self.assertRaises(ValueError):commit_page(root,session,page,'c'*64,'tele_raw')

    def test_atomic_publish_never_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'terminal.json';exclusive_json(path,{'status':'first'})
            with self.assertRaises(FileExistsError):exclusive_json(path,{'status':'second'})
            self.assertIn('first',path.read_text())

    def test_full_initialization_ast_matches_gpu_passed_smoke(self):
        root=Path(__file__).parent
        smoke=ast.parse((root/'smoke.py').read_text(encoding='utf-8'))
        main=next(x for x in smoke.body if isinstance(x,ast.FunctionDef) and x.name=='main')
        body=next(x for x in main.body if isinstance(x,ast.Try)).body
        start=next(i for i,x in enumerate(body) if isinstance(x,ast.Expr) and ast.unparse(x).startswith('sys.path.insert'))
        end=next(i for i,x in enumerate(body) if isinstance(x,ast.Assign) and ast.unparse(x).startswith('hybrid_class ='))
        runtime=ast.parse((root/'runtime_load.py').read_text(encoding='utf-8'))
        actual=next(x for x in runtime.body if isinstance(x,ast.FunctionDef)).body[:-1]
        self.assertEqual([ast.dump(x) for x in body[start:end+1]],[ast.dump(x) for x in actual])

    def test_each_loader_has_its_own_full_budget(self):
        self.assertEqual(phase_limit('load_tele'),630)
        self.assertEqual(phase_limit('load_paddle'),630)
        self.assertEqual(phase_limit('page'),930)
        tree=ast.parse((Path(__file__).parent/'full_worker.py').read_text())
        phases=[x.args[0].value for x in ast.walk(tree) if isinstance(x,ast.Call) and
                isinstance(x.func,ast.Name) and x.func.id=='stage' and isinstance(x.args[0],ast.Constant)]
        self.assertIn('load_tele',phases);self.assertIn('load_paddle',phases)

    def test_unknown_oom_and_supervisor_stops_cannot_resume(self):
        for reason,oom,blocked,exit_code in [('foreign_gpu_process',False,False,0),
              ('supervisor_error',False,False,0),(None,True,False,0),(None,False,True,1),
              (None,False,False,137)]:
            with self.subTest(reason=reason,oom=oom,blocked=blocked),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)
                exclusive_json(path/'EXIT.json',{'reason':reason,'container':{'State':{
                    'Running':False,'OOMKilled':oom,'Status':'exited','ExitCode':exit_code}}})
                exclusive_json(path/'SESSION_RESULT.json',{'complete':True,'blocked':blocked})
                with self.assertRaises(ValueError):validate_session_exit(path)

    def test_clean_completion_and_explicit_pause_can_resume(self):
        for reason,result,exit_code in [(None,{'complete':True},0),
                 (None,{'complete':False,'pause_reason':'PauseAfterFallback'},2),('page_timeout',None,137)]:
            with self.subTest(reason=reason,result=result),tempfile.TemporaryDirectory() as folder:
                path=Path(folder)
                exclusive_json(path/'EXIT.json',{'reason':reason,'container':{'State':{
                    'Running':False,'OOMKilled':False,'Status':'exited','ExitCode':exit_code}}})
                if result is not None:exclusive_json(path/'SESSION_RESULT.json',result)
                validate_session_exit(path)

    def test_incomplete_and_unrecognized_pages_are_not_silently_skipped(self):
        with tempfile.TemporaryDirectory() as folder:
            root,session,page=self.setup_run(folder)
            self.assertEqual(pending_pages(root,[page],'b'*64,'tele_raw'),[page])
            exclusive_json(root/'terminals/unrecognized.json',{})
            with self.assertRaises(ValueError):pending_pages(root,[page],'b'*64,'tele_raw')
