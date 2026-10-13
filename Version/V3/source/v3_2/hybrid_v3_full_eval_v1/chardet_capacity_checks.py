"""Focused synthetic capacity gates; no model or subprocess execution."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import asset_binding as a
from .core import ContractError,canonical,read,sha
class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.serial=0
    def fixture(self,components=(),include=False,role='paddle'):
        self.serial+=1;p=self.root/str(self.serial);p.mkdir();lock=a.chardet_lock()
        aux={'native.fixture':'a'*64}
        if include:aux.update(lock['members'])
        r={'schema':2,'assets':{'auxiliary_models':{'files':aux}},'environments':{},'v31':{'components':list(components)},'parameter_ledger':[],'native':{'capacity':{'passed':True,'conservative_upper_bound':0}}}
        if include and role:r['environments'][role]={'site_packages':'/site','image_files':{'/site/'+n:h for n,h in lock['members'].items()},'asset_roles':['auxiliary_models']}
        for name in ['tele','fasttext','onnx',*components]+(['chardet'] if include else []):
            e={'component':name,'stored_elements':1,'unique_trainable':1,'buffers':0,'model_files_sha256':canonical(aux),'method':'exact_auxiliary_inventory','measurement_source_sha256':'1'*64}
            if name=='chardet':e.update(lock['counts'],relevant_members=lock['members'],classification_lock_sha256=a.CHARDET_LOCK_SHA)
            row={k:e[k] for k in ('component','stored_elements','unique_trainable','buffers','model_files_sha256')}
            row.update(model_asset='auxiliary_models',evidence={'asset':'parameter_evidence','path':name+'.json','sha256':''})
            r['parameter_ledger'].append(row);self.save(p,row,e)
        return r,{'parameter_evidence':p,'auxiliary_models':p/'aux'}
    def save(self,p,row,e):
        file=p/row['evidence']['path'];file.write_text(json.dumps(e));row['evidence']['sha256']=sha(file)
    def change_count(self,r,roots,name,count):
        row=next(x for x in r['parameter_ledger'] if x['component']==name);e=read(roots['parameter_evidence']/row['evidence']['path'])
        for x in (row,e):x.update(stored_elements=count,unique_trainable=count)
        self.save(roots['parameter_evidence'],row,e)
    def test_native_single_both_and_schema1(self):
        for components,inc,n in [((),False,3),(('layout',),True,5),(('formula',),True,5),(('layout','formula'),True,6)]:
            r,roots=self.fixture(components,inc);self.assertEqual(len(r['parameter_ledger']),n);self.assertEqual(a.capacity(r,roots),(n-1)+23199744 if inc else n)
        r,roots=self.fixture();r['schema']=1;r['environments']['paddle']={}
        self.assertEqual(a.capacity(r,roots),3)
    def test_row_delete_duplicate_wrong_component(self):
        for fault in ('delete','duplicate','wrong'):
            r,roots=self.fixture(('layout','formula'),True);r['mode']='pass-through'
            if fault=='delete':r['parameter_ledger'].pop()
            elif fault=='duplicate':r['parameter_ledger'].append(copy.deepcopy(r['parameter_ledger'][-1]))
            else:r['parameter_ledger'][-1]['component']='other'
            with self.subTest(fault=fault),self.assertRaises(ContractError):a.capacity(r,roots)
    def test_partial_image_or_asset(self):
        for count in (1,2,3):
            for target in ('image','asset'):
                r,roots=self.fixture(include=True)
                d=r['environments']['paddle']['image_files'] if target=='image' else r['assets']['auxiliary_models']['files']
                for n in list(a.chardet_lock()['members'])[count:]:d.pop('/site/'+n if target=='image' else n)
                with self.subTest(count=count,target=target),self.assertRaises(ContractError):a.capacity(r,roots)
    def test_hash_and_member_provenance(self):
        for target in ('image','asset','evidence'):
            r,roots=self.fixture(include=True);name=next(iter(a.chardet_lock()['members']))
            if target=='image':r['environments']['paddle']['image_files']['/site/'+name]='f'*64
            elif target=='asset':r['assets']['auxiliary_models']['files'][name]='f'*64
            else:
                row=r['parameter_ledger'][-1];e=read(roots['parameter_evidence']/'chardet.json');e['relevant_members'][name]='f'*64;self.save(roots['parameter_evidence'],row,e)
            with self.subTest(target=target),self.assertRaises(ContractError):a.capacity(r,roots)
    def test_exact_counts_method_and_classification_lock(self):
        for key,values in {'stored_elements':[0,23297028,23297030],'unique_trainable':[0,23199743,23199745],'buffers':[0,90720,90722],'static_elements':[0,6563,6565],'method':['loaded_unique_parameters'],'classification_lock_sha256':['f'*64]}.items():
            for value in values:
                r,roots=self.fixture(include=True);row=r['parameter_ledger'][-1];e=read(roots['parameter_evidence']/'chardet.json');e[key]=value
                if key in row:row[key]=value
                self.save(roots['parameter_evidence'],row,e)
                with self.subTest(key=key,value=value),self.assertRaises(ContractError):a.capacity(r,roots)
        with patch.object(a,'sha',return_value='f'*64),self.assertRaises(ContractError):a.chardet_lock()
    def test_assets_only_and_multiple_images_once(self):
        r,roots=self.fixture(include=True,role=None);self.assertEqual(a.capacity(r,roots),23199747)
        r,roots=self.fixture(include=True,role='native');r['environments']['paddle']=copy.deepcopy(r['environments']['native'])
        self.assertEqual(a.capacity(r,roots),23199747)
    def test_paddle_cannot_erase_all_members(self):
        r,roots=self.fixture();r['environments']['paddle']={'site_packages':'/site','image_files':{},'asset_roles':[]}
        with self.assertRaises(ContractError):a.capacity(r,roots)
    def test_asset_roles_required(self):
        r,roots=self.fixture(include=True);r['environments']['paddle']['asset_roles']=[]
        with self.assertRaises(ContractError):a.capacity(r,roots)
    def test_four_billion_boundary_and_old_passed(self):
        r,roots=self.fixture(('layout','formula'),True)
        self.change_count(r,roots,'tele',4_000_000_000-23199744-4);self.assertEqual(a.capacity(r,roots),4_000_000_000)
        self.change_count(r,roots,'tele',4_000_000_001-23199744-4)
        with self.assertRaises(ContractError):a.capacity(r,roots)
        self.change_count(r,roots,'tele',4_000_000_000-4)
        with self.assertRaises(ContractError):a.capacity(r,roots)
    def test_load_boundary_verifies_aux_even_asset_only(self):
        r,roots=self.fixture(include=True,role=None)
        r['assets']['parameter_evidence']={'files':{'fixture.json':'0'*64}}
        r['environments']['native']={'resolved_python':str(Path(a.sys.executable).resolve()),'packages':{},'image_files':{},'asset_roles':['parameter_evidence']}
        seen=[]
        with patch.object(a,'verify_tree',side_effect=lambda root,files:seen.append((str(root),files))),patch.object(a,'capacity',return_value=23199747):
            a.load_boundary(r,'native')
        self.assertTrue(any(path.endswith('auxiliary_models') and files==r['assets']['auxiliary_models']['files'] for path,files in seen))
        with patch.object(a,'verify_tree',side_effect=ContractError('tampered aux')),self.assertRaises(ContractError):a.load_boundary(r,'native')
    def test_requirements_publishes_exact_identity(self):
        from .asset_binder import requirements
        out=self.root/'requirements.json';r=requirements(out);self.assertEqual(r['chardet_capacity']['counts']['unique_trainable'],23199744)
if __name__=='__main__':unittest.main()
