"""CPU tests of production V32 routing and observation. No model construction."""
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import tempfile
from unittest.mock import patch
from PIL import Image
from .core import ContractError, canonical, write
from .request_binding import BindingGraph
from .v32_dispatch import Dispatcher
from .v32_observer import EngineObserver
from .v32_protocol import CONFIG_SHA, IDENTITY, ContentRejected, validate
from .v32_parameters import enumerate_model
from .v32_cluster import BASE, PREPARED, PYTHON, descriptor, mount_evidence

def request():
    return {'run_id':'r','page_id':'p','slot_id':'p:native:0','request_id':'r/p/text/1','input_sha256':'a'*64,
            'crop':{'mode':'RGB'},'png_sha256':'b'*64,'model_sha256':'c'*64,'config_sha256':CONFIG_SHA,'source_sha256':'d'*64}

def response(q,text=' readable '):
    tensors={k:{'dtype':'float32','shape':[1],'device':'cpu','bytes_sha256':'f'*64} for k in ('pixel_values','image_grid_thw')}
    payload={'prompt_token_ids':[1,2],'mm_features':[{'modality':'image','tensors':tensors}]}
    ev={'identity':{k:q[k] for k in IDENTITY},'submitted':True,'request_id':'internal',
        'external_req_id':'external','output_request_id':'external','payload':payload,'payload_sha256':canonical(payload),
        'prompt_length':2,'prompt_ids_sha256':canonical([1,2])}
    return {**q,'raw_text':text,'engine_evidence':ev,'generation':{'finished':True,'finish_reason':'stop',
        'stop_reason':None,'token_ids':[7,248044],'eos_token_id':248044,'config_sha256':CONFIG_SHA}}

def cluster_env():
    root='/data/test/v32-ovis-text/runtime-003';venv=root+'/venv'
    dirs=[venv,'/usr/lib/python3.12','/usr/lib/gcc/x86_64-linux-gnu/11','/usr/include']
    mounts=[{'source':p,'destination':p,'read_only':True,'type':'directory'} for p in dirs]
    mounts.append({'source':'/usr/bin/python3.12','destination':'/usr/bin/python3.12','read_only':True,'type':'file','sha256':PYTHON})
    files={'/usr/bin/python3.12':PYTHON};packages={'torch':'2.11.0','vllm':'0.22.1','transformers':'5.5.1'}
    d={'schema':1,'base_image':BASE,'runtime_root':root,'prepared':{'path':root+'/PREPARED.json','sha256':PREPARED,'installed_files':82431},
       'mounts':mounts,'files':files,'links':{},'packages':packages,'include_system_site_packages':False}
    return {'kind':'cluster_ro_bind','image':BASE,'python':venv+'/bin/python','resolved_python':'/usr/bin/python3.12',
            'site_packages':venv+'/lib/python3.12/site-packages','image_files':files,'image_links':{},'packages':packages,
            'host_environment_binding':{'path':'/data/test/binding.json','sha256':'f'*64,'descriptor_sha256':canonical(d),'descriptor':d}}

class Checks(unittest.TestCase):
    def route(self,mode='on',category='text',text='good',error=None):
        image=Image.new('RGB',(2,2));graph=BindingGraph('r',{'page_id':'p','file_sha256':'a'*64},'c'*64,lambda x:x)
        record=graph.register(image,'content','p:native:0');record['category']=category;calls=[]
        def native(images,*args,**kwargs):
            self.assertIs(images[0],image);calls.append('tele');record.update(stage='completed',raw_content='native');return ['native']
        def expert(im,r):
            self.assertIs(im,image);calls.append('ovis')
            if error:raise error
            q=request();q['crop']=record['crop'];return q,response(q,text)
        backend=NS(batch_predict=native);d=Dispatcher(graph,backend,mode,NS(text=expert))
        result=backend.batch_predict([image],['prompt'],[None],None)
        self.assertEqual(record['stage'],'completed');return calls,result,record,d
    def test_text_success_never_native(self):
        calls,result,r,d=self.route();self.assertEqual((calls,result),(['ovis'],['good']))
        self.assertEqual(r['actual_content_source'],'ovis');self.assertFalse(d.events[0]['native_attempted'])
    def test_exact_type_non_targets_off_pass(self):
        for mode,kind in [('off','text'),('pass-through','text'),('on','title'),('on','table'),('on','equation'),('on','Text')]:
            with self.subTest(mode=mode,kind=kind):self.assertEqual(self.route(mode,kind)[0],['tele'])
    def test_content_reject_one_native_same_object(self):
        for text in ['   ','<table><tr><td>1</td></tr></table>','![x](x.png)']:
            self.assertEqual(self.route(text=text)[0],['ovis','tele'])
    def test_system_errors_never_fallback(self):
        for error in [ContractError('identity'),TimeoutError('deadline'),RuntimeError('worker')]:
            with self.assertRaises(type(error)):self.route(error=error)
    def test_raw_eos_never_appended(self):
        q=request();r=response(q);r['generation']['token_ids']=[7]
        with self.assertRaises(ContractError):validate(q,r)
        self.assertEqual(r['generation']['token_ids'],[7])
    def test_length_only_named_content_failure(self):
        q=request();r=response(q);r['generation'].update(token_ids=[7],finish_reason='length')
        with self.assertRaises(ContentRejected):validate(q,r)
        r['run_id']='other'
        with self.assertRaises(ContractError) as c:validate(q,r)
        self.assertNotIsInstance(c.exception,ContentRejected)
    def observer(self,data=True,length=2):
        tensor={'dtype':'float32','shape':[1],'device':'cpu','bytes_sha256':'e'*64}
        f=NS(modality='image',identifier='im',mm_hash='hash',mm_position={'offset':1},
             data=NS(get_data=lambda:{'pixel_values':tensor,'image_grid_thw':tensor}) if data else None)
        q=NS(request_id='external',external_req_id=None,prompt_token_ids=[1]*length,mm_features=[f],
             sampling_params=NS(max_tokens=16384,temperature=0,n=1,ignore_eos=False,stop=None,stop_token_ids=[]),
             pooling_params=None,prompt_embeds=None)
        submitted=[];processed=[]
        def process(*args,**kwargs):processed.append(q);return q
        engine=NS(input_processor=NS(process_inputs=process),engine_core=NS(add_request=lambda r:submitted.append(r)))
        o=EngineObserver(NS(llm_engine=engine),CONFIG_SHA);o.begin(request())
        return o,q,processed,submitted
    def enqueue(self,o,q):
        self.assertIs(o.process_inputs(),q);q.external_req_id=q.request_id;q.request_id='internal';o.add_request(q)
        return o.finish(NS(request_id='external',prompt_token_ids=q.prompt_token_ids))
    def test_observed_same_object_and_id_mapping(self):
        o,q,p,s=self.observer();r=self.enqueue(o,q)
        self.assertIs(s[0],q);self.assertEqual(len(p),1);self.assertEqual(r['request_id'],'internal')
    def test_length_gate_before_enqueue(self):
        o,q,p,s=self.observer(length=16385)
        with self.assertRaises(ContractError):o.process_inputs()
        self.assertEqual(len(p),1);self.assertEqual(s,[])
    def test_unobserved_or_mutated_input_rejected(self):
        o,q,p,s=self.observer()
        with self.assertRaises(ContractError):o.add_request(q)
        o.process_inputs();q.external_req_id=q.request_id;q.request_id='internal';q.prompt_token_ids.append(99)
        with self.assertRaises(ContractError):o.add_request(q)
        self.assertEqual(s,[])
    def test_cache_requires_observed_successful_submission(self):
        o,q,p,s=self.observer(data=False)
        with self.assertRaises(ContractError):o.process_inputs()
        o,q,p,s=self.observer();self.enqueue(o,q);q.request_id='next';q.external_req_id=None;q.mm_features[0].data=None
        o.begin(request());o.process_inputs();q.external_req_id='next';q.request_id='internal2';o.add_request(q)
        r=o.finish(NS(request_id='next',prompt_token_ids=q.prompt_token_ids))
        self.assertTrue(r['payload']['mm_features'][0]['cache_hit']);self.assertEqual(r['payload']['mm_features'][0]['reference'],'internal')
    def test_missing_tensor_rejected(self):
        o,q,p,s=self.observer();q.mm_features[0].data=NS(get_data=lambda:{'pixel_values':{}})
        with self.assertRaises(ContractError):o.process_inputs()
    def test_cluster_mounts_and_tamper(self):
        env=cluster_env();d=descriptor(env)
        state={'Id':'cid','Mounts':[{'Source':m['source'],'Destination':m['destination'],'RW':False} for m in d['mounts']]+[
            {'Source':'/data/test/binding.json','Destination':'/cluster-environment.json','RW':False}]}
        self.assertEqual(mount_evidence(state,env)['cid'],'cid');state['Mounts'][0]['RW']=True
        with self.assertRaises(ContractError):mount_evidence(state,env)
        env['host_environment_binding']['descriptor']['mounts'][0]['source']='/usr/lib'
        with self.assertRaises(ContractError):descriptor(env)
    def test_broad_host_mount_disallowed_even_rehashed(self):
        env=cluster_env();d=env['host_environment_binding']['descriptor'];d['mounts'][0].update(source='/usr/lib',destination='/usr/lib')
        env['host_environment_binding']['descriptor_sha256']=canonical(d)
        with self.assertRaises(ContractError):descriptor(env)
    def test_real_storage_overlap_and_buffers(self):
        import numpy as np
        a=np.arange(10,dtype=np.float32)
        class Tensor:
            device='cpu';dtype='float32';requires_grad=False
            def __init__(self,lo,hi):self.array=a[lo:hi];self.shape=self.array.shape;self.lo=lo
            def is_contiguous(self):return self.array.flags.c_contiguous
            def numel(self):return self.array.size
            def element_size(self):return self.array.itemsize
            def storage_offset(self):return self.lo
            def untyped_storage(self):return NS(data_ptr=lambda:a.ctypes.data,nbytes=lambda:a.nbytes)
        model=NS(named_parameters=lambda **kw:[('a',Tensor(0,6)),('alias',Tensor(0,6)),('slice',Tensor(4,10))],
                 named_buffers=lambda **kw:[('buffer',Tensor(1,3))])
        r=enumerate_model(model);self.assertEqual(r['unique_trainable'],10);self.assertEqual(r['stored_elements'],18);self.assertEqual(r['buffers'],2)
    def test_cumulative_deadline_cannot_reset(self):
        from .v32_cluster import bound_budget
        r={'v32':{'host_absolute_deadline':1000}}
        self.assertEqual(bound_budget(r,{'total_seconds':900,'cleanup_seconds':240},500)['total_seconds'],500)
        with self.assertRaises(ContractError):bound_budget(r,{'total_seconds':900,'cleanup_seconds':240},800)
    def test_bootstrap_rejected_for_pages(self):
        from .v32_binding import capacity
        r={'v32':{'parameter_status':'bootstrap_header','execution':'pages','header_stored_elements':852985920},
           'parameter_ledger':[],'assets':{'ovis_model':{}}}
        with patch('hybrid_v3_full_eval_v1.asset_binding.capacity',return_value=10):
            with self.assertRaises(ContractError):capacity(r,{})
    def test_postprocess_final_adoption_and_tamper(self):
        from .v32_dispatch import geometry,assembly,verify
        calls,output,r,d=self.route();raw=[{'type':'text','bbox':[0,0,1,1],'content':output[0]}]
        d.before=geometry(raw);post_calls=[]
        def post(blocks):post_calls.append(1);return blocks
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);d.post(raw,post,folder)
            self.assertEqual(post_calls,[1]);write(folder/'REQUEST_BINDINGS.json',[r])
            md=folder/'p.md';md.write_text(output[0]);middle=folder/'p_middle.json';write(middle,raw)
            write(folder/'V32_ASSEMBLY.json',assembly(folder,md,middle));self.assertTrue(verify(folder))
            md.write_text('tampered')
            with self.assertRaises(ContractError):verify(folder)
    def test_geometry_and_mixed_evidence_rejected(self):
        from .v32_dispatch import geometry,verify
        calls,output,r,d=self.route();raw=[{'type':'text','bbox':[0,0,1,1],'content':output[0]}]
        d.before=geometry(raw);raw[0]['bbox']=[1,1,2,2]
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ContractError):d.post(raw,lambda b:b,temp)
            write(Path(temp)/'V31_TRANSACTIONS.json',[])
            with self.assertRaises(ContractError):verify(temp)

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    stream=io.StringIO();r=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (out/'checks.log').write_text(stream.getvalue());write(out/'CPU_CHECKS.json',{'tests':r.testsRun,'passed':r.wasSuccessful(),
        'failures':len(r.failures),'errors':len(r.errors),'model_loads':0,'gpu_runs':0,'storage_fixture':'real NumPy shared storage; not a Torch runtime acceptance'})
    print(stream.getvalue());return 0 if r.wasSuccessful() else 1

if __name__=='__main__':raise SystemExit(main())
