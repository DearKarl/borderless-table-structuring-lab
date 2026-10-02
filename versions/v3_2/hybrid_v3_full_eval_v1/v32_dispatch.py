"""Exact text routing at the original prepared-crop batch boundary."""
import copy
from pathlib import Path
from .core import ContractError, canonical, read, sha, write
from .request_binding import rgb_binding
from .v32_protocol import ContentRejected, validate

def geometry(blocks):return [{k:v for k,v in dict(b).items() if k!='content'} for b in blocks]

def assembly(folder,md,middle):
    folder=Path(folder);value=read(folder/'V32_CONTENT.json');tree=read(middle);markdown=Path(md).read_text(encoding='utf-8')
    def matches(node,text,path=()):
        if isinstance(node,dict):return [p for k,v in node.items() for p in matches(v,text,path+(k,))]
        if isinstance(node,list):return [p for k,v in enumerate(node) for p in matches(v,text,path+(k,))]
        return [list(path)] if isinstance(node,str) and text and text in node else []
    adopted=[]
    for e in value['events']:
        if e['actual_content_source']!='ovis':continue
        index=int(e['slot_id'].rsplit(':',1)[1]);raw=value['raw_blocks'][index]
        candidates=[(i,b) for i,b in enumerate(value['post_blocks']) if geometry([b])==geometry([raw])]
        if len(candidates)!=1:raise ContractError('Ovis post-process location ambiguous')
        i,b=candidates[0];text=b.get('content');locations=matches(tree,text)
        if not text or not locations or text not in markdown:raise ContractError('Ovis final middle/Markdown adoption unproven')
        adopted.append({'slot_id':e['slot_id'],'post_index':i,'post_content':text,'middle_paths':locations,
                        'markdown_offset':markdown.index(text),'raw_to_post':'original frozen post_process'})
    return {'content_sha256':sha(folder/'V32_CONTENT.json'),
            'files':{Path(p).relative_to(folder).as_posix():sha(p) for p in (md,middle)},
            'actual_adoptions':adopted,'quality_claim':False}

class Dispatcher:
    def __init__(self,graph,backend,mode,expert):
        self.graph,self.backend,self.mode,self.expert=graph,backend,mode,expert
        self.original=backend.batch_predict;self.events=[];self.before=None;self.calls=0
        if mode!='off':backend.batch_predict=self.batch
    def prepare(self,blocks,returned):
        if self.before is not None:raise ContractError('Repeated V32 preparation')
        self.before=copy.deepcopy(geometry(blocks))
        for image,index in zip(returned[0],returned[3]):
            self.graph.objects[id(image)][1]['category']=blocks[index].type
    def batch(self,images,prompts='',sampling_params=None,priority=None,**kwargs):
        from collections.abc import Sequence
        records=[self.graph.objects.get(id(im)) for im in images]
        if not records or any(p is None or p[0] is not im for im,p in zip(images,records)):
            raise ContractError('Unbound V32 dispatch image')
        if all(p[1]['kind']=='layout' for p in records):return self.original(images,prompts,sampling_params,priority,**kwargs)
        if any(p[1]['kind']!='content' for p in records):raise ContractError('Mixed V32 dispatch kinds')
        def item(value,i):return value[i] if isinstance(value,Sequence) and not isinstance(value,str) else value
        outputs=[]
        for i,(image,pair) in enumerate(zip(images,records)):
            r=pair[1]
            if r['stage']!='prepared' or r['crop']!=rgb_binding(image):raise ContractError('Changed/reused V32 slot')
            event={'slot_id':r['slot_id'],'request_id':r['request_id'],'category':r['category'],
                   'actual_content_source':'tele','protocol_accepted':False,'quality_selected':False,
                   'native_attempted':False,'native_primary':{'status':'not_run'}}
            if self.mode=='on' and r['category']=='text':
                if self.expert is None:raise ContractError('Ovis service missing')
                self.calls+=1
                request,response=self.expert.text(image,r)
                event.update(ovis_request=request,ovis_response=response)
                try:cleaned=validate(request,response)
                except ContentRejected as exc:event['rejection']=str(exc)
                else:
                    self.graph.external_completion(image,request,response,cleaned)
                    event.update(actual_content_source='ovis',protocol_accepted=True,selected_content=cleaned)
                    self.events.append(event);outputs.append(cleaned);continue
            result=self.original([image],item(prompts,i),item(sampling_params,i),item(priority,i),**kwargs)
            if len(result)!=1 or r['stage']!='completed' or r['raw_content']!=result[0]:raise ContractError('Native fallback binding differs')
            event.update(native_attempted=True,native_primary={'status':'executed','raw_content':result[0]},selected_content=result[0])
            self.events.append(event);outputs.append(result[0])
        return outputs
    def post(self,blocks,post,folder):
        if geometry(blocks)!=self.before:raise ContractError('V32 changed geometry/type/order')
        raw=copy.deepcopy([dict(b) for b in blocks])
        if self.mode=='off':
            for r in self.graph.slots.values():
                self.events.append({'slot_id':r['slot_id'],'request_id':r['request_id'],'category':r['category'],
                    'actual_content_source':'tele','protocol_accepted':False,'quality_selected':False,'native_attempted':True,
                    'native_primary':{'status':'executed','raw_content':r['raw_content']},'selected_content':r['raw_content']})
        for index,b in enumerate(raw):
            r=self.graph.slots.get(self.graph.page['page_id']+':native:'+str(index))
            if r is not None and (r['stage']!='completed' or r['raw_content']!=b.get('content')):raise ContractError('V32 final raw assignment differs')
        result=post(blocks)
        write(Path(folder)/'V32_CONTENT.json',{'mode':self.mode,'prepared_geometry':self.before,'raw_blocks':raw,
            'post_blocks':copy.deepcopy([dict(b) for b in result]),'events':self.events,'original_post_calls':1})
        return result
    def restore(self):self.backend.batch_predict=self.original

def verify(folder):
    folder=Path(folder)
    if any((folder/n).exists() for n in ('V31_TRANSACTIONS.json','PRIMARY_CONTENT.json','V31_FINAL_RAW.json')):
        raise ContractError('Mixed V31/V32 page evidence')
    value=read(folder/'V32_CONTENT.json');requests=read(folder/'REQUEST_BINDINGS.json')
    raw=value['raw_blocks'];events=value['events'];bindings={r['slot_id']:r for r in requests if r['kind']=='content'}
    if value['original_post_calls']!=1 or geometry(raw)!=value['prepared_geometry']:raise ContractError('V32 geometry/post evidence differs')
    if len(events)!=len(bindings) or len({e['slot_id'] for e in events})!=len(events):raise ContractError('V32 slot audit coverage differs')
    for e in events:
        r=bindings[e['slot_id']];index=int(e['slot_id'].rsplit(':',1)[1])
        if r['stage']!='completed' or e['request_id']!=r['request_id'] or r['raw_content']!=e['selected_content'] or raw[index].get('content')!=r['raw_content']:
            raise ContractError('V32 content/request chain differs')
        if e['quality_selected'] is not False:raise ContractError('V32 cannot assert quality selection')
        if e['actual_content_source']=='ovis':
            if (value['mode']!='on' or e['category']!='text' or e['native_attempted'] or e['native_primary']!={'status':'not_run'}
                or e['protocol_accepted'] is not True or r.get('actual_content_source')!='ovis'
                or validate(e['ovis_request'],e['ovis_response'])!=e['selected_content']
                or r.get('external_completion')!=e['ovis_response']):raise ContractError('V32 external completion differs')
        elif e['actual_content_source']=='tele':
            if not e['native_attempted'] or not r.get('processor_tensors') or not r.get('termination'):raise ContractError('V32 native trace missing')
            if 'ovis_response' in e:
                try:validate(e['ovis_request'],e['ovis_response'])
                except ContentRejected as exc:
                    if str(exc)!=e.get('rejection'):raise ContractError('V32 fallback reason changed')
                else:raise ContractError('Successful Ovis incorrectly fell back')
        else:raise ContractError('Unknown V32 content source')
    assembly=read(folder/'V32_ASSEMBLY.json')
    if assembly['content_sha256']!=sha(folder/'V32_CONTENT.json'):raise ContractError('V32 assembly content binding differs')
    for rel,h in assembly['files'].items():
        from .core import closed
        if sha(closed(folder,rel))!=h:raise ContractError('V32 final assembly changed')
    files=list(assembly['files'])
    md=next(n for n in files if n.endswith('.md'));middle=next(n for n in files if n.endswith('_middle.json'))
    if globals()['assembly'](folder,folder/md,folder/middle)!=assembly:raise ContractError('V32 final adoption chain differs')
    return True
