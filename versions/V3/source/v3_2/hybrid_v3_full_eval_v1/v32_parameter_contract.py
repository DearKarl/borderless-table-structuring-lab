"""Fixed Ovis header/fusion/alias/buffer contract. Pure stdlib; never learns from a new run."""
import hashlib
import json
import math
from pathlib import Path
from .core import ContractError,canonical

CONTRACT_SHA='1aee939dba621ac7f602c65b84fc21f009813f8c0bcd156c4fde14d6f22bf71a'
HEADER_SHA='dd13611d0977715c509ca38263376f195e516ddd1f64758652e86b5ba6abd091'
ROW_FIELDS={'name','shape','dtype','device','numel','storage_address','storage_bytes','storage_offset','byte_interval','requires_grad'}
SIZES={'torch.bfloat16':2,'torch.float32':4}
def require(ok,message):
    if not ok:raise ContractError('Ovis parameter mapping: '+message)
def integer(value,minimum=0):return type(value) is int and value>=minimum
def fixed_json(name,digest,cap):
    path=Path(__file__).with_name(name);raw=path.read_bytes()
    require(len(raw)<=cap and hashlib.sha256(raw).hexdigest()==digest,'fixed data SHA changed: '+name)
    return json.loads(raw)
def load_contract():
    return fixed_json('v32_parameter_contract.json',CONTRACT_SHA,512*1024),fixed_json('v32_parameter_header.json',HEADER_SHA,128*1024)
def inventory(actual,fixed,label):
    require(isinstance(actual,dict) and set(actual)=={'rows','stored_elements','unique_elements','aliases','structure_sha256'},label+' inventory fields differ')
    rows=actual['rows'];require(isinstance(rows,list) and len(rows)==len(fixed['stable_rows']),label+' row count differs')
    names=set();groups={};stable=[];stored=0
    for row in rows:
        require(isinstance(row,dict) and set(row)==ROW_FIELDS,label+' row fields differ')
        name=row['name'];shape=row['shape'];dtype=row['dtype']
        require(isinstance(name,str) and name not in names,label+' duplicate/invalid name');names.add(name)
        require(isinstance(shape,list) and all(integer(v,1) for v in shape),label+' invalid shape')
        require(isinstance(dtype,str) and dtype in SIZES,label+' unsupported dtype')
        require(integer(row['numel'],1) and row['numel']==math.prod(shape),label+' shape/numel differs')
        require(type(row['requires_grad']) is bool,label+' requires_grad type differs')
        require(row['device']=='cuda:0' and integer(row['storage_address'],1),label+' device/address differs')
        size=SIZES[dtype];length=row['numel']*size
        require(type(row['storage_offset']) is int and row['storage_offset']==0,label+' nonzero offset')
        require(isinstance(row['byte_interval'],list) and len(row['byte_interval'])==2 and all(integer(v) for v in row['byte_interval']) and row['byte_interval']==[0,length],label+' incomplete byte interval')
        require(integer(row['storage_bytes'],1) and row['storage_bytes']==length,label+' storage padding/size differs')
        key=(row['device'],row['storage_address'])
        group=groups.setdefault(key,{'dtype':dtype,'size':size,'storage_bytes':length,'names':[],'intervals':[]})
        require((group['dtype'],group['storage_bytes'])==(dtype,length),label+' partial/mixed storage sharing')
        group['names'].append(name);group['intervals'].append(list(row['byte_interval']));stored+=row['numel']
        stable.append({k:v for k,v in row.items() if k not in ('device','storage_address','storage_bytes')})
    aliases=[g['names'] for g in groups.values()];unique=0;unique_bytes=0
    for group in groups.values():
        merged=[]
        for lo,hi in sorted(group['intervals']):
            if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
            else:merged.append([lo,hi])
        total=sum(hi-lo for lo,hi in merged);require(total%group['size']==0,label+' unaligned union')
        unique+=total//group['size'];unique_bytes+=total
    structure=canonical({'rows':stable,'aliases':aliases})
    require(stable==fixed['stable_rows'],label+' stable rows differ from fixed contract')
    require(aliases==fixed['aliases']==actual['aliases'],label+' alias groups differ')
    require(integer(actual['stored_elements']) and actual['stored_elements']==stored==fixed['stored_elements'],label+' named sum differs')
    require(integer(actual['unique_elements']) and actual['unique_elements']==unique==fixed['unique_elements'],label+' union count differs')
    require(structure==fixed['structure_sha256']==actual['structure_sha256'],label+' structure SHA differs')
    return {'rows':stable,'aliases':aliases,'groups':set(groups),'stored':stored,'unique':unique,'unique_bytes':unique_bytes,'structure_sha256':structure}
def header_mapping(header,contract,parameters):
    require(isinstance(header,dict) and len(header)==473,'header tensor set differs')
    intervals=[];elements=0
    for name,row in header.items():
        require(isinstance(name,str) and isinstance(row,dict) and set(row)=={'dtype','shape','data_offsets'},'header fields differ')
        shape=row['shape'];offsets=row['data_offsets']
        require(row['dtype']=='BF16' and isinstance(shape,list) and all(integer(v,1) for v in shape),'header dtype/shape differs')
        require(isinstance(offsets,list) and len(offsets)==2 and all(integer(v) for v in offsets),'header offset types differ')
        count=math.prod(shape);require(offsets[1]-offsets[0]==count*2,'header payload length differs');elements+=count;intervals.append(offsets)
    cursor=0
    for lo,hi in sorted(intervals):require(lo==cursor,'header payload gap/overlap');cursor=hi
    require(elements==852985920 and cursor==1705971840,'header payload total differs')
    rows={r['name']:r for r in parameters['rows']};used=[];loaded=[];categories={};exceptions=set(contract['fp32_exceptions'])
    require(len(exceptions)==18,'FP32 exception count differs')
    mapping=contract['mapping'];require(isinstance(mapping,list) and len(mapping)==401,'mapping cardinality differs')
    for item in mapping:
        require(set(item)=={'loaded_name','header_names','axis','kind'},'mapping fields differ')
        name=item['loaded_name'];inputs=item['header_names']
        require(name in rows and name not in loaded and isinstance(inputs,list) and inputs and all(n in header for n in inputs),'mapping name/coverage differs')
        row=rows[name];shapes=[header[n]['shape'] for n in inputs]
        require(all(s[1:]==shapes[0][1:] for s in shapes),'fusion trailing dimensions differ')
        if len(inputs)==1:require(item['axis'] is None and row['shape']==shapes[0],'direct shape differs')
        else:require(item['axis']==0 and row['shape']==[sum(s[0] for s in shapes),*shapes[0][1:]],'axis-zero fused shape differs')
        require(row['numel']==sum(math.prod(s) for s in shapes),'fusion numel not conserved')
        require(row['dtype']==('torch.float32' if name in exceptions else 'torch.bfloat16'),'unapproved header/live dtype change')
        used.extend(inputs);loaded.append(name);categories[item['kind']]=categories.get(item['kind'],0)+1
    require(len(used)==len(set(used))==473 and set(used)==set(header),'header not consumed exactly once')
    require(loaded==[g[0] for g in parameters['aliases']] and categories==contract['mapping_counts'],'independent loaded coverage differs')
    alias=contract['parameter_alias'];require([g for g in parameters['aliases'] if len(g)>1]==[alias],'unique tied parameter alias differs')
    require(set(rows)-set(loaded)=={alias[1]},'unexplained loaded parameter name')
    a,b=(rows[n] for n in alias)
    require(all(a[k]==b[k] for k in ('shape','dtype','numel','storage_offset','byte_interval','requires_grad')),'tied parameter structure differs')
    require(a['shape']==[248320,1024] and a['dtype']=='torch.bfloat16','embedding/head shape differs')
    require({n for n,r in rows.items() if r['dtype']=='torch.float32'}==exceptions and sum(rows[n]['numel'] for n in exceptions)==288,'FP32 exception elements differ')
    return {'header_tensors':473,'independent_loaded_parameters':401,'named_loaded_parameters':402,'categories':categories,'header_elements':elements,'header_payload_bytes':cursor}
def validate_loaded_count(count):
    require(isinstance(count,dict),'count must be an object');contract,header=load_contract()
    require(count.get('method')=='loaded_unique_parameters','count method differs')
    require(type(count.get('header_stored_elements')) is int and count['header_stored_elements']==852985920 and type(count.get('header_tensor_count')) is int and count['header_tensor_count']==473,'header summary differs')
    require(isinstance(count.get('identity'),dict) and integer(count.get('pid'),1),'current count identity/pid missing')
    identity=count['identity'];require(set(identity)=={'run_id','load_id','model_sha256','source_sha256','environment_sha256'} and all(isinstance(v,str) and v for v in identity.values()),'current identity fields differ')
    p=inventory(count.get('parameters'),contract['parameters'],'parameters');b=inventory(count.get('buffer_inventory'),contract['buffer_inventory'],'buffers')
    require(not(p['groups']&b['groups']),'parameter/buffer storage overlap')
    exp=contract['expected'];combined=canonical([p['structure_sha256'],b['structure_sha256']])
    for key,value in [('stored_elements',p['stored']),('unique_trainable',p['unique']),('buffers',b['unique'])]:require(integer(count.get(key)) and count[key]==value,'top-level '+key+' differs')
    require(combined==exp['combined_structure_sha256']==count.get('structure_sha256'),'combined structure differs')
    require(len(p['groups'])==401 and len(b['groups'])==26 and p['unique_bytes']==1705972416 and b['unique_bytes']==134742112,'storage byte/group totals differ')
    require(sum(r['requires_grad'] for r in p['rows'])==218 and sum(not r['requires_grad'] for r in p['rows'])==184,'frozen/trainable enumeration differs')
    require(all(r['requires_grad'] is False for r in b['rows']),'buffer requires_grad differs')
    coverage=header_mapping(header,contract,p)
    require(p['unique']==coverage['header_elements'] and p['unique_bytes']==coverage['header_payload_bytes']+288*2,'parameter element/dtype byte relation differs')
    byname={r['name']:r for r in b['rows']};classes=contract['buffer_classes'];classified=[]
    for key,item in classes.items():
        names=item['names'];require(all(n in byname and byname[n]['shape']==item['shape'] for n in names),'buffer class differs');classified+=names
        expected_dtype='torch.float32' if key=='attention_scales' else 'torch.bfloat16'
        require(all(byname[n]['dtype']==expected_dtype for n in names),'buffer class dtype differs')
    require(len(classified)==len(set(classified))==31 and set(classified)==set(byname),'buffer class coverage differs')
    require(b['unique']==8192*32+1048576*64+6*4,'buffer construction total differs')
    return {'contract_sha256':CONTRACT_SHA,'header_sha256':HEADER_SHA,'source_provenance':contract['provenance'],
        'raw_canonical_sha256':canonical(count),'raw_hash_encoding':'core.canonical of the current unmodified RPC result',
        'current_identity':dict(identity),'current_pid':count['pid'],'coverage':coverage,'parameter_aliases':p['aliases'],'buffer_aliases':b['aliases'],
        'fp32_exceptions':contract['fp32_exceptions'],'buffer_classes':classes,'recomputed':{'stored_elements':p['stored'],'unique_trainable':p['unique'],'buffers':b['unique'],
        'parameter_unique_bytes':p['unique_bytes'],'buffer_unique_bytes':b['unique_bytes'],'parameter_structure_sha256':p['structure_sha256'],'buffer_structure_sha256':b['structure_sha256'],'structure_sha256':combined},
        'mapping_passed':True,'request_or_page_acceptance':False}
