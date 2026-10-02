"""Same-load worker.get_model parameter enumeration; contiguous storage interval union."""
import os
from .core import ContractError, canonical

def enumerate_model(model):
    def inventory(items):
        rows=[];groups={};stored=0
        for name,p in items:
            if not p.is_contiguous():raise ContractError('Noncontiguous parameter/buffer layout requires explicit mapping: '+name)
            n=int(p.numel());size=int(p.element_size());storage=p.untyped_storage()
            address=int(storage.data_ptr());offset=int(p.storage_offset());start=offset*size;end=start+n*size
            if start<0 or end>storage.nbytes():raise ContractError('Storage slice outside allocation')
            key=(str(p.device),address)
            if key in groups and groups[key]['dtype']!=str(p.dtype):raise ContractError('Mixed dtype storage alias')
            group=groups.setdefault(key,{'dtype':str(p.dtype),'size':size,'intervals':[],'names':[]})
            group['intervals'].append((start,end));group['names'].append(name);stored+=n
            rows.append({'name':name,'shape':list(p.shape),'dtype':str(p.dtype),'device':str(p.device),
                         'numel':n,'storage_address':address,'storage_bytes':int(storage.nbytes()),
                         'storage_offset':offset,'byte_interval':[start,end], 'requires_grad':bool(p.requires_grad)})
        unique=0
        for g in groups.values():
            merged=[]
            for lo,hi in sorted(g['intervals']):
                if merged and lo<=merged[-1][1]:merged[-1][1]=max(hi,merged[-1][1])
                else:merged.append([lo,hi])
            total=sum(hi-lo for lo,hi in merged)
            if total%g['size']:raise ContractError('Unaligned storage union')
            unique+=total//g['size']
        aliases=[g['names'] for g in groups.values()]
        stable=[{k:v for k,v in r.items() if k not in ('device','storage_address','storage_bytes')} for r in rows]
        return {'rows':rows,'stored_elements':stored,'unique_elements':unique,'aliases':aliases,
                'structure_sha256':canonical({'rows':stable,'aliases':aliases})}
    parameters=inventory(model.named_parameters(remove_duplicate=False))
    buffers=inventory(model.named_buffers(remove_duplicate=False))
    return {'parameters':parameters,'buffer_inventory':buffers,'unique_trainable':parameters['unique_elements'],
            'stored_elements':parameters['stored_elements'],'buffers':buffers['unique_elements'],
            'structure_sha256':canonical([parameters['structure_sha256'],buffers['structure_sha256']])}

def count_loaded_worker(worker,identity):
    from .v32_gpu_identity_source import actual_cuda_identity
    model=worker.get_model()
    result=enumerate_model(model)
    result['resource_audit']=worker.v32_resources.result()
    result.update(identity=identity,pid=os.getpid(),device=actual_cuda_identity(),method='loaded_unique_parameters',
                  header_stored_elements=852985920,header_tensor_count=473)
    return result
