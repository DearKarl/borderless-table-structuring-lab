"""Fresh UUID -> numeric index, with CUDA driver identity proof before model load."""
import csv
import ctypes
import io
import os
import re
import subprocess
import uuid

def bus_key(value):
    match=re.fullmatch(r'([0-9a-fA-F]+):([0-9a-fA-F]+):([0-9a-fA-F]+)\.([0-9a-fA-F]+)',value)
    if not match:raise RuntimeError('invalid PCI bus ID')
    return tuple(int(x,16) for x in match.groups())

def parse_snapshot(text,selected_uuid):
    rows=[];seen_index=set();seen_uuid=set()
    for fields in csv.reader(io.StringIO(text)):
        if len(fields)!=6:raise RuntimeError('unexpected GPU snapshot schema')
        index,identity,bus,name,total,used=[x.strip() for x in fields]
        if not index.isdigit() or not re.fullmatch(r'GPU-[0-9a-fA-F-]{36}',identity):
            raise RuntimeError('invalid GPU identity')
        bus_key(bus)
        if int(index) in seen_index or identity in seen_uuid:raise RuntimeError('ambiguous GPU mapping')
        seen_index.add(int(index));seen_uuid.add(identity)
        rows.append({'index':int(index),'uuid':identity,'pci_bus_id':bus,'name':name,
                     'memory_total_mib':int(total),'memory_used_mib':int(used)})
    selected=[r for r in rows if r['uuid']==selected_uuid]
    if len(selected)!=1:raise RuntimeError('requested UUID not uniquely present')
    row=selected[0]
    if 'A100' not in row['name'] or row['memory_total_mib']<24000 or row['memory_used_mib']!=0:
        raise RuntimeError('requested UUID not freshly idle A100')
    return {'selected':row,'all_devices':rows,'raw_nvidia_smi':text}

def fresh_snapshot(selected_uuid):
    text=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name,memory.total,memory.used',
                         '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20).stdout
    return parse_snapshot(text,selected_uuid)

def numeric_environment(env,selected,output):
    if env.get('CUDA_VISIBLE_DEVICES')!=selected['uuid']:
        raise RuntimeError('original worker UUID binding mismatch')
    return dict(env,CUDA_VISIBLE_DEVICES=str(selected['index']),CUDA_DEVICE_ORDER='PCI_BUS_ID',
                V32_EXPECTED_GPU_UUID=selected['uuid'],V32_EXPECTED_PCI_BUS=selected['pci_bus_id'],
                V32_EXPECTED_GPU_INDEX=str(selected['index']),V32_GPU_EVIDENCE=str(output))

def actual_cuda_identity():
    # Driver API enumerates visibility in this very worker process, without
    # importing torch/vLLM, constructing a model, or allocating a CUDA context.
    cuda=ctypes.CDLL('libcuda.so.1')
    def call(name,argtypes,*args):
        fn=getattr(cuda,name);fn.argtypes=argtypes;fn.restype=ctypes.c_int
        code=fn(*args)
        if code!=0:raise RuntimeError(name+' failed: '+str(code))
    call('cuInit',[ctypes.c_uint],0)
    count=ctypes.c_int();call('cuDeviceGetCount',[ctypes.POINTER(ctypes.c_int)],ctypes.byref(count))
    if count.value!=1:raise RuntimeError('exactly one visible CUDA device required')
    device=ctypes.c_int();call('cuDeviceGet',[ctypes.POINTER(ctypes.c_int),ctypes.c_int],ctypes.byref(device),0)
    raw_uuid=(ctypes.c_ubyte*16)()
    uuid_api='cuDeviceGetUuid_v2' if hasattr(cuda,'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid'
    call(uuid_api,[ctypes.c_void_p,ctypes.c_int],ctypes.byref(raw_uuid),device.value)
    bus=ctypes.create_string_buffer(64)
    call('cuDeviceGetPCIBusId',[ctypes.c_void_p,ctypes.c_int,ctypes.c_int],bus,64,device.value)
    return {'count':count.value,'logical_device':0,'uuid':'GPU-'+str(uuid.UUID(bytes=bytes(raw_uuid))),
            'pci_bus_id':bus.value.decode('ascii'),'uuid_api':uuid_api,
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
            'CUDA_DEVICE_ORDER':os.environ.get('CUDA_DEVICE_ORDER')}

def validate_identity(actual,expected_uuid,expected_bus,expected_index):
    if actual['count']!=1 or actual['logical_device']!=0:raise RuntimeError('visible device count/index mismatch')
    if actual['CUDA_DEVICE_ORDER']!='PCI_BUS_ID':raise RuntimeError('CUDA_DEVICE_ORDER mismatch')
    if actual['CUDA_VISIBLE_DEVICES']!=str(expected_index):raise RuntimeError('numeric visibility mismatch')
    if actual['uuid'].lower()!=expected_uuid.lower():raise RuntimeError('visible CUDA UUID mismatch; no model load')
    if bus_key(actual['pci_bus_id'])!=bus_key(expected_bus):raise RuntimeError('visible PCI bus mismatch; no model load')
    return actual
