"""Check the container's sole visible CUDA device before loading any model."""
import ctypes
import json
import os
import uuid

def verify_device(out):
    result={'expected_uuid':os.environ['CUDA_VISIBLE_DEVICES'],'calls':[]}
    cuda=ctypes.CDLL('libcuda.so.1')
    def call(name,*args):
        rc=getattr(cuda,name)(*args);result['calls'].append({'call':name,'return_code':rc})
        if rc:
            text=ctypes.c_char_p()
            if cuda.cuGetErrorString(rc,ctypes.byref(text))==0 and text.value:result['error']=text.value.decode()
            (out/'ACTUAL_GPU_DEVICE.json').write_text(json.dumps(result,indent=2))
            raise RuntimeError(str(result))
    call('cuInit',0);count=ctypes.c_int();call('cuDeviceGetCount',ctypes.byref(count));result['device_count']=count.value
    assert count.value==1
    device=ctypes.c_int();call('cuDeviceGet',ctypes.byref(device),0)
    class UUID(ctypes.Structure):_fields_=[('bytes',ctypes.c_ubyte*16)]
    value=UUID();call('cuDeviceGetUuid_v2' if hasattr(cuda,'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid',ctypes.byref(value),device)
    result['actual_uuid']='GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)))
    assert result['actual_uuid']==result['expected_uuid']
    (out/'ACTUAL_GPU_DEVICE.json').write_text(json.dumps(result,indent=2))
