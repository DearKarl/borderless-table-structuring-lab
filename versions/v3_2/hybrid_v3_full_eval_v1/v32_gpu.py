"""Container-local NVML index followed by actual CUDA UUID/PCI proof."""
import os
from .core import ContractError, write
from .v32_gpu_identity_source import actual_cuda_identity, validate_identity

def adapt(control,out):
    import pynvml
    expected=control['runtime']['gpu_uuids'][-1]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=expected:raise ContractError('Ovis initial UUID visibility mismatch')
    pynvml.nvmlInit()
    try:
        handle=pynvml.nvmlDeviceGetHandleByUUID(expected)
        index=int(pynvml.nvmlDeviceGetIndex(handle));uid=pynvml.nvmlDeviceGetUUID(handle)
        pci=pynvml.nvmlDeviceGetPciInfo(handle).busId
        if isinstance(uid,bytes):uid=uid.decode()
        if isinstance(pci,bytes):pci=pci.decode()
        if uid.lower()!=expected.lower():raise ContractError('NVML UUID mismatch')
    finally:pynvml.nvmlShutdown()
    os.environ['CUDA_VISIBLE_DEVICES']=str(index);os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    actual=validate_identity(actual_cuda_identity(),expected,pci,index)
    value={'nvml':{'uuid':uid,'index':index,'pci_bus_id':pci},'cuda':actual}
    write(out/'experts/GPU_NUMERIC_MAPPING.json',value)
    return value
