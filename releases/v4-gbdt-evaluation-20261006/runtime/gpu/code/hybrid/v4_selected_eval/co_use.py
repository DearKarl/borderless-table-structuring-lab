"""Exact shared GPU query reuse; no old A/B state or runtime imports."""
from copy import deepcopy
import re
import time
import xml.etree.ElementTree as ET
from .dependencies import need as require
GPU0="GPU-PUBLIC-REQUIRES-BINDING-0"
def parse_co_use_xml(raw, uuids, version):
    """Narrow GPU0 parser: retain observed load/process fields, never fake idle."""
    require(uuids == [GPU0], 'Only the explicitly selected GPU0 UUID')
    require(len(raw.encode()) <= 1024**2 and '<!ENTITY' not in raw.upper(), 'Unsafe/large GPU XML')
    raw = re.sub(r'<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v[0-9]+\.dtd">', '', raw)
    require('<!DOCTYPE' not in raw.upper(), 'Unsupported GPU XML DTD')
    root = ET.fromstring(raw)
    require(root.findtext('driver_version') == version, 'XML driver version differs')
    found = {}
    def number(text, unit=''):
        require(isinstance(text, str), 'Missing GPU number')
        match = re.fullmatch(r'\s*([0-9]+)'+(r'\s*'+re.escape(unit) if unit else '')+r'\s*', text)
        require(match is not None, 'Unsupported GPU numeric field')
        return int(match.group(1))
    for gpu in root.findall('gpu'):
        uid = gpu.findtext('uuid')
        if uid != GPU0:
            continue
        require(uid not in found, 'Duplicate selected UUID')
        minor = number(gpu.findtext('minor_number'))
        require(0 <= minor < 255 and gpu.findtext('mig_mode/current_mig') == 'Disabled', 'GPU minor/MIG differs')
        memory = number(gpu.findtext('fb_memory_usage/used'), 'MiB')
        utilization = number(gpu.findtext('utilization/gpu_util'), '%')
        require(utilization <= 100, 'Invalid GPU utilization')
        processes = gpu.find('processes')
        require(processes is not None and not (processes.text or '').strip(), 'Missing/unknown GPU process evidence')
        rows = []
        for process in processes:
            require(process.tag == 'process_info', 'Unknown GPU process record')
            row = {field.tag:field.text for field in process}
            require(len(row) == len(process), 'Duplicate process evidence field')
            require(number(row.get('pid')) > 0, 'Invalid observed process PID')
            rows.append(row)
        found[uid] = dict(uuid=uid, minor=minor, memory_mib=memory, utilization=utilization,
                          mig='Disabled', processes=rows, driver_version=version)
    require(set(found) == {GPU0}, 'Selected UUID missing from real XML')
    return found

def install_co_use_query(backend, amendment_sha):
    require(not getattr(backend, '_nju_gpu0_co_use', False), 'Do not re-install GPU amendment')
    first_mapping = []
    def query(uuids, ceiling, version, deadline, runner=None):
        require(uuids == [GPU0] and ceiling == 0, 'Unexpected co-use query scope')
        runner = runner or backend.subprocess.run
        result = runner(['nvidia-smi','-q','-x'], check=True, text=True, capture_output=True, timeout=deadline.bound(10))
        observed = parse_co_use_xml(result.stdout, uuids, version)
        gpu = observed[GPU0]
        require(backend.kernel_version() == version, 'Kernel driver changed')
        devices = backend.device_stats(gpu['minor'])
        backend.validate_devices(devices, gpu['minor'])
        mapping = dict(minor=gpu['minor'], devices=devices, driver_version=version)
        if first_mapping:
            require(mapping == first_mapping[0], 'GPU mapping changed across acquire/start/release')
        else:
            first_mapping.append(deepcopy(mapping))
        # Original Guardian persists this truthful XML summary inside both
        # admission/start evidence and PHYSICAL_RELEASE.resource.xml.
        return dict(observed_monotonic=time.monotonic(), gpus=observed, xml_sha256=backend.canonical(result.stdout),
            policy=dict(selected_gpu_uuid=GPU0, co_resident_tasks_allowed=True, global_idle_required=False,
                        terminate_other_tasks_allowed=False, runtime_amendment_sha256=amendment_sha))
    backend.query_xml = query
    backend._nju_gpu0_co_use = True
