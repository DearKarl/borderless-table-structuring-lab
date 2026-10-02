"""Host supervisor: no GPU exposure, immutable output, owned container only."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/srv/hybrid-research')
BASE = ROOT / 'inference/hybrid-v2-tele-base/attempt-001'
IMAGE = 'sha256:787dd9a15d41c442e78a5faa9cdc721f154cbf3d4cf8624edb41026c3052b75a'


def save(path, data):
    with path.open('x') as f: json.dump(data, f, indent=2)


def main():
    if sys.platform != 'linux': raise RuntimeError('Linux host required')
    code = BASE / 'environment-probe-v2'
    lock = json.loads((code / 'CODE_LOCK.json').read_bytes())
    for name, digest in lock.items():
        if Path(name).name != name or hashlib.sha256((code/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Code lock mismatch')
    output = BASE / 'environment-probe-output-v2'; output.mkdir(exist_ok=False)
    envroot = ROOT / 'inference/environments'
    source = ROOT / 'inference/models/tele-source/TeleOCR-9921cffe380efe4e2fa010258b3d0c3cb70bab2d'
    command = ['docker', 'create', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--cpus', '2', '--memory', '8g', '--pids-limit', '128',
        '--tmpfs', '/tmp:rw,size=512m']
    for key, value in {'CUDA_VISIBLE_DEVICES':'', 'HOME':'/output', 'FTLANG_CACHE':'/output/ftlang',
        'HF_HUB_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1', 'PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONNOUSERSITE':'1', 'OMP_NUM_THREADS':'2', 'OPENBLAS_NUM_THREADS':'2'}.items():
        command += ['--env', key+'='+value]
    mounts = [(code, '/code', True), (output, '/output', False), (source, '/source', True)]
    mounts += [(envroot/name, str(envroot/name), True) for name in
               ('tele-v2', 'tele-v3', 'cpython-3.12.14-20260901')]
    for src, dst, ro in mounts:
        command += ['--mount', f'type=bind,src={src},dst={dst}'+(',readonly' if ro else '')]
    command += [IMAGE, str(envroot/'tele-v3/bin/python'), '-B', '/code/environment_probe.py',
                '--source', '/source', '--output', '/output/ENVIRONMENT_RESULT.json']
    cid = subprocess.check_output(command, text=True).strip()
    save(output/'START.json', {'cid':cid, 'command':command, 'code_lock':lock})
    timeout = False; cli = None
    try:
        with (output/'stdout.log').open('x') as stdout, (output/'stderr.log').open('x') as stderr:
            try: cli = subprocess.run(['docker','start','--attach',cid], stdout=stdout, stderr=stderr, timeout=180).returncode
            except subprocess.TimeoutExpired: timeout = True
    finally:
        state = json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        if state['State']['Running']: subprocess.run(['docker','kill',cid],check=True,capture_output=True)
        subprocess.run(['docker','wait',cid],check=True,capture_output=True)
        state = json.loads(subprocess.check_output(['docker','inspect',cid],text=True))[0]
        save(output/'EXIT.json', {'cli_exit':cli,'timeout':timeout,'container':state})
    if state['HostConfig']['Devices'] or state['HostConfig']['DeviceRequests']: raise RuntimeError('Unexpected GPU access')
    print(json.dumps({'output':str(output),'exit_code':state['State']['ExitCode'],'timeout':timeout}))
    return 0 if cli == 0 and not timeout and not state['State']['OOMKilled'] else 1


if __name__ == '__main__': raise SystemExit(main())
