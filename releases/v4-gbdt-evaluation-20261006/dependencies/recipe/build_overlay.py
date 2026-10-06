"""Root-only preparation recipe. Downloads wheels; never modifies a Python env."""
import argparse
import email
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import zipfile

PINS = {'scikit-learn':'1.9.1','scipy':'1.18.1','joblib':'1.6.0',
        'threadpoolctl':'3.7.0','narwhals':'2.26.0','cloudpickle':'3.1.2'}
TOPS = {'sklearn','scikit_learn.libs','scipy','scipy.libs','joblib','threadpoolctl.py',
        'narwhals','cloudpickle'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    a=p.parse_args();root=Path(a.output).resolve();root.mkdir(parents=False,exist_ok=False)
    wheels=root/'wheels';wheels.mkdir();overlay=root/'site';overlay.mkdir()
    command=[sys.executable,'-m','pip','download','--disable-pip-version-check',
        '--only-binary=:all:','--no-deps','--implementation','cp','--python-version','312',
        '--abi','cp312','--abi','abi3','--abi','none',
        '--platform','manylinux_2_28_x86_64','--platform','manylinux_2_27_x86_64',
        '--platform','manylinux_2_17_x86_64','--platform','manylinux2014_x86_64',
        '--dest',str(wheels),*[k+'=='+v for k,v in PINS.items()]]
    # Failure remains failure. No version substitutions, source build or native install.
    subprocess.run(command,check=True,timeout=600)
    records=[];seen=set()
    for wheel in sorted(wheels.iterdir()):
        if wheel.suffix!='.whl':raise ValueError('Nonwheel download')
        with zipfile.ZipFile(wheel) as z:
            names=z.namelist();metadata=[n for n in names if n.endswith('.dist-info/METADATA')]
            if len(metadata)!=1:raise ValueError('Ambiguous wheel metadata')
            m=email.message_from_bytes(z.read(metadata[0]));name=m['Name'].lower().replace('_','-')
            if name in seen or PINS.get(name)!=m['Version']:raise ValueError('Unpinned wheel')
            seen.add(name);info=metadata[0].split('/')[0]
            for member in z.infolist():
                rel=PurePosixPath(member.filename)
                if rel.is_absolute() or '..' in rel.parts or '\\' in member.filename:
                    raise ValueError('Unsafe wheel path')
                if rel.parts[0] not in TOPS|{info}:raise ValueError('Unexpected overlay top-level path: '+str(rel))
                if stat.S_ISLNK(member.external_attr>>16):raise ValueError('Linked wheel member')
                target=overlay.joinpath(*rel.parts)
                if member.is_dir():target.mkdir(parents=True,exist_ok=True);continue
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write(z.read(member))
            records.append(dict(name=name,version=m['Version'],wheel=wheel.name,
                sha256=sha(wheel),requires_dist=m.get_all('Requires-Dist',[])))
    if seen!=set(PINS):raise ValueError('Incomplete wheels')
    files={f.relative_to(overlay).as_posix():sha(f) for f in sorted(overlay.rglob('*')) if f.is_file()}
    receipt=dict(schema='v4_gbdt_overlay_v1',target='cp312-linux-x86_64',pins=PINS,
        wheels=records,files=files,native_packages_modified=False,dependency_probe_passed=False)
    (root/'OVERLAY.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    if os.name=='posix':
        for f in overlay.rglob('*'):f.chmod(0o555 if f.is_dir() else 0o444)
        overlay.chmod(0o555)
    print(json.dumps(dict(overlay=str(overlay),manifest=str(root/'OVERLAY.json'),manifest_sha256=sha(root/'OVERLAY.json'),status='built_not_verified')))

if __name__=='__main__':main()
