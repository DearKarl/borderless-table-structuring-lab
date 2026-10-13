"""Fail-closed native CPU dependency evidence. Never installs or trains anything."""
import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import sys
import sysconfig
from .contracts import checked_json, frozen_model, MODEL_SHA, MARGINS
from hybrid.v4_input_selector.worker_contract import file_sha
from hybrid.v4_input_selector.scale_policy import ScalePolicy

IMAGE='sha256:5dd2b6a864a9565cde9216c8da3db3a8b05c93cbdca9ef417f3956841ce0722f'
PYTHON='/opt/v31-native/bin/python'
ABI='cpython-312-x86_64-linux-gnu'
PINS={'scikit-learn':'1.9.1','scipy':'1.18.1','joblib':'1.6.0',
      'threadpoolctl':'3.7.0','narwhals':'2.26.0','cloudpickle':'3.1.2'}
MODULES={'scikit-learn':'sklearn','scipy':'scipy','joblib':'joblib',
         'threadpoolctl':'threadpoolctl','narwhals':'narwhals','cloudpickle':'cloudpickle'}

def need(condition,message):
    if not condition:raise ValueError(message)

def inventory(root):
    root=Path(root)
    need(root.is_dir() and not root.is_symlink(),'Overlay missing or linked')
    paths=list(root.rglob('*'));need(not any(p.is_symlink() for p in paths),'Linked overlay entry')
    return {p.relative_to(root).as_posix():file_sha(p) for p in sorted(paths) if p.is_file()}

def install_path(root,manifest,manifest_sha):
    """Append a verified, isolated directory; native packages keep precedence."""
    value=checked_json(manifest,manifest_sha)
    need(value.get('schema')=='v4_gbdt_overlay_v1' and value.get('pins')==PINS,'Overlay pins differ')
    need(value.get('target')=='cp312-linux-x86_64','Wrong binary target')
    need(inventory(root)==value['files'],'Overlay bytes/inventory differ')
    allowed={'sklearn','scikit_learn.libs','scipy','scipy.libs','joblib','threadpoolctl.py','narwhals','cloudpickle'}
    for name in value['files']:
        top=name.split('/')[0]
        need(top in allowed or any(top==k.replace('-','_')+'-'+v+'.dist-info' for k,v in PINS.items()),'Forbidden package in overlay')
    need(sysconfig.get_config_var('SOABI')==ABI and sys.executable==PYTHON,'Original native interpreter/ABI required')
    sys.path.append(str(Path(root).resolve()))
    return value

def native_origins():
    rows={}
    for name in ('numpy','torch','pypdfium2'):
        spec=importlib.util.find_spec(name)
        need(spec is not None and spec.origin,'Native package missing: '+name)
        origin=Path(spec.origin).resolve()
        need(origin.is_relative_to('/opt/v31-native/lib/python3.12/site-packages'),'Native package shadowed: '+name)
        rows[name]=dict(path=str(origin),sha256=file_sha(origin))
    return rows

def probe(root,manifest,manifest_sha,model,fixture,fixture_sha):
    before=native_origins();install_path(root,manifest,manifest_sha)
    after=native_origins();need(before==after,'Overlay changed native package origins')
    import numpy
    need(numpy.__version__=='2.5.3','Native numpy version differs')
    rows={'numpy':dict(version=numpy.__version__,path=str(Path(numpy.__file__).resolve()))}
    for distribution,name in MODULES.items():
        module=importlib.import_module(name);version=importlib.metadata.version(distribution)
        origin=Path(module.__file__).resolve()
        need(version==PINS[distribution] and origin.is_relative_to(Path(root).resolve()),'Dependency version/origin differs: '+name)
        rows[name]=dict(version=version,path=str(origin))
    fixture_value=checked_json(fixture,fixture_sha)
    need(fixture_value['model_sha256']==MODEL_SHA and fixture_value['margins']==list(MARGINS),'Fixture identity differs')
    policy=frozen_model(model);actual=[]
    for margin in MARGINS:
        diagnostic=ScalePolicy(policy.estimator,dict(policy.config,learned_enabled=True,margin=margin),policy.identity)
        actual.append([diagnostic.predict(row) for row in fixture_value['action_features']])
    expected=fixture_value['predictions']
    need(len(actual)==len(expected),'Prediction count differs')
    for arow,erow in zip(actual,expected):
        need(len(arow)==len(erow),'Fixture row count differs')
        for a,e in zip(arow,erow):
            need({k:v for k,v in a.items() if k!='predicted_gains'}=={k:v for k,v in e.items() if k!='predicted_gains'},'Prediction decision parity failed')
            need(set(a['predicted_gains'])==set(e['predicted_gains']),'Prediction action keys differ')
            need(all(math.isclose(v,e['predicted_gains'][k],rel_tol=1e-12,abs_tol=1e-12) for k,v in a['predicted_gains'].items()),'Numeric prediction parity failed')
    need(any(x['predicted_gains'] for row in actual for x in row),'No estimator prediction executed')
    return dict(schema='v4_native_gbdt_probe_v1',passed=True,interpreter=sys.executable,
        abi=sysconfig.get_config_var('SOABI'),python=sys.version,modules=rows,native_origins=after,
        overlay_sha256=manifest_sha,fixture_sha256=fixture_sha,model_sha256=MODEL_SHA,
        predictions=actual,parity=True,recognition_calls=0,model_fits=0)

def require_receipt(receipt,*,overlay_sha,fixture_sha):
    """Envelope is host-written after Docker inspect + successful process exit."""
    need(receipt.get('schema')=='v4_native_gbdt_probe_envelope_v1','Successful native dependency receipt required')
    need(receipt.get('passed') is True and receipt.get('error') is None,'Dependency envelope failed')
    need(receipt.get('image')==IMAGE and receipt.get('exit_code')==0 and receipt.get('container_absent') is True,'Dependency container not positively verified/released')
    need(receipt.get('network')=='none' and receipt.get('readonly') is True and receipt.get('gpu_devices')==[],'Dependency CPU isolation differs')
    result=receipt.get('result',{})
    need(result.get('schema')=='v4_native_gbdt_probe_v1' and result.get('passed') is True and result.get('parity') is True,'Native prediction did not pass')
    need(result.get('interpreter')==PYTHON and result.get('abi')==ABI and result.get('model_sha256')==MODEL_SHA,'Native model/interpreter binding differs')
    need(result.get('overlay_sha256')==overlay_sha and result.get('fixture_sha256')==fixture_sha,'Dependency evidence binding differs')
    need(result.get('recognition_calls')==0 and result.get('model_fits')==0,'CPU probe exceeded scope')
    expected=dict(numpy='2.5.3',**{MODULES[k]:v for k,v in PINS.items()})
    need({k:v.get('version') for k,v in result.get('modules',{}).items()}==expected,'Dependency receipt incomplete')
    return result

def main():
    p=argparse.ArgumentParser()
    for name in ('overlay','overlay-manifest','overlay-sha','model','fixture','fixture-sha'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(json.dumps(probe(a.overlay,a.overlay_manifest,a.overlay_sha,a.model,a.fixture,a.fixture_sha),sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
