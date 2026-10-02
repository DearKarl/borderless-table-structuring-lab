"""Exact code and runtime binding for the new cache-disabled native controller."""
from pathlib import Path

from .cache import CSV_CODE, checked, key, read, sha
from . import page_paths

FRESH_CODE = '77ad2239439b012536b53c55d3761ae8e7221037e451af15cac0210b3a82f2f0'
COMPATIBILITY_SHA = '6c364e03b71100b48a258efc55d2be988921658f8594d8c019b7f4a121741768'
PATH_POLICY_SHA = 'ea09a0e945eea0f9505501fc333c33b34d4bf631f41dc6784c6fc0d56dbbc2db'
POLICY = {'M_workers':1,'P_workers':1,'arm_order':['mineru','paddle'],'shard_size':8}
SOURCE_RUNTIME_SHAS = {'mineru':'6904e03db6e026f99321058f442792067de2f2164dd5f8a7d82884c149d732ac',
                       'paddle':'99bce2a7f7c56d6490456a4f1df0828a4b8a6109c84dddf5286ba56451d40dc1'}
RUNTIME_DELTA_FIELDS = {'full_code_lock_sha256','input_manifest_sha256','unified_system_freeze_sha256',
                        'native_controller_sha256','native_code_compatibility_sha256','path_policy_sha256',
                        'source_runtime_key','source_runtime_artifact','native_inference_signature'}


def verify_code(code,freeze):
    code=Path(code);execution=freeze['native_execution']
    assert set(execution)=={'profile','code_lock','compatibility'}
    assert execution['profile']=='fresh-empty-v1' and freeze['execution_policy']==POLICY
    assert checked(execution['code_lock']).resolve()==(code/'CODE_LOCK.json').resolve()
    assert execution['code_lock']['sha256']==FRESH_CODE
    members=read(code/'CODE_LOCK.json')['files']
    assert len(members)==16 and len({r['path'] for r in members})==16
    for item in members:
        assert Path(item['path']).name==item['path'] and item['path'] not in ('.','..')
        p=code/item['path'];assert p.is_file() and not p.is_symlink() and sha(p)==item['sha256']
    assert checked(freeze['native_controller']).resolve()==(code/'native_controller.py').resolve()
    assert checked(execution['compatibility']).resolve()==(code/'SOURCE_COMPATIBILITY.json').resolve()
    assert execution['compatibility']['sha256']==COMPATIBILITY_SHA
    proof=read(code/'SOURCE_COMPATIBILITY.json')
    assert proof['kind']=='HYBRID-NATIVE-EMPTY-v1'
    assert proof['inference_parameter_changes'] is False and proof['old_predictions_adopted'] is False
    assert proof['load_timeout']==600 and proof['page_timeout']==900
    assert proof['policy_sha256']==sha(code/'watchdog_policy.py')
    assert PATH_POLICY_SHA==sha(code/'page_paths.py')==sha(Path(page_paths.__file__))
    return proof


def make_runtime(original,arm,freeze,manifest_sha256):
    assert arm in ('mineru','paddle') and original['arm']==arm
    assert original['full_code_lock_sha256']==CSV_CODE, 'Unknown source runtime code'
    source=freeze['native_bindings'][arm]['runtime_artifact']
    assert source['sha256']==SOURCE_RUNTIME_SHAS[arm], 'Unapproved source runtime'
    assert read(checked(source))==original
    runtime={**original,'full_code_lock_sha256':FRESH_CODE,
             'input_manifest_sha256':manifest_sha256,'unified_system_freeze_sha256':key(freeze),
             'native_controller_sha256':freeze['native_controller']['sha256'],
             'native_code_compatibility_sha256':COMPATIBILITY_SHA,'path_policy_sha256':PATH_POLICY_SHA,
             'source_runtime_key':key(original),'source_runtime_artifact':source}
    inference={k:v for k,v in original.items() if k not in RUNTIME_DELTA_FIELDS}
    assert inference=={k:v for k,v in runtime.items() if k not in RUNTIME_DELTA_FIELDS}
    runtime['native_inference_signature']=key({'inference_inputs':inference,
        'source_runtime_key':key(original),'native_code_lock_sha256':FRESH_CODE,
        'code_compatibility_sha256':COMPATIBILITY_SHA,'path_policy_sha256':PATH_POLICY_SHA})
    return runtime


def verify_capacity_binding(freeze):
    """A separate derived receipt is required; old CSV inventory is not relabeled."""
    binding=read(checked(freeze['fresh_capacity_binding']))
    assert binding['kind']=='HYBRID-FRESH-EMPTY-CAPACITY-v1'
    assert binding['native_code_lock_sha256']==FRESH_CODE
    assert binding['native_execution']==freeze['native_execution']
    assert binding['parameter_inventory']==freeze['parameter_inventory']
    assert binding['environment_coverage_artifacts']==freeze['environment_coverage_artifacts']
    assert binding['source_runtime_artifacts']=={arm:freeze['native_bindings'][arm]['runtime_artifact'] for arm in ('mineru','paddle')}
    assert binding['model_environment_resources_unchanged'] is True
    assert binding['inference_signatures']=={arm:make_runtime(read(checked(freeze['native_bindings'][arm]['runtime_artifact'])),arm,freeze,freeze['input_manifest_sha256'])['native_inference_signature'] for arm in ('mineru','paddle')}
    inventory=read(checked(freeze['parameter_inventory']))
    assert binding['total_parameter_upper_bound']==inventory['total_parameter_upper_bound']<=4_000_000_000
    return binding
