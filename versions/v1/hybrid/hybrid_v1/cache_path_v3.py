"""Admit only the reviewed MinerU path-v3 run, for complete development reuse."""
from pathlib import Path

from .cache import CSV_CODE, PATH_CODE, checked, key, read, sha

ROOT = Path('/srv/hybrid-research')
OLD_RUN = ROOT/'artifacts/official-full-20260921-v1/native/mineru/native-csv-v2'
PATH_RUN = ROOT/'artifacts/official-full-20260921-v1/native/mineru/native-path-v3'
CODE = ROOT/'inference/remote-full-native-v1/full-path-v3'
RUNTIME_SHA = 'c450c977d33c4a7e8579e3e27d033d00e7a22fcdd0264eedd2d3063a604fa77e'
COMPAT_SHA = '4f6838ba39d0e2fe60827923673d6e50cf8683386be63309c078d3cab739aada'
MAP_SHA = 'f3d7801b45232dce8963a7ab9e3bd430ab5cf3bf7237d4de6fce087ce22ee060'
PREFLIGHT_SHA = 'cd8358af726902639bb0ff6279ffb51afcca4d79042434315582380aaa2c44ed'
OLD_RUNTIME_SHA = '6904e03db6e026f99321058f442792067de2f2164dd5f8a7d82884c149d732ac'
TRANSFORMS_SHA = '181d7b5defc5f61361203d540bc3c3698e0bf23931234eabfa81afbba58df0f3'
SOURCE_EVIDENCE_SHA = '20ef95ae40a251c6f1131f8d9223108d42d1127fc3bcb34c2913b5d8eef4f1b3'


def pinned(path,digest):
    assert path.is_file() and not path.is_symlink() and sha(path)==digest, 'Unapproved path-v3 artifact'
    return read(path)


def verify_run(run,runtime,pages):
    assert Path(run).resolve()==PATH_RUN.resolve(), 'Only the reviewed native-path-v3 run is admitted'
    assert runtime['arm']=='mineru' and runtime['full_code_lock_sha256']==PATH_CODE
    assert pinned(PATH_RUN/'RUNTIME.json',RUNTIME_SHA)==runtime
    compatibility=pinned(PATH_RUN/'COMPATIBILITY_RECEIPT.json',COMPAT_SHA)
    assert compatibility['kind']=='FULL-PATH-UTF8-v3'
    assert compatibility['continuation_allowed'] is True and compatibility['pause_reasons']==[]
    assert compatibility['old_code_lock_sha256']==CSV_CODE and compatibility['new_code_lock_sha256']==PATH_CODE
    assert compatibility['new_runtime_key']==key(runtime)
    source_path=checked(compatibility['source_runtime'])
    assert source_path.resolve()==(OLD_RUN/'RUNTIME.json').resolve()
    old=pinned(source_path,OLD_RUNTIME_SHA)
    assert old['arm']=='mineru' and old['full_code_lock_sha256']==CSV_CODE
    assert compatibility['old_runtime_key']==key(old)
    old_inputs={k:v for k,v in old.items() if k!='full_code_lock_sha256'}
    assert old_inputs=={k:v for k,v in runtime.items() if k!='full_code_lock_sha256'}, 'Inference configuration changed'
    lock=pinned(CODE/'CODE_LOCK.json',PATH_CODE)
    members={row['path']:row for row in lock['files']}
    assert len(members)==len(lock['files'])==18
    for name,item in members.items():
        assert Path(name).name==name and name not in ('.','..')
        path=CODE/name
        assert path.is_file() and not path.is_symlink() and sha(path)==item['sha256']
    assert members['PATH_TRANSFORMS.json']['sha256']==TRANSFORMS_SHA
    assert compatibility['native_inference_signature']==key({'runtime_inputs':old_inputs,'path_transforms_sha256':TRANSFORMS_SHA})
    assert compatibility['inference_configuration_unchanged'] is True
    assert compatibility['terminal_count']==1481 and compatibility['planned_count']==170
    assert compatibility['reviewed_path_failures']==['000172','000174']
    assert compatibility['actual_started_pages_retried'] is False and compatibility['original_page_id_preserved'] is True
    assert compatibility['dispatcher_exit_proof']['original_exited'] is True
    dispatcher_path=checked(compatibility['dispatcher'])
    assert dispatcher_path.resolve()==(ROOT/'receipts/FULL_NATIVE_v1/mineru/native-csv-v2/DISPATCH.json').resolve()
    dispatcher=read(dispatcher_path)
    assert dispatcher['pid']==325300 and str(dispatcher['proc_start_ticks'])=='25362850'
    owned=compatibility['owned_containers_stopped']
    assert len(owned)==175 and len({x['cid'] for x in owned})==175
    assert {x['shard'] for x in owned}=={f'{i:06d}' for i in range(175)}
    for item in owned:
        state=item['state']
        assert state['Status']=='exited' and not state['Running'] and not state['OOMKilled']
        assert state['ExitCode']==(1 if item['shard'] in ('000172','000174') else 0)
    evidence_path=checked(compatibility['source_evidence'])
    assert evidence_path.resolve()==(CODE/'SOURCE_EVIDENCE.json').resolve()
    evidence=pinned(evidence_path,SOURCE_EVIDENCE_SHA)
    assert len(evidence['files'])==10
    for item in evidence['files']:
        source=OLD_RUN/item['relative_path']
        assert source.resolve().is_relative_to(OLD_RUN.resolve())
        checked({'path':str(source),'sha256':item['sha256'],'bytes':item['bytes']})
    mapping=pinned(PATH_RUN/'PAGE_FILE_MAP.json',MAP_SHA)
    proof=pinned(PATH_RUN/'NAME_PREFLIGHT.json',PREFLIGHT_SHA)
    assert checked(proof['mapping']).resolve()==(PATH_RUN/'PAGE_FILE_MAP.json').resolve()
    assert proof['passed'] is True and proof['original_page_id_preserved'] is True
    assert proof['pages']==1651 and proof['filesystem_name_max']>=255
    assert proof['original_primary_names_preserved'] is True and proof['max_original_primary_utf8_bytes']==242
    assert proof['changed_component_counts']=={'directory':0,'event':0,'terminal':0,'primary':0}
    assert mapping['schema']=='page-paths-v3' and mapping['encoding']=='utf-8' and mapping['component_limit']==255
    assert mapping['original_page_id_preserved'] is True
    rows={row['page_id']:row for row in mapping['pages']}
    assert len(rows)==len(mapping['pages'])==len(pages)==1651 and rows.keys()==pages.keys()
    for pid,row in rows.items():
        assert pid not in ('','.','..') and not any(x in pid for x in ('/','\\','\x00'))
        assert row['input_sha256']==pages[pid]['input_sha256']
        for field,suffix in [('directory',''),('event','.started.json'),('terminal','.json'),('primary','.md')]:
            assert row[field]==pid+suffix and len(row[field].encode('utf-8'))<=255
    return compatibility,old
