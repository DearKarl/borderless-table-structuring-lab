"""Verify selected package bytes and reconstruction provenance without inference."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from .package_model import checkpoint_layout,SOURCES,reconstruction_source
from .portable_load import digest


def main():
    parser=argparse.ArgumentParser()
    for name in ['packages','protocol','nomination','output']:parser.add_argument('--'+name,required=True)
    args=parser.parse_args();protocol=json.loads(Path(args.protocol).read_text());nomination=json.loads(Path(args.nomination).read_text())
    if nomination['protocol_sha256']!=digest(args.protocol):raise ValueError('Nomination belongs to another protocol')
    versions={};repo=Path(__file__).resolve().parents[4]
    for version in ['V8.4','V8.5']:
        package=Path(args.packages)/version;manifest=json.loads((package/'MANIFEST.json').read_text());chosen=nomination['selected_checkpoints'][version]
        if manifest['format']!='tdatr_native_lora_complete_v1' or manifest['version']!=version:raise ValueError('Package identity differs')
        if (manifest['execution_protocol_sha256']!=digest(args.protocol) or manifest['nomination_sha256']!=digest(args.nomination) or
            manifest['checkpoint_sha256']!=chosen['checkpoint_sha256'] or manifest['selected_fusion_epoch']!=chosen['epoch'] or
            manifest['selected_dev_teds']!=chosen['dev_teds']):raise ValueError('Selected package provenance differs')
        if manifest['base_file_hashes']!=protocol['bindings']['model_hashes'] or manifest['runtime_packages']!=protocol['bindings']['runtime_packages']:
            raise ValueError('Package base/runtime provenance differs')
        if manifest['completed_fit_updates']!=7500 or manifest['completed_fit_exposures']!=120000 or manifest['formal_fits_for_this_version']!=1:
            raise ValueError('Package completed-fit claims differ')
        weights=package/'adapted.safetensors'
        if digest(weights)!=chosen['checkpoint_sha256'] or weights.stat().st_size!=manifest['checkpoint_bytes']:raise ValueError('Selected weights changed')
        categories,count=checkpoint_layout(weights,version)
        if categories!=manifest['component_parameters'] or count!=manifest['tensor_count']:raise ValueError('Package tensor manifest differs')
        expected=set(SOURCES)|{'versions/__init__.py','versions/tdatr_adaptation/__init__.py','versions/generator_pilot/__init__.py'}
        if set(manifest['reconstruction_source_hashes'])!=expected:raise ValueError('Reconstruction source closure differs')
        for name,expected_sha in manifest['reconstruction_source_hashes'].items():
            path=(package/'code'/name).resolve()
            if not path.is_relative_to((package/'code').resolve()) or digest(path)!=expected_sha:raise ValueError('Packaged source changed')
            if name in SOURCES:
                if digest(reconstruction_source(repo,name))!=expected_sha:raise ValueError('Source differs from packaging implementation')
                if not name.endswith('/portable_load.py') and protocol['frozen_source_hashes'][name]!=expected_sha:
                    raise ValueError('Scientific module differs from frozen runtime')
            elif path.read_bytes()!=b'':raise ValueError('Unexpected package initialization side effects')
        manifest['manifest_sha256']=digest(package/'MANIFEST.json')
        manifest['exact_weight_and_source_bytes_verified']=True
        manifest['complete_FP32_storage_verified']=True
        versions[version]=manifest
    result=dict(status='passed',at_utc=datetime.now(timezone.utc).isoformat(),versions=versions,
        execution_protocol_sha256=digest(args.protocol),nomination_sha256=digest(args.nomination),
        new_model_loads=0,new_inference_calls=0,GPU_allocations=0,
        verification_scope='Selected checkpoint/source bytes, FP32 tensor storage and protocol/DEV provenance; no final numerical reload or quality rerun.')
    with Path(args.output).open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='passed',packages=2,new_model_loads=0)))


if __name__=='__main__':main()
