"""Package a DEV-selected complete adapted checkpoint without private bindings."""
import argparse
import json
import math
from pathlib import Path
import shutil
from .portable_load import digest

SOURCES=[
    'versions/tdatr_adaptation/portable_load.py','versions/tdatr_adaptation/model.py',
    'versions/tdatr_adaptation/backbone.py','versions/tdatr_adaptation/sgcl.py',
    'versions/tdatr_adaptation/inputs.py','versions/tdatr_adaptation/cell_tokens.py',
    'versions/generator_pilot/large_adapters.py','versions/generator_pilot/supervision.py']


def reconstruction_source(repo,name):
    """Resolve the public V8 layout while retaining canonical packaged paths."""
    direct=repo/name
    if direct.is_file():return direct
    if name.startswith(('versions/generator_pilot/','versions/tdatr_adaptation/')):
        relocated=repo/'versions/V8/source'/Path(name).parent.name/Path(name).name
        if relocated.is_file():return relocated
    raise FileNotFoundError('Missing frozen reconstruction module: '+name)


def checkpoint_layout(path,version):
    """Inspect exact FP32 storage without importing torch or allocating tensors."""
    def unique_pairs(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Repeated checkpoint header key')
            value[key]=item
        return value
    with Path(path).open('rb') as stream:
        prefix=stream.read(8)
        if len(prefix)!=8:raise ValueError('Truncated checkpoint prefix')
        length=int.from_bytes(prefix,'little')
        if not 1<=length<=2**22:raise ValueError('Unexpected checkpoint header size')
        encoded=stream.read(length)
        if len(encoded)!=length:raise ValueError('Truncated checkpoint header')
        tensors=json.loads(encoded,object_pairs_hook=unique_pairs)
    tensors={name:item for name,item in tensors.items() if name!='__metadata__'}
    expected={}
    for layer in range(28):
        for projection,(input_dim,output_dim) in dict(q_proj=(1024,2048),k_proj=(1024,1024),v_proj=(1024,1024),o_proj=(2048,1024)).items():
            prefix=f'model.language_model.layers.{layer}.self_attn.{projection}'
            for kind in ['lora']+(['initial'] if version=='V8.5' else []):
                expected[prefix+'.'+kind+'_a']=[8,input_dim]
                expected[prefix+'.'+kind+'_b']=[output_dim,8]
    if not set(expected)<=set(tensors):raise ValueError('Incomplete attention reconstruction factors')
    counts={};intervals=[]
    for name,item in tensors.items():
        shape=item['shape']
        if item['dtype']!='F32' or any(type(size) is not int or size<1 for size in shape):raise ValueError('Invalid FP32 tensor shape')
        if name in expected:
            if shape!=expected[name]:raise ValueError('Reconstruction tensor shape differs')
            group='initial_compensation' if name.endswith(('.initial_a','.initial_b')) else 'attention_adapters'
        elif name.startswith('model.visual.merger.'):group='visual_merger'
        elif name.startswith('tdatr_localizer.'):group='localizer'
        else:raise ValueError('Unexpected checkpoint tensor')
        count=math.prod(shape);low,high=item['data_offsets']
        if type(low) is not int or type(high) is not int or low<0 or high-low!=4*count:raise ValueError('Invalid tensor byte interval')
        counts[group]=counts.get(group,0)+count;intervals.append((low,high))
    end=0
    for low,high in sorted(intervals):
        if low!=end:raise ValueError('Overlapping or incomplete tensor byte coverage')
        end=high
    if end+8+length!=Path(path).stat().st_size:raise ValueError('Checkpoint byte coverage differs')
    required=dict(attention_adapters=2293760,visual_merger=31464704,localizer=7396205)
    if version=='V8.5':required['initial_compensation']=2293760
    if counts!=required:raise ValueError('Incomplete adapted checkpoint components')
    return counts,len(tensors)


def main():
    parser=argparse.ArgumentParser()
    for name in ['protocol','nomination','training','checkpoint','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--version',choices=['V8.4','V8.5'],required=True);args=parser.parse_args()
    protocol=json.loads(Path(args.protocol).read_text());nomination=json.loads(Path(args.nomination).read_text())
    training=json.loads(Path(args.training).read_text());checkpoint=Path(args.checkpoint)
    if protocol['status']!='frozen' or training['status']!='complete' or training['version']!=args.version:
        raise ValueError('One completed frozen fit required')
    if training['optimizer_updates']!=7500 or training['exposures']!=120000:raise ValueError('Formal fit did not complete')
    if training['protocol_sha256']!=digest(args.protocol) or nomination['protocol_sha256']!=digest(args.protocol):
        raise ValueError('Fit/selection protocol binding differs')
    selected=nomination['selected_checkpoints'][args.version]
    if digest(checkpoint)!=selected['checkpoint_sha256']:raise ValueError('Only the frozen DEV checkpoint can be packaged')
    epoch=selected['epoch'];record=next(row for row in training['checkpoints'] if row['label']=='fusion-epoch-'+str(epoch))
    if not record['DEV_selection_eligible'] or record['sha256']!=digest(checkpoint):raise ValueError('Checkpoint eligibility differs')
    repo=Path(__file__).resolve().parents[4];output=Path(args.output)
    if output.exists():raise FileExistsError('Model package cannot overwrite an existing artifact')
    # All scientific reconstruction modules must be byte-identical to training.
    for name in SOURCES:
        if not name.endswith('/portable_load.py') and digest(reconstruction_source(repo,name))!=protocol['frozen_source_hashes'][name]:
            raise ValueError('Scientific reconstruction source differs from the formal runtime: '+name)
    categories,tensor_count=checkpoint_layout(checkpoint,args.version)
    output.mkdir(parents=True,exist_ok=False);shutil.copyfile(checkpoint,output/'adapted.safetensors')
    if digest(output/'adapted.safetensors')!=selected['checkpoint_sha256']:raise ValueError('Copied checkpoint changed')
    hashes={}
    for name in SOURCES:
        target=output/'code'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(reconstruction_source(repo,name),target);hashes[name]=digest(target)
    for name in ['versions/__init__.py','versions/tdatr_adaptation/__init__.py','versions/generator_pilot/__init__.py']:
        target=output/'code'/name;target.write_bytes(b'');hashes[name]=digest(target)
    manifest=dict(format='tdatr_native_lora_complete_v1',version=args.version,
        method='Ordinary LoRA with TDATR-style adapted detail/fusion' if args.version=='V8.4' else 'LoRA-GA with identical TDATR-style adapted detail/fusion',
        selected_fusion_epoch=epoch,selected_optimizer_updates=3750+1250*epoch,selected_training_exposures=60000+20000*epoch,
        completed_fit_updates=7500,completed_fit_exposures=120000,formal_fits_for_this_version=1,user_fixed_seed=None,
        checkpoint_sha256=digest(checkpoint),checkpoint_bytes=checkpoint.stat().st_size,tensor_count=tensor_count,component_parameters=categories,
        base_file_hashes=protocol['bindings']['model_hashes'],runtime_packages=protocol['bindings']['runtime_packages'],
        reconstruction_source_hashes=hashes,execution_protocol_sha256=digest(args.protocol),nomination_sha256=digest(args.nomination),
        shared_exposures_sha256=protocol['shared_plan']['exposures_sha256'],common_inputs_sha256=protocol['bindings']['common_inputs_sha256'],
        dev_selection='Mean full256-source TEDS; earlier fusion epoch on exact ties',selected_dev_teds=selected['dev_teds'],
        benchmark_used_for_selection=False,localization='Own generated OTSL replay; no GT; separate additional forward; boxes do not rewrite text',
        rank=8,alpha=16,base_precision='BF16',trainable_precision='FP32',forward_autocast='BF16',maximum_sequence_tokens=16384,
        backend='transformers',default_native_200DPI_processing=True,presence_frequency_penalties_applied=False,historical_vllm_equivalence=False,
        scientific_modules_exact_to_training=True,portable_wrapper_added_after_training_freeze=True,
        package_validation='Exact selected checkpoint/source hashes, tensor shapes, FP32 byte coverage and component counts; no additional model reload or quality measurement',
        generic_peft_load_supported=False,merged_BF16_equivalence_claimed=False,weights_published_to_git=False,
        raw_data_or_predictions_included=False)
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text(f'''# {args.version} complete adapted checkpoint

One completed 7,500-update,120,000-exposure fit; DEV selected fusion epoch {epoch}.
Contains all attention adapters, the trained visual merger and full localization
head. V8.5 additionally contains the initial factors needed for compensation.

Add this package's `code` directory to `PYTHONPATH`, then import
`versions.tdatr_adaptation.portable_load`. Call
`load(package, original_base, device="cuda:0")` to receive model, processor,
feature capture and manifest. Base assets, reconstruction modules, package and
runtime are checked before use. Pair `torch.inference_mode()` with
`generation_context(model)` and the unchanged default TeleOCR table helper,
prompt, processor and greedy decoder. Inputs are native predicted table crops.
The original base remains responsible for layout and non-table regions.

`localize(model, processor, native_values, raw_otsl, capture)` performs an extra
forward on the model's own output. Invalid structures or overlong replays can
leave localization unavailable; retain the unchanged text. Call `capture.close()`
when done. No geometry labels are needed at inference.

Generic final-factor-only loading is incorrect for V8.5: its forward includes
the original base plus `2 * (B A x - B0 A0 x)`. No BF16 merged equivalence is
claimed. This is an adapted TDATR reproduction, and exposure count is twice the
old three-pass V8 controls. One fit does not establish seed stability.
Weights stay in the local artifact package; Git publication concerns metadata
and source, not automatic weight upload.
''')
    print(json.dumps(dict(version=args.version,checkpoint_sha256=manifest['checkpoint_sha256'],component_parameters=categories)))


if __name__=='__main__':main()
