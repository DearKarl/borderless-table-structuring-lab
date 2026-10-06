"""Selected bridge with original physical Guardian release/admission lifecycle."""
from hybrid.v4_input_selector.linux_provider import *
from .host_manifest import check_mounts

class CalibrationGuardian(Guardian):
    metadata_keys=('plan','projection','rounding','overlay_manifest','dependency_receipt')
    def __init__(self,*args,manifest,**kwargs):
        super().__init__(*args,**kwargs)
        self.manifest=manifest

    def inspect_state(self,state):
        actual=super().inspect_state(state)
        values=[v for v in state['Config']['Env'] if v.startswith('V4_SELECTOR_RUNTIME_SHA256=')]
        require(values==['V4_SELECTOR_RUNTIME_SHA256='+self.manifest['runtime']['sha256']],'Container runtime environment differs')
        return actual

    def inspect_owned(self, allocation, end):
        require(allocation == self.allocation and self.owned is None and (self.probe or self.binding is not None), 'Acquisition order/identity differs')
        run = self.bank.root/'runs'/allocation
        with self.alarm(end):
            self.ipc = Path(tempfile.mkdtemp(prefix='v4p-', dir='/tmp')).resolve()
            os.chmod(self.ipc, 0o700)
            require(len(str(self.ipc/'control.sock').encode()) < 108, 'Host Unix socket path exceeds bound')
            self.mounts = [(str(Path(self.config['code_root']).resolve()), self.config['code_root'], False),
                (str(run/'control'), str(run/'control'), False), (str(run/'output'), str(run/'output'), True),
                (str(self.ipc), '/v4-ipc', False)]
            roots = dict(control=str(run/'control'), output=str(run/'output'), input='/unmounted', vendor='/unmounted', model='/unmounted', aux='/unmounted')
            if not self.probe:
                roots.update(input=str(self.bank.input_root), **self.config['roots'])
                self.mounts.extend((path,path,False) for key,path in roots.items() if key in ('input','vendor','model','aux'))
                self.mounts.append(self.v31.gpu_backend.driver_mount(self.binding))
                # The already observed small weight receipt is a named read-only
                # file, not the private master or a directory-wide evidence mount.
                weight_ref = self.bank.template['runtime_binding']['weight_receipt']
                weight = checked_json(weight_ref)
                require(weight.get('schema') == 'v4_weight_observation_v1' and weight.get('sha256') == self.bank.template['model_files']['model.safetensors']
                        and weight.get('model_loads') == 0 and weight.get('path') == str(Path(roots['model'])/'model.safetensors'), 'Weight observation does not bind the staged native model')
                self.mounts.append((weight_ref['path'],weight_ref['path'],False))
            m=self.manifest
            self.mounts.extend([(m['gbdt_root'],m['gbdt_root'],False),(m['overlay_root'],'/v4-deps',False)])
            for key in self.metadata_keys:
                self.mounts.append((m[key]['path'],m[key]['path'],False))
            check_mounts(m,[source for source,_,_ in self.mounts])
            require(not any(private == Path(source) or private.is_relative_to(Path(source)) for private in self.bank.private_paths for source,_,_ in self.mounts), 'Private host data would be mounted')
            self.image_identity = self.v31.image_identity.resolve_image_identity(NATIVE_IMAGE, self.deadline(end), self.runner)
            self.container_args = ['/opt/v31-native/bin/python','-B','-m','hybrid.v4_selected_eval.bridge',
                '--global-end',str(self.global_end),'--initial-work-end',str(end)]
            for key,value in roots.items():
                self.container_args.extend(['--'+key,value])
            if self.probe:
                self.container_args.append('--lifecycle-probe')
            flags = self.v31.lifecycle.limits(self.config['resources'])
            flags += ['--restart','no','--init=false','--user',str(os.getuid())+':'+str(os.getgid()),'--label','v4-bank='+self.bank.master['bank_id'], '--env','PYTHONPATH='+self.config['code_root'],
                      '--env','PYTHONUNBUFFERED=1','--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1','--env','CUDA_DEVICE_ORDER=PCI_BUS_ID']
            if self.probe:
                flags += ['--runtime','runc','--env','NVIDIA_VISIBLE_DEVICES=void','--env','CUDA_VISIBLE_DEVICES=','--env','NVIDIA_DRIVER_CAPABILITIES=']
            else:
                flags += self.v31.gpu_backend.launch_flags({'schema':2,'gpu_backend':{'kind':'manual','host_binding':self.config['driver_binding']}}, self.binding['uuid'], self.binding)
            flags += ['--env','V4_SELECTOR_RUNTIME_SHA256='+self.manifest['runtime']['sha256'],'--env','PYTHONDONTWRITEBYTECODE=1']
            flags += self.v31.lifecycle.mount_args(self.mounts, str(run/'output'))
            # Override an image entrypoint rather than silently appending to it.
            flags += ['--entrypoint', self.container_args[0], NATIVE_IMAGE, *self.container_args[1:]]
            self.owned = self.v31.lifecycle.OwnedDocker(allocation, self.deadline(end), self.output, runner=self.runner)
            self.owned.admission = self.admission
            self.owned.create(flags)
            state = self.owned.inspect()
            actual = self.inspect_state(state)
            atomic_json(self.output/'CONTAINER_INSPECT.json', state)
        return dict(job_id=state['Config']['Labels']['v4-bank'], image=state['Image'],
            gpu_uuid=None if self.probe else self.binding['uuid'], owner_token=state['Config']['Labels']['hybrid-v3-owner'],
            observed_at=self.wall(), container_id=state['Id'], inspected_at_utc=datetime.now(timezone.utc).isoformat(),
            mounts=[dict(source=a,target=z,writable=w) for a,z,w in actual])
