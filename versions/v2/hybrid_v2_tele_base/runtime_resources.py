"""Record Python resource opens and bind native auxiliary model loaders."""
import json
import os
from pathlib import Path
import sys
from .region_protocol import RunBlocked


class ResourceAudit:
    def __init__(self, identity_files):
        self.allowed = {str(Path(x['path']).resolve()) for x in identity_files}
        self.opened, self.native = set(), []
        def hook(event, args):
            if event != 'open' or not isinstance(args[0], (str,bytes,os.PathLike)): return
            path = Path(os.fsdecode(args[0]))
            if path.suffix.lower() in ('.onnx','.ftz','.bin','.safetensors','.pdparams','.pdiparams','.npz','.model'):
                self.opened.add(str(path.resolve()))
        sys.addaudithook(hook)

    def bind_tele_loaders(self):
        import fasttext
        import onnxruntime
        load, session = fasttext.load_model, onnxruntime.InferenceSession
        def check(path, kind):
            if not isinstance(path,(str,os.PathLike)): raise RunBlocked('Unbound in-memory auxiliary model')
            resolved = str(Path(path).resolve())
            if resolved not in self.allowed: raise RunBlocked('Unbound auxiliary model: '+resolved)
            self.native.append({'kind':kind,'path':resolved})
        def fasttext_load(path, *args, **kwargs):
            check(path,'fasttext'); return load(path,*args,**kwargs)
        def onnx_load(path, *args, **kwargs):
            check(path,'onnxruntime'); return session(path,*args,**kwargs)
        fasttext.load_model, onnxruntime.InferenceSession = fasttext_load, onnx_load

    def result(self):
        return {'opened':sorted(self.opened),'native_loads':self.native,
                'unknown_resources':sorted(self.opened-self.allowed)}
