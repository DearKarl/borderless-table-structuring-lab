"""Expert-only resource opens: integrity hashing never authorizes runtime loads."""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import threading
from . import core

_SHA = core.sha
_SHA_CODE = _SHA.__code__
_SUFFIXES = ('.onnx','.ftz','.bin','.safetensors','.pdparams','.pdiparams','.npz','.model')

def normalized(path):
    if not isinstance(path,(str,bytes,os.PathLike)):
        raise core.ContractError('Resource path must be filesystem-backed')
    try:
        value=os.fsdecode(path)
        if not value or '\x00' in value:raise ValueError('empty/NUL path')
        return str(Path(value).resolve())
    except (ValueError,OSError,RuntimeError) as exc:
        raise core.ContractError('Invalid resource path') from exc

def locked_paths(rows):
    result={}
    for path,h in rows:
        if not Path(os.fsdecode(path)).is_absolute():raise core.ContractError('Resource lock path must be absolute')
        name=normalized(path);core.digest(h)
        if name in result and result[name]!=h:raise core.ContractError('Conflicting resource path lock')
        result[name]=h
    return result

def boundary_paths(runtime):
    env=runtime['environments']['paddle']
    rows=list(env['image_files'].items())
    # load_boundary also checks auxiliary_models independently when chardet is present.
    roles=set(env['asset_roles'])|{'auxiliary_models'}
    for role in roles:
        core.identifier(role)
        for name,h in runtime['assets'][role]['files'].items():
            rows.append((core.closed(Path('/assets')/role,name),h))
    return locked_paths(rows)

class ExpertResourceAudit:
    def __init__(self,identity_files):
        self.allowed=locked_paths((x['path'],x['sha256']) for x in identity_files)
        self.opened=set();self.runtime_opened=set();self.native=[]
        self.verification_reads=[];self.boundaries=[]
        self._local=threading.local();self._lock=threading.RLock()
        sys.addaudithook(self._hook)

    @staticmethod
    def _sha_read(path,mode,flags):
        if type(flags) is not int or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):return False
        if mode is not None and (not isinstance(mode,str) or any(c in mode for c in 'wax+')):return False
        frame=sys._getframe(1)
        try:
            while frame is not None:
                if frame.f_code is _SHA_CODE and frame.f_globals is _SHA.__globals__:
                    return normalized(frame.f_locals.get('path'))==path
                frame=frame.f_back
            return False
        finally:del frame

    def _hook(self,event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        if Path(os.fsdecode(args[0])).suffix.lower() not in _SUFFIXES:return
        path=normalized(args[0]);boundary=getattr(self._local,'boundary',None)
        verification=(boundary is not None and path in boundary['expected'] and len(args)>=3 and self._sha_read(path,args[1],args[2]))
        with self._lock:
            self.opened.add(path)
            if verification:
                self.verification_reads.append({'path':path,'expected_sha256':boundary['expected'][path],
                    'component':boundary['record']['component'],'boundary':boundary['record']['boundary'],
                    'status':'pending','verified':False})
                boundary['reads'].append(self.verification_reads[-1])
            else:self.runtime_opened.add(path)

    @contextmanager
    def integrity_boundary(self,runtime,component):
        if getattr(self._local,'boundary',None) is not None:raise core.ContractError('Nested expert integrity boundary')
        expected=boundary_paths(runtime)
        with self._lock:
            record={'boundary':len(self.boundaries)+1,'component':component,'role':'paddle','status':'pending','verified':False}
            self.boundaries.append(record)
        state={'expected':expected,'record':record,'reads':[]};self._local.boundary=state
        try:
            yield
        except BaseException as exc:
            with self._lock:
                record.update(status='failed',verified=False,error={'type':type(exc).__name__,'message':str(exc)})
                for row in state['reads']:row.update(status='boundary_failed',verified=False,error_type=type(exc).__name__)
            raise
        else:
            with self._lock:
                record.update(status='verified',verified=True)
                for row in state['reads']:row.update(status='verified',verified=True)
        finally:self._local.boundary=None

    def result(self):
        with self._lock:
            return {'opened':sorted(self.opened),'native_loads':list(self.native),
                    'runtime_opened':sorted(self.runtime_opened),
                    'verification_reads':[dict(r) for r in self.verification_reads],
                    'integrity_boundaries':[dict(r) for r in self.boundaries],
                    'unknown_resources':sorted(self.runtime_opened-set(self.allowed))}
