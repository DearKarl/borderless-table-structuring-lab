"""Root-configured child entry with the unchanged owned-process lifecycle.

The existing provider bridge/guardian/admission budgets remain caller-owned.
Root stages SELECTOR_RUNTIME.json in readonly control and supplies its hash in
V4_SELECTOR_RUNTIME_SHA256. No old A/B module, state, output or cache is imported.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
from hybrid.v4_input_selector import provider_bridge as frozen
from hybrid.v4_input_selector.supervisor import OwnedWorker, enable_subreaper, process_identity
from hybrid.v4_input_selector.worker_contract import recv_message
from .contracts import checked_json


class SelectedOwnedWorker(OwnedWorker):
    def start(self,deadline):
        config_path=Path(self.args.contract).parent/'SELECTOR_RUNTIME.json'
        config=checked_json(config_path,os.environ['V4_SELECTOR_RUNTIME_SHA256'])
        if config.get('schema')!='v4_selector_runtime_v1' or config.get('activated') is not True:
            raise ValueError('Root-bound selector runtime is not activated')
        values=config['arguments']
        required={'mode','projection','projection-sha','model','code-root','overlay','overlay-manifest',
                  'overlay-sha','dependency-receipt','dependency-receipt-sha','fixture-sha'}
        optional={'manifest-sha','certificate','certificate-sha','plan','plan-sha','rounding-evidence','rounding-evidence-sha','experiment-auth','experiment-auth-sha'}
        if not required<=set(values) or set(values)-required-optional:
            raise ValueError('Unknown/missing selector runtime arguments')
        self.subreaper=enable_subreaper()
        parent,child=socket.socketpair();self.sock=parent
        argv=[sys.executable,'-B','-m','hybrid.v4_selected_eval.entry']
        for key,value in [('contract',self.args.contract),('contract-sha',self.contract_sha),
            ('vendor-root',self.args.vendor_root),('model-root',self.args.model_root),
            ('aux-root',self.args.aux_root),('input-root',self.args.input_root),
            ('output-root',self.output_root),('owner-token',self.owner_token),('socket-fd',child.fileno()),*values.items()]:
            argv.extend(['--'+key,str(value)])
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:
            self.logs=[(self.log_root/'stdout.log').open('xb'),(self.log_root/'stderr.log').open('xb')]
            self.process=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=self.logs[0],stderr=self.logs[1],
                start_new_session=True,pass_fds=(child.fileno(),),env=env)
            self.identity=process_identity(self.process.pid)
            if self.identity['pgid']!=self.process.pid or self.identity['sid']!=self.process.pid or self.identity['uid']!=os.getuid():
                raise RuntimeError('Worker is not in owned process group')
        finally:child.close()
        ready=recv_message(self.sock,deadline=deadline)
        if (ready.get('kind')!='ready' or ready.get('owner_token')!=self.owner_token
                or ready.get('contract_sha256')!=self.contract_sha or ready.get('pid')!=self.process.pid
                or ready.get('pgid')!=self.process.pid or ready.get('projection_sha256')!=values['projection-sha']):
            raise RuntimeError('Unbound selected worker handshake')
        return dict(process=self.identity,handshake=ready,subreaper=self.subreaper)


def main():
    original=frozen.Bridge
    frozen.Bridge=lambda *a,**k:original(*a,worker_factory=SelectedOwnedWorker,**k)
    return frozen.main()

if __name__=='__main__':raise SystemExit(main())
