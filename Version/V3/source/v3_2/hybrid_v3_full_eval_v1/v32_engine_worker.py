"""Instrument the actual vLLM worker without changing generation."""
from .v32_resources import resource_audit
from .core import read,ContractError

from vllm.v1.worker.gpu_worker import Worker
class AuditedWorker(Worker):
    def __init__(self,*args,**kwargs):
        runtime=read('/control.json')['runtime']
        self.v32_resources=resource_audit(runtime)
        super().__init__(*args,**kwargs)
    def load_model(self,*args,**kwargs):
        from .asset_binding import load_boundary
        runtime=read('/control.json')['runtime']
        with self.v32_resources.integrity_boundary(runtime):load_boundary(runtime,'ovis')
        result=super().load_model(*args,**kwargs)
        if self.v32_resources.result()['unknown_resources']:raise ContractError('Unknown actual Ovis worker model resource')
        return result
    def v32_count_loaded_parameters(self, identity):
        from .v32_parameters import count_loaded_worker
        return count_loaded_worker(self, identity)
