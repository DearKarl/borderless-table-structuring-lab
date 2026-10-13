"""Instrument only the existing vLLM GPU worker's model-load resource boundary."""
from contextlib import contextmanager
from .expert_resource_audit import ExpertResourceAudit
from .core import ContractError, read

class OvisResourceAudit(ExpertResourceAudit):
    @contextmanager
    def integrity_boundary(self,runtime,component='ovis'):
        proxy={**runtime,'environments':{'paddle':runtime['environments']['ovis']}}
        with super().integrity_boundary(proxy,component):
            self.boundaries[-1]['role']='ovis'
            yield

def resource_audit(runtime):
    rows=[{'path':'/assets/'+role+'/'+n,'sha256':h} for role in ('ovis_model','auxiliary_models')
          for n,h in runtime['assets'][role]['files'].items()]
    # Image-resident auxiliaries remain individual resources, not directory exemptions.
    for row in runtime['native']['identity_files']:
        if 'image_path' in row and row.get('image_role')=='ovis':rows.append({'path':row['image_path'],'sha256':row['sha256']})
    return OvisResourceAudit(rows)

