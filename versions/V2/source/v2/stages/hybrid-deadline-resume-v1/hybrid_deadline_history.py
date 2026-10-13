"""Read-only prior-proof composition for this exact Hybrid native pause."""
import importlib.util
from pathlib import Path
from hybrid_deadline_evidence import verify_original


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def bind_prior_history(score,here,base,pins):
    guards=load('hybrid_deadline_prior_guard',here/'guard_score_gate.py')
    first=load('hybrid_deadline_prior_external',here/'external_score_gate.py')
    second=load('hybrid_deadline_prior_external2',here/'external2_score_gate.py')
    def history(pair,pages,binding):
        verify_original(base,pins)
        recovery=score.verify_recovery(base,pair,pages,binding)
        resource=score.verify_resource_recovery(base,pair,pages,binding)
        one=first.load_external_pins(base,binding);two=second.load_external2_pins(base,binding)
        guard=guards.verify_guard_recovery(base,pair,pages,binding,continuation=one,allow_session4=True,second_continuation=two,allow_session5=True)
        external=first.verify_external_recovery(base,pair,pages,binding,continuation=two,allow_session5=True)
        external2=second.verify_external2_recovery(base,pair,pages,binding,native_deadline=pins)
        tele=score.verify_deadline_recovery(base,pair,pages,binding)
        for name,value in [('resource_recovery',resource),('guard_recovery',guard),('external_recovery',external),('external2_recovery',external2),('native_deadline_continuation',tele)]:
            recovery[name]=value;recovery['evidence'].update(value['evidence'])
        return recovery
    # Only this independent host preflight gets the extended historical proof composition.
    score.all_recovery=history
