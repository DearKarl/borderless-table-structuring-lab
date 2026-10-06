"""Fresh-process diagnostic of the unchanged real model; emits no policy."""
import argparse
import json
import sys
from hybrid.v4_input_selector.scale_policy import ScalePolicy
from .contracts import MARGINS, frozen_model

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--margin',type=float,required=True,choices=MARGINS)
    a=p.parse_args(); original=frozen_model(a.model)
    candidate=ScalePolicy(original.estimator,dict(original.config,learned_enabled=True,margin=a.margin),original.identity)
    maps=json.load(sys.stdin)['action_features']
    print(json.dumps(dict(diagnostic_only=True,deployment_enabled=False,
        predictions=[candidate.predict(m) for m in maps]),sort_keys=True))

if __name__=='__main__':main()
