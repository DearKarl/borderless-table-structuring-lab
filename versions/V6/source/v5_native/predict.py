"""Load a named V6 package and predict one direct regional scale, without OCR."""
import argparse
import json
from pathlib import Path

from .direct_controller import DirectScaleModel
from .io_utils import read_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True);p.add_argument('--kind',choices=('table','equation'),required=True)
    p.add_argument('--features',required=True);p.add_argument('--width',type=int,required=True);p.add_argument('--height',type=int,required=True)
    a=p.parse_args();path=Path(a.package).resolve();config=read_json(path)
    if a.kind not in config['kinds']:raise ValueError('Task type is not enabled by this version')
    if a.width<1 or a.height<1:raise ValueError('Positive native crop dimensions are required')
    binding=config['controllers'][a.kind];model_path=(path.parent/binding['file']).resolve()
    if not model_path.is_relative_to(path.parent):raise ValueError('Checkpoint must be inside this package')
    model=DirectScaleModel(model_path,binding['sha256']);features=read_json(a.features)
    print(json.dumps({'version':config['version'],'kind':a.kind,'decision':model.choose(features,a.width,a.height)},indent=2))


if __name__=='__main__':main()
