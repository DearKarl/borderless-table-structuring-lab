"""Four matched policies operating on one already generated fixed candidate."""
import hashlib,json
from pathlib import Path
from .features import FEATURE_NAMES,simple_rule
from .model import NetGainModel

def decisions(features,model):
    learned=model.decide(features)
    return {'V7.1':{'accepted':False,'reason':'native_baseline','model_calls':0,'model_rows':0},
            'V7.2':{'accepted':features['common_valid'],'reason':'common_validity','model_calls':0,'model_rows':0},
            'V7.3':{'accepted':simple_rule(features),'reason':'simple_structure_rule','model_calls':0,'model_rows':0},
            'V7.4':learned}

def load_package(path):
    path=Path(path);config=json.loads(path.read_text())
    if config['contract']!='v7_fixed_table_net_gain_v1' or config['features']!=FEATURE_NAMES:raise ValueError('configuration_contract')
    protocol=path.parent/'PROTOCOL.json'
    if hashlib.sha256(protocol.read_bytes()).hexdigest()!=config['protocol_sha256']:raise ValueError('protocol_hash')
    model=NetGainModel(path.parent/config['checkpoint'],config['checkpoint_sha256']) if config['checkpoint'] else None
    return config,model
