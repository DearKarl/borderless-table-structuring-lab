"""Five source-group OOF fits, one final net-gain fit; no validation/test inputs."""
import argparse,datetime,hashlib,json,math,os,threading,time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from .features import FEATURE_NAMES
from .model import NetGainModel

PARAMETERS=dict(loss='squared_error',learning_rate=.05,max_iter=100,max_leaf_nodes=7,
                min_samples_leaf=10,l2_regularization=1,early_stopping=False,random_state=0)
MARGINS=[0.,.0025,.005,.01,.02,'infinity']
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def group_folds(rows):
    groups=sorted({r['source_group'] for r in rows},key=lambda g:(hashlib.sha256(g.encode()).hexdigest(),g))
    mapping={g:i%5 for i,g in enumerate(groups)}
    return [mapping[r['source_group']] for r in rows]

def choose_margin(rows,predictions):
    choices=[]
    for margin in MARGINS:
        threshold=math.inf if margin=='infinity' else margin
        accepts=[r['required_features_known'] and r['common_valid'] and math.isfinite(float(p)) and p>threshold for r,p in zip(rows,predictions)]
        score=sum(r['candidate_quality'] if yes else r['native_quality'] for r,yes in zip(rows,accepts))/len(rows)
        choices.append({'margin':margin,'oof_mean_teds':float(score),'accepted':int(sum(accepts)),
                        'replacements':int(sum(yes and r['raw_output_changed'] for r,yes in zip(rows,accepts)))})
    best=max(c['oof_mean_teds'] for c in choices)
    selected=min((c for c in choices if best-c['oof_mean_teds']<=1e-6),
        key=lambda c:(c['replacements'],-(math.inf if c['margin']=='infinity' else c['margin'])))
    return selected,choices

def export_trees(model):
    return [[{k:node[k].item() for k in ('value','feature_idx','num_threshold','left','right','is_leaf')}
             for node in group[0].nodes] for group in model._predictors]

def saved_predictions(checkpoint,rows):
    out=[]
    for row in rows:
        score=checkpoint['baseline']
        for tree in checkpoint['trees']:
            i=0
            while not tree[i]['is_leaf']:
                n=tree[i];i=n['left'] if row[n['feature_idx']]<=n['num_threshold'] else n['right']
            score+=tree[i]['value']
        out.append(score)
    return out

def fit(path,protocol_path,output,events,resume_completed_folds=False):
    protocol=json.loads(Path(protocol_path).read_text());bundle=json.loads(Path(path).read_text())
    assert bundle['split']=='train' and bundle['scope']=='external_prescribed_training_only'
    rows=bundle['rows'];assert len(rows)==112 and len({r['region_id'] for r in rows})==112
    assert protocol['features']==FEATURE_NAMES and protocol['model_parameters']==PARAMETERS
    output=Path(output);output.mkdir(parents=True,exist_ok=resume_completed_folds)
    x=np.asarray([r['features'] for r in rows],dtype=np.float64)
    y=np.asarray([r['target_net_gain'] for r in rows],dtype=np.float64)
    assert x.shape==(112,34) and np.isfinite(x).all() and np.isfinite(y).all()
    folds=group_folds(rows);oof=np.full(112,np.nan);logs=[]
    def emit(event,**extra):
        with Path(events).open('a',encoding='utf-8') as f:f.write(json.dumps({'at':utc(),'event':event,'pid':os.getpid(),'device':'cpu',**extra})+'\n')
    split={'at':utc(),'method':'sha256(source_group UTF-8) ascending, tie group lexical; round robin modulo5',
           'train_sha256':digest(path),'rows':[{'region_id':r['region_id'],'source_group':r['source_group'],'fold':folds[i]} for i,r in enumerate(rows)]}
    if resume_completed_folds:
        old_split=json.loads((output/'FOLDS.json').read_text())
        assert old_split['train_sha256']==split['train_sha256'] and old_split['rows']==split['rows']
        assert not (output/'net_gain_gbdt.json').exists() and all((output/('fold_'+str(i)+'.json')).exists() for i in range(5))
        completed=[json.loads(line) for line in Path(events).read_text().splitlines() if line.strip()]
        completed=[r for r in completed if r['event']=='model_fit_completed' and r['fit'].startswith('fold_')]
        assert len(completed)==5 and len({r['fit'] for r in completed})==5
        logs.extend({k:v for k,v in r.items() if k not in ('event','at')} for r in completed)
        emit('OOF_recovery_from_saved_folds',completed_folds_refitted=0)
    else:
        write(output/'FOLDS.json',split);emit('training_folds_frozen',sha256=digest(output/'FOLDS.json'))
    def one_fit(label,mask):
        start=time.monotonic();started=utc();emit('model_fit_started',fit=label,rows=int(mask.sum()))
        def stop():emit('model_fit_failed',fit=label,reason='1800second_hard_timeout');os._exit(124)
        timer=threading.Timer(1800,stop);timer.daemon=True;timer.start()
        try:model=HistGradientBoostingRegressor(**PARAMETERS).fit(x[mask],y[mask],sample_weight=np.ones(int(mask.sum())))
        except BaseException as exc:emit('model_fit_failed',fit=label,error=type(exc).__name__+':'+str(exc));raise
        finally:timer.cancel()
        log={'fit':label,'pid':os.getpid(),'device':'cpu','started_at':started,'finished_at':utc(),'elapsed_seconds':time.monotonic()-start,
             'rows':int(mask.sum()),'source_groups':len({r['source_group'] for i,r in enumerate(rows) if mask[i]})}
        logs.append(log);emit('model_fit_completed',**{k:v for k,v in log.items() if k not in ('pid','device')});return model
    for fold in range(5):
        held=np.asarray(folds)==fold;mask=~held
        assert not {rows[i]['source_group'] for i in np.where(mask)[0]} & {rows[i]['source_group'] for i in np.where(held)[0]}
        if resume_completed_folds:
            saved=json.loads((output/('fold_'+str(fold)+'.json')).read_text());assert saved['held_fold']==fold
            oof[held]=saved_predictions(saved,x[held])
        else:
            model=one_fit('fold_'+str(fold),mask);oof[held]=model.predict(x[held])
            write(output/('fold_'+str(fold)+'.json'),{'baseline':float(model._baseline_prediction[0,0]),'trees':export_trees(model),'held_fold':fold})
    assert np.isfinite(oof).all()
    selected,choices=choose_margin(rows,oof)
    write(output/'OOF.json',{'at':utc(),'train_sha256':digest(path),'folds_sha256':digest(output/'FOLDS.json'),'selection':selected,'choices':choices,
        'rows':[{'region_id':r['region_id'],'fold':folds[i],'predicted_gain':float(oof[i]),'target_gain':r['target_net_gain']} for i,r in enumerate(rows)]})
    emit('margin_selected_training_only',**selected)
    model=one_fit('final',np.ones(112,dtype=bool))
    checkpoint={'contract':'v7_fixed_table_net_gain_v1','features':FEATURE_NAMES,'parameters':PARAMETERS,'preprocessing':'None; finite parser placeholders/flags defined by frozen features.py',
        'margin':selected['margin'],'baseline':float(model._baseline_prediction[0,0]),'trees':export_trees(model),
        'train_sha256':digest(path),'protocol_sha256':digest(protocol_path),'folds_sha256':digest(output/'FOLDS.json'),'oof_sha256':digest(output/'OOF.json'),
        'training_rows':112,'source_groups':len({r['source_group'] for r in rows}),'region_weight':1}
    target=output/'net_gain_gbdt.json';write(target,checkpoint)
    loaded=NetGainModel(target,digest(target));actual=np.asarray(loaded.predict(x.tolist()));expected=model.predict(x)
    difference=float(np.max(np.abs(actual-expected)));assert difference<=1e-10
    threshold=math.inf if selected['margin']=='infinity' else selected['margin'];assert np.array_equal(actual>threshold,expected>threshold)
    receipt={'at':utc(),'final_checkpoint_sha256':digest(target),'margin':selected['margin'],'final_models':1,'fold_models':5,'fit_logs':logs,
        'fit_seconds_sum':sum(r['elapsed_seconds'] for r in logs),'numpy':np.__version__,'sklearn':sklearn.__version__,
        'device':'cpu','threads':1,'export_max_abs_error':difference,'identical_loaded_threshold_decisions':True,
        'recovered_OOF_from_saved_folds':resume_completed_folds,'completed_folds_refitted':0,
        'train_sha256':digest(path),'validation_or_test_access':False,'model_calls_for_export_validation':loaded.calls,'rows_for_export_validation':loaded.rows_predicted}
    write(output/'TRAINING_RECEIPT.json',receipt);return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--events',required=True);p.add_argument('--resume-completed-folds',action='store_true');a=p.parse_args()
    print(json.dumps(fit(a.train,a.protocol,a.output,a.events,a.resume_completed_folds),indent=2))
