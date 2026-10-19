"""Exact saved preprocessing/head; no online weights or rank fitting on test data."""
import subprocess
import sys
import time
from experiments.anatomy_core import *
from experiments.anatomy_train import load_anatomy_head
from experiments.weak_core import validate_weak_bundle


def load_anatomy_predictor(bundle,config,device):
    meta=validate_weak_bundle(bundle,config)
    if meta['python']!=list(sys.version_info[:2]) or meta['torch']!=torch.__version__: raise ValueError('Use the pinned training runtime.')
    if config.get('ensemble'):
        members=[]
        for eid in config['train_dependencies']:
            path=bundle/'members'/eid;cfg=json.loads((path/'assets.json').read_text())['experiment']
            members.append(load_anatomy_predictor(path,cfg,device))
        return dict(config=config,members=members)
    return dict(config=config,encoder=torch.jit.load(str(bundle/'encoder.ts'),map_location=device).eval(),
        head=load_anatomy_head(bundle/'model.pt',config,device),reference=torch.load(bundle/'reference.pt',map_location='cpu',weights_only=True).numpy())


@torch.inference_mode()
def anatomy_probability(predictor,uid,series,image_root,device,flip=False):
    cfg=predictor['config']
    if cfg.get('ensemble'):
        if cfg['method']=='tta':
            member=predictor['members'][0]
            return (anatomy_probability(member,uid,series,image_root,device,False)+anatomy_probability(member,uid,series,image_root,device,True))/2
        values=[]
        for member in predictor['members']:
            p=anatomy_probability(member,uid,series,image_root,device)
            if cfg['method']=='rank': p=reference_cdf(member['reference'],p[None])[0]
            values.append(p)
        return np.mean(values,axis=0)
    tokens,mask,coords=anatomy_tokens(uid,series,image_root,predictor['encoder'],cfg,device,flip=flip)
    return predictor['head'](tokens[None].to(device),mask[None].to(device),coords[None].to(device)).sigmoid().cpu().numpy()[0]


def submit_anatomy_experiment(config,data,bundle,out):
    data,bundle,out=Path(data),Path(bundle),Path(out);started=time.monotonic()
    validate_weak_bundle(bundle,config)
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(bundle/'wheels'),
        'pydicom==3.0.2','python-gdcm==3.2.6'])
    torch.set_num_threads(1);torch.backends.mha.set_fastpath_enabled(False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    predictor=load_anatomy_predictor(bundle,config,device)
    test=pd.read_csv(data/'test.csv');series=pd.read_csv(data/'test_series.csv');columns=pd.read_csv(data/'sample_submission.csv',nrows=0).columns
    if not len(test) or test.StudyInstanceUID.isna().any() or not test.StudyInstanceUID.is_unique or not series.SeriesInstanceUID.is_unique:
        raise ValueError('Invalid test metadata.')
    probabilities=[anatomy_probability(predictor,uid,series,data/'test_series',device) for uid in test.StudyInstanceUID]
    out.mkdir(parents=True,exist_ok=True);result=save_submission(test.StudyInstanceUID.tolist(),probabilities,columns,out/'submission.csv')
    (out/'inference_summary.json').write_text(json.dumps(dict(experiment=config,studies=len(test),labels=12,
        seconds=time.monotonic()-started,manifest_sha256=file_hash(bundle/'assets.json'),internet_required=False),indent=2))
    return result
