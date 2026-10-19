"""Recursive fixed-weight inference: R01 = .5 E09 + .25 S11 + .25 S12."""
from experiments.core import *
from experiments.weak_core import *
from experiments.spatial_core import *
from experiments.weak_submit import load_weak_predictor
from experiments.spatial_submit import load_spatial_predictor
import subprocess
import sys
import time


def predictor_leaves(bundle, weight=1.):
    bundle=Path(bundle)
    config=json.loads((bundle/'assets.json').read_text())['experiment']
    if not config.get('ensemble'): return [(bundle,config,weight)]
    dependencies=config['train_dependencies']
    if not dependencies: raise ValueError('Empty ensemble.')
    return [leaf for eid in dependencies for leaf in
            predictor_leaves(bundle/'members'/eid,weight/len(dependencies))]


def submit_followup_experiment(config,data,bundle,out):
    from tqdm.auto import tqdm
    data,bundle,out=Path(data),Path(bundle),Path(out); started=time.monotonic()
    validate_weak_bundle(bundle,config)
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(bundle/'wheels'),
                           'pydicom==3.0.2','python-gdcm==3.2.6'])
    torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    members=[]
    for path,cfg,weight in predictor_leaves(bundle):
        loader=load_spatial_predictor if cfg.get('architecture')=='spatial_v1' else load_weak_predictor
        members.append((loader(path,cfg,device),weight))
    test=pd.read_csv(data/'test.csv'); series=pd.read_csv(data/'test_series.csv')
    sample=pd.read_csv(data/'sample_submission.csv',nrows=0)
    if (not len(test) or test.StudyInstanceUID.isna().any() or not test.StudyInstanceUID.is_unique
        or not series.SeriesInstanceUID.is_unique or not set(test.StudyInstanceUID)<=set(series.StudyInstanceUID)):
        raise ValueError('Invalid test metadata.')
    predictions=[]
    with torch.inference_mode():
        for uid in tqdm(test.StudyInstanceUID,desc=config['id']+' test'):
            total=np.zeros(12,dtype=np.float64)
            for (encoder,head,transform,cfg),weight in members:
                spatial=cfg.get('architecture')=='spatial_v1'
                tokens,mask,coords=(spatial_tokens if spatial else weak_tokens)(uid,series,data/'test_series',encoder,transform,cfg,device)
                args=[tokens[None].to(device),mask[None].to(device)]
                if spatial: args.append(coords[None].to(device))
                total+=weight*head(*args).sigmoid().cpu().numpy()[0]
            predictions.append(total)
    out.mkdir(parents=True,exist_ok=True)
    frame=save_submission(test.StudyInstanceUID.tolist(),predictions,sample.columns,out/'submission.csv')
    (out/'inference_summary.json').write_text(json.dumps(dict(experiment=config,studies=len(test),labels=12,
        seconds=time.monotonic()-started,manifest_sha256=file_hash(bundle/'assets.json'),internet_required=False),indent=2),encoding='utf-8')
    return frame
