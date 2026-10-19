"""Offline prediction using the exact trained token/coordinate pipeline."""
import subprocess
import sys
import time
from experiments.core import *
from experiments.weak_core import validate_weak_bundle
from experiments.spatial_core import *


def load_spatial_predictor(bundle,config,device):
    manifest=validate_weak_bundle(bundle,config)
    if manifest['python']!=list(sys.version_info[:2]) or manifest['torch']!=torch.__version__:
        raise ValueError('Use the pinned training runtime.')
    saved=torch.load(bundle/'model.pt',map_location='cpu',weights_only=True)
    if saved['experiment']!=config or saved['labels']!=LABELS: raise ValueError('Wrong trained head.')
    head=SpatialHead(config,saved['state']['mean'],saved['state']['std'])
    head.load_state_dict(saved['state'],strict=True)
    encoder=torch.jit.load(str(bundle/'encoder.ts'),map_location=device).eval()
    return encoder,head.to(device).eval(),make_transform(config),config


def submit_spatial_experiment(config,data,bundle,out):
    from tqdm.auto import tqdm
    data,bundle,out=Path(data),Path(bundle),Path(out); started=time.monotonic()
    validate_weak_bundle(bundle,config)
    subprocess.check_call([sys.executable,'-m','pip','install','-q','--no-index','--no-deps','--find-links',str(bundle/'wheels'),
                           'pydicom==3.0.2','python-gdcm==3.2.6'])
    torch.set_num_threads(1); torch.backends.mha.set_fastpath_enabled(False)
    device='cuda' if torch.cuda.is_available() else 'cpu'; members=[]
    for eid in config['train_dependencies'] if config.get('ensemble') else [None]:
        path=bundle/'members'/eid if eid else bundle
        cfg=json.loads((path/'assets.json').read_text())['experiment']
        members.append(load_spatial_predictor(path,cfg,device))
    test=pd.read_csv(data/'test.csv'); series=pd.read_csv(data/'test_series.csv')
    sample=pd.read_csv(data/'sample_submission.csv',nrows=0)
    if not len(test) or test.StudyInstanceUID.isna().any() or not test.StudyInstanceUID.is_unique or not series.SeriesInstanceUID.is_unique:
        raise ValueError('Invalid test metadata.')
    predictions=[]
    with torch.inference_mode():
        for uid in tqdm(test.StudyInstanceUID,desc=config['id']+' test'):
            probs=[]
            for encoder,head,transform,cfg in members:
                tokens,mask,coords=spatial_tokens(uid,series,data/'test_series',encoder,transform,cfg,device)
                probs.append(head(tokens[None].to(device),mask[None].to(device),coords[None].to(device)).sigmoid().cpu().numpy()[0])
            predictions.append(np.mean(probs,axis=0))
    out.mkdir(parents=True,exist_ok=True)
    frame=save_submission(test.StudyInstanceUID.tolist(),predictions,sample.columns,out/'submission.csv')
    (out/'inference_summary.json').write_text(json.dumps(dict(experiment=config,studies=len(test),labels=12,
        seconds=time.monotonic()-started,manifest_sha256=file_hash(bundle/'assets.json'),internet_required=False),indent=2),encoding='utf-8')
    return frame
