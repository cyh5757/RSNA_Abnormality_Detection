"""Gold-held-out heads, saved fold models, fixed-reference rank fusion."""
import shutil
import time
from experiments.anatomy_core import *
from experiments.weak_core import make_targets,find_report_labels,find_training_run,validate_weak_bundle,weighted_bce
from experiments.weak_train import write_weak_metrics,write_weak_manifest
from experiments.spatial_train import spatial_predict
from experiments.train import export_encoder


def anatomy_statistics(tokens,mask,index):
    total=torch.zeros(tokens.shape[-1],dtype=torch.float64);squares=total.clone();n=0
    for batch in torch.as_tensor(index).long().split(16):
        v=quadrant_mean(tokens[batch].double())[mask[batch]]
        if not torch.isfinite(v).all(): raise ValueError('Invalid training features.')
        total+=v.sum(0);squares+=v.square().sum(0);n+=len(v)
    if n<2: raise ValueError('Insufficient training features.')
    return (total/n).float(),((squares-total.square()/n)/(n-1)).clamp_min(0).sqrt().clamp_min(.01).float()


def fit_anatomy_head(tokens,mask,coords,y,weights,index,seed,config):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(seed)
    index=torch.as_tensor(index).long()
    head=AnatomyHead(config,*anatomy_statistics(tokens,mask,index)).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay']);history=[]
    for epoch in range(1,config['epochs']+1):
        head.train();numerator=denominator=0.;steps=0
        for batch in index[torch.randperm(len(index))].split(config['batch_size']):
            weight=weights[batch].sum().item()
            if weight==0: continue
            optimizer.zero_grad(set_to_none=True)
            logits=head(tokens[batch].to(device),mask[batch].to(device),coords[batch].to(device))
            loss=weighted_bce(logits,y[batch].to(device),weights[batch].to(device))
            if not torch.isfinite(loss): raise ValueError('Non-finite head loss.')
            loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
            numerator+=loss.item()*weight;denominator+=weight;steps+=1
        if not steps: raise ValueError('No supervised batches.')
        history.append(dict(epoch=epoch,train_loss=numerator/denominator,optimizer_steps=steps))
    return head.eval(),history


def save_anatomy_head(head,config,path,**extra):
    torch.save(dict(state={k:v.detach().cpu() for k,v in head.state_dict().items()},experiment=config,labels=LABELS,**extra),path)


def load_anatomy_head(path,config,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['experiment']!=config or saved['labels']!=LABELS: raise ValueError('Wrong saved anatomy head.')
    head=AnatomyHead(config,saved['state']['mean'],saved['state']['std'])
    head.load_state_dict(saved['state'],strict=True)
    return head.to(device).eval()


def frame_metrics(frame,truth,config,out):
    runs=[]
    for seed,rows in frame.groupby('seed'):
        y=truth.loc[rows.StudyInstanceUID,LABELS].to_numpy(dtype=np.float32)
        logits=torch.logit(torch.tensor(rows[LABELS].to_numpy(),dtype=torch.float32).clamp(1e-7,1-1e-7))
        scores=[dict(fold=fold,**metric_scores(y[rows.fold.to_numpy()==fold],logits[rows.fold.to_numpy()==fold])) for fold in range(3)]
        runs.append(dict(seed=int(seed),**metric_scores(y,logits),folds=scores))
    frame.to_csv(out/'oof_predictions.csv',index=False)
    return write_weak_metrics(config,runs,out)


def evaluate_anatomy_cv(tokens,mask,coords,y,w,gold,ids,config,out):
    gi,wi=np.flatnonzero(gold),np.flatnonzero(~gold); gids=np.asarray(ids)[gold];folds=make_folds(gids.tolist(),y[gi].numpy())
    pd.DataFrame(dict(StudyInstanceUID=gids,fold=folds)).to_csv(out/'folds.csv',index=False)
    (out/'fold_models').mkdir();frames=[];ranks=[];history=[]
    for seed in SEEDS:
        for fold in range(3):
            train_index=np.concatenate((wi,gi[folds!=fold]));valid=gi[folds==fold]
            head,h=fit_anatomy_head(tokens,mask,coords,y,w,train_index,seed,config)
            probabilities=spatial_predict(head,tokens,mask,coords,valid).sigmoid().numpy()
            # Reference contains only training weak studies; never ranks held-out gold together.
            reference=spatial_predict(head,tokens,mask,coords,wi).sigmoid().numpy()
            frame=pd.DataFrame(probabilities,columns=LABELS)
            frame.insert(0,'fold',fold);frame.insert(0,'seed',seed);frame.insert(0,'StudyInstanceUID',np.asarray(ids)[valid]);frames.append(frame)
            ranked=frame.copy();ranked[LABELS]=reference_cdf(reference,probabilities);ranks.append(ranked)
            save_anatomy_head(head,config,out/'fold_models'/f'{seed}_{fold}.pt',fold=fold,seed=seed,validation_ids=np.asarray(ids)[valid].tolist())
            history.extend(dict(seed=seed,fold=fold,**row) for row in h)
            print(f'{config["id"]}: seed={seed} fold={fold} complete',flush=True)
            del head
    frame=pd.concat(frames,ignore_index=True);pd.concat(ranks,ignore_index=True).to_csv(out/'oof_rank_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    truth=pd.DataFrame(y.numpy(),columns=LABELS,index=ids)
    return frame_metrics(frame,truth,config,out)


def anatomy_signature(config):
    return {k:config[k] for k in ('crop','resize','interpolation','neighbors','centers','representation','field_mm','canonical','pipeline_sha256')}


def checked_anatomy_run(eid):
    path=find_training_run(eid);cfg=json.loads((path/'bundle/assets.json').read_text())['experiment']
    meta=validate_weak_bundle(path/'bundle',cfg)
    return path,cfg,meta


def read_anatomy_cache(path,meta):
    if file_hash(path/'tokens.pt')!=meta['tokens_sha256']: raise ValueError('Corrupt cached features.')
    return torch.load(path/'tokens.pt',map_location='cpu',weights_only=True)


def extract_anatomy_cache(ids,series,image_root,encoder,config,flip=False):
    count=config['centers']*config['neighbors'];n=len(ids)
    tokens=torch.zeros(n,6,count,4,config['feature_dim'],dtype=torch.float16)
    mask=torch.zeros(n,6,count,dtype=torch.bool);coords=torch.zeros(n,6,count,3)
    audit=[];started=time.monotonic();encoder=encoder.cuda().eval()
    for i,uid in enumerate(ids):
        if time.monotonic()-started>7*3600: raise TimeoutError('Feature extraction exceeded 7h; no partial result accepted.')
        tokens[i],mask[i],coords[i]=anatomy_tokens(uid,series,image_root,encoder,config,'cuda',audit,flip)
        if i%50==0: print(f'features {i}/{n} elapsed={(time.monotonic()-started)/60:.1f} min',flush=True)
    encoder.cpu();torch.cuda.empty_cache()
    return dict(ids=list(ids),tokens=tokens,mask=mask,coords=coords),audit


def blend_oof(frames):
    keys=['StudyInstanceUID','seed','fold'];ordered=[f.sort_values(['seed','StudyInstanceUID']).reset_index(drop=True) for f in frames]
    if not ordered or any(not ordered[0][keys].equals(f[keys]) for f in ordered[1:]):
        raise ValueError('Ensemble OOF patients/seeds/folds differ.')
    result=ordered[0].copy();result[LABELS]=np.mean([f[LABELS].to_numpy() for f in ordered],axis=0)
    return result


def train_anatomy_fusion(config,data,out):
    parents=[checked_anatomy_run(eid) for eid in config['train_dependencies']]
    paths=[p[0] for p in parents]
    if config['method']=='tta':
        path,cfg,meta=parents[0]
        if file_hash(path/'oof_predictions.csv')!=meta['oof_sha256']['oof_predictions.csv']: raise ValueError('Corrupt OOF predictions.')
        frame=pd.read_csv(path/'oof_predictions.csv');ids=frame.StudyInstanceUID.drop_duplicates().tolist()
        encoder=torch.jit.load(str(path/'bundle/encoder.ts'),map_location='cpu')
        cache,_=extract_anatomy_cache(ids,pd.read_csv(data/'train_series.csv'),data/'train_series',encoder,cfg,flip=True)
        lookup={uid:i for i,uid in enumerate(ids)}
        for (seed,fold),rows in frame.groupby(['seed','fold']):
            headpath=path/'fold_models'/f'{seed}_{fold}.pt'
            if file_hash(headpath)!=meta['fold_models_sha256'][headpath.name]: raise ValueError('Corrupt fold head.')
            saved=torch.load(headpath,map_location='cpu',weights_only=True)
            if set(saved['validation_ids'])!=set(rows.StudyInstanceUID): raise ValueError('Wrong validation fold.')
            head=load_anatomy_head(headpath,cfg,'cuda')
            indices=[lookup[uid] for uid in rows.StudyInstanceUID]
            p=spatial_predict(head,cache['tokens'],cache['mask'],cache['coords'],indices).sigmoid().numpy()
            frame.loc[rows.index,LABELS]=(rows[LABELS].to_numpy()+p)/2
    else:
        filename='oof_rank_predictions.csv' if config['method']=='rank' else 'oof_predictions.csv'
        for path,cfg,meta in parents:
            if file_hash(path/filename)!=meta['oof_sha256'][filename]: raise ValueError('Corrupt OOF predictions.')
        frame=blend_oof([pd.read_csv(p/filename) for p in paths])
    metrics=frame_metrics(frame,pd.read_csv(data/'train.csv').set_index('StudyInstanceUID'),config,out)
    shutil.copy2(paths[0]/'folds.csv',out/'folds.csv');bundle=out/'bundle'
    for path,cfg,meta in parents: shutil.copytree(path/'bundle',bundle/'members'/cfg['id'])
    shutil.copytree(paths[0]/'bundle/wheels',bundle/'wheels')
    write_weak_manifest(config,bundle,4350,58,dict(mixing=config['method'],members=config['train_dependencies']))
    return metrics


def train_anatomy_experiment(config,data,out):
    data,out=Path(data),Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory.')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(42);torch.backends.mha.set_fastpath_enabled(False)
    if config.get('ensemble'): return train_anatomy_fusion(config,data,out)
    if not torch.cuda.is_available(): raise RuntimeError('A Kaggle GPU is required.')
    table,y,w,gold,groups,audit=make_targets(pd.read_csv(data/'train.csv'),pd.read_csv(find_report_labels(config)),config)
    if len(table)!=4350 or int(gold.sum())!=58: raise ValueError('Training population changed.')
    y,w=torch.from_numpy(y),torch.from_numpy(w);ids=table.StudyInstanceUID.tolist()
    (out/'label_audit.json').write_text(json.dumps(audit,indent=2))
    parent,old,meta=checked_anatomy_run(config['train_dependencies'][0]);bundle=out/'bundle';bundle.mkdir()
    encoder=QuadrantSwin(False)
    old_encoder=torch.jit.load(str(parent/'bundle/encoder.ts'),map_location='cpu')
    state={k:v for k,v in old_encoder.state_dict().items() if k.startswith(('features.','norm.'))}
    encoder.load_state_dict(state,strict=True);del old_encoder
    if config.get('adapt_stage'):
        from experiments.anatomy_adapt import find_adaptation,adapt_recipe
        cp=find_adaptation(config['adapt_stage'])
        progress=json.loads((cp/'progress.json').read_text())
        if progress['checkpoint_sha256']!=file_hash(cp/'checkpoint.pt') or progress['recipe']!=adapt_recipe(config):
            raise ValueError('Wrong adaptation checkpoint.')
        saved=torch.load(cp/'checkpoint.pt',map_location='cpu',weights_only=True)
        if saved['epoch']!=config['backbone_epochs'] or saved['segment']!=3: raise ValueError('Incomplete adaptation epoch.')
        encoder.load_state_dict(saved['best_encoder'],strict=True)
        (out/'backbone_audit.json').write_text(json.dumps(progress,indent=2));del saved
    encoder=export_encoder(encoder,dict(config,slices=16),bundle/'encoder.ts');encoder_hash=file_hash(bundle/'encoder.ts')
    started=time.monotonic();reused=False
    if old.get('architecture')=='anatomy_v1' and anatomy_signature(old)==anatomy_signature(config) and not config.get('adapt_stage'):
        cache=read_anatomy_cache(parent,meta)
        # Re-export may have a different archive hash; compare original tensor states, not archive bytes.
        shutil.copy2(parent/'bundle/encoder.ts',bundle/'encoder.ts');encoder_hash=file_hash(bundle/'encoder.ts');reused=True
        if cache['encoder_sha256']!=encoder_hash or cache['ids']!=ids or cache['view_signature']!=anatomy_signature(config):
            raise ValueError('Incompatible cached anatomy features.')
    else:
        cache,geometry=extract_anatomy_cache(ids,pd.read_csv(data/'train_series.csv'),data/'train_series',encoder,config)
        pd.DataFrame(geometry).to_csv(out/'selection_audit.csv',index=False)
    cache.update(view_signature=anatomy_signature(config),encoder_sha256=encoder_hash)
    torch.save(cache,out/'tokens.pt');del encoder
    tokens,mask,coords=(cache[k] for k in ('tokens','mask','coords'))
    timing=dict(feature_seconds=time.monotonic()-started,cached_studies=len(ids) if reused else 0,images=int(mask.sum()),gpu=torch.cuda.get_device_name(0))
    metrics=evaluate_anatomy_cv(tokens,mask,coords,y,w,gold,ids,config,out)
    head,history=fit_anatomy_head(tokens,mask,coords,y,w,np.arange(len(ids)),42,config)
    save_anatomy_head(head,config,bundle/'model.pt')
    reference=spatial_predict(head,tokens,mask,coords,np.flatnonzero(~gold)).sigmoid()
    torch.save(reference,bundle/'reference.pt');pd.DataFrame(history).to_csv(out/'final_fit_history.csv',index=False)
    shutil.copytree(parent/'bundle/wheels',bundle/'wheels')
    write_weak_manifest(config,bundle,4350,58,dict(tokens_sha256=file_hash(out/'tokens.pt'),folds_sha256=file_hash(out/'folds.csv'),
        head_parameters=sum(p.numel() for p in head.parameters()),label_audit=audit,
        fold_models_sha256={p.name:file_hash(p) for p in (out/'fold_models').glob('*.pt')},
        oof_sha256={name:file_hash(out/name) for name in ('oof_predictions.csv','oof_rank_predictions.csv')}))
    (out/'timings.json').write_text(json.dumps(timing,indent=2));return metrics
