"""Train coordinate-aware heads and export offline bundles."""
import shutil
import subprocess
import sys
import time
from experiments.core import *
from experiments.weak_core import make_targets, find_report_labels, find_training_run, validate_weak_bundle, weighted_bce, per_label_weighted_bce
from experiments.weak_train import write_weak_metrics, write_weak_manifest, train_weak_ensemble, freeze_swin_except_last_stage
from experiments.train import export_encoder
from experiments.spatial_core import *


def feature_statistics(tokens, mask, indices):
    total=torch.zeros(tokens.shape[-1],dtype=torch.float64)
    squares=torch.zeros_like(total); count=0
    for batch in torch.as_tensor(indices,dtype=torch.long).split(32):
        values=tokens[batch][mask[batch]].double()
        if not torch.isfinite(values).all(): raise ValueError('Invalid training features.')
        total+=values.sum(0); squares+=(values*values).sum(0); count+=len(values)
    if count<2: raise ValueError('Insufficient training features.')
    mean=total/count
    std=((squares-total.square()/count)/(count-1)).clamp_min(0).sqrt().clamp_min(.01)
    return mean.float(),std.float()


def fit_spatial_head(tokens,mask,coords,y,weights,train_index,seed,config):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    index=torch.as_tensor(train_index,dtype=torch.long)
    head=SpatialHead(config,*feature_statistics(tokens,mask,index)).to(device)
    optimizer=torch.optim.AdamW(head.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    history=[]
    for epoch in range(1,config['epochs']+1):
        head.train(); loss_sum=weight_sum=0.; steps=0
        order=index[torch.randperm(len(index))]
        for batch in order.split(config['batch_size']):
            if weights[batch].sum().item() == 0: continue
            optimizer.zero_grad(set_to_none=True)
            logits=head(tokens[batch].to(device),mask[batch].to(device),coords[batch].to(device))
            loss_fn=per_label_weighted_bce if config.get('loss_reduction')=='per_label' else weighted_bce
            loss=loss_fn(logits,y[batch].to(device),weights[batch].to(device))
            if not torch.isfinite(loss): raise ValueError('Non-finite head loss.')
            loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(),5.); optimizer.step()
            w=1. if config.get('loss_reduction')=='per_label' else weights[batch].sum().item()
            loss_sum+=loss.item()*w; weight_sum+=w; steps+=1
        if not steps: raise ValueError('No supervised head batches.')
        history.append(dict(epoch=epoch,train_loss=loss_sum/weight_sum,optimizer_steps=steps))
    return head.eval(),history


@torch.inference_mode()
def spatial_predict(head,tokens,mask,coords,index):
    device=next(head.parameters()).device
    return torch.cat([head(tokens[b].to(device),mask[b].to(device),coords[b].to(device)).cpu()
                      for b in torch.as_tensor(index,dtype=torch.long).split(32)])


def evaluate_spatial_cv(tokens,mask,coords,y,weights,gold,ids,config,out):
    gold_index=np.flatnonzero(gold); weak_index=np.flatnonzero(~gold)
    gold_ids=np.asarray(ids)[gold].tolist(); gold_y=y[gold_index].numpy()
    folds=make_folds(gold_ids,gold_y)
    pd.DataFrame(dict(StudyInstanceUID=gold_ids,fold=folds)).to_csv(out/'folds.csv',index=False)
    runs=[]; frames=[]; histories=[]
    for seed in SEEDS:
        oof=torch.full((len(gold_index),12),float('nan')); scores=[]
        for fold in range(3):
            train_index=np.concatenate((weak_index,gold_index[folds!=fold])); valid=gold_index[folds==fold]
            head,history=fit_spatial_head(tokens,mask,coords,y,weights,train_index,seed,config)
            logits=spatial_predict(head,tokens,mask,coords,valid); oof[folds==fold]=logits
            scores.append(dict(fold=fold,**metric_scores(gold_y[folds==fold],logits)))
            histories.extend(dict(seed=seed,fold=fold,**row) for row in history)
            print(f"seed={seed} fold={fold} AUC={scores[-1]['macro_auc']:.5f}",flush=True)
            del head
        if not torch.isfinite(oof).all(): raise ValueError('Incomplete OOF predictions.')
        runs.append(dict(seed=seed,**metric_scores(gold_y,oof),folds=scores))
        frame=pd.DataFrame(oof.sigmoid().numpy(),columns=LABELS)
        frame.insert(0,'fold',folds); frame.insert(0,'seed',seed); frame.insert(0,'StudyInstanceUID',gold_ids)
        frames.append(frame)
    pd.concat(frames,ignore_index=True).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame(histories).to_csv(out/'history.csv',index=False)
    return write_weak_metrics(config,runs,out)


def weak_finetune_split(gold, groups):
    weak=np.flatnonzero(~gold)
    valid=weak[np.array([int(g[:8],16)%10==0 for g in groups[weak]])]
    train_index=np.setdiff1d(weak,valid)
    if not len(train_index) or not len(valid) or gold[train_index].any() or gold[valid].any(): raise ValueError('Invalid FT split.')
    return train_index,valid


def finetune_spatial(encoder,cache,table,y,weights,gold,groups,series,data,config,out):
    train_index,valid=weak_finetune_split(gold,groups)
    torch.manual_seed(42)
    warmup_epochs=config.get('head_warmup_epochs',0)
    if warmup_epochs:
        # Fresh head: the S11 final head has seen all gold and must NEVER initialize FT.
        encoder.eval().requires_grad_(False)
        head,warmup=fit_spatial_head(cache['tokens'],cache['mask'],cache['coords'],y,weights,
            train_index,42,dict(config,epochs=warmup_epochs,lr=.001))
        pd.DataFrame(warmup).to_csv(out/'warmup_history.csv',index=False)
    else:
        head=SpatialHead(config,*feature_statistics(cache['tokens'],cache['mask'],train_index)).cuda()
    encoder=encoder.cuda(); parameters=freeze_swin_except_last_stage(encoder)
    head=head.cuda()
    optimizer=torch.optim.AdamW([dict(params=parameters,lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda'); transform=make_transform(config)
    deadline=time.monotonic()+7*3600; best=float('inf'); history=[]; best_epoch=None

    def predict(index):
        if time.monotonic()>deadline: raise TimeoutError('Spatial FT exceeded 7h budget.')
        images,mask,coords=spatial_views(table.StudyInstanceUID.iloc[index],series,data/'train_series',transform,config)
        with torch.autocast('cuda',dtype=torch.float16):
            features=torch.cat([encoder(chunk.cuda()) for chunk in images[mask].split(8)])
            device_mask=mask.cuda(); tokens=features.new_zeros(*mask.shape,config['feature_dim'])
            tokens[device_mask]=features
            return head(tokens[None],device_mask[None],coords[None].cuda())

    for epoch in range(1,config['backbone_epochs']+1):
        encoder.eval(); encoder.features[-1].train(); head.train()
        order=np.random.default_rng(42+epoch).permutation(train_index); total=0.
        for start in range(0,len(order),8):
            batch=order[start:start+8]; optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss=weighted_bce(predict(index),y[index:index+1].cuda(),weights[index:index+1].cuda())
                if not torch.isfinite(loss): raise ValueError('Non-finite FT loss.')
                scaler.scale(loss/len(batch)).backward(); total+=loss.item()
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_([*parameters,*head.parameters()],5.)
            scaler.step(optimizer); scaler.update()
            if start%200==0: print(f'FT epoch={epoch} studies={start}/{len(order)}',flush=True)
        encoder.eval(); head.eval(); value=denominator=0.
        with torch.inference_mode():
            for index in valid:
                w=weights[index:index+1].sum().item()
                value+=weighted_bce(predict(index),y[index:index+1].cuda(),weights[index:index+1].cuda()).item()*w
                denominator+=w
        score=value/denominator
        if not np.isfinite(score): raise ValueError('Invalid weak validation loss.')
        history.append(dict(epoch=epoch,train_loss=total/len(order),weak_validation_bce=score))
        if score<best:
            best,best_epoch=score,epoch
            torch.save({k:v.detach().cpu() for k,v in encoder.state_dict().items()},out/'backbone_best.pt')
    encoder.load_state_dict(torch.load(out/'backbone_best.pt',map_location='cpu',weights_only=True))
    pd.DataFrame(history).to_csv(out/'backbone_history.csv',index=False)
    (out/'backbone_audit.json').write_text(json.dumps(dict(training_weak_studies=len(train_index),
        validation_weak_studies=len(valid),gold_studies_used_for_backbone=0,best_epoch=best_epoch,
        validation_bce=best,head_warmup_epochs=warmup_epochs,warmup_gold_studies=0,
        trainable_parameters=sum(p.numel() for p in parameters))),encoding='utf-8')
    return encoder.cpu().eval().requires_grad_(False)


def compatible_spatial_cache(cache,config,encoder_hash,ids):
    return (cache.get('view_signature')==spatial_signature(config) and cache.get('encoder_sha256')==encoder_hash
            and cache.get('ids')==list(ids))


def train_spatial_experiment(config,encoder_factory,data,out):
    from tqdm.auto import tqdm
    data,out=Path(data),Path(out)
    if out.exists() and any(out.iterdir()): raise FileExistsError('Choose an empty output directory.')
    out.mkdir(parents=True,exist_ok=True)
    if config.get('ensemble'): return train_weak_ensemble(config,data,out)
    if not torch.cuda.is_available(): raise RuntimeError('A Kaggle GPU is required.')
    torch.set_num_threads(1); torch.manual_seed(42); torch.backends.mha.set_fastpath_enabled(False)
    table,y,w,gold,groups,audit=make_targets(pd.read_csv(data/'train.csv'),pd.read_csv(find_report_labels(config)),config)
    if len(table)!=4350 or int(gold.sum())!=58: raise ValueError('Training population changed.')
    y,w=torch.from_numpy(y),torch.from_numpy(w)
    audit.update(source='pilkwang/rsna-knee-llm-labels/versions/1',sha256=config['labels_sha256'])
    (out/'label_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    series=pd.read_csv(data/'train_series.csv')
    if not series.SeriesInstanceUID.is_unique or not set(table.StudyInstanceUID)<=set(series.StudyInstanceUID):
        raise ValueError('Invalid MRI series metadata.')
    bundle=out/'bundle'; bundle.mkdir(); previous=None; cache=None
    if config['train_dependencies']:
        previous=find_training_run(config['train_dependencies'][0])
        meta=json.loads((previous/'bundle/assets.json').read_text())
        validate_weak_bundle(previous/'bundle',meta['experiment'])
        if file_hash(previous/'tokens.pt')!=meta['tokens_sha256']: raise ValueError('Corrupt cached tokens.')
        cache=torch.load(previous/'tokens.pt',map_location='cpu',weights_only=True)
        shutil.copy2(previous/'bundle/encoder.ts',bundle/'encoder.ts')
        encoder=torch.jit.load(str(bundle/'encoder.ts'),map_location='cpu').eval()
    else:
        encoder=export_encoder(encoder_factory(),dict(config,slices=16),bundle/'encoder.ts')
    ids=table.StudyInstanceUID.tolist()
    backbone_started=time.monotonic()
    if config['finetune']:
        if not compatible_spatial_cache(cache,config,file_hash(bundle/'encoder.ts'),ids): raise ValueError('FT requires matching cache.')
        eager=encoder_factory(pretrained=False); eager.load_state_dict(encoder.state_dict(),strict=True)
        eager=finetune_spatial(eager,cache,table,y,w,gold,groups,series,data,config,out)
        encoder=export_encoder(eager,dict(config,slices=16),bundle/'encoder.ts'); del eager
        cache=None
    backbone_seconds=time.monotonic()-backbone_started if config['finetune'] else 0.
    encoder_hash=file_hash(bundle/'encoder.ts'); started=time.monotonic(); reused=False
    torch.cuda.reset_peak_memory_stats()
    if cache is not None and compatible_spatial_cache(cache,config,encoder_hash,ids):
        tokens,mask,coords=(cache[k] for k in ('tokens','mask','coords')); reused=True
    else:
        cache=None; encoder=encoder.cuda().eval(); transform=make_transform(config); rows=[]; selection_audit=[]
        for uid in tqdm(ids,desc=config['id']+' MRI features'):
            if time.monotonic()-started>7*3600: raise TimeoutError('Feature extraction exceeded 7h budget.')
            rows.append(spatial_tokens(uid,series,data/'train_series',encoder,transform,config,'cuda',audit=selection_audit))
        pd.DataFrame(selection_audit).to_csv(out/'selection_audit.csv',index=False)
        tokens,mask,coords=[torch.stack([row[i] for row in rows]) for i in range(3)]
        del rows
    if not torch.isfinite(tokens).all() or not torch.isfinite(coords).all(): raise ValueError('Invalid feature cache.')
    torch.save(dict(ids=ids,tokens=tokens,mask=mask,coords=coords,view_signature=spatial_signature(config),encoder_sha256=encoder_hash),out/'tokens.pt')
    timing=dict(feature_seconds=time.monotonic()-started,backbone_seconds=backbone_seconds,images=int(mask.sum()),cached_studies=len(ids) if reused else 0,
                peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/1024**2,gpu=torch.cuda.get_device_name(0),
                physical_coordinate_tokens=int(((coords[...,2]>.5)&mask).sum()),index_fallback_tokens=int(((coords[...,2]<=.5)&mask).sum()))
    del cache,encoder; torch.cuda.empty_cache()
    metrics=evaluate_spatial_cv(tokens,mask,coords,y,w,gold,ids,config,out)
    head,history=fit_spatial_head(tokens,mask,coords,y,w,np.arange(len(table)),42,config)
    torch.save(dict(state={k:v.detach().cpu() for k,v in head.state_dict().items()},experiment=config,labels=LABELS),bundle/'model.pt')
    pd.DataFrame(history).to_csv(out/'final_fit_history.csv',index=False)
    if previous: shutil.copytree(previous/'bundle/wheels',bundle/'wheels')
    else:
        (bundle/'wheels').mkdir()
        subprocess.check_call([sys.executable,'-m','pip','download','-q','--only-binary=:all:','--no-deps','--dest',str(bundle/'wheels'),
                               'pydicom==3.0.2','python-gdcm==3.2.6'])
    write_weak_manifest(config,bundle,len(table),int(gold.sum()),dict(head_parameters=sum(p.numel() for p in head.parameters()),
        tokens_sha256=file_hash(out/'tokens.pt'),folds_sha256=file_hash(out/'folds.csv'),label_audit=audit))
    (out/'timings.json').write_text(json.dumps(timing,indent=2),encoding='utf-8')
    print(json.dumps(dict(macro_auc=metrics['macro_auc_mean'],timing=timing),indent=2),flush=True)
