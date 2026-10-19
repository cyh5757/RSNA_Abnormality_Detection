"""Weak-only full encoder adaptation, four bounded resumable jobs per epoch."""
import hashlib
import time
from experiments.anatomy_core import *
from experiments.anatomy_train import (checked_anatomy_run,read_anatomy_cache,fit_anatomy_head)
from experiments.weak_core import make_targets,find_report_labels,weighted_bce
from experiments.spatial_train import weak_finetune_split


def adapt_recipe(config):
    keys=('pipeline_sha256','labels_sha256')
    return hashlib.sha256(json.dumps({k:config[k] for k in keys},sort_keys=True).encode()).hexdigest()


def segment_indices(gold,groups,epoch,segment):
    if epoch<1 or segment not in range(4): raise ValueError('Invalid adaptation stage.')
    train_index,valid=weak_finetune_split(gold,groups)
    order=np.random.default_rng(42+epoch).permutation(train_index)
    # Partition accumulation batches, avoiding optimizer steps split across jobs.
    batches=[order[i:i+8] for i in range(0,len(order),8)]
    selected=np.array_split(np.arange(len(batches)),4)[segment]
    return (np.concatenate([batches[i] for i in selected]) if len(selected) else np.array([],dtype=int)),valid


def validate_checkpoint(checkpoint,config,epoch,segment):
    prior=(epoch,segment-1) if segment else (epoch-1,3)
    if checkpoint.get('recipe')!=adapt_recipe(config) or (checkpoint.get('epoch'),checkpoint.get('segment'))!=prior:
        raise ValueError('Checkpoint recipe or predecessor stage mismatch.')


def find_adaptation(stage,input_root=Path('/kaggle/input')):
    slug='rsna-knee-'+stage.replace('_','-')+'-train'
    paths=[input_root/slug/'adapt',input_root/'notebooks/choiyoungwhan'/slug/'adapt',input_root/'kernels/choiyoungwhan'/slug/'adapt']
    found=[p for p in paths if (p/'progress.json').is_file()]
    if len(found)!=1: raise ValueError('Attach exactly the pinned adaptation stage '+stage)
    return found[0]


def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu()
    if isinstance(value,dict): return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu_tree(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu_tree(v) for v in value)
    return value


def adapt_anatomy(config,epoch,segment,previous_stage,data,out):
    from torch.utils.checkpoint import checkpoint as recompute
    data,out=Path(data),Path(out)
    if not torch.cuda.is_available(): raise RuntimeError('Adaptation requires a GPU.')
    if out.exists() and any(out.iterdir()): raise FileExistsError('Output is not empty.')
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    table,y,w,gold,groups,audit=make_targets(pd.read_csv(data/'train.csv'),pd.read_csv(find_report_labels(config)),config)
    if len(table)!=4350 or int(gold.sum())!=58: raise ValueError('Training population changed.')
    train_index,valid=segment_indices(gold,groups,epoch,segment)
    if not len(train_index): raise ValueError('Empty adaptation segment.')
    y,w=torch.from_numpy(y).cuda(),torch.from_numpy(w).cuda();series=pd.read_csv(data/'train_series.csv')
    base,base_cfg,meta=checked_anatomy_run('u03_patch_attention')
    origin=file_hash(base/'bundle/encoder.ts');encoder=QuadrantSwin(False)
    saved=None
    if previous_stage:
        folder=find_adaptation(previous_stage);progress=json.loads((folder/'progress.json').read_text())
        if file_hash(folder/'checkpoint.pt')!=progress['checkpoint_sha256']: raise ValueError('Corrupt checkpoint.')
        saved=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
        validate_checkpoint(saved,config,epoch,segment)
        if saved['origin_sha256']!=origin: raise ValueError('Adaptation origin changed.')
        encoder.load_state_dict(saved['encoder'])
        head=AnatomyHead(config,saved['head']['mean'],saved['head']['std']);head.load_state_dict(saved['head'])
    else:
        if (epoch,segment)!=(1,0): raise ValueError('Missing continuation checkpoint.')
        encoder.load_state_dict(torch.jit.load(str(base/'bundle/encoder.ts'),map_location='cpu').state_dict())
        cache=read_anatomy_cache(base,meta);weak_train,_=weak_finetune_split(gold,groups)
        head,_=fit_anatomy_head(cache['tokens'],cache['mask'],cache['coords'],y.cpu(),w.cpu(),weak_train,42,dict(config,epochs=1))
        del cache
    encoder=encoder.cuda().requires_grad_(True);head=head.cuda()
    optimizer=torch.optim.AdamW([dict(params=encoder.parameters(),lr=1e-5),dict(params=head.parameters(),lr=1e-4)],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    if saved:
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    best=saved['best_score'] if saved else float('inf');best_encoder=saved['best_encoder'] if saved else None
    best_epoch=saved['best_epoch'] if saved else None;history=saved['history'] if saved else []
    started=time.monotonic();deadline=started+7*3600

    def predict(index,training):
        if time.monotonic()>deadline: raise TimeoutError('Adaptation segment exceeded 7h; resume previous completed segment.')
        images,mask,coords=anatomy_views(table.StudyInstanceUID.iloc[index],series,data/'train_series',config,limit=8)
        with torch.autocast('cuda',dtype=torch.float16):
            values=[]
            for chunk in images[mask].split(4):
                chunk=chunk.cuda()
                values.append(recompute(encoder,chunk,use_reentrant=False) if training else encoder(chunk))
            features=torch.cat(values);mask=mask.cuda();tokens=features.new_zeros(*mask.shape,4,config['feature_dim']);tokens[mask]=features
            return head(tokens[None],mask[None],coords[None].cuda())

    encoder.train();head.train();total=0.;steps=0
    for start in range(0,len(train_index),8):
        batch=train_index[start:start+8];optimizer.zero_grad(set_to_none=True)
        for index in batch:
            loss=weighted_bce(predict(index,True),y[index:index+1],w[index:index+1])
            if not torch.isfinite(loss): raise ValueError('Non-finite adaptation loss.')
            scaler.scale(loss/len(batch)).backward();total+=loss.item()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_([*encoder.parameters(),*head.parameters()],5.)
        scaler.step(optimizer);scaler.update();steps+=1
        if start%80==0: print(f'full FT epoch={epoch} segment={segment+1}/4 studies={start}/{len(train_index)}',flush=True)
    validation=None
    if segment==3:
        encoder.eval();head.eval();numerator=denominator=0.
        with torch.inference_mode():
            for index in valid:
                mass=w[index:index+1].sum().item()
                numerator+=weighted_bce(predict(index,False),y[index:index+1],w[index:index+1]).item()*mass;denominator+=mass
        validation=numerator/denominator
        if not np.isfinite(validation): raise ValueError('Non-finite weak validation loss.')
        if validation<best: best,best_encoder,best_epoch=validation,cpu_tree(encoder.state_dict()),epoch
    history.append(dict(epoch=epoch,segment=segment,studies=len(train_index),optimizer_steps=steps,
        train_loss=total/len(train_index),weak_validation_bce=validation,seconds=time.monotonic()-started))
    record=dict(recipe=adapt_recipe(config),origin_sha256=origin,epoch=epoch,segment=segment,
        encoder=cpu_tree(encoder.state_dict()),head=cpu_tree(head.state_dict()),optimizer=cpu_tree(optimizer.state_dict()),scaler=scaler.state_dict(),
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),best_score=best,best_encoder=best_encoder,best_epoch=best_epoch,history=history)
    torch.save(record,out/'checkpoint.pt')
    progress=dict(recipe=record['recipe'],epoch=epoch,segment=segment,complete=True,checkpoint_sha256=file_hash(out/'checkpoint.pt'),
        origin_sha256=origin,gold_studies_used_for_backbone=0,training_weak_studies=len(weak_finetune_split(gold,groups)[0]),
        validation_weak_studies=len(valid),trainable_parameters=sum(p.numel() for p in encoder.parameters()),best_epoch=best_epoch,
        validation_bce=None if best==float('inf') else best,history=history)
    (out/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
