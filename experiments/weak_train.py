"""Train report-supervised ablations and save fully offline inference bundles."""
import shutil
import subprocess
import sys
import time
from experiments.core import *
from experiments.weak_core import *
from experiments.train import export_encoder


def write_weak_metrics(config, runs, out):
    metrics = {'experiment': config, 'runs': runs,
               'macro_auc_mean': float(np.mean([r['macro_auc'] for r in runs])),
               'macro_auc_seed_std': float(np.std([r['macro_auc'] for r in runs])),
               'fold_auc_mean': float(np.mean([f['macro_auc'] for r in runs for f in r['folds']])),
               'bce_mean': float(np.mean([r['bce'] for r in runs])),
               'note': 'Gold58 development OOF; public teacher independence is not established. Seed SD is not a CI.'}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    return metrics


def evaluate_weak_cv(tokens, mask, y, weights, gold, ids, config, out):
    gold_index = np.flatnonzero(gold)
    weak_index = np.flatnonzero(~gold)
    gold_ids = np.asarray(ids)[gold].tolist()
    gold_y = y[gold_index].cpu().numpy()
    folds = make_folds(gold_ids, gold_y)
    pd.DataFrame({'StudyInstanceUID': gold_ids, 'fold': folds}).to_csv(out / 'folds.csv', index=False)
    runs, predictions, histories = [], [], []
    for seed in SEEDS:
        oof = torch.full((len(gold_index), 12), float('nan'))
        scores = []
        for fold in range(3):
            train_index = np.concatenate([weak_index, gold_index[folds != fold]])
            valid = gold_index[folds == fold]
            head, history = fit_weak_head(tokens, mask, y, weights, train_index, seed, config)
            with torch.inference_mode():
                logits = head(tokens[valid], mask[valid]).cpu()
            oof[folds == fold] = logits
            scores.append({'fold': fold, **metric_scores(gold_y[folds == fold], logits)})
            histories.extend({'seed': seed, 'fold': fold, **row} for row in history)
            print(f"seed={seed} fold={fold} AUC={scores[-1]['macro_auc']:.5f}", flush=True)
            del head
        if not torch.isfinite(oof).all():
            raise ValueError('Incomplete gold OOF predictions.')
        runs.append({'seed': seed, **metric_scores(gold_y, oof), 'folds': scores})
        frame = pd.DataFrame(oof.sigmoid().numpy(), columns=LABELS)
        frame.insert(0, 'fold', folds)
        frame.insert(0, 'seed', seed)
        frame.insert(0, 'StudyInstanceUID', gold_ids)
        predictions.append(frame)
    pd.concat(predictions, ignore_index=True).to_csv(out / 'oof_predictions.csv', index=False)
    pd.DataFrame(histories).to_csv(out / 'history.csv', index=False)
    return write_weak_metrics(config, runs, out)


def freeze_swin_except_last_stage(encoder):
    encoder.eval().requires_grad_(False)
    encoder.features[-1].requires_grad_(True)
    encoder.norm.requires_grad_(True)
    return [p for p in encoder.parameters() if p.requires_grad]


def finetune_weak_encoder(encoder, cache, table, y, weights, gold, groups, series, data, config, out):
    device = 'cuda'
    weak = np.flatnonzero(~gold)
    valid = weak[np.array([int(group[:8], 16) % 10 == 0 for group in groups[weak]])]
    train_index = np.setdiff1d(weak, valid)
    if not len(valid) or not len(train_index) or gold[train_index].any():
        raise ValueError('Invalid weak-only encoder split.')
    torch.manual_seed(42)
    encoder = encoder.to(device)
    parameters = freeze_swin_except_last_stage(encoder)
    statistics = cache['tokens'][train_index][cache['mask'][train_index]]
    head = WeakHead(config, statistics.mean(0), statistics.std(0).clamp_min(.01)).to(device)
    del statistics
    optimizer = torch.optim.AdamW([{'params': parameters, 'lr': 1e-5},
                                  {'params': head.parameters(), 'lr': 1e-4}], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda')
    transform = make_transform(config)
    deadline = time.monotonic() + 7 * 3600
    best, best_epoch, history = float('inf'), None, []
    checkpoint = out / 'backbone_best.pt'

    def predict(index):
        if time.monotonic() > deadline:
            raise TimeoutError('Fine-tuning exceeded its 7h budget; no incomplete submission model will be used.')
        images, mask, _ = study_views(table.StudyInstanceUID.iloc[index], series, data / 'train_series', transform, config)
        mask = mask.to(device)
        with torch.autocast('cuda', dtype=torch.float16):
            valid_images = images[mask.cpu()].to(device)
            features = torch.cat([encoder(batch) for batch in valid_images.split(16)])
            tokens = features.new_zeros(*mask.shape, config['feature_dim'])
            tokens[mask] = features
            return head(tokens[None], mask[None])

    targets, confidence = torch.from_numpy(y).to(device), torch.from_numpy(weights).to(device)
    for epoch in range(1, config['backbone_epochs'] + 1):
        encoder.eval()
        encoder.features[-1].train()
        head.train()
        order = np.random.default_rng(42 + epoch).permutation(train_index)
        training_loss = 0.
        for start in range(0, len(order), 8):
            batch = order[start:start+8]
            optimizer.zero_grad(set_to_none=True)
            for index in batch:
                loss = weighted_bce(predict(index), targets[index:index+1], confidence[index:index+1])
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite backbone training loss.')
                scaler.scale(loss / len(batch)).backward()
                training_loss += loss.item()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([*parameters, *head.parameters()], 5.)
            scaler.step(optimizer)
            scaler.update()
            if start % 400 == 0:
                print(f'Backbone epoch={epoch} studies={start}/{len(order)}', flush=True)
        encoder.eval()
        head.eval()
        validation_loss = validation_weight = 0.
        with torch.inference_mode():
            for index in valid:
                weight = confidence[index:index+1].sum().item()
                loss = weighted_bce(predict(index), targets[index:index+1], confidence[index:index+1]).item()
                validation_loss += loss * weight
                validation_weight += weight
        score = validation_loss / validation_weight
        if not np.isfinite(score):
            raise ValueError('Non-finite backbone validation loss.')
        history.append({'epoch': epoch, 'train_loss': training_loss / len(order), 'weak_validation_bce': score})
        if score < best:
            best, best_epoch = score, epoch
            torch.save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, checkpoint)
        print(f'Backbone epoch={epoch} weak validation BCE={score:.6f}', flush=True)
    encoder.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    pd.DataFrame(history).to_csv(out / 'backbone_history.csv', index=False)
    report = {'training_weak_studies': len(train_index), 'validation_weak_studies': len(valid),
              'gold_studies_used_for_backbone': 0, 'best_epoch': best_epoch,
              'validation_bce': best, 'trainable_parameters': sum(p.numel() for p in parameters)}
    (out / 'backbone_audit.json').write_text(json.dumps(report, indent=2))
    del head, optimizer, scaler, targets, confidence
    return encoder.cpu().eval().requires_grad_(False)


def write_weak_manifest(config, bundle, table_count, gold_count, extra=None):
    manifest = {'experiment': config, 'labels': LABELS, 'python': list(sys.version_info[:2]),
                'torch': torch.__version__, 'final_training_studies': table_count,
                'gold_training_studies': gold_count, 'weak_training_studies': table_count-gold_count,
                'files': {p.relative_to(bundle).as_posix(): file_hash(p) for p in sorted(bundle.rglob('*'))
                          if p.is_file() and p != bundle / 'assets.json'}, **(extra or {})}
    (bundle / 'assets.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def train_weak_ensemble(config, data, out):
    members = [find_training_run(eid) for eid in config['train_dependencies']]
    frames = [pd.read_csv(path / 'oof_predictions.csv').sort_values(['seed','StudyInstanceUID']).reset_index(drop=True)
              for path in members]
    pd.testing.assert_frame_equal(frames[0][['StudyInstanceUID','seed','fold']], frames[1][['StudyInstanceUID','seed','fold']])
    frame = frames[0].copy()
    frame[LABELS] = (frames[0][LABELS].to_numpy() + frames[1][LABELS].to_numpy()) / 2
    truth = pd.read_csv(data / 'train.csv').set_index('StudyInstanceUID')
    runs = []
    for seed, rows in frame.groupby('seed'):
        y = truth.loc[rows.StudyInstanceUID, LABELS].to_numpy(dtype=np.float32)
        logits = torch.logit(torch.from_numpy(rows[LABELS].to_numpy(dtype=np.float32)).clamp(1e-7, 1-1e-7))
        scores = [{'fold': fold, **metric_scores(y[rows.fold.to_numpy() == fold], logits[rows.fold.to_numpy() == fold])}
                  for fold in range(3)]
        runs.append({'seed': int(seed), **metric_scores(y, logits), 'folds': scores})
    frame.to_csv(out / 'oof_predictions.csv', index=False)
    shutil.copy2(members[0] / 'folds.csv', out / 'folds.csv')
    metrics = write_weak_metrics(config, runs, out)
    bundle = out / 'bundle'
    metadata = []
    for eid, path in zip(config['train_dependencies'], members):
        meta = json.loads((path / 'bundle/assets.json').read_text())
        validate_weak_bundle(path / 'bundle', meta['experiment'])
        metadata.append(meta)
        shutil.copytree(path / 'bundle', bundle / 'members' / eid)
    shutil.copytree(members[0] / 'bundle/wheels', bundle / 'wheels')
    if any(m['final_training_studies'] != config['expected_training_studies'] for m in metadata):
        raise ValueError('Ensemble members use different training populations.')
    write_weak_manifest(config, bundle, config['expected_training_studies'], 58,
                        {'members': config['train_dependencies'], 'mixing': 'fixed probability mean 0.5/0.5'})
    print(json.dumps(metrics, indent=2), flush=True)


def train_weak_experiment(config, encoder_factory, data, out):
    from tqdm.auto import tqdm
    data, out = Path(data), Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Choose an empty output directory.')
    out.mkdir(parents=True, exist_ok=True)
    if config.get('ensemble'):
        return train_weak_ensemble(config, data, out)
    if not torch.cuda.is_available():
        raise RuntimeError('Run this notebook with a Kaggle GPU.')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.mha.set_fastpath_enabled(False)
    label_path = find_report_labels(config)
    table, y, weights, gold, groups, audit = make_targets(pd.read_csv(data / 'train.csv'), pd.read_csv(label_path), config)
    if int(gold.sum()) != 58 or len(table) != config['expected_training_studies']:
        raise ValueError('Training population changed; review the label audit and pinned source.')
    audit.update({'source': 'pilkwang/rsna-knee-llm-labels/versions/1', 'sha256': file_hash(label_path)})
    (out / 'label_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit), flush=True)
    series = pd.read_csv(data / 'train_series.csv')
    if not series.SeriesInstanceUID.is_unique or not set(table.StudyInstanceUID) <= set(series.StudyInstanceUID):
        raise ValueError('Missing or duplicate MRI series metadata.')
    bundle = out / 'bundle'
    bundle.mkdir()
    previous, cache = None, None
    if config['train_dependencies']:
        previous = find_training_run(config['train_dependencies'][0])
        previous_meta = json.loads((previous / 'bundle/assets.json').read_text())
        validate_weak_bundle(previous / 'bundle', previous_meta['experiment'])
        if file_hash(previous / 'tokens.pt') != previous_meta['tokens_sha256']:
            raise ValueError('Invalid cached feature artifact.')
        cache = torch.load(previous / 'tokens.pt', map_location='cpu', weights_only=True)
        shutil.copy2(previous / 'bundle/encoder.ts', bundle / 'encoder.ts')
        encoder = torch.jit.load(str(bundle / 'encoder.ts'), map_location='cpu').eval()
    else:
        encoder = export_encoder(encoder_factory(), {**config, 'slices': 16}, bundle / 'encoder.ts')
    if config['finetune']:
        if cache['ids'] != table.StudyInstanceUID.tolist() or cache['view_signature'] != view_signature(config):
            raise ValueError('Fine-tuning requires the matching preceding feature cache.')
        eager = encoder_factory(pretrained=False)
        eager.load_state_dict(encoder.state_dict(), strict=True)
        eager = finetune_weak_encoder(eager, cache, table, y, weights, gold, groups, series, data, config, out)
        encoder = export_encoder(eager, {**config, 'slices': 16}, bundle / 'encoder.ts')
        cache = None
        del eager
    encoder = encoder.to('cuda').eval()
    encoder_hash = file_hash(bundle / 'encoder.ts')
    cached_index = {}
    if cache is not None and cache['encoder_sha256'] == encoder_hash and cache['view_signature'] == view_signature(config):
        if len(cache['ids']) != len(set(cache['ids'])):
            raise ValueError('Duplicate cached study IDs.')
        cached_index = {uid: i for i, uid in enumerate(cache['ids'])}
    transform, rows = make_transform(config), []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for uid in tqdm(table.StudyInstanceUID, desc=config['id'] + ' MRI features'):
        if uid in cached_index:
            index = cached_index[uid]
            rows.append(tuple(cache[key][index] for key in ('tokens', 'mask', 'positions')))
        else:
            rows.append(weak_tokens(uid, series, data / 'train_series', encoder, transform, config, 'cuda'))
    tokens, mask, positions = [torch.stack([row[i] for row in rows]).clone() for i in range(3)]
    if not torch.isfinite(tokens).all():
        raise ValueError('Non-finite feature cache.')
    timing = {'feature_seconds': time.monotonic()-started, 'images': int(mask.sum()),
              'cached_studies': len(set(table.StudyInstanceUID) & cached_index.keys()),
              'peak_cuda_allocated_mb': torch.cuda.max_memory_allocated()/1024**2,
              'gpu': torch.cuda.get_device_name(0)}
    torch.save({'ids': table.StudyInstanceUID.tolist(), 'tokens': tokens, 'mask': mask, 'positions': positions,
                'view_signature': view_signature(config), 'encoder_sha256': encoder_hash}, out / 'tokens.pt')
    del cache, rows, encoder
    torch.cuda.empty_cache()
    tokens, mask = tokens.to('cuda'), mask.to('cuda')
    targets, confidence = torch.from_numpy(y).to('cuda'), torch.from_numpy(weights).to('cuda')
    metrics = evaluate_weak_cv(tokens, mask, targets, confidence, gold, table.StudyInstanceUID.tolist(), config, out)
    head, history = fit_weak_head(tokens, mask, targets, confidence, np.arange(len(table)), 42, config)
    torch.save({'state': {k:v.detach().cpu() for k,v in head.state_dict().items()}, 'experiment': config,
                'labels': LABELS}, bundle / 'model.pt')
    pd.DataFrame(history).to_csv(out / 'final_fit_history.csv', index=False)
    wheels = bundle / 'wheels'
    if previous:
        shutil.copytree(previous / 'bundle/wheels', wheels)
    else:
        wheels.mkdir()
        subprocess.check_call([sys.executable, '-m', 'pip', 'download', '-q', '--only-binary=:all:', '--no-deps',
                               '--dest', str(wheels), 'pydicom==3.0.2', 'python-gdcm==3.2.6'])
    write_weak_manifest(config, bundle, len(table), int(gold.sum()),
                        {'head_parameters': sum(p.numel() for p in head.parameters()),
                         'tokens_sha256': file_hash(out / 'tokens.pt'),
                         'folds_sha256': file_hash(out / 'folds.csv'), 'label_audit': audit})
    (out / 'timings.json').write_text(json.dumps(timing, indent=2))
    print(json.dumps({'macro_auc': metrics['macro_auc_mean'], 'timing': timing}), flush=True)
