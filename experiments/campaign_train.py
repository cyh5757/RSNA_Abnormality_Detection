"""Reuse CV and fold-only statistics; only head fitting and blend weights differ."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from experiments.core import LABELS, metric_scores
from experiments.weak_core import find_training_run
from experiments.weak_train import train_weak_ensemble, write_weak_metrics
from experiments.spatial_train import train_spatial_experiment, fit_spatial_head as _base_fit
from experiments.campaign_core import DropoutSpatialHead, SeedMeanHead, blend_oof, blend_weights


def fit_campaign_head(tokens, mask, coords, y, weights, train_index, seed, config):
    offsets = config.get('head_seed_offsets', [0])
    if not offsets or len(set(offsets)) != len(offsets) or any(type(x) != int or x < 0 for x in offsets):
        raise ValueError('Seed offsets must be distinct nonnegative integers.')
    heads, history = [], []
    for offset in offsets:
        head, rows = _base_fit(tokens, mask, coords, y, weights, train_index, seed + offset, config)
        heads.append(head)
        history.extend(dict(row, member_seed=seed + offset) for row in rows)
    return (heads[0] if len(heads) == 1 else SeedMeanHead(heads).eval()), history


def train_campaign_experiment(config, encoder_factory, data, out):
    data, out = Path(data), Path(out)
    if config.get('ensemble'):
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Choose an empty output directory.')
        out.mkdir(parents=True, exist_ok=True)
        # Reuse the existing bundle-copy/hash/population checks before replacing the equal blend.
        train_weak_ensemble(config, data, out)
        frames = [pd.read_csv(find_training_run(eid) / 'oof_predictions.csv') for eid in config['train_dependencies']]
        frame = blend_oof(frames, blend_weights(config))
        truth = pd.read_csv(data / 'train.csv').set_index('StudyInstanceUID')
        runs = []
        for seed, rows in frame.groupby('seed'):
            y = truth.loc[rows.StudyInstanceUID, LABELS].to_numpy(np.float32)
            logits = torch.logit(torch.from_numpy(rows[LABELS].to_numpy(np.float32)).clamp(1e-7, 1 - 1e-7))
            scores = [dict(fold=fold, **metric_scores(y[rows.fold == fold], logits[rows.fold.to_numpy() == fold])) for fold in range(3)]
            runs.append(dict(seed=int(seed), **metric_scores(y, logits), folds=scores))
        frame.to_csv(out / 'oof_predictions.csv', index=False)
        metrics = write_weak_metrics(config, runs, out)
        path = out / 'bundle/assets.json'; manifest = json.loads(path.read_text())
        manifest['mixing'] = dict(method='fixed_probability', weights=blend_weights(config).tolist())
        path.write_text(json.dumps(manifest, indent=2))
        return metrics
    namespace = train_spatial_experiment.__globals__
    previous = {name: namespace[name] for name in ('fit_spatial_head', 'SpatialHead')}
    try:
        namespace.update(fit_spatial_head=fit_campaign_head, SpatialHead=DropoutSpatialHead)
        return train_spatial_experiment(config, encoder_factory, data, out)
    finally:
        namespace.update(previous)
