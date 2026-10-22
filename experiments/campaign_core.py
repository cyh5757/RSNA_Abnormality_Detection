"""Small head/ensemble additions; historical training sources stay unchanged."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from experiments.core import LABELS
from experiments.spatial_core import SpatialHead
from experiments.plane_depth_core import PlaneDepthHead


def blend_weights(config):
    count = len(config['train_dependencies'])
    weights = np.asarray(config.get('blend_weights', [1 / count] * count if count else []), dtype=float)
    if (not count or weights.shape != (count,) or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1., rtol=0, atol=1e-8)):
        raise ValueError('Blend weights must be finite, nonnegative, and sum to one.')
    return weights


def blend_oof(frames, weights):
    weights = blend_weights(dict(train_dependencies=list(range(len(frames))), blend_weights=weights))
    keys = ['StudyInstanceUID', 'seed', 'fold']
    aligned = []
    for frame in frames:
        if frame.columns.tolist() != [*keys, *LABELS] or frame.duplicated(keys[:2]).any() or not len(frame):
            raise ValueError('Invalid OOF columns, empty input, or duplicate study/seed.')
        frame = frame.sort_values(keys[:2]).reset_index(drop=True)
        if frame[keys].isna().any().any() or (aligned and not frame[keys].equals(aligned[0][keys])):
            raise ValueError('OOF study/seed/fold alignment differs.')
        values = frame[LABELS].to_numpy(float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('Invalid OOF probabilities.')
        aligned.append(frame)
    result = aligned[0].copy()
    result[LABELS] = sum(w * frame[LABELS].to_numpy(float) for w, frame in zip(weights, aligned))
    return result


def drop_slices(mask, probability):
    if (mask.ndim != 3 or mask.dtype != torch.bool or not 0 <= probability < 1
            or not mask.flatten(1).any(1).all()):
        raise ValueError('Invalid slice dropout mask/probability.')
    if probability == 0:
        return mask
    keep = mask & (torch.rand(mask.shape, device=mask.device) >= probability)
    empty = mask.any(-1) & ~keep.any(-1)
    # Retain the first valid token only when random dropping emptied a present slot.
    rescue = torch.zeros_like(mask).scatter_(-1, mask.long().argmax(-1, keepdim=True), True)
    return keep | (rescue & empty[..., None])


class DropoutSpatialHead(SpatialHead):
    def forward(self, tokens, mask, coords):
        if self.training and self.config.get('slice_dropout', 0):
            mask = drop_slices(mask, self.config['slice_dropout'])
        return super().forward(tokens, mask, coords)


class SeedMeanHead(nn.Module):
    def __init__(self, heads):
        super().__init__()
        if not heads:
            raise ValueError('A seed ensemble needs trained heads.')
        self.heads = nn.ModuleList(heads)

    def forward(self, tokens, mask, coords):
        probabilities = torch.stack([head(tokens, mask, coords).sigmoid() for head in self.heads]).mean(0)
        return torch.logit(probabilities.clamp(1e-7, 1 - 1e-7))


def restore_head(config, state):
    offsets = config.get('head_seed_offsets', [0])
    if len(offsets) > 1:
        head = SeedMeanHead([DropoutSpatialHead(config, state[f'heads.{i}.mean'], state[f'heads.{i}.std'])
                             for i in range(len(offsets))])
    else:
        cls = PlaneDepthHead if 'plane_depth_mode' in config else DropoutSpatialHead
        head = cls(config, state['mean'], state['std'])
    head.load_state_dict(state, strict=True)
    return head
