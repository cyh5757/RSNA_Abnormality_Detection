"""Locally adapted public training methods; never load competition-trained public weights."""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import torch

from experiments.core import file_hash


def average_backbone(source, expected_config):
    source = Path(source)
    receipt = json.loads((source/'local_run.json').read_text(encoding='utf-8'))
    if receipt.get('status') != 'complete' or receipt['config'] != expected_config:
        raise ValueError('Checkpoint averaging requires the completed matching training recipe.')
    run = source/'run'
    audit = json.loads((run/'backbone_audit.json').read_text(encoding='utf-8'))
    if audit['gold_studies_used_for_backbone'] != 0:
        raise ValueError('Gold was used for backbone training.')
    rows = json.loads((run/'backbone_candidates.json').read_text(encoding='utf-8'))
    if len(rows) != 3 or len({r['epoch'] for r in rows}) != 3:
        raise ValueError('Expected three distinct weak-validation checkpoints.')
    averaged = None
    for row in rows:
        if row['file'] != f"backbone_epoch_{row['epoch']:03d}.pt" or not math.isfinite(row['validation_bce']):
            raise ValueError('Invalid checkpoint selection metadata.')
        path = run/row['file']
        if file_hash(path) != row['sha256']:
            raise ValueError('Checkpoint hash mismatch.')
        state = torch.load(path, map_location='cpu', weights_only=True)
        if averaged is None:
            averaged = {k: v.clone() for k, v in state.items()}
            for value in averaged.values():
                if value.is_floating_point():
                    value.div_(3)
        else:
            if state.keys() != averaged.keys():
                raise ValueError('Checkpoint parameter names differ.')
            for key, value in state.items():
                target = averaged[key]
                if target.shape != value.shape or target.dtype != value.dtype:
                    raise ValueError('Checkpoint parameter shape or dtype differs.')
                if value.is_floating_point():
                    target.add_(value / 3)
                elif not torch.equal(target, value):
                    raise ValueError('Checkpoint non-floating buffers differ.')
        if any(not torch.isfinite(v).all() for v in state.values()):
            raise ValueError('Non-finite checkpoint weights.')
    if any(not torch.isfinite(v).all() for v in averaged.values()):
        raise ValueError('Non-finite averaged weights.')
    return averaged, dict(method='mean_of_three_lowest_weak_validation_BCE_checkpoints',
        source_run=str(source.resolve()), source_receipt_sha256=file_hash(source/'local_run.json'),
        checkpoints=rows, gold_studies_used_for_backbone=0,
        note='Selection uses weak validation only; evaluate separately from the best single checkpoint.')


@contextmanager
def use_averaged_backbone(state, audit):
    if state is None:
        yield
        return
    from experiments import spatial_train
    previous = spatial_train.finetune_spatial

    def restore(encoder, cache, table, y, weights, gold, groups, series, data, config, out):
        encoder.load_state_dict(state, strict=True)
        torch.save(state, out/'backbone_best.pt')
        (out/'backbone_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        return encoder.cpu().eval().requires_grad_(False)

    spatial_train.finetune_spatial = restore
    try:
        yield
    finally:
        spatial_train.finetune_spatial = previous
