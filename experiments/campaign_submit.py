"""Reuse offline inference, with weighted leaves and exact T/seed head loading."""
import json
import sys
from pathlib import Path
import torch

from experiments.core import LABELS, make_transform
from experiments.weak_core import validate_weak_bundle
from experiments.campaign_core import blend_weights, restore_head
from experiments.followup_submit import submit_followup_experiment


def campaign_leaves(bundle, weight=1.):
    bundle = Path(bundle)
    config = json.loads((bundle / 'assets.json').read_text())['experiment']
    if not config.get('ensemble'):
        return [(bundle, config, weight)]
    return [leaf for eid, fraction in zip(config['train_dependencies'], blend_weights(config))
            for leaf in campaign_leaves(bundle / 'members' / eid, weight * float(fraction))]


def load_campaign_predictor(bundle, config, device):
    bundle = Path(bundle)
    manifest = validate_weak_bundle(bundle, config)
    if manifest['python'] != list(sys.version_info[:2]) or manifest['torch'] != torch.__version__:
        raise ValueError('Use the pinned training runtime.')
    saved = torch.load(bundle / 'model.pt', map_location='cpu', weights_only=True)
    if saved['experiment'] != config or saved['labels'] != LABELS:
        raise ValueError('Wrong trained head.')
    head = restore_head(config, saved['state']).to(device).eval()
    encoder = torch.jit.load(str(bundle / 'encoder.ts'), map_location=device).eval()
    return encoder, head, make_transform(config), config


def submit_campaign_experiment(config, data, bundle, out):
    namespace = submit_followup_experiment.__globals__
    previous = {name: namespace[name] for name in ('predictor_leaves', 'load_spatial_predictor')}
    try:
        namespace.update(predictor_leaves=campaign_leaves, load_spatial_predictor=load_campaign_predictor)
        return submit_followup_experiment(config, data, bundle, out)
    finally:
        namespace.update(previous)
