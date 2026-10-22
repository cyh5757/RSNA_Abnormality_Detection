"""Finite V experiments selected before execution; no automatic weight search."""
import hashlib
import json
from pathlib import Path

from experiments.build_notebooks import ROOT
from experiments.followup_configs import INPUTS

CAMPAIGN_INPUTS = dict(INPUTS,
    r01_e09_s13_blend=ROOT/'outputs/automation/followup-20261019',
    s02_slice5=ROOT/'outputs/automation/spatial',
    t01_position_sum=ROOT/'outputs/automation/plane-depth-20261019')


def campaign_configs():
    folder = Path(__file__).parent
    names = ('core.py', 'backbones.py', 'train.py', 'weak_core.py', 'weak_train.py', 'weak_submit.py',
             'spatial_core.py', 'spatial_train.py', 'spatial_submit.py', 'spatial_notebooks.py',
             'followup_submit.py', 'followup_notebooks.py', 'plane_depth_core.py', 'build_notebooks.py',
             'campaign_core.py', 'campaign_train.py', 'campaign_submit.py', 'campaign_configs.py', 'campaign_notebooks.py')
    digest = hashlib.sha256(b''.join((folder/name).read_bytes() for name in names)).hexdigest()
    def parent(eid):
        return json.loads((CAMPAIGN_INPUTS[eid]/eid/'train/run/bundle/assets.json').read_text())['experiment']
    old = parent('s10_gate')
    base = dict(old, pipeline_sha256=digest, feature_pipeline_sha256=old['pipeline_sha256'],
                finetune=False, ensemble=False, train_dependencies=['s10_gate'], comparison_parent='s10_gate')
    blends = [dict(base, id=eid, ensemble=True, comparison_parent='r01_e09_s13_blend',
                   train_dependencies=['r01_e09_s13_blend', second], blend_weights=[.8, .2])
              for eid, second in [('v00_gate_blend', 's10_gate'), ('v01_depth_blend', 't01_position_sum'),
                                  ('v02_slice_blend', 's02_slice5')]]
    fresh = dict(base, id='v05_convnext', backbone='convnext_tiny', train_dependencies=[])
    fresh.pop('feature_pipeline_sha256')
    return blends + [dict(base, id='v03_slice_dropout', slice_dropout=.15),
                     dict(base, id='v04_seed_mean', head_seed_offsets=[0, 1, 2]), fresh,
                     dict(base, id='v06_convnext_blend', ensemble=True, comparison_parent='r01_e09_s13_blend',
                          train_dependencies=['r01_e09_s13_blend', 'v05_convnext'], blend_weights=[.8, .2])]
