"""Fixed S11 features, seeds, folds, loss and parameter count; position alone varies."""
import hashlib
from pathlib import Path
from experiments.followup_configs import saved_config


def plane_depth_configs():
    root = Path(__file__).parent
    names = ('core.py', 'backbones.py', 'train.py', 'weak_core.py', 'weak_train.py',
             'spatial_core.py', 'spatial_train.py', 'spatial_submit.py', 'spatial_notebooks.py',
             'build_notebooks.py', 'plane_depth_core.py', 'plane_depth_configs.py', 'plane_depth_notebooks.py')
    digest = hashlib.sha256(b''.join((root / name).read_bytes() for name in names)).hexdigest()
    old = saved_config('s11_cross')
    base = dict(old, pipeline_sha256=digest, feature_pipeline_sha256=old['pipeline_sha256'],
                train_dependencies=['s11_cross'], finetune=False, ensemble=False)
    return [dict(base, id=eid, plane_depth_mode=mode, comparison_parent=parent)
            for eid, mode, parent in (
                ('t00_position_control', 'none', 's11_cross'),
                ('t01_position_sum', 'sum', 't00_position_control'),
                ('t02_position_joint', 'joint', 't01_position_sum'))]
