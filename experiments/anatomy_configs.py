"""Predeclared public-method ablations; never select weights using Gold58."""
import hashlib
from pathlib import Path
from experiments.followup_configs import saved_config


def anatomy_configs():
    root=Path(__file__).parent
    names=('core.py','backbones.py','train.py','weak_core.py','weak_train.py','spatial_core.py','spatial_train.py',
        'build_notebooks.py','spatial_notebooks.py','anatomy_core.py','anatomy_train.py','anatomy_adapt.py',
        'anatomy_submit.py','anatomy_configs.py','anatomy_notebooks.py','run_anatomy_queue.py')
    digest=hashlib.sha256(b''.join((root/name).read_bytes() for name in names)).hexdigest()
    base=dict(saved_config('s11_cross'),architecture='anatomy_v1',pipeline_sha256=digest,
        patch_attention=False,field_mm=None,canonical=False,method='model',finetune=False,ensemble=False)
    base.pop('feature_pipeline_sha256',None)
    specs=[('u00_control','s11_cross',{}),
        ('u01_mm_crop','u00_control',dict(field_mm=140.)),
        ('u02_orientation','u01_mm_crop',dict(canonical=True)),
        ('u03_patch_attention','u02_orientation',dict(patch_attention=True)),
        ('u04_full_ft1','u03_patch_attention',dict(backbone_epochs=1,adapt_stage='a01_s04')),
        ('u05_full_ft3','u04_full_ft1',dict(backbone_epochs=3,adapt_stage='a03_s04')),
        ('u06_flip_tta','u05_full_ft3',dict(ensemble=True,method='tta')),
        ('u07_probability_blend','u05_full_ft3',dict(ensemble=True,method='probability')),
        ('u08_rank_blend','u07_probability_blend',dict(ensemble=True,method='rank'))]
    result=[]
    for eid,parent,changes in specs:
        cfg=dict(base if not result else result[-1]);cfg.update(changes,id=eid,comparison_parent=parent,train_dependencies=[parent])
        if eid in ('u04_full_ft1','u05_full_ft3'): cfg['train_dependencies']=['u03_patch_attention',cfg['adapt_stage']]
        if eid in ('u07_probability_blend','u08_rank_blend'): cfg['train_dependencies']=['u03_patch_attention','u05_full_ft3']
        result.append(cfg)
    return result
