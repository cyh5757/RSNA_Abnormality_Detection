"""R ablations inherit saved, completed configs, never regenerated historical hashes."""
import hashlib
import json
from pathlib import Path
from experiments.build_notebooks import ROOT

INPUTS={eid:ROOT/'outputs/automation'/suite for eid,suite in (
    ('e09_protocols','weak-labels'),('s10_gate','spatial'),('s11_cross','spatial'),
    ('s12_finetune','spatial'),('s13_ensemble','spatial'))}


def saved_config(eid):
    return json.loads((INPUTS[eid]/eid/'train/run/bundle/assets.json').read_text(encoding='utf-8'))['experiment']


def followup_configs(include_conditional=False):
    root=Path(__file__).parent
    names=('core.py','backbones.py','train.py','weak_core.py','weak_train.py','weak_submit.py',
           'spatial_core.py','spatial_train.py','spatial_submit.py','spatial_notebooks.py',
           'followup_configs.py','followup_submit.py','followup_notebooks.py','build_notebooks.py')
    digest=hashlib.sha256(b''.join((root/n).read_bytes() for n in names)).hexdigest()
    def variant(eid,parent,**changes):
        old=saved_config(parent)
        return dict(old,id=eid,comparison_parent=parent,pipeline_sha256=digest,
            feature_pipeline_sha256=old['pipeline_sha256'],center_range=[.25,.75],
            train_dependencies=[parent],**changes)
    configs=[
        variant('r01_e09_s13_blend','s13_ensemble'),
        variant('r04_confidence','s10_gate',weak_min_confidence=.8),
        variant('r05_label_loss','s10_gate',loss_reduction='per_label'),
        variant('r02_wide','s10_gate'),
        variant('r03_dense','s10_gate'),
        variant('r06_warmup','s11_cross',head_warmup_epochs=1)]
    configs[0]['train_dependencies']=['e09_protocols','s13_ensemble']
    configs[3]['center_range']=[.1,.9];configs[4]['centers']=9
    configs[5].update(finetune=True,comparison_parent='s12_finetune')
    if include_conditional:
        configs.append(dict(configs[5],id='r07_long_ft',comparison_parent='r06_warmup',backbone_epochs=3))
    return configs
