"""Fixed, branched ablations; no leaderboard-driven automatic selection."""
import hashlib
from pathlib import Path
from experiments.weak_configs import weak_configs


def spatial_configs():
    root=Path(__file__).resolve().parent
    names=('core.py','backbones.py','weak_core.py','weak_train.py','spatial_core.py',
           'spatial_train.py','spatial_submit.py','spatial_notebooks.py','spatial_configs.py')
    digest=hashlib.sha256(b''.join((root/name).read_bytes() for name in names)).hexdigest()
    base=dict(weak_configs()[4],architecture='spatial_v1',representation='single',neighbors=5,
              centers=5,hidden_dim=128,heads=4,sequence=False,position='none',plane_embedding=False,
              fusion='concat',mm_unit=3.,batch_size=16,pipeline_sha256=digest)
    specs=[('s00_triplet',None,dict(representation='triplet',neighbors=3)),
           ('s01_slice3',0,dict(representation='single',neighbors=3)),
           ('s02_slice5',1,dict(neighbors=5)),('s03_slice10',2,dict(neighbors=10)),
           ('s04_sequence',2,dict(sequence=True)),('s05_position',4,dict(position='additive')),
           ('s06_rope_index',4,dict(position='rope_index')),('s07_rope_mm',4,dict(position='rope_mm')),
           ('s08_plane',7,dict(plane_embedding=True)),('s09_mlp',8,dict(fusion='mlp')),
           ('s10_gate',8,dict(fusion='gate')),('s11_cross',8,dict(fusion='cross')),
           ('s12_finetune',11,dict(finetune=True)),('s13_ensemble',11,dict(ensemble=True))]
    configs=[]
    for eid,parent,changes in specs:
        cfg=dict(base if parent is None else configs[parent])
        cfg.update(changes,id=eid,train_dependencies=[] if parent is None else [configs[parent]['id']])
        if eid=='s13_ensemble': cfg['train_dependencies']=['s11_cross','s12_finetune']
        configs.append(cfg)
    return configs
