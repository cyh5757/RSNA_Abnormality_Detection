"""Standalone R notebooks, including mixed weak/spatial nested ensemble inference."""
import json
from pathlib import Path
from experiments.spatial_notebooks import build_spatial, render_spatial
from experiments.build_notebooks import module_source


def render_followup(config,mode):
    nb=render_spatial(config,mode)
    nb['cells'][0]['source']=[f"# {config['id']} — {mode}\n\nGold58 development experiment; pinned parent {config['comparison_parent']}.\n"]
    if mode=='submit':
        # Both head families are required by R01. Keep training functions out of inference.
        def cell(src): return dict(cell_type='code',metadata={},outputs=[],execution_count=None,source=src.splitlines(keepends=True))
        nb['cells'][3]=cell(module_source('weak_core.py',('fit_weak_head','make_targets','find_report_labels','find_training_run')))
        nb['cells'][-2:-1]=[
            cell(module_source('weak_submit.py',('submit_weak_experiment',))),
            cell(module_source('spatial_submit.py',('submit_spatial_experiment',))),
            cell(module_source('followup_submit.py'))]
        nb['cells'][-1]=cell("submit_followup_experiment(EXPERIMENT,find_data(),find_bundle(),Path('/kaggle/working'))\n")
    return nb


def build_followup(configs,output_root,training_version=None):
    build_spatial(configs,output_root,training_version)
    for config in configs:
        for mode in ('train','submit'):
            folder=Path(output_root)/config['id']/mode
            (folder/(mode+'.ipynb')).write_text(json.dumps(render_followup(config,mode),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
            if config['id']=='r01_e09_s13_blend':
                path=folder/'kernel-metadata.json';meta=json.loads(path.read_text())
                meta['enable_gpu']=False
                path.write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8')
