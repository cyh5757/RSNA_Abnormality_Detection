"""Standalone notebooks, including bounded adaptation checkpoint stages."""
import argparse
import ast
import json
from pathlib import Path
from experiments.build_notebooks import ROOT,module_source
from experiments.spatial_notebooks import build_spatial


def anatomy_source(filename,exclude=()):
    # The legacy renderer removes top-level local imports only; this suite also has lazy ones.
    tree=ast.parse(module_source(filename,exclude))
    class Standalone(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            return None if (node.module or '').startswith('experiments.') else node
    return ast.unparse(Standalone().visit(tree))+'\n'


def render_anatomy(config,mode):
    def cell(kind,text):
        return dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),**(dict(outputs=[],execution_count=None) if kind=='code' else {}))
    cells=[cell('markdown',f"# {config['id']} — {mode}\n\nPhysical crop / coarse patch attention ablation. Gold58 development CV is not the public leaderboard.\n"),
        cell('code','EXPERIMENT = '+repr(config)+'\n')]
    if mode=='train': cells.append(cell('code',"import subprocess, sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','pydicom==3.0.2','python-gdcm==3.2.6'])\n"))
    elif mode!='submit': raise ValueError(mode)
    sources=[('core.py',('fit_model',)),('backbones.py',()),('weak_core.py',('study_views','weak_tokens','WeakHead','fit_weak_head','view_signature')),
        ('spatial_core.py',()),('train.py',('evaluate_cv','train_experiment')),
        ('weak_train.py',('evaluate_weak_cv','finetune_weak_encoder','train_weak_experiment','train_weak_ensemble')),
        ('spatial_train.py',('train_spatial_experiment','finetune_spatial','evaluate_spatial_cv','fit_spatial_head')),
        ('anatomy_core.py',()),('anatomy_train.py',()),('anatomy_adapt.py',()),('anatomy_submit.py',())]
    for filename,exclude in sources: cells.append(cell('code',anatomy_source(filename,exclude)))
    entry=("train_anatomy_experiment(EXPERIMENT,find_data(),Path('/kaggle/working/run'))\n" if mode=='train' else
        "submit_anatomy_experiment(EXPERIMENT,find_data(),find_bundle(),Path('/kaggle/working'))\n")
    cells.append(cell('code',entry))
    return dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)


def render_adaptation(config,stage):
    nb=render_anatomy(config,'train')
    nb['cells'][0]['source']=[f"# {stage['id']} — full-backbone adaptation, epoch {stage['epoch']} segment {stage['segment']+1}/4\n",
        'Gold58 excluded; optimizer and random state resume from pinned predecessor.\n']
    nb['cells'][-1]['source']=[f"adapt_anatomy(EXPERIMENT,{stage['epoch']},{stage['segment']},{stage['previous']!r},find_data(),Path('/kaggle/working/adapt'))\n"]
    return nb


def build_anatomy(configs,output_root,training_version=None):
    build_spatial(configs,output_root,training_version)
    for cfg in configs:
        for mode in ('train','submit'):
            path=Path(output_root)/cfg['id']/mode/(mode+'.ipynb')
            path.write_text(json.dumps(render_anatomy(cfg,mode),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')


def build_adaptation(config,stage,folder,sources):
    from experiments.build_notebooks import DOCKER
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'train.ipynb').write_text(json.dumps(render_adaptation(config,stage),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    metadata=dict(id='choiyoungwhan/rsna-knee-'+stage['id'].replace('_','-')+'-train',title=f"RSNA Knee {stage['id']} adaptation",
        code_file='train.ipynb',language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
        competition_sources=['rsna-knee-abnormality-detection'],dataset_sources=['pilkwang/rsna-knee-llm-labels'],kernel_sources=sources,docker_image=DOCKER)
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    from experiments.anatomy_configs import anatomy_configs
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'notebooks/anatomy')
    args=parser.parse_args();build_anatomy(anatomy_configs(),args.output)
