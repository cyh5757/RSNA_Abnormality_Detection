"""Separate builder: never rewrites the active weak-label notebook sources."""
import argparse
import inspect
import json
from pathlib import Path
from experiments.build_notebooks import ROOT, DOCKER, module_source
from experiments.backbones import swin_t_encoder
from experiments.spatial_configs import spatial_configs


def render_spatial(config,mode):
    def cell(kind,text):
        return dict(cell_type=kind,metadata={},source=text.splitlines(keepends=True),
                    **(dict(outputs=[],execution_count=None) if kind=='code' else {}))
    cells=[cell('markdown',f"# {config['id']} — {mode}\n\nStudy-level weak supervision; slice coordinates and masks are preserved.\n"),
           cell('code','EXPERIMENT = '+repr(config)+'\n')]
    if mode=='train':
        cells.append(cell('code',"import subprocess, sys\nsubprocess.check_call([sys.executable,'-m','pip','install','-q','pydicom==3.0.2','python-gdcm==3.2.6'])\n"))
    elif mode!='submit': raise ValueError(mode)
    cells.append(cell('code',module_source('core.py',('fit_model',))))
    excluded=('study_views','weak_tokens','WeakHead','fit_weak_head','view_signature')
    if mode=='submit': excluded+=('make_targets','find_report_labels','find_training_run','weighted_bce')
    cells.append(cell('code',module_source('weak_core.py',excluded)))
    cells.append(cell('code',module_source('spatial_core.py')))
    if mode=='train':
        cells.append(cell('code',inspect.getsource(swin_t_encoder).replace('def swin_t_encoder','def build_encoder',1)))
        cells.append(cell('code',module_source('train.py',('evaluate_cv','train_experiment'))))
        cells.append(cell('code',module_source('weak_train.py',('evaluate_weak_cv','finetune_weak_encoder','train_weak_experiment'))))
        cells.append(cell('code',module_source('spatial_train.py')))
        cells.append(cell('code',"train_spatial_experiment(EXPERIMENT,build_encoder,find_data(),Path('/kaggle/working/run'))\n"))
    else:
        cells.append(cell('code',module_source('spatial_submit.py')))
        cells.append(cell('code',"submit_spatial_experiment(EXPERIMENT,find_data(),find_bundle(),Path('/kaggle/working'))\n"))
    return dict(nbformat=4,nbformat_minor=4,metadata=dict(kernelspec=dict(name='python3',display_name='Python 3',language='python')),cells=cells)


def build_spatial(configs,output_root,training_version=None):
    for cfg in configs:
        slug='rsna-knee-'+cfg['id'].replace('_','-')
        for mode in ('train','submit'):
            folder=Path(output_root)/cfg['id']/mode; folder.mkdir(parents=True,exist_ok=True)
            (folder/(mode+'.ipynb')).write_text(json.dumps(render_spatial(cfg,mode),ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
            sources=['choiyoungwhan/rsna-knee-'+eid.replace('_','-')+'-train/1' for eid in cfg['train_dependencies']] if mode=='train' else [f'choiyoungwhan/{slug}-train/{training_version or 1}']
            metadata=dict(id=f'choiyoungwhan/{slug}-{mode}',title=f"RSNA Knee {cfg['id']} {mode}",code_file=mode+'.ipynb',
                language='python',kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=mode=='train',
                competition_sources=['rsna-knee-abnormality-detection'],
                dataset_sources=['pilkwang/rsna-knee-llm-labels'] if mode=='train' and not cfg['ensemble'] else [],
                kernel_sources=sources,docker_image=DOCKER)
            (folder/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    print(f'Generated {len(configs)} spatial train/submit pairs.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,default=ROOT/'notebooks/experiments')
    args=parser.parse_args(); build_spatial(spatial_configs(),args.output)
