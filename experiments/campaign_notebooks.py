"""Standalone V notebooks built from the existing mixed-family R renderer."""
import inspect
import json
from pathlib import Path

from experiments.build_notebooks import module_source
from experiments.backbones import convnext_tiny_encoder
from experiments.followup_notebooks import render_followup
from experiments.spatial_notebooks import build_spatial


def render_campaign(config, mode):
    nb = render_followup(config, mode)
    def cell(source):
        return dict(cell_type='code', metadata={}, outputs=[], execution_count=None, source=source.splitlines(keepends=True))
    additions = [cell(module_source('plane_depth_core.py')), cell(module_source('campaign_core.py'))]
    if mode == 'train':
        additions.append(cell('_base_fit = fit_spatial_head\n' + module_source('campaign_train.py')))
        final = "train_campaign_experiment(EXPERIMENT,build_encoder,find_data(),Path('/kaggle/working/run'))\n"
        if config['backbone'] == 'convnext_tiny':
            indices = [i for i, c in enumerate(nb['cells']) if 'def build_encoder(' in ''.join(c['source'])]
            if len(indices) != 1: raise ValueError('Missing encoder factory.')
            nb['cells'][indices[0]] = cell(inspect.getsource(convnext_tiny_encoder).replace('def convnext_tiny_encoder', 'def build_encoder', 1))
    else:
        additions.append(cell(module_source('campaign_submit.py')))
        final = "submit_campaign_experiment(EXPERIMENT,find_data(),find_bundle(),Path('/kaggle/working'))\n"
    nb['cells'][-1:] = [*additions, cell(final)]
    nb['cells'][0]['source'] = [f"# {config['id']} — {mode}\n\n",
        "Fixed research campaign. Gold58 development OOF; no leaderboard score claim.\n",
        "Private training / Internet OFF inference. No LLM or agent calls.\n"]
    return nb


def build_campaign(configs, output_root, training_version=None):
    build_spatial(configs, output_root, training_version)
    for cfg in configs:
        for mode in ('train', 'submit'):
            folder = Path(output_root)/cfg['id']/mode
            (folder/(mode+'.ipynb')).write_text(json.dumps(render_campaign(cfg, mode), ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
            path = folder/'kernel-metadata.json'; metadata = json.loads(path.read_text())
            metadata['enable_gpu'] = mode == 'submit' or not cfg['ensemble']
            path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
