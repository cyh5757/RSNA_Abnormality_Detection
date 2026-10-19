"""Standalone train / offline-submit pairs with one identical head definition."""
import argparse
import json
from pathlib import Path
from experiments.build_notebooks import ROOT, module_source
from experiments.spatial_notebooks import build_spatial, render_spatial
from experiments.plane_depth_configs import plane_depth_configs


def render_plane_depth(config, mode):
    nb = render_spatial(config, mode)
    nb['cells'][0]['source'] = [
        f"# {config['id']} — {mode}\n\n",
        "방향(범주) × slice 깊이(연속 좌표) 실험입니다. 이미지 내부 patch의 x/y 좌표와는 다릅니다.\n",
        "S11의 고정 Swin feature, 5장 이웃, 3 folds × 3 seeds, loss, head 파라미터 수를 유지합니다.\n",
        "T00: 기존 방향 embedding + mm RoPE. T01: 깊이 sin/cos 추가. T02: 방향×깊이 상호작용 추가.\n",
        "학습 후 생성된 bundle로 Internet OFF 제출 notebook을 실행하면 submission.csv가 생성됩니다.\n"]
    source = module_source('plane_depth_core.py') + '\nSpatialHead = PlaneDepthHead\n'
    cell = dict(cell_type='code', metadata={}, source=source.splitlines(keepends=True),
                execution_count=None, outputs=[])
    indices = [i for i, c in enumerate(nb['cells']) if 'class SpatialHead(' in ''.join(c['source'])]
    if len(indices) != 1:
        raise ValueError('Expected exactly one base SpatialHead definition.')
    nb['cells'].insert(indices[0] + 1, cell)
    return nb


def build_plane_depth(configs, output_root, training_version=None):
    build_spatial(configs, output_root, training_version)
    for cfg in configs:
        for mode in ('train', 'submit'):
            path = Path(output_root) / cfg['id'] / mode / (mode + '.ipynb')
            path.write_text(json.dumps(render_plane_depth(cfg, mode), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'notebooks/plane_depth')
    args = parser.parse_args()
    build_plane_depth(plane_depth_configs(), args.output)
