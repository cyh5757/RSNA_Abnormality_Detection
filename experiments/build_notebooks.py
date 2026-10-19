"""Generate standalone, single-experiment Kaggle train/submit notebooks."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path

from experiments import backbones

ROOT = Path(__file__).resolve().parents[1]
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
BACKBONES = {
    "resnet18": (512, 256, "bilinear", backbones.resnet18_encoder),
    "dinov2": (384, 256, "bicubic", backbones.dinov2_encoder),
    "swin_t": (768, 232, "bicubic", backbones.swin_t_encoder),
    "convnext_tiny": (768, 236, "bilinear", backbones.convnext_tiny_encoder),
    "vit_b16": (768, 256, "bilinear", backbones.vit_b16_encoder),
}


def experiment_configs(attention_backbone="swin_t"):
    rows = [("e00_resnet18_3", "resnet18", 3, "mean", False, False),
            ("e01_resnet18_5", "resnet18", 5, "mean", False, False),
            ("e02a_dinov2_5", "dinov2", 5, "mean", False, False),
            ("e02b_swin_5", "swin_t", 5, "mean", False, False),
            ("e02c_convnext_5", "convnext_tiny", 5, "mean", False, False),
            ("e02d_vit_5", "vit_b16", 5, "mean", False, False),
            ("e03_attention", attention_backbone, 5, "attention", False, False),
            ("e04_position", attention_backbone, 5, "attention", True, False),
            ("e05_plane", attention_backbone, 5, "attention", True, True)]
    configs = []
    for name, backbone, slices, aggregation, position, plane in rows:
        dim, resize, interpolation, factory = BACKBONES[backbone]
        pipeline = (ROOT / "experiments/core.py").read_bytes() + inspect.getsource(factory).encode()
        configs.append({"id": name, "backbone": backbone, "feature_dim": dim, "slices": slices,
                        "crop": 224, "resize": resize, "interpolation": interpolation,
                        "aggregation": aggregation, "position": position, "plane_embedding": plane,
                        "epochs": 5, "lr": .001, "weight_decay": .1,
                        "pipeline_sha256": hashlib.sha256(pipeline).hexdigest()})
    return configs


def all_experiment_configs(attention_backbone="swin_t"):
    from experiments.weak_configs import weak_configs
    return experiment_configs(attention_backbone) + weak_configs()


EXPERIMENTS = experiment_configs()


def module_source(filename, exclude=()):
    text = (ROOT / "experiments" / filename).read_text(encoding="utf-8")
    nodes = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.ImportFrom) and (node.module or '').startswith("experiments."):
            continue
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in exclude:
            continue
        start = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]]) - 1
        nodes.append("\n".join(text.splitlines()[start:node.end_lineno]))
    return "\n\n".join(nodes) + "\n"


def render_notebook(config, mode):
    def cell(kind, text):
        return {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {})}
    title = f"# {config['id']} — {mode}\n\n"
    if config.get('suite') == 'weak':
        title += ('보고서 약라벨 비교 실험입니다. 이 실험의 전 단계 출력과 고정된 공개 라벨을 연결합니다. '
                  'gold58 개발용 CV이며 최종 제출 모델에는 허용된 학습 자료만 사용합니다.\n' if mode == 'train' else
                  '이 실험의 완료된 학습 출력만 연결합니다. Internet OFF로 실행해 submission.csv를 생성합니다.\n')
        cells = [cell('markdown', title), cell('code', 'EXPERIMENT = ' + repr(config) + '\n')]
        if mode == 'train':
            cells.append(cell('code', "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom==3.0.2', 'python-gdcm==3.2.6'])\n"))
        excluded = ('fit_model',) if mode == 'train' else ('fit_model', 'make_folds', 'metric_scores')
        cells.append(cell('code', module_source('core.py', excluded)))
        weak_excluded = () if mode == 'train' else ('make_targets', 'find_report_labels', 'find_training_run', 'weighted_bce', 'fit_weak_head')
        cells.append(cell('code', module_source('weak_core.py', weak_excluded)))
        if mode == 'train':
            cells.append(cell('code', inspect.getsource(backbones.swin_t_encoder).replace('def swin_t_encoder', 'def build_encoder', 1)))
            cells.append(cell('code', module_source('train.py', ('evaluate_cv', 'train_experiment'))))
            cells.append(cell('code', module_source('weak_train.py')))
            cells.append(cell('code', "train_weak_experiment(EXPERIMENT, build_encoder, find_data(), Path('/kaggle/working/run'))\n"))
        elif mode == 'submit':
            cells.append(cell('code', module_source('weak_submit.py')))
            cells.append(cell('code', "submit_weak_experiment(EXPERIMENT, find_data(), find_bundle(), Path('/kaggle/working'))\n"))
        else:
            raise ValueError(mode)
        return {'nbformat': 4, 'nbformat_minor': 4,
                'metadata': {'kernelspec': {'name':'python3', 'display_name':'Python 3', 'language':'python'}}, 'cells': cells}
    title += ("이 Notebook은 이 실험만 학습합니다. GPU·인터넷 ON, 대회 데이터를 연결하고 Run All 하세요. "
              "OOF 평가 후 전체 gold로 최종 모델을 만들고 run/bundle에 저장합니다.\n" if mode == "train" else
              "이 Notebook은 이 실험의 완료된 학습 버전만 사용합니다. GPU·인터넷 OFF, 대회 데이터와 해당 학습 출력을 연결하세요. "
              "Save Version → Run All 후 출력의 submission.csv를 확인하고 대회에 제출하세요.\n")
    cells = [cell("markdown", title), cell("code", "EXPERIMENT = " + repr(config) + "\n")]
    if mode == "train":
        cells.append(cell("code", "import subprocess, sys\nsubprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pydicom==3.0.2', 'python-gdcm==3.2.6'])\n"))
    excluded = () if mode == "train" else ("fit_model", "make_folds", "metric_scores")
    cells.append(cell("code", module_source("core.py", excluded)))
    if mode == "train":
        factory = BACKBONES[config["backbone"]][3]
        cells.append(cell("code", inspect.getsource(factory).replace("def " + factory.__name__, "def build_encoder", 1)))
        cells.append(cell("code", module_source("train.py")))
        cells.append(cell("code", "train_experiment(EXPERIMENT, build_encoder, find_data(), Path('/kaggle/working/run'))\n"))
    elif mode == "submit":
        cells.append(cell("code", module_source("submit.py")))
        cells.append(cell("code", "submit_experiment(EXPERIMENT, find_data(), find_bundle(), Path('/kaggle/working'))\n"))
    else:
        raise ValueError(mode)
    return {"nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": cells}


def build_all(configs, output_root, training_version=None):
    for config in configs:
        slug = "rsna-knee-" + config["id"].replace("_", "-")
        for mode in ("train", "submit"):
            folder = Path(output_root) / config["id"] / mode
            folder.mkdir(parents=True, exist_ok=True)
            notebook = render_notebook(config, mode)
            (folder / (mode + ".ipynb")).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            metadata_path = folder / "kernel-metadata.json"
            version = training_version or 1
            if training_version is None and mode == "submit" and metadata_path.exists():
                previous = json.loads(metadata_path.read_text(encoding="utf-8"))
                version = int(previous["kernel_sources"][0].rsplit("/", 1)[1])
            metadata = {"id": f"choiyoungwhan/{slug}-{mode}", "title": f"RSNA Knee {config['id']} {mode}",
                        "code_file": mode + ".ipynb", "language": "python", "kernel_type": "notebook",
                        "is_private": True, "enable_gpu": True, "enable_internet": mode == "train",
                        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [],
                        "kernel_sources": [] if mode == "train" else [f"choiyoungwhan/{slug}-train/{version}"],
                        "docker_image": DOCKER}
            if config.get('suite') == 'weak' and mode == 'train':
                if not config.get('ensemble'):
                    metadata['dataset_sources'] = ['pilkwang/rsna-knee-llm-labels']
                metadata['kernel_sources'] = ['choiyoungwhan/rsna-knee-' + eid.replace('_','-') + '-train/1'
                                              for eid in config['train_dependencies']]
            metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(configs)} independent train/submit notebook pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attention-backbone", choices=BACKBONES, default="swin_t")
    parser.add_argument("--experiment")
    parser.add_argument("--training-version", type=int)
    args = parser.parse_args()
    configs = all_experiment_configs(args.attention_backbone)
    if args.experiment:
        configs = [c for c in configs if c["id"] == args.experiment]
        if not configs:
            parser.error("Unknown experiment ID.")
    if args.training_version is not None and (not args.experiment or args.training_version < 1):
        parser.error("--training-version requires one --experiment and a positive version.")
    build_all(configs, ROOT / "notebooks/experiments", args.training_version)
