"""One controlled change per report-supervised experiment."""
import hashlib
import inspect
from pathlib import Path
from experiments.backbones import swin_t_encoder


WEAK_IDS = ['e06_gold_control', 'e07_weak_labels', 'e08_triplet', 'e09_protocols',
            'e10_class_mil', 'e11_finetune', 'e12_ensemble']


def weak_configs():
    root = Path(__file__).resolve().parent
    source = (root / 'core.py').read_bytes() + (root / 'weak_core.py').read_bytes()
    digest = hashlib.sha256(source + inspect.getsource(swin_t_encoder).encode()).hexdigest()
    configs = []
    for index, eid in enumerate(WEAK_IDS):
        configs.append({'id': eid, 'suite': 'weak', 'backbone': 'swin_t', 'feature_dim': 768,
                        'slices': 5, 'crop': 224, 'resize': 232, 'interpolation': 'bicubic',
                        'aggregation': 'class_mil' if index >= 4 else 'mean',
                        'triplet': index >= 2, 'protocols': index >= 3, 'weak_labels': index >= 1,
                        'finetune': index == 5, 'ensemble': index == 6, 'backbone_epochs': 1,
                        'epochs': 20, 'batch_size': 64, 'lr': .001, 'weight_decay': .01,
                        'expected_training_studies': 58 if index == 0 else 4350,
                        'labels_sha256': '6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920',
                        'train_dependencies': WEAK_IDS[4:6] if index == 6 else WEAK_IDS[max(0,index-1):index],
                        'pipeline_sha256': digest})
    return configs
