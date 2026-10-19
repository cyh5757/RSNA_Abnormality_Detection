import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments.core import LABELS
from experiments.weak_core import WeakHead, make_targets, weighted_bce, study_views, fit_weak_head


class WeakExperimentTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, **changes):
        return dict(dict(feature_dim=4, protocols=False, triplet=False, slices=3, crop=4,
                         aggregation='mean', weak_labels=True, epochs=2, batch_size=2,
                         lr=.001, weight_decay=.01), **changes)

    def test_targets_mask_unknown_exclude_overlap_and_keep_gold(self):
        rows = []
        for uid, report, gold in [('a', 'Gold report', True), ('b', '  GOLD  report ', False),
                                  ('c', 'known weak', False), ('d', 'unknown weak', False)]:
            rows.append(dict(StudyInstanceUID=uid, Report=report,
                             **{label: int(i % 2) if gold else np.nan for i, label in enumerate(LABELS)}))
        labels = []
        for uid in ('b', 'c', 'd'):
            row = {'StudyInstanceUID': uid}
            for label in LABELS:
                row.update({label: .8, label+'__conf': .9,
                            label+'__verdict': 'UNK' if uid == 'd' else 'YES'})
            labels.append(row)
        source = pd.DataFrame(labels)
        table, y, w, gold, groups, audit = make_targets(pd.DataFrame(rows), source, self.config())
        self.assertEqual(table.StudyInstanceUID.tolist(), ['a', 'c'])
        np.testing.assert_array_equal(y[0], np.arange(12) % 2)
        self.assertTrue((w[0] == 3).all())
        self.assertEqual(audit['weak_gold_report_overlap'], 1)
        self.assertEqual(audit['weak_studies'], 1)
        with self.assertRaises(ValueError):
            make_targets(pd.DataFrame(rows), pd.concat([source, source.iloc[:1]]), self.config())
        source.loc[0, 'ACL'] = 1.5
        with self.assertRaises(ValueError):
            make_targets(pd.DataFrame(rows), source, self.config())

    def test_unknown_targets_have_no_gradient(self):
        logits = torch.tensor([[1., 2., 3.]], requires_grad=True)
        y = torch.tensor([[.8, .5, .2]])
        weights = torch.tensor([[1., 0., 2.]])
        weighted_bce(logits, y, weights).backward()
        self.assertEqual(logits.grad[0, 1].item(), 0)
        self.assertNotEqual(logits.grad[0, 0].item(), 0)

    def test_triplets_stay_in_series_and_missing_protocol_is_masked(self):
        series = pd.DataFrame([{'StudyInstanceUID':'study', 'SeriesInstanceUID':'sag',
                                'Anatomical_Plane':'Sagittal', 'Fluid_Sensitive':True, 'Fat_Suppression':True}])
        paths = [Path(str(i)) for i in range(5)]
        def read(path, transform):
            return transform(torch.full((3, 4, 4), float(path.name)))
        with patch('experiments.weak_core.ordered_slices', return_value=(paths, np.linspace(0, 1, 5))), \
             patch('experiments.weak_core.image_tensor', side_effect=read):
            x, mask, pos = study_views('study', series, Path('.'), lambda x:x,
                                      self.config(protocols=True, triplet=True))
        self.assertEqual(tuple(x.shape), (6, 3, 3, 4, 4))
        self.assertEqual(mask.sum().item(), 3)
        torch.testing.assert_close(x[0, 0, :, 0, 0], torch.tensor([0., 1., 2.]))
        torch.testing.assert_close(x[0, 2, :, 0, 0], torch.tensor([2., 3., 4.]))
        self.assertFalse(mask[1:].any())

    def test_class_attention_ignores_missing_slots_and_padded_values(self):
        cfg = self.config(protocols=True, aggregation='class_mil')
        head = WeakHead(cfg, torch.zeros(4), torch.ones(4)).eval()
        x = torch.randn(2, 6, 3, 4)
        mask = torch.ones(2, 6, 3, dtype=torch.bool)
        mask[:, 2] = False
        first = head(x, mask)
        x[~mask] = float('nan')
        torch.testing.assert_close(head(x, mask), first)
        torch.testing.assert_close(head(x[:, :, [2, 0, 1]], mask[:, :, [2, 0, 1]]), first)
        self.assertEqual(tuple(first.shape), (2, 12))
        self.assertTrue(torch.isfinite(first).all())

    def test_validation_features_and_targets_never_affect_training(self):
        torch.manual_seed(3)
        x = torch.randn(6, 3, 3, 4)
        mask = torch.ones(6, 3, 3, dtype=torch.bool)
        y, w = torch.rand(6, 12), torch.ones(6, 12)
        first, _ = fit_weak_head(x, mask, y, w, np.arange(4), 42, self.config())
        x[4:], y[4:] = 1e5, 1.
        second, _ = fit_weak_head(x, mask, y, w, np.arange(4), 42, self.config())
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key])

    def test_new_notebooks_are_standalone_and_submit_has_no_training_or_network(self):
        import ast
        from experiments.build_notebooks import render_notebook
        from experiments.weak_configs import weak_configs
        for cfg in weak_configs():
            for mode in ('train', 'submit'):
                notebook = render_notebook(cfg, mode)
                code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
                ast.parse(code)
                self.assertNotIn('from experiments', code)
                # Execute all definitions, excluding the final training/submission and install cells.
                namespace = {}
                for cell in notebook['cells']:
                    source = ''.join(cell['source'])
                    if cell['cell_type']=='code' and not source.startswith(('train_weak_experiment(', 'submit_weak_experiment(', 'import subprocess, sys')):
                        exec(compile(source, '<notebook>', 'exec'), namespace)
                self.assertEqual(namespace['EXPERIMENT'], cfg)
                self.assertIn('WeakHead', namespace)
                if mode == 'submit':
                    self.assertNotIn('def fit_weak_head', code)
                    self.assertNotIn('def build_encoder', code)
                    self.assertIn('--no-index', code)

    def test_partial_finetuning_has_gradients_only_in_last_swin_stage(self):
        from experiments.backbones import swin_t_encoder
        from experiments.weak_train import freeze_swin_except_last_stage
        model = swin_t_encoder(pretrained=False)
        trainable = freeze_swin_except_last_stage(model)
        self.assertGreater(len(trainable), 0)
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.requires_grad, name.startswith(('features.7.', 'norm.')))
        model(torch.randn(1, 3, 32, 32)).sum().backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable))
        self.assertTrue(all(p.grad is None for p in model.features[0].parameters()))

    def test_weak_bundle_export_reload_and_integrity(self):
        from experiments.weak_core import validate_weak_bundle
        from experiments.weak_train import write_weak_manifest
        from experiments.weak_submit import load_weak_predictor
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.config(feature_dim=3)
            encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
            torch.jit.trace(encoder, torch.randn(1, 3, 4, 4)).save(str(root / 'encoder.ts'))
            head = WeakHead(cfg, torch.zeros(3), torch.ones(3)).eval()
            torch.save({'state':head.state_dict(), 'experiment':cfg, 'labels':LABELS}, root / 'model.pt')
            (root / 'wheels').mkdir()
            for name in ('pydicom-3.0.2-py3-none-any.whl','python_gdcm-3.2.6-test.whl'):
                (root / 'wheels' / name).write_bytes(b'wheel')
            write_weak_manifest(cfg, root, 58, 58)
            with patch('experiments.weak_submit.make_transform', return_value=lambda x:x):
                loaded_encoder, loaded_head, _, _ = load_weak_predictor(root, cfg, 'cpu')
            x = loaded_encoder(torch.rand(6, 3, 4, 4)).reshape(1, 3, 2, 3)
            mask = torch.ones(1, 3, 2, dtype=torch.bool)
            torch.testing.assert_close(head(x, mask), loaded_head(x, mask))
            (root / 'model.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                validate_weak_bundle(root, cfg)

    def test_ensemble_aligns_oof_and_bundles_both_actual_models(self):
        from experiments.weak_configs import weak_configs
        from experiments.weak_train import train_weak_ensemble, write_weak_manifest
        from experiments.weak_core import validate_weak_bundle
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = weak_configs()[-1]
            truth = pd.DataFrame({'StudyInstanceUID':[f's{i}' for i in range(58)]})
            for j, name in enumerate(LABELS):
                truth[name] = (np.arange(58)+j) % 2
            truth.to_csv(root / 'train.csv', index=False)
            members = {}
            for member_index, eid in enumerate(cfg['train_dependencies']):
                run = root / eid
                bundle = run / 'bundle'
                (bundle / 'wheels').mkdir(parents=True)
                for name in ('encoder.ts', 'model.pt', 'wheels/pydicom-3.0.2-test.whl', 'wheels/python_gdcm-3.2.6-test.whl'):
                    (bundle / name).write_bytes(f'{eid}-{name}'.encode())
                write_weak_manifest(dict(self.config(), id=eid), bundle, 4350, 58)
                rows = []
                for seed in (42,2026,3407):
                    frame = truth.copy()
                    frame[LABELS] = .1 + .1*member_index + .6*truth[LABELS]
                    frame['seed'], frame['fold'] = seed, np.arange(58)%3
                    rows.append(frame)
                pd.concat(rows).sample(frac=1, random_state=member_index).to_csv(run/'oof_predictions.csv', index=False)
                pd.DataFrame({'StudyInstanceUID':truth.StudyInstanceUID, 'fold':np.arange(58)%3}).to_csv(run/'folds.csv', index=False)
                members[eid] = run
            output = root/'ensemble'
            output.mkdir()
            with patch('experiments.weak_train.find_training_run', side_effect=members.__getitem__):
                train_weak_ensemble(cfg, root, output)
            actual = pd.read_csv(output/'oof_predictions.csv')
            expected = .15 + .6*truth.set_index('StudyInstanceUID').loc[actual.StudyInstanceUID, LABELS].to_numpy()
            np.testing.assert_allclose(actual[LABELS], expected)
            manifest = validate_weak_bundle(output/'bundle', cfg)
            self.assertEqual(manifest['members'], cfg['train_dependencies'])
            for eid in members:
                self.assertEqual((output/'bundle/members'/eid/'encoder.ts').read_bytes(),
                                 (members[eid]/'bundle/encoder.ts').read_bytes())


if __name__ == '__main__':
    unittest.main()
