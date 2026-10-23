import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from experiments import spatial_train


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Linear(8, 8))
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, images):
        return self.norm(self.features(images.mean(dim=(-2, -1))))


@unittest.skipUnless(torch.cuda.is_available(), 'Real CUDA optimizer integration test')
class EarlyStoppingTests(unittest.TestCase):
    def run_curve(self, scores, patience, extra=None):
        torch.set_num_threads(1)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.manual_seed(7)
        encoder = TinyEncoder()
        config = dict(feature_dim=8, hidden_dim=16, heads=4, sequence=True,
                      position='rope_index', plane_embedding=True, fusion='concat',
                      representation='single', neighbors=1, centers=1, crop=4, resize=4,
                      interpolation='bicubic', mm_unit=3., backbone_epochs=len(scores))
        if patience is not None:
            config['early_stopping_patience'] = patience
        config.update(extra or {})
        cache = dict(tokens=torch.randn(4, 6, 1, 8), mask=torch.ones(4, 6, 1, dtype=torch.bool),
                     coords=torch.zeros(4, 6, 1, 3))
        table = pd.DataFrame(dict(StudyInstanceUID=['train1', 'train2', 'valid', 'gold']))
        gold = np.array([False, False, False, True])
        groups = np.array(['00000001', '00000002', '0000000a', '00000003'])
        images = torch.randn(6, 1, 3, 4, 4)
        snapshots = []
        real_bce = spatial_train.weighted_bce

        def views(uid, *args):
            self.assertNotEqual(uid, 'gold')
            self.assertEqual(uid == 'valid', not torch.is_grad_enabled())
            return images, cache['mask'][0], cache['coords'][0]

        def loss(logits, target, weights):
            if torch.is_grad_enabled():
                return real_bce(logits, target, weights)
            snapshots.append({k: v.detach().cpu().clone() for k, v in encoder.state_dict().items()})
            return torch.tensor(scores[len(snapshots)-1])

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(spatial_train, 'spatial_views', side_effect=views), \
             patch.object(spatial_train, 'weighted_bce', side_effect=loss):
            out = Path(tmp)
            result = spatial_train.finetune_spatial(encoder, cache, table, torch.zeros(4, 12),
                torch.ones(4, 12), gold, groups, pd.DataFrame(), out, config, out)
            history = pd.read_csv(out/'backbone_history.csv')
            audit = json.loads((out/'backbone_audit.json').read_text())
            saved = torch.load(out/'backbone_best.pt', weights_only=True)
        return result, history, audit, snapshots, saved

    def test_three_consecutive_non_improvements_reset_on_new_best_and_restore_best(self):
        result, history, audit, snapshots, saved = self.run_curve(
            [.60, .59, .59, .60, .58, .58, .60, .59, .57], 3)
        self.assertEqual(len(history), 8)
        self.assertEqual(audit['best_epoch'], 5)
        self.assertEqual(audit['stop_reason'], 'early_stopping')
        self.assertEqual(audit['epochs_without_improvement'], 3)
        self.assertFalse(result.training)
        self.assertTrue(all(not p.requires_grad for p in result.parameters()))
        for key, value in result.state_dict().items():
            torch.testing.assert_close(value, snapshots[4][key], rtol=0, atol=0)
            torch.testing.assert_close(saved[key], snapshots[4][key], rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(snapshots[4][k], snapshots[-1][k]) for k in saved))

    def test_improving_curve_reaches_cap_and_legacy_run_keeps_fixed_epoch_count(self):
        for scores, patience, reason, best_epoch in (
                ([.6, .5, .4], 3, 'max_epochs', 3),
                ([.5, .6, .7, .8, .9], None, 'fixed_epochs', 1)):
            with self.subTest(patience=patience):
                result, history, audit, snapshots, _ = self.run_curve(scores, patience)
                self.assertEqual(len(history), len(scores))
                self.assertEqual(audit['stop_reason'], reason)
                self.assertEqual(audit['best_epoch'], best_epoch)
                for key, value in result.state_dict().items():
                    torch.testing.assert_close(value, snapshots[best_epoch-1][key], rtol=0, atol=0)

    def test_nonfinite_validation_fails_instead_of_counting_as_a_plateau(self):
        with self.assertRaisesRegex(ValueError, 'validation loss'):
            self.run_curve([.6, float('nan'), .5], 3)

    def test_public_recipe_updates_earlier_encoder_layers_with_checkpointing(self):
        _, _, audit, snapshots, _ = self.run_curve([.6, .5, .4], 3,
            dict(backbone_scope='full', gradient_checkpointing=True,
                 backbone_schedule='warmup_cosine', keep_backbone_checkpoints=3))
        self.assertFalse(torch.equal(snapshots[0]['features.0.weight'], snapshots[-1]['features.0.weight']))
        self.assertEqual(audit['backbone_scope'], 'full')


class EarlyStoppingValidationTests(unittest.TestCase):
    def test_invalid_limits_are_rejected_before_gpu_training(self):
        for changes in ({'early_stopping_patience': 0}, {'early_stopping_patience': -1},
                        {'early_stopping_patience': True}, {'backbone_epochs': 0},
                        {'backbone_timeout_hours': float('nan')}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config = dict(backbone_epochs=30, early_stopping_patience=3)
                config.update(changes)
                spatial_train.finetune_spatial(*([None]*9), config, None)


if __name__ == '__main__':
    unittest.main()
