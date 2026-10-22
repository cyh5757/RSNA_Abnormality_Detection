import json
import ast
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.core import LABELS


class CampaignCoreTests(unittest.TestCase):
    def test_weighted_oof_aligns_ids_and_preserves_fold_contract(self):
        from experiments.campaign_core import blend_oof
        a = pd.DataFrame([[uid, 42, fold, *([p] * 12)] for uid, fold, p in [('a', 0, .2), ('b', 1, .8)]],
                         columns=['StudyInstanceUID', 'seed', 'fold', *LABELS])
        b = a.iloc[::-1].copy(); b[LABELS] = .6
        result = blend_oof([a, b], [.8, .2])
        np.testing.assert_allclose(result[LABELS].iloc[:, 0], [.28, .76])
        b.loc[b.StudyInstanceUID == 'a', 'fold'] = 2
        with self.assertRaises(ValueError): blend_oof([a, b], [.8, .2])
        with self.assertRaises(ValueError): blend_oof([a, a], [.8, .3])
        with self.assertRaises(ValueError): blend_oof([a, a.iloc[:1]], [.8, .2])
        bad = a.copy(); bad.loc[0, LABELS[0]] = float('nan')
        with self.assertRaises(ValueError): blend_oof([a, bad], [.8, .2])

    def test_slice_dropout_keeps_each_present_slot_and_never_adds_padding(self):
        from experiments.campaign_core import drop_slices
        mask = torch.zeros(4, 6, 8, dtype=torch.bool); mask[:, 0, :4] = True; mask[:, 3, 7] = True
        torch.manual_seed(42)
        result = drop_slices(mask, .999)
        self.assertTrue(torch.equal(result.any(-1), mask.any(-1)))
        self.assertFalse((result & ~mask).any())
        self.assertLess(result.sum(), mask.sum())
        self.assertTrue(torch.equal(drop_slices(mask, 0), mask))
        with self.assertRaises(ValueError): drop_slices(mask, 1.)

    def test_seed_head_averages_probabilities_and_roundtrips(self):
        from experiments.campaign_core import SeedMeanHead, restore_head
        from experiments.spatial_core import SpatialHead
        config = dict(feature_dim=4, hidden_dim=8, heads=2, fusion='gate', sequence=False,
                      plane_embedding=False, position='none', head_seed_offsets=[0, 1, 2])
        heads = [SpatialHead(config, torch.zeros(4), torch.ones(4)) for _ in range(3)]
        head = SeedMeanHead(heads).eval()
        x = torch.randn(2, 6, 3, 4); mask = torch.ones(2, 6, 3, dtype=torch.bool); coords = torch.zeros(2, 6, 3, 3)
        expected = torch.stack([h(x, mask, coords).sigmoid() for h in heads]).mean(0)
        torch.testing.assert_close(head(x, mask, coords).sigmoid(), expected)
        loaded = restore_head(config, head.state_dict()).eval()
        torch.testing.assert_close(loaded(x, mask, coords), head(x, mask, coords))

    def test_nested_weights_keep_r01_leaf_ratios(self):
        from experiments.campaign_submit import campaign_leaves
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = {'': dict(ensemble=True, train_dependencies=['r01', 's10'], blend_weights=[.8, .2]),
                       'members/r01': dict(ensemble=True, train_dependencies=['e09', 's13']),
                       'members/r01/members/s13': dict(ensemble=True, train_dependencies=['s11', 's12'])}
            for name in ['members/s10', 'members/r01/members/e09', 'members/r01/members/s13/members/s11', 'members/r01/members/s13/members/s12']:
                configs[name] = dict(id=name.split('/')[-1], ensemble=False)
            for name, cfg in configs.items():
                path = root / name; path.mkdir(parents=True, exist_ok=True)
                (path/'assets.json').write_text(json.dumps(dict(experiment=cfg)))
            actual = {cfg['id']: weight for _, cfg, weight in campaign_leaves(root)}
            self.assertEqual(actual, dict(e09=.4, s11=.2, s12=.2, s10=.2))

    def test_t01_restores_position_head(self):
        from experiments.campaign_core import restore_head
        from experiments.plane_depth_core import PlaneDepthHead
        cfg = dict(feature_dim=4, hidden_dim=8, heads=2, fusion='cross', sequence=True,
                   plane_embedding=True, position='rope_mm', mm_unit=3., plane_depth_mode='sum')
        original = PlaneDepthHead(cfg, torch.zeros(4), torch.ones(4)).eval()
        loaded = restore_head(cfg, original.state_dict()).eval()
        self.assertIsInstance(loaded, PlaneDepthHead)
        x = torch.randn(2, 6, 3, 4); mask = torch.ones(2, 6, 3, dtype=torch.bool); coords = torch.randn(2, 6, 3, 3)
        torch.testing.assert_close(original(x, mask, coords), loaded(x, mask, coords))


class CampaignNotebookTests(unittest.TestCase):
    def test_standalone_notebooks_use_exact_heads_and_offline_submission(self):
        from experiments.campaign_configs import campaign_configs
        from experiments.campaign_notebooks import build_campaign
        with tempfile.TemporaryDirectory() as tmp:
            build_campaign(campaign_configs(), Path(tmp))
            for path in Path(tmp).rglob('*.ipynb'):
                nb = json.loads(path.read_text(encoding='utf-8')); namespace = {}
                for cell in nb['cells']:
                    if cell['cell_type'] != 'code': continue
                    source = ''.join(cell['source']); ast.parse(source)
                    self.assertNotIn('from experiments', source)
                    if source.startswith(('import subprocess, sys', 'train_campaign_experiment(', 'submit_campaign_experiment(')):
                        continue
                    exec(source, namespace)
                cfg = namespace['EXPERIMENT']
                meta = json.loads((path.parent/'kernel-metadata.json').read_text(encoding='utf-8'))
                self.assertTrue(meta['is_private'])
                if path.stem == 'submit':
                    self.assertFalse(meta['enable_internet'])
                    self.assertTrue(meta['enable_gpu'])
                    self.assertIn('restore_head', namespace)
                    self.assertNotIn('fit_campaign_head', namespace)
                else:
                    self.assertEqual(meta['enable_gpu'], not cfg['ensemble'])
                self.assertIn('PlaneDepthHead', namespace)

    def test_convnext_uses_fresh_features_and_matched_sampling(self):
        from experiments.campaign_configs import campaign_configs
        from experiments.followup_configs import saved_config
        configs = {c['id']: c for c in campaign_configs()}
        cfg = configs['v05_convnext']; parent = saved_config('s10_gate')
        self.assertEqual(cfg['train_dependencies'], [])
        self.assertNotIn('feature_pipeline_sha256', cfg)
        for name in ('crop', 'resize', 'interpolation', 'representation', 'neighbors', 'centers', 'fusion', 'labels_sha256'):
            self.assertEqual(cfg[name], parent[name])
        self.assertEqual(cfg['backbone'], 'convnext_tiny')

    def test_training_seed_mean_matches_reloaded_head_and_train_only_statistics(self):
        from experiments.campaign_train import fit_campaign_head
        from experiments.campaign_core import restore_head
        cfg = dict(feature_dim=4, hidden_dim=8, heads=2, fusion='gate', sequence=False,
                   plane_embedding=False, position='none', head_seed_offsets=[0, 1, 2],
                   lr=.001, weight_decay=.1, epochs=1, batch_size=2)
        x = torch.randn(5, 6, 3, 4); x[3:] = 1000
        mask = torch.ones(5, 6, 3, dtype=torch.bool); coords = torch.zeros(5, 6, 3, 3)
        y = torch.zeros(5, 12); weight = torch.ones(5, 12)
        model, history = fit_campaign_head(x, mask, coords, y, weight, np.arange(3), 42, cfg)
        self.assertEqual({row['member_seed'] for row in history}, {42, 43, 44})
        for head in model.heads:
            self.assertLess(head.mean.abs().max(), 2.)
        loaded = restore_head(cfg, model.state_dict()).eval()
        torch.testing.assert_close(model(x, mask, coords), loaded(x, mask, coords))


if __name__ == '__main__': unittest.main()
