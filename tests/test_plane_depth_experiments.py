import ast
import contextlib
import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from experiments.core import LABELS
from experiments.plane_depth_core import PlaneDepthHead
from experiments.plane_depth_configs import plane_depth_configs
from experiments.plane_depth_notebooks import render_plane_depth
from experiments.spatial_core import SpatialHead, spatial_signature


class PlaneDepthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def config(self, mode='joint'):
        cfg = next(c for c in plane_depth_configs() if c['plane_depth_mode'] == mode)
        return dict(cfg, feature_dim=3, hidden_dim=16, crop=4, epochs=2, batch_size=2)

    def inputs(self):
        torch.manual_seed(17)
        x = torch.randn(2, 6, 4, 3)
        mask = torch.ones(2, 6, 4, dtype=torch.bool)
        mask[:, 2] = False
        coords = torch.zeros(2, 6, 4, 3)
        coords[..., 0] = torch.arange(4.)
        coords[..., 1] = torch.arange(4.) * 3.
        coords[..., 2] = 1.
        return x, mask, coords

    def test_control_reproduces_s11_outputs_and_gradients_exactly(self):
        cfg = self.config('none')
        torch.manual_seed(42)
        old = SpatialHead(cfg, torch.zeros(3), torch.ones(3)).eval()
        new = PlaneDepthHead(cfg, torch.zeros(3), torch.ones(3)).eval()
        new.load_state_dict(old.state_dict(), strict=True)
        x, mask, coords = self.inputs()
        a, b = old(x, mask, coords), new(x, mask, coords)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward()
        b.square().sum().backward()
        for p, q in zip(old.parameters(), new.parameters()):
            if p.grad is None:
                self.assertIsNone(q.grad)
            else:
                torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)

    def test_encoding_is_plane_conditioned_but_protocol_independent(self):
        head = PlaneDepthHead(self.config(), torch.zeros(3), torch.ones(3))
        pos = torch.arange(4.).expand(2, 6, -1)
        joint = head.position_features(pos)
        torch.testing.assert_close(joint[:, 0], joint[:, 1])
        self.assertFalse(torch.allclose(joint[:, 0], joint[:, 2]))
        self.assertFalse(torch.allclose(joint[:, :, 0], joint[:, :, 1]))
        joint.square().sum().backward()
        self.assertGreater(head.plane.weight.grad.abs().sum().item(), 0)
        shared = PlaneDepthHead(self.config('sum'), torch.zeros(3), torch.ones(3))
        torch.testing.assert_close(shared.position_features(pos)[:, 0], shared.position_features(pos)[:, 2])

    def test_positions_travel_with_images_padding_is_ignored_and_fallback_matches(self):
        x, mask, coords = self.inputs()
        order = [3, 0, 2, 1]
        for mode in ('none', 'sum', 'joint'):
            head = PlaneDepthHead(self.config(mode), torch.zeros(3), torch.ones(3)).eval()
            with torch.no_grad():
                first = head(x, mask, coords)
                torch.testing.assert_close(first, head(x[:, :, order], mask[:, :, order], coords[:, :, order]), atol=2e-6, rtol=2e-5)
                self.assertGreater((first - head(x, mask, coords[:, :, order])).abs().max().item(), 1e-7)
                fallback = coords.clone()
                fallback[..., 1] = 9999.
                fallback[..., 2] = 0.
                # coord_kind is zero at initialization; mm/3 equals ordinal index here.
                torch.testing.assert_close(first, head(x, mask, fallback))
                bad_x, bad_coords = x.clone(), coords.clone()
                bad_x[~mask], bad_coords[~mask] = float('nan'), float('nan')
                torch.testing.assert_close(first, head(bad_x, mask, bad_coords))
                with self.assertRaises(ValueError):
                    head(x, torch.zeros_like(mask), coords)

    def test_configs_preserve_features_training_and_parameter_count(self):
        from experiments.followup_configs import saved_config
        configs = plane_depth_configs()
        base = saved_config('s11_cross')
        counts = []
        for cfg in configs:
            self.assertEqual(spatial_signature(cfg), spatial_signature(base))
            for key in ('epochs', 'lr', 'weight_decay', 'batch_size', 'backbone', 'neighbors', 'centers'):
                self.assertEqual(cfg[key], base[key])
            self.assertEqual(cfg['train_dependencies'], ['s11_cross'])
            head = PlaneDepthHead(cfg, torch.zeros(768), torch.ones(768))
            counts.append(sum(p.numel() for p in head.parameters()))
        self.assertEqual(len(set(counts)), 1)
        legacy = SpatialHead(base, torch.zeros(768), torch.ones(768))
        self.assertEqual(counts[0], sum(p.numel() for p in legacy.parameters()))

    def namespace(self, cfg, mode):
        ns = {}
        for cell in render_plane_depth(cfg, mode)['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            ast.parse(source)
            self.assertNotIn('from experiments', source)
            if not source.startswith(('import subprocess, sys', 'train_spatial_experiment(', 'submit_spatial_experiment(')):
                exec(compile(source, '<standalone>', 'exec'), ns)
        self.assertIs(ns['SpatialHead'], ns['PlaneDepthHead'])
        return ns

    def test_standalone_training_updates_head_without_using_heldout_targets(self):
        cfg = self.config()
        ns = self.namespace(cfg, 'train')
        x, mask, coords = self.inputs()
        x, mask, coords = x.repeat(3, 1, 1, 1), mask.repeat(3, 1, 1), coords.repeat(3, 1, 1, 1)
        y, weight = torch.rand(6, 12), torch.ones(6, 12)
        with patch('torch.cuda.is_available', return_value=False):
            first, history = ns['fit_spatial_head'](x, mask, coords, y, weight, np.arange(4), 42, cfg)
            x[4:], coords[4:], y[4:] = 9999., 9999., 1.
            second, _ = ns['fit_spatial_head'](x, mask, coords, y, weight, np.arange(4), 42, cfg)
        self.assertEqual(len(history), 2)
        for key, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[key])
        torch.manual_seed(42)
        untrained = ns['SpatialHead'](cfg, first.mean, first.std)
        self.assertFalse(torch.equal(first.plane.weight, untrained.plane.weight))

    def test_all_offline_notebooks_load_same_head_and_generate_csv(self):
        from experiments.weak_train import write_weak_manifest
        for mode in ('none', 'sum', 'joint'):
            cfg = self.config(mode)
            self.namespace(cfg, 'train')
            ns = self.namespace(cfg, 'submit')
            self.assertNotIn('fit_spatial_head', ns)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                bundle = root / 'bundle'
                (bundle / 'wheels').mkdir(parents=True)
                for name in ('pydicom-3.0.2-test.whl', 'python_gdcm-3.2.6-test.whl'):
                    (bundle / 'wheels' / name).write_bytes(b'wheel')
                encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
                torch.jit.trace(encoder, torch.rand(1, 3, 4, 4)).save(str(bundle / 'encoder.ts'))
                head = PlaneDepthHead(cfg, torch.zeros(3), torch.ones(3)).eval()
                torch.save(dict(state=head.state_dict(), experiment=cfg, labels=LABELS), bundle / 'model.pt')
                write_weak_manifest(cfg, bundle, 4350, 58)
                pd.DataFrame(dict(StudyInstanceUID=['u'])).to_csv(root / 'test.csv', index=False)
                pd.DataFrame(dict(StudyInstanceUID=['u'], SeriesInstanceUID=['s'])).to_csv(root / 'test_series.csv', index=False)
                pd.DataFrame(columns=['StudyInstanceUID', *LABELS]).to_csv(root / 'sample_submission.csv', index=False)
                x, mask, coords = self.inputs()
                ns['spatial_tokens'] = lambda *args: (x[0], mask[0], coords[0])
                ns['make_transform'] = lambda cfg: lambda image: image
                with patch('subprocess.check_call') as install:
                    frame = ns['submit_spatial_experiment'](cfg, root, bundle, root / 'out')
                self.assertIn('--no-index', install.call_args.args[0])
                expected = head(x[:1], mask[:1], coords[:1]).sigmoid().detach().numpy()[0]
                np.testing.assert_allclose(frame[LABELS].to_numpy()[0], expected, rtol=1e-5)
                self.assertTrue((root / 'out/submission.csv').is_file())

    def test_queue_waits_for_lock_and_checks_predecessor_before_upload(self):
        from experiments.run_plane_depth_queue import wait_and_execute, verify_finished_followup
        from experiments.run_spatial_queue import RunnerBusy
        calls = []

        @contextlib.contextmanager
        def lock(path):
            calls.append(path)
            if len(calls) == 1:
                raise RunnerBusy()
            yield

        callback, sleep = Mock(), Mock()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('experiments.run_plane_depth_queue.exclusive_lock', lock):
                wait_and_execute(root / 'lock', root / 'status.json', 60, callback, sleep)
            sleep.assert_called_once_with(60)
            callback.assert_called_once()
            with self.assertRaises(RuntimeError):
                verify_finished_followup(root)

    def test_predecessor_requires_verified_outputs_and_respects_r07_decision(self):
        from experiments.run_plane_depth_queue import verify_finished_followup
        stage = dict(downloaded=True, version=1, error='')
        job = dict(verified=True, train=stage, submit=stage)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'prior'
            bundle = root / 'r01/train/run/bundle'
            bundle.mkdir(parents=True)
            (bundle / 'assets.json').write_text(json.dumps({'experiment': {'id': 'r01'}}))
            (root / 'state.json').write_text(json.dumps({'jobs': {'r01': job}}))
            with patch('experiments.run_plane_depth_queue.followup_configs', return_value=[{'id': 'r01'}]), \
                 patch('experiments.run_plane_depth_queue.verify_results') as verify:
                with self.assertRaises(RuntimeError):
                    verify_finished_followup(root)
                (root / 'r07_decision.json').write_text(json.dumps({'status': 'deferred'}))
                verify_finished_followup(root)
                verify.assert_called()
                (root / 'r07_decision.json').write_text(json.dumps({'status': 'eligible'}))
                with self.assertRaises(FileNotFoundError):
                    verify_finished_followup(root)
                r07 = root.with_name(root.name + '-r07')
                bundle = r07 / 'r07_long_ft/train/run/bundle'
                bundle.mkdir(parents=True)
                (bundle / 'assets.json').write_text(json.dumps({'experiment': {'id': 'r07_long_ft'}}))
                (r07 / 'state.json').write_text(json.dumps({'jobs': {'r07_long_ft': job}}))
                verify_finished_followup(root)
                broken = dict(job, train=dict(stage, version=None))
                (r07 / 'state.json').write_text(json.dumps({'jobs': {'r07_long_ft': broken}}))
                with self.assertRaises(RuntimeError):
                    verify_finished_followup(root)
            (root / 'state.json').write_text(json.dumps({'jobs': {}}))
            with self.assertRaises(RuntimeError):
                verify_finished_followup(root)

    def test_available_quota_schedule_requires_only_verified_s11_dependencies(self):
        from experiments.run_plane_depth_queue import check_scheduling_predecessor
        configs = plane_depth_configs()
        with patch('experiments.run_plane_depth_queue.verify_finished_followup') as verify:
            check_scheduling_predecessor(Path('previous'), configs, False)
            verify.assert_called_once_with(Path('previous'))
            verify.reset_mock()
            check_scheduling_predecessor(Path('previous'), configs, True)
            verify.assert_not_called()
            changed = [dict(configs[0], train_dependencies=['r02_wide'])]
            with self.assertRaises(ValueError):
                check_scheduling_predecessor(Path('previous'), changed, True)

    def test_available_quota_reserves_train_and_submit_and_defers_without_sleep(self):
        from experiments.run_plane_depth_queue import require_available_gpu, QuotaDeferred
        api = Mock()
        quota = api.quota_view.return_value
        quota.gpu_quota.total_time_allowed = timedelta(hours=30)
        quota.gpu_quota.time_used = timedelta(hours=28)
        quota.gpu_quota.time_reserved = timedelta(hours=.8)
        quota.quota_refresh_time = '2026-10-24'
        with tempfile.TemporaryDirectory() as tmp:
            status = Path(tmp) / 'quota.json'
            # 1.2h remains: enough for training alone, not the full 1.25h pair.
            with self.assertRaises(QuotaDeferred):
                require_available_gpu(api, 'train', status)
            self.assertEqual(api.quota_view.call_count, 1)
            saved = json.loads(status.read_text())
            self.assertEqual(saved['status'], 'deferred_for_quota')
            self.assertEqual(saved['required_hours'], 1.25)
            self.assertAlmostEqual(saved['remaining_hours'], 1.2)
            # An already-trained model can still finish its offline submit step.
            require_available_gpu(api, 'submit', status)
            self.assertEqual(json.loads(status.read_text())['status'], 'ready')
            quota.gpu_quota.time_reserved = timedelta(hours=.75)
            require_available_gpu(api, 'train', status)
            self.assertEqual(json.loads(status.read_text())['status'], 'ready')
            with self.assertRaises(ValueError):
                require_available_gpu(api, 'unknown', status)


if __name__ == '__main__':
    unittest.main()
