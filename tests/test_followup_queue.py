import ast
from datetime import timedelta
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class FollowupQueueTest(unittest.TestCase):
    def test_r07_respects_both_ft_and_total_session_budget(self):
        from experiments.run_followup_queue import r07_decision
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);child=root/'child';parent=root/'parent';child.mkdir();parent.mkdir()
            def metrics(score):
                return dict(macro_auc_mean=score,runs=[dict(seed=seed,macro_auc=score,
                    per_label_auc=dict.fromkeys(['MCL','Lateral Meniscus','Fracture'],score)) for seed in [42,2026,3407]])
            (child/'metrics.json').write_text(json.dumps(metrics(.75)))
            (parent/'metrics.json').write_text(json.dumps(metrics(.73)))
            def decision(ft,features):
                (child/'timings.json').write_text(json.dumps(dict(backbone_seconds=ft*3600,feature_seconds=features*3600)))
                return r07_decision(child,parent)
            self.assertEqual(decision(2.5,1)['status'],'deferred') # 9.5h total, but 7.5h FT exceeds 7h limit
            self.assertEqual(decision(2.,1)['status'],'eligible')
            self.assertEqual(decision(2.,4.)['status'],'deferred') # total >10.5h

    def test_quota_waits_without_upload_or_purchase(self):
        from experiments.run_followup_queue import wait_for_gpu
        def quota(hours): return SimpleNamespace(gpu_quota=SimpleNamespace(
            total_time_allowed=timedelta(hours=30),time_used=timedelta(hours=30-hours),time_reserved=timedelta()),quota_refresh_time='next')
        api=Mock(); api.quota_view.side_effect=[quota(1),quota(5)]
        sleep=Mock()
        with tempfile.TemporaryDirectory() as tmp:
            wait_for_gpu(api,2,Path(tmp)/'quota.json',sleep=sleep)
        sleep.assert_called_once_with(600)
        api.kernels_push.assert_not_called()

    def test_verified_inputs_and_pins(self):
        from experiments.run_followup_queue import resolve_verified_inputs
        from experiments.followup_configs import INPUTS
        resolved=resolve_verified_inputs()
        self.assertEqual(set(resolved),set(INPUTS))
        for item in resolved.values():
            self.assertTrue(item['job']['verified'])
            self.assertGreater(item['job']['train']['version'],0)
            self.assertEqual(len(item['assets_sha256']),64)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'state.json').write_text(json.dumps({'jobs':{'s10_gate':{'verified':False}}}))
            with self.assertRaises(ValueError): resolve_verified_inputs({'s10_gate':p})

    def test_configs_and_standalone_notebooks(self):
        from experiments.followup_configs import followup_configs
        from experiments.followup_notebooks import build_followup
        configs=followup_configs(); byid={c['id']:c for c in configs}
        self.assertEqual([c['id'][:3] for c in configs],['r01','r04','r05','r02','r03','r06'])
        self.assertEqual(byid['r06_warmup']['train_dependencies'],['s11_cross'])
        self.assertEqual(byid['r04_confidence']['weak_min_confidence'],.8)
        with tempfile.TemporaryDirectory() as tmp:
            build_followup(configs,Path(tmp))
            for path in Path(tmp).rglob('*.ipynb'):
                nb=json.loads(path.read_text()); ns={}
                for cell in nb['cells']:
                    if cell['cell_type']!='code': continue
                    src=''.join(cell['source']); ast.parse(src)
                    self.assertNotIn('from experiments',src)
                    if not src.startswith(('import subprocess, sys','train_spatial_experiment(','submit_followup_experiment(')):
                        exec(src,ns)
                self.assertIn('per_label_weighted_bce',ns)
                if path.stem=='submit':
                    self.assertIn('WeakHead',ns);self.assertIn('SpatialHead',ns)
                    self.assertNotIn('fit_spatial_head',ns)
                    meta=json.loads((path.parent/'kernel-metadata.json').read_text())
                    self.assertFalse(meta['enable_internet'])


if __name__=='__main__': unittest.main()
