import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class CampaignQueueTests(unittest.TestCase):
    def test_pin_refuses_changed_plan_without_overwriting(self):
        from experiments.run_campaign_queue import pin_campaign
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pin_campaign(root, {'plan': ['v00'], 'source': 'old'})
            pin_campaign(root, {'plan': ['v00'], 'source': 'old'})
            with self.assertRaises(ValueError): pin_campaign(root, {'plan': ['v00'], 'source': 'new'})
            self.assertEqual(json.loads((root/'inputs.json').read_text())['source'], 'old')

    def test_gpu_budget_reserves_submission_and_treats_cpu_blends_as_free(self):
        from experiments.run_campaign_queue import campaign_gpu_hours
        self.assertEqual(campaign_gpu_hours({'id': 'v00_gate_blend', 'ensemble': True}, 'train'), 0)
        self.assertEqual(campaign_gpu_hours({'id': 'v00_gate_blend', 'ensemble': True}, 'submit'), .5)
        self.assertGreaterEqual(campaign_gpu_hours({'id': 'v05_convnext', 'ensemble': False}, 'train'), 7.5)
        self.assertGreater(campaign_gpu_hours({'id': 'v04_seed_mean', 'ensemble': False, 'head_seed_offsets': [0, 1, 2]}, 'train'),
                           campaign_gpu_hours({'id': 'v03_slice_dropout', 'ensemble': False}, 'train'))

    def test_source_edit_during_quota_wait_blocks_upload(self):
        from experiments.run_campaign_queue import guarded_gpu_wait
        checks = iter([None, ValueError('source changed')])
        def check():
            result = next(checks)
            if result: raise result
        with patch('experiments.run_campaign_queue.wait_for_gpu') as wait:
            with self.assertRaises(ValueError): guarded_gpu_wait(object(), 1, Path('unused.json'), check)
            wait.assert_called_once()

    def test_reports_wait_for_partial_training_outputs(self):
        from experiments.run_campaign_queue import refresh_reports
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'outputs/automation/partial/item/train/run').mkdir(parents=True)
            (root/'outputs/automation/partial/item/train/run/metrics.json').write_text('{}')
            output = root/'report'; output.mkdir()
            with patch('experiments.run_campaign_queue.ROOT', root), patch('scripts.analyze_experiments.main') as analyze:
                self.assertFalse(refresh_reports(output))
                analyze.assert_not_called()
                self.assertEqual(json.loads((output/'analysis_status.json').read_text())['status'], 'waiting_for_artifacts')

    def test_adaptation_checks_campaign_source_after_quota_wait_before_upload(self):
        from experiments.run_campaign_queue import guarded_adaptation_stages
        import experiments.run_anatomy_queue as anatomy
        changed = False
        def wait(*args):
            nonlocal changed
            changed = True
        def check():
            if changed: raise ValueError('campaign source changed')
        state = {'jobs': {'u03_patch_attention': {'verified': True, 'train': {'ref': 'u03', 'version': 1}}}}
        stage = dict(id='a01_s01', epoch=1, segment=0, previous=None)
        api = Mock()
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(anatomy, 'adaptation_stages', return_value=[stage]), \
             patch.object(anatomy, 'assert_current_source'), patch.object(anatomy, 'check_version'), \
             patch.object(anatomy, 'build_adaptation', return_value={'id': 'a01'}), \
             patch.object(anatomy, 'checked_gpu_wait', side_effect=wait) as original:
            with self.assertRaisesRegex(ValueError, 'campaign source changed'):
                guarded_adaptation_stages(api, {}, Path(tmp), state, 120, check)
            original.assert_called_once()
            self.assertIs(anatomy.checked_gpu_wait, original)
            api.kernels_push.assert_not_called()
            self.assertEqual(state.get('adapt_jobs', {}), {})


if __name__ == '__main__': unittest.main()
