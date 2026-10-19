import json
import tempfile
import unittest
from requests.exceptions import Timeout, HTTPError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments.build_notebooks import experiment_configs
from experiments import run_queue


class FakeKaggle:
    def __init__(self):
        self.uploads = []
        self.versions = {}
        self.interrupt = False
        self.fail = False

    def kernels_push(self, folder):
        folder = Path(folder)
        meta = json.loads((folder / 'kernel-metadata.json').read_text())
        ref = meta['id']
        version = self.versions.get(ref, 6) + 1
        self.versions[ref] = version
        self.uploads.append(meta)
        return SimpleNamespace(version_number=version, error='', invalid_kernel_sources=[],
                               invalid_dataset_sources=[], invalid_competition_sources=[], invalid_model_sources=[])

    def kernels_status(self, ref):
        if self.interrupt:
            self.interrupt = False
            raise KeyboardInterrupt()
        return SimpleNamespace(status='ERROR' if self.fail else 'COMPLETE', failure_message='test failure')

    def kernels_output(self, ref, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)


class RunQueueTest(unittest.TestCase):
    def test_read_timeout_then_success_and_exhaustion(self):
        self.assertTrue(hasattr(run_queue, 'retry_kaggle_read'))
        from unittest.mock import Mock
        call=Mock(side_effect=[Timeout('network'), Timeout('network'), 7]); sleep=Mock()
        self.assertEqual(run_queue.retry_kaggle_read(call, sleep=sleep),7)
        self.assertEqual([x.args[0] for x in sleep.call_args_list],[5,10])
        call=Mock(side_effect=Timeout('network'))
        with self.assertRaises(Timeout): run_queue.retry_kaggle_read(call, attempts=3, sleep=sleep)
        self.assertEqual(call.call_count,3)

    def test_nontransient_error_is_not_retried(self):
        self.assertTrue(hasattr(run_queue, 'retry_kaggle_read'))
        from unittest.mock import Mock
        for error in (ValueError('configuration'),HTTPError(response=SimpleNamespace(status_code=403,headers={}))):
            call=Mock(side_effect=error)
            with self.assertRaises(type(error)): run_queue.retry_kaggle_read(call,sleep=Mock())
            self.assertEqual(call.call_count,1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.api = FakeKaggle()
        self.configs = experiment_configs()[6:]

    def run_queue(self, configs=None):
        with patch.object(run_queue, 'latest_version', side_effect=lambda api, ref: api.versions[ref]), \
             patch.object(run_queue, 'verify_results', return_value={'macro_auc': .6, 'bce': .7}):
            run_queue.run(self.api, configs or self.configs, self.root, poll_seconds=0)

    def test_orders_jobs_pins_returned_version_and_resume_does_not_upload_again(self):
        self.run_queue()
        self.assertEqual(len(self.api.uploads), 6)
        for train, submit in zip(self.api.uploads[::2], self.api.uploads[1::2]):
            self.assertTrue(train['enable_internet'])
            self.assertFalse(submit['enable_internet'])
            self.assertEqual(submit['kernel_sources'], [train['id'] + '/7'])
        self.assertTrue((self.root / 'summary.csv').exists())
        self.run_queue()
        self.assertEqual(len(self.api.uploads), 6)

    def test_interrupt_after_upload_resumes_existing_job(self):
        self.api.interrupt = True
        with self.assertRaises(KeyboardInterrupt):
            self.run_queue()
        state = json.loads((self.root / 'state.json').read_text())
        self.assertEqual(state['jobs'][self.configs[0]['id']]['train']['version'], 7)
        self.run_queue()
        self.assertEqual(len(self.api.uploads), 6)

    def test_failed_job_stops_before_submission_and_next_experiment(self):
        self.api.fail = True
        with self.assertRaisesRegex(RuntimeError, 'ERROR'):
            self.run_queue()
        self.assertEqual(len(self.api.uploads), 1)

    def test_unknown_push_outcome_is_not_automatically_retried(self):
        with patch.object(self.api, 'kernels_push', side_effect=ConnectionError('lost response')):
            with self.assertRaises(ConnectionError):
                self.run_queue()
        with self.assertRaisesRegex(RuntimeError, 'upload outcome'):
            self.run_queue()
        self.assertEqual(self.api.uploads, [])

    def test_changed_experiment_config_cannot_reuse_run(self):
        self.run_queue()
        changed = [dict(c, epochs=10) for c in self.configs]
        with self.assertRaisesRegex(ValueError, 'new run name'):
            self.run_queue(changed)

    def test_remote_version_change_is_rejected(self):
        with patch.object(run_queue, 'latest_version', return_value=8):
            with self.assertRaisesRegex(RuntimeError, 'version changed'):
                run_queue.wait_for_job(self.api, 'owner/slug', 7, self.root, 0)

    def test_weak_queue_pins_dependencies_and_never_attaches_reports_to_submission(self):
        from experiments.weak_configs import weak_configs
        configs = weak_configs()
        self.run_queue(configs)
        self.assertEqual(len(self.api.uploads), 14)
        for config, train, submit in zip(configs, self.api.uploads[::2], self.api.uploads[1::2]):
            expected = ['choiyoungwhan/rsna-knee-' + eid.replace('_','-') + '-train/7'
                        for eid in config['train_dependencies']]
            self.assertEqual(train['kernel_sources'], expected)
            self.assertEqual(submit['dataset_sources'], [])
            self.assertEqual(submit['kernel_sources'], [train['id']+'/7'])
        self.run_queue(configs)
        self.assertEqual(len(self.api.uploads), 14)

    def test_weak_queue_missing_dependency_cannot_start(self):
        from experiments.weak_configs import weak_configs
        with self.assertRaisesRegex(RuntimeError, 'dependency'):
            self.run_queue(weak_configs()[1:2])
        self.assertEqual(self.api.uploads, [])

    def test_bad_download_can_be_retried_without_new_gpu_jobs(self):
        with patch.object(run_queue, 'latest_version', side_effect=lambda api, ref: api.versions[ref]), \
             patch.object(run_queue, 'verify_results', side_effect=ValueError('incomplete download')):
            with self.assertRaises(ValueError):
                run_queue.run(self.api, self.configs[:1], self.root, 0)
        state = json.loads((self.root / 'state.json').read_text())
        self.assertFalse(state['jobs'][self.configs[0]['id']]['submit']['downloaded'])
        self.run_queue(self.configs[:1])
        self.assertEqual(len(self.api.uploads), 2)

    def test_verification_rejects_mismatched_model_and_invalid_probabilities(self):
        import pandas as pd
        config = self.configs[0]
        train, submit = self.root / 'train', self.root / 'submit'
        (train / 'run/bundle').mkdir(parents=True)
        submit.mkdir()
        metrics = {'experiment': config, 'macro_auc_mean': .6, 'bce_mean': .7}
        assets = {'experiment': config, 'final_training_studies': 58,
                  'files': {'model.pt': 'head-hash', 'encoder.ts': 'encoder-hash'}}
        summary = {'experiment': config, 'internet_required': False, 'studies': 1,
                   'model_sha256': 'head-hash', 'encoder_sha256': 'encoder-hash'}
        run_queue.write_json(train / 'run/metrics.json', metrics)
        run_queue.write_json(train / 'run/bundle/assets.json', assets)
        run_queue.write_json(submit / 'inference_summary.json', summary)
        frame = pd.DataFrame([['study', *([.5]*12)]], columns=['StudyInstanceUID', *run_queue.LABELS])
        frame.to_csv(submit / 'submission.csv', index=False)
        with patch.object(run_queue, 'ROOT', self.root):
            self.assertEqual(run_queue.verify_results(config, train, submit)['macro_auc'], .6)
            summary['model_sha256'] = 'wrong-model'
            run_queue.write_json(submit / 'inference_summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'final training model'):
                run_queue.verify_results(config, train, submit)
            summary['model_sha256'] = 'head-hash'
            run_queue.write_json(submit / 'inference_summary.json', summary)
            frame.iloc[0, 1] = 1.1
            frame.to_csv(submit / 'submission.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'probabilities'):
                run_queue.verify_results(config, train, submit)


if __name__ == '__main__':
    unittest.main()
