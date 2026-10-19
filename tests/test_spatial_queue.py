import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import run_queue
from experiments.run_spatial_queue import predecessor_complete, exclusive_lock, RunnerBusy, wait_and_run
from experiments.spatial_configs import spatial_configs
from experiments.spatial_notebooks import build_spatial
from experiments.weak_configs import WEAK_IDS
from tests.test_run_queue import FakeKaggle


class SpatialQueueTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.previous=self.root/'previous'; self.previous.mkdir()

    def state(self,complete=True):
        jobs={eid:dict(verified=complete,train=dict(version=1,downloaded=complete),submit=dict(version=1,downloaded=complete)) for eid in WEAK_IDS}
        (self.previous/'state.json').write_text(json.dumps(dict(jobs=jobs)))

    def test_gate_requires_all_seven_verified_stages(self):
        self.assertFalse(predecessor_complete(self.previous))
        self.state(False); self.assertFalse(predecessor_complete(self.previous))
        self.state(); self.assertTrue(predecessor_complete(self.previous))
        state=json.loads((self.previous/'state.json').read_text()); state['jobs']['e12_ensemble']['submit']['downloaded']=False
        (self.previous/'state.json').write_text(json.dumps(state)); self.assertFalse(predecessor_complete(self.previous))

    def test_os_lock_blocks_second_owner_and_releases_after_exception(self):
        lock=self.root/'runner.lock'
        with exclusive_lock(lock):
            with self.assertRaises(RunnerBusy):
                with exclusive_lock(lock): pass
        with exclusive_lock(lock): pass

    def test_stopped_incomplete_predecessor_never_starts_callback(self):
        self.state(False)
        with patch('experiments.run_spatial_queue.verify_predecessor') as verify:
            callback=unittest.mock.Mock()
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                wait_and_run(self.previous,self.root/'runner.lock',10,callback)
            callback.assert_not_called(); verify.assert_not_called()

    def test_active_predecessor_waits_then_success_runs_once_under_lock(self):
        self.state(False)
        active=exclusive_lock(self.root/'runner.lock'); active.__enter__()
        def release(_):
            self.state(); active.__exit__(None,None,None)
        calls=[]
        def callback():
            with self.assertRaises(RunnerBusy):
                with exclusive_lock(self.root/'runner.lock'): pass
            calls.append(1)
        with patch('experiments.run_spatial_queue.time.sleep',side_effect=release), \
             patch('experiments.run_spatial_queue.verify_predecessor'):
            wait_and_run(self.previous,self.root/'runner.lock',10,callback)
        self.assertEqual(calls,[1])

    def test_spatial_jobs_pin_dependencies_and_resume_without_uploads(self):
        api=FakeKaggle(); configs=spatial_configs()
        with patch.object(run_queue,'latest_version',side_effect=lambda api,ref:api.versions[ref]), \
             patch.object(run_queue,'verify_results',return_value=dict(macro_auc=.6,bce=.7)):
            run_queue.run(api,configs,self.root/'new',0,builder=build_spatial)
            self.assertEqual(len(api.uploads),28)
            for cfg,train,submit in zip(configs,api.uploads[::2],api.uploads[1::2]):
                self.assertEqual(train['kernel_sources'],['choiyoungwhan/rsna-knee-'+eid.replace('_','-')+'-train/7' for eid in cfg['train_dependencies']])
                self.assertEqual(submit['kernel_sources'],[train['id']+'/7'])
                self.assertFalse(submit['enable_internet']); self.assertEqual(submit['dataset_sources'],[])
            run_queue.run(api,configs,self.root/'new',0,builder=build_spatial)
            self.assertEqual(len(api.uploads),28)


if __name__=='__main__': unittest.main()
