import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_local_public_methods import predecessor_ready, verify_pins, prepare_queue


class PublicQueueTests(unittest.TestCase):
    def test_predecessor_failure_or_identity_change_never_starts_next_training(self):
        identity = dict(config={'id': 'baseline'}, input_hashes={'labels': 'abc'},
                        source_hashes={'train.py': 'v1'}, started_at='start', pid=123)
        self.assertFalse(predecessor_ready(dict(identity, status='running'), identity))
        self.assertTrue(predecessor_ready(dict(identity, status='complete'), identity))
        for record in (dict(identity, status='failed'), dict(identity, status='complete', pid=456)):
            with self.assertRaises(ValueError): predecessor_ready(record, identity)

    def test_snapshot_survives_workspace_changes_but_rejects_changed_inputs_or_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('experiments/__init__.py', 'experiments/core.py', 'scripts/__init__.py',
                         'scripts/train_local.py', 'scripts/run_local_public_methods.py',
                         'data/raw/train.csv', 'data/raw/train_series.csv',
                         'data/weak_labels/report_labels_v2.csv', 'data/local-assets/s11/run/bundle/assets.json'):
                path = root/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('original')
            predecessor = root/'outputs/local/earlystop-p3-rtx5090-train/local_run.json'
            predecessor.parent.mkdir(parents=True)
            predecessor.write_text(json.dumps(dict(status='running', config={}, input_hashes={},
                                                   source_hashes={}, started_at='start', pid=123)))
            queue = root/'queue'
            prepare_queue(root, queue)
            (root/'experiments/core.py').write_text('workspace changed')
            verify_pins(queue)
            with self.assertRaises(FileExistsError): prepare_queue(root, queue)
            target = root/'data/raw/train.csv'; target.write_text('changed')
            with self.assertRaises(ValueError): verify_pins(queue)
            target.write_text('original')
            (queue/'source/experiments/core.py').write_text('snapshot changed')
            with self.assertRaises(ValueError): verify_pins(queue)
