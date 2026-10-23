import json
from pathlib import Path
import tempfile
import unittest

import torch

from experiments.core import file_hash
from experiments.public_finetune import average_backbone
from experiments.spatial_train import retain_backbone, warmup_cosine_scale


class PublicFinetuningTests(unittest.TestCase):
    def test_schedule_warms_up_then_decays_to_floor_without_negative_learning_rate(self):
        self.assertAlmostEqual(warmup_cosine_scale(0, 100), .2)
        self.assertAlmostEqual(warmup_cosine_scale(4, 100), 1.)
        self.assertAlmostEqual(warmup_cosine_scale(5, 100), 1.)
        self.assertGreater(warmup_cosine_scale(30, 100), warmup_cosine_scale(90, 100))
        self.assertAlmostEqual(warmup_cosine_scale(100, 100), .01)
        self.assertAlmostEqual(warmup_cosine_scale(110, 100), .01)
        with self.assertRaises(ValueError): warmup_cosine_scale(1, 0)

    def test_average_uses_best_three_weak_checkpoints_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            config = {'recipe': 'test'}
            (root/'local_run.json').write_text(json.dumps(dict(status='complete', config=config)))
            (run/'backbone_audit.json').write_text(json.dumps(dict(gold_studies_used_for_backbone=0)))
            model = torch.nn.Linear(1, 1, bias=False)
            for epoch, score in enumerate([.6, .4, .5, .7, .3], 1):
                with torch.no_grad(): model.weight.fill_(epoch)
                retain_backbone(model, run, epoch, score, 3)
            state, audit = average_backbone(root, config)
            self.assertEqual([r['epoch'] for r in audit['checkpoints']], [5, 2, 3])
            torch.testing.assert_close(state['weight'], torch.tensor([[10/3]]))
            with self.assertRaisesRegex(ValueError, 'matching training recipe'):
                average_backbone(root, {'recipe': 'wrong'})
            target = run/audit['checkpoints'][0]['file']
            target.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                average_backbone(root, config)

    def test_average_rejects_nonfinite_weights_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; run.mkdir()
            (root/'local_run.json').write_text(json.dumps(dict(status='complete', config={})))
            (run/'backbone_audit.json').write_text(json.dumps(dict(gold_studies_used_for_backbone=0)))
            rows = []
            for epoch in range(1, 4):
                name = f'backbone_epoch_{epoch:03d}.pt'
                torch.save({'weight': torch.tensor([float('nan')])}, run/name)
                rows.append(dict(epoch=epoch, file=name, sha256=file_hash(run/name), validation_bce=.5))
            (run/'backbone_candidates.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                average_backbone(root, {})


if __name__ == '__main__':
    unittest.main()
