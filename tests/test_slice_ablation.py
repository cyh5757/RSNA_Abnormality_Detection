import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "slice_ablation", ROOT / "notebooks/experiments/slice_ablation/slice_ablation.py")
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


class SliceAblationTest(unittest.TestCase):
    def test_sampling_adds_intermediate_slices_without_widening_coverage(self):
        self.assertEqual(experiment.slice_indices(9, 3).tolist(), [2, 4, 6])
        self.assertEqual(experiment.slice_indices(9, 5).tolist(), [2, 3, 4, 5, 6])
        self.assertEqual(experiment.slice_indices(2, 5).tolist(), [0, 1])
        self.assertEqual(experiment.slice_indices(1, 5).tolist(), [0])
        with self.assertRaises(ValueError):
            experiment.slice_indices(0, 5)

    def test_cache_alignment_uses_ids_and_rejects_ambiguous_or_invalid_data(self):
        features = torch.stack([torch.full((1536,), 8.), torch.full((1536,), 3.)])
        cache = {"study_ids": ["b", "a"], "features": features}
        aligned = experiment.align_features(cache, ["a", "b"])
        self.assertEqual(aligned[:, 0].tolist(), [3., 8.])
        for bad_ids in (["a", "a"], ["a", "c"]):
            with self.assertRaises(ValueError):
                experiment.align_features(cache, bad_ids)
        with self.assertRaises(ValueError):
            experiment.align_features({**cache, "study_ids": ["a", "a"]}, ["a", "b"])
        features[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            experiment.align_features(cache, ["a", "b"])

    def test_folds_are_stable_by_study_id_and_cover_each_study_once(self):
        ids = [f"study-{i:03}" for i in range(60)]
        y = np.array([[i % 2, (i // 2) % 2] for i in range(60)], dtype=np.float32)
        folds = experiment.make_folds(ids, y)
        reordered = experiment.make_folds(ids[::-1], y[::-1])
        self.assertEqual(dict(zip(ids, folds)), dict(zip(ids[::-1], reordered)))
        self.assertEqual(np.bincount(folds).tolist(), [20, 20, 20])
        for fold in range(3):
            self.assertTrue(set(np.array(ids)[folds == fold]).isdisjoint(
                np.array(ids)[folds != fold]))
        with self.assertRaises(ValueError):
            experiment.make_folds(ids, np.zeros_like(y))

    def test_validation_features_do_not_change_training_statistics_or_weights(self):
        torch.set_num_threads(1)
        x = torch.arange(32, dtype=torch.float32).reshape(8, 4) / 10
        y = torch.tensor([[i % 2, (i // 2) % 2] for i in range(8)], dtype=torch.float32)
        valid = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
        logits, history, state = experiment.fit_head(x, y, valid, 42)
        extended = torch.cat([valid, torch.full((1, 4), 1e6)])
        other, _, other_state = experiment.fit_head(x, y, extended, 42)
        torch.testing.assert_close(logits, other[:2])
        torch.testing.assert_close(state["mean"], x.mean(0))
        torch.testing.assert_close(state["std"], x.std(0).clamp_min(.01))
        for key in state["head"]:
            torch.testing.assert_close(state["head"][key], other_state["head"][key])
        self.assertEqual(len(history), 5)
        self.assertTrue(torch.isfinite(logits).all())

    def test_evaluation_writes_one_oof_prediction_per_study_and_seed(self):
        torch.set_num_threads(1)
        ids = [f"study-{i:03}" for i in range(60)]
        y = np.array([[i % 2, (i // 2) % 2] for i in range(60)], dtype=np.float32)
        x = torch.from_numpy(np.column_stack([y, np.arange(60) / 60]).astype(np.float32))
        folds = experiment.make_folds(ids, y)
        with tempfile.TemporaryDirectory() as tmp:
            metrics = experiment.evaluate(x, y, ids, ["a", "b"], folds, Path(tmp), seeds=(42,))
            predictions = pd.read_csv(Path(tmp) / "oof_predictions.csv")
            self.assertEqual(len(predictions), 60)
            self.assertTrue(predictions.StudyInstanceUID.is_unique)
            self.assertEqual(predictions.StudyInstanceUID.tolist(), ids)
            self.assertTrue(np.isfinite(predictions[["a", "b"]]).all().all())
            self.assertEqual(len(metrics["runs"][0]["folds"]), 3)
            self.assertGreaterEqual(metrics["macro_auc_mean"], 0.)
            self.assertLessEqual(metrics["macro_auc_mean"], 1.)


if __name__ == "__main__":
    unittest.main()
