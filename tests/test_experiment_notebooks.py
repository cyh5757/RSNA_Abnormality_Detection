import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.core import StudyHead, fit_model, validate_bundle


def config(mode="mean", position=False, plane=False):
    return {"id": "test", "backbone": "toy", "feature_dim": 4, "slices": 5,
            "aggregation": mode, "position": position, "plane_embedding": plane,
            "epochs": 5, "lr": .001, "weight_decay": .1}


class ExperimentCoreTest(unittest.TestCase):
    def test_encoder_export_accepts_float32_roundoff_but_rejects_changed_features(self):
        from experiments.train import export_encoder

        class TraceOffset(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, x):
                return x.mean((2, 3)) * 0 + (self.offset if torch.jit.is_tracing() else 0)

        with tempfile.TemporaryDirectory() as tmp:
            cfg = {'crop': 4, 'slices': 5}
            export_encoder(TraceOffset(1.4e-5), cfg, Path(tmp) / 'ok.ts')
            with self.assertRaises(AssertionError):
                export_encoder(TraceOffset(.01), cfg, Path(tmp) / 'bad.ts')

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_mean_keeps_plane_slots_and_ignores_padding(self):
        head = StudyHead(config(), torch.zeros(12), torch.ones(12))
        with torch.no_grad():
            head.classifier.weight.copy_(torch.eye(12))
            head.classifier.bias.zero_()
        x = torch.zeros(1, 3, 5, 4)
        mask = torch.zeros(1, 3, 5, dtype=torch.bool)
        mask[0, 0, :2] = True
        mask[0, 1, 0] = True
        x[0, 0, 0], x[0, 0, 1], x[0, 1, 0] = 2., 4., 8.
        x[~mask] = float("nan")
        result = head(x, mask, torch.zeros(1, 3, 5))
        torch.testing.assert_close(result, torch.tensor([[3.]*4 + [8.]*4 + [0.]*4]))

    def test_attention_respects_mask_and_position_pairing(self):
        torch.manual_seed(12)
        x = torch.randn(2, 3, 5, 4)
        mask = torch.ones(2, 3, 5, dtype=torch.bool)
        mask[0, 2] = False
        positions = torch.linspace(0, 1, 5).expand(2, 3, 5)
        perm = [4, 2, 0, 3, 1]
        for use_pos in (False, True):
            head = StudyHead(config("attention", use_pos, use_pos), torch.zeros(4), torch.ones(4)).eval()
            original = head(x, mask, positions)
            changed = x.clone()
            changed[~mask] = float("nan")
            torch.testing.assert_close(head(changed, mask, positions), original)
            paired = head(x[:, :, perm], mask[:, :, perm], positions[:, :, perm])
            torch.testing.assert_close(paired, original, atol=1e-6, rtol=1e-5)
            moved = head(x[:, :, perm], mask[:, :, perm], positions)
            if use_pos:
                self.assertGreater((moved - original).abs().max().item(), 1e-5)
            else:
                torch.testing.assert_close(moved, original, atol=1e-6, rtol=1e-5)

    def test_validation_values_never_change_fitted_model(self):
        torch.manual_seed(5)
        x = torch.randn(10, 3, 5, 4)
        mask = torch.ones(10, 3, 5, dtype=torch.bool)
        pos = torch.linspace(0, 1, 5).expand(10, 3, 5)
        y = torch.tensor([[i % 2]*12 for i in range(10)], dtype=torch.float32)
        for mode in ("mean", "attention"):
            first, history = fit_model(x, mask, pos, y, np.arange(8), 42, config(mode))
            changed = x.clone()
            changed[8:] = 1e6
            second, _ = fit_model(changed, mask, pos, y, np.arange(8), 42, config(mode))
            for key, value in first.state_dict().items():
                torch.testing.assert_close(value, second.state_dict()[key])
            self.assertEqual(len(history), 5)

    def test_bundle_rejects_other_experiment_corruption_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['encoder.ts', 'model.pt', 'wheels/pydicom-3.0.2-py3-none-any.whl',
                     'wheels/python_gdcm-3.2.6-cp313-cp313-manylinux.whl']
            for name in names:
                target = root / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(b'toy model')
            manifest = {"experiment": config(), "files": {name: hashlib.sha256(b"toy model").hexdigest() for name in names}}
            (root / "assets.json").write_text(json.dumps(manifest))
            self.assertEqual(validate_bundle(root, config()), manifest)
            with self.assertRaises(ValueError):
                validate_bundle(root, {**config(), "id": "different"})
            (root / "encoder.ts").write_bytes(b"modified")
            with self.assertRaises(ValueError):
                validate_bundle(root, config())
            (root / "encoder.ts").write_bytes(b'toy model')
            manifest["files"]["../outside"] = "bad"
            (root / "assets.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                validate_bundle(root, config())

    def test_bundle_requires_hashes_for_every_consumed_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = ['encoder.ts', 'model.pt', 'wheels/pydicom-3.0.2-py3-none-any.whl',
                     'wheels/python_gdcm-3.2.6-cp313-cp313-manylinux.whl']
            for name in names:
                target = root / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(b'toy')
            hashes = {name: hashlib.sha256(b'toy').hexdigest() for name in names}
            for omitted in names:
                manifest = {'experiment': config(), 'files': {name: digest for name, digest in hashes.items() if name != omitted}}
                (root / 'assets.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError, msg=omitted):
                    validate_bundle(root, config())

    def test_export_reload_and_dicom_to_submission(self):
        import pandas as pd
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid
        from experiments.core import LABELS, PLANES, save_submission, study_tokens
        from experiments.train import export_encoder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            encoder = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(1))
            cfg = {**config(), "feature_dim": 3, "crop": 12}
            exported = export_encoder(encoder, cfg, root / "encoder.ts")
            rows = []
            for plane in PLANES:
                folder = root / "study" / plane
                folder.mkdir(parents=True)
                rows.append({"StudyInstanceUID": "study", "SeriesInstanceUID": plane,
                             "Anatomical_Plane": plane, "Fluid_Sensitive": True, "Fat_Suppression": True})
                for index in range(3):
                    meta = FileMetaDataset()
                    meta.TransferSyntaxUID = ExplicitVRLittleEndian
                    meta.MediaStorageSOPClassUID = generate_uid()
                    meta.MediaStorageSOPInstanceUID = generate_uid()
                    ds = FileDataset(str(folder / f'{2-index}.dcm'), {}, file_meta=meta, preamble=b'\0'*128)
                    ds.Rows = ds.Columns = 12
                    ds.SamplesPerPixel = 1
                    ds.PhotometricInterpretation = 'MONOCHROME2'
                    ds.BitsAllocated = ds.BitsStored = 16
                    ds.HighBit, ds.PixelRepresentation = 15, 0
                    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                    ds.ImagePositionPatient = [0, 0, index*2]
                    ds.InstanceNumber = index+1
                    ds.PixelData = np.arange(144, dtype=np.uint16).reshape(12, 12).tobytes()
                    ds.save_as(ds.filename, enforce_file_format=True)
            tokens, mask, positions = study_tokens('study', pd.DataFrame(rows), root,
                                                   exported, lambda x: x, cfg, 'cpu')
            self.assertEqual(mask.sum().item(), 9)
            torch.testing.assert_close(positions[0, :3], torch.tensor([0., .5, 1.]))
            head = StudyHead(cfg, torch.zeros(9), torch.ones(9)).eval()
            with torch.no_grad():
                probabilities = head(tokens[None], mask[None], positions[None]).sigmoid().numpy()
            frame = save_submission(['study'], probabilities, ['StudyInstanceUID', *LABELS], root / 'submission.csv')
            self.assertEqual(frame.shape, (1, 13))
            with self.assertRaises(ValueError):
                save_submission(['study'], np.full((1, 12), np.nan), frame.columns, root / 'bad.csv')

    def test_each_generated_notebook_is_standalone_and_single_experiment(self):
        import ast
        from experiments.build_notebooks import EXPERIMENTS, render_notebook
        self.assertGreaterEqual(len(EXPERIMENTS), 9)
        for cfg in EXPERIMENTS:
            for mode in ('train', 'submit'):
                notebook = render_notebook(cfg, mode)
                code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
                tree = ast.parse(code)
                self.assertEqual(sum(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENT' for t in n.targets) for n in tree.body), 1)
                self.assertNotIn('from experiments', code)
                if mode == 'submit':
                    self.assertNotIn('torch.hub.load', code)
                    self.assertNotIn('def fit_model', code)
                    self.assertIn('--no-index', code)


if __name__ == "__main__":
    unittest.main()
