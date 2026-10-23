import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts import train_local
from experiments import spatial_train
from experiments.spatial_core import spatial_signature
from experiments.core import LABELS, file_hash
from experiments.weak_train import write_weak_manifest


class LocalTrainingTests(unittest.TestCase):
    def parent_config(self):
        return dict(id='s11_cross', backbone='swin_t', architecture='spatial_v1',
                    finetune=False, ensemble=False, pipeline_sha256='original',
                    representation='single', neighbors=5, centers=5, crop=224,
                    resize=232, interpolation='bicubic', backbone_epochs=1)

    def test_local_recipe_reuses_parent_features_without_reusing_gold_head(self):
        parent = self.parent_config()
        config = train_local.local_config(parent)
        self.assertEqual(spatial_signature(config), spatial_signature(parent))
        self.assertTrue(config['finetune'])
        self.assertEqual(config['head_warmup_epochs'], 1)
        self.assertEqual(config['train_dependencies'], ['s11_cross'])
        self.assertNotEqual(config['id'], parent['id'])
        self.assertFalse(parent['finetune'])
        with self.assertRaises(ValueError):
            train_local.local_config(dict(parent, finetune=True))
        with self.assertRaises(ValueError):
            train_local.local_config(dict(parent, id='s12_finetune'))

    def test_smoke_subset_excludes_gold_and_keeps_weak_validation_separate(self):
        gold = np.array([True, False, False, False, False])
        groups = np.array(['00000001', '00000001', '0000000a', '00000002', '00000003'])
        indices = train_local.smoke_indices(gold, groups)
        np.testing.assert_array_equal(indices, [1, 3, 2])
        self.assertFalse(gold[indices].any())
        with self.assertRaises(ValueError):
            train_local.smoke_indices(gold[:3], groups[:3])

    def test_early_stopping_is_a_separate_recipe_with_the_same_frozen_parent(self):
        parent = self.parent_config()
        config = train_local.local_config(parent, early_stopping=True)
        self.assertEqual(spatial_signature(config), spatial_signature(parent))
        self.assertEqual(config['train_dependencies'], ['s11_cross'])
        self.assertEqual(config['early_stopping_patience'], 3)
        self.assertEqual(config['backbone_epochs'], 30)
        self.assertGreater(config['backbone_timeout_hours'], 30*2)
        original = train_local.local_config(parent)
        self.assertNotEqual(config['id'], original['id'])
        self.assertEqual(original['backbone_epochs'], 1)
        self.assertNotIn('early_stopping_patience', original)

    def test_local_paths_are_scoped_and_restored_on_failure(self):
        old_labels, old_parent = spatial_train.find_report_labels, spatial_train.find_training_run
        with self.assertRaisesRegex(RuntimeError, 'stop'):
            with train_local.local_paths(Path('labels.csv'), Path('parent')):
                self.assertEqual(spatial_train.find_report_labels({}), Path('labels.csv'))
                self.assertEqual(spatial_train.find_training_run('s11_cross'), Path('parent'))
                with self.assertRaises(ValueError):
                    spatial_train.find_training_run('other')
                raise RuntimeError('stop')
        self.assertIs(spatial_train.find_report_labels, old_labels)
        self.assertIs(spatial_train.find_training_run, old_parent)

    def test_image_check_matches_selected_protocol_and_rejects_missing_dicoms(self):
        table = pd.DataFrame(dict(StudyInstanceUID=['a']))
        series = pd.DataFrame([
            dict(StudyInstanceUID='a', SeriesInstanceUID='selected',
                 Anatomical_Plane='Sagittal', Fat_Suppression=True, Fluid_Sensitive=True),
            dict(StudyInstanceUID='a', SeriesInstanceUID='unused',
                 Anatomical_Plane='Sagittal', Fat_Suppression=True, Fluid_Sensitive=False)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                train_local.check_images(table, series, root)
            folder = root/'a/selected'
            folder.mkdir(parents=True)
            (folder/'slice.dcm').write_bytes(b'presence check only')
            self.assertEqual(train_local.check_images(table, series, root), 1)
            bad = series.copy()
            bad.loc[0, 'SeriesInstanceUID'] = '../outside'
            with self.assertRaises(ValueError):
                train_local.check_images(table, bad, root)
            with self.assertRaises(ValueError):
                train_local.check_images(table, series.iloc[:0], root)

    def test_output_reservation_never_overwrites_and_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'run'
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                with train_local.run_record(output, {'mode': 'smoke'}):
                    raise RuntimeError('interrupted')
            record = json.loads((output/'local_run.json').read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertIn('interrupted', record['error'])
            with self.assertRaises(FileExistsError):
                with train_local.run_record(output, {'mode': 'train'}):
                    self.fail('existing output was reused')

    def test_preflight_checks_real_parent_labels_cache_and_partial_smoke_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, parent = root/'data', root/'parent'
            data.mkdir()
            bundle = parent/'bundle'
            (bundle/'wheels').mkdir(parents=True)
            ids = [f'1.2.{i:05d}' for i in range(4350)]
            train = pd.DataFrame(dict(StudyInstanceUID=ids, Report=[f'report {i}' for i in ids]))
            for label in LABELS:
                train[label] = [0.]*58 + [np.nan]*4292
            train.to_csv(data/'train.csv', index=False)
            labels = pd.DataFrame({'StudyInstanceUID': ids[58:]})
            for label in LABELS:
                labels[label], labels[label+'__conf'], labels[label+'__verdict'] = .7, .9, 'YES'
            labels_path = root/'labels.csv'
            labels.to_csv(labels_path, index=False)
            config = dict(self.parent_config(), weak_labels=True, feature_dim=3,
                          labels_sha256=file_hash(labels_path))
            for name in ('encoder.ts', 'model.pt', 'wheels/pydicom-3.0.2-test.whl',
                         'wheels/python_gdcm-3.2.6-test.whl'):
                (bundle/name).write_bytes(b'hash validation fixture')
            cache = dict(ids=ids, tokens=torch.zeros(4350, 6, 1, 3),
                         mask=torch.ones(4350, 6, 1, dtype=torch.bool), coords=torch.zeros(4350, 6, 1, 3),
                         encoder_sha256=file_hash(bundle/'encoder.ts'), view_signature=spatial_signature(config))
            torch.save(cache, parent/'tokens.pt')
            write_weak_manifest(config, bundle, 4350, 58, dict(tokens_sha256=file_hash(parent/'tokens.pt')))
            _, _, _, gold, groups, _ = train_local.make_targets(train, labels, config)
            selected = train_local.smoke_indices(gold, groups)
            series = pd.DataFrame(dict(StudyInstanceUID=ids, SeriesInstanceUID=['s'+i for i in ids],
                                       Anatomical_Plane='Sagittal', Fat_Suppression=True, Fluid_Sensitive=True))
            series.to_csv(data/'train_series.csv', index=False)
            for index in selected:
                folder = data/'train_series'/ids[index]/('s'+ids[index])
                folder.mkdir(parents=True)
                (folder/'one.dcm').write_bytes(b'presence only')
            result = train_local.validate_inputs(data, labels_path, parent, 'smoke')
            self.assertEqual(result['selected_series'], 3)
            with self.assertRaises(FileNotFoundError):
                train_local.validate_inputs(data, labels_path, parent, 'train')
            original = labels_path.read_bytes()
            labels_path.write_bytes(original+b'\n')
            with self.assertRaisesRegex(ValueError, 'label hash'):
                train_local.validate_inputs(data, labels_path, parent, 'smoke')
            labels_path.write_bytes(original)
            (parent/'tokens.pt').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'cache hash'):
                train_local.validate_inputs(data, labels_path, parent, 'smoke')


if __name__ == '__main__':
    unittest.main()
