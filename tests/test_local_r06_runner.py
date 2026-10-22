import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from scripts.run_local_r06 import extract_series


class LocalR06RunnerTests(unittest.TestCase):
    def test_extracts_only_selected_series_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            name = 'train_series/a/b/one.dcm'
            content = b'\0' * 128 + b'DICM' + b'image'
            (data / name).parent.mkdir(parents=True)
            (data / name).write_bytes(content)
            with ZipFile(root / 'images.zip', 'w') as archive:
                archive.writestr(name, content)
                archive.writestr('train_series/a/b/two.dcm', content)
                archive.writestr('train_series/unused/b/other.dcm', content)
            self.assertEqual(extract_series(root / 'images.zip', data, {'train_series/a/b'}), 2)
            self.assertEqual((data / name).read_bytes(), content)
            self.assertTrue((data / 'train_series/a/b/two.dcm').is_file())
            self.assertFalse((data / 'train_series/unused').exists())

    def test_conflicting_existing_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = 'train_series/a/b/one.dcm'
            (root / name).parent.mkdir(parents=True)
            (root / name).write_bytes(b'existing')
            with ZipFile(root / 'images.zip', 'w') as archive:
                archive.writestr(name, b'\0' * 128 + b'DICM')
            with self.assertRaises(FileExistsError):
                extract_series(root / 'images.zip', root, {'train_series/a/b'})
            self.assertEqual((root / name).read_bytes(), b'existing')

    def test_missing_series_or_path_traversal_prevents_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('train_series/other/b/one.dcm', 'train_series/a/b/../../escape.dcm'):
                with self.subTest(name=name):
                    with ZipFile(root / 'images.zip', 'w') as archive:
                        archive.writestr(name, b'\0' * 128 + b'DICM')
                    with self.assertRaises(ValueError):
                        extract_series(root / 'images.zip', root / 'data', {'train_series/a/b'})
                    self.assertFalse((root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
