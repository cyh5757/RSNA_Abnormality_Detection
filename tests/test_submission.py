import csv
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SubmissionTest(unittest.TestCase):
    def test_saved_submission_covers_test_studies_with_valid_probabilities(self):
        output = ROOT / "outputs/submission/submission.csv"
        self.assertTrue(output.is_file(), "Submission notebook must produce submission.csv")
        with (ROOT / "data/raw/sample_submission.csv").open(newline="", encoding="utf-8-sig") as stream:
            expected_columns = next(csv.reader(stream))
        with (ROOT / "data/raw/test.csv").open(newline="", encoding="utf-8-sig") as stream:
            expected_ids = [row["StudyInstanceUID"] for row in csv.DictReader(stream)]
        with output.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            self.assertEqual(reader.fieldnames, expected_columns)
            rows = list(reader)
        self.assertEqual([row["StudyInstanceUID"] for row in rows], expected_ids)
        self.assertEqual(len({row["StudyInstanceUID"] for row in rows}), len(rows))
        for row in rows:
            for label in expected_columns[1:]:
                value = float(row[label])
                self.assertTrue(math.isfinite(value) and 0 <= value <= 1, label)


if __name__ == "__main__":
    unittest.main()
