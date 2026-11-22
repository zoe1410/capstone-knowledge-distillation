import tempfile
import unittest
from pathlib import Path

import torch

from Student_extraction import StudentWithHead
from test import load_student


class CheckpointTest(unittest.TestCase):
    def test_loads_original_and_legacy_classifier_widths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pth"
            for widths, state_field in (((512, 256), "student_state_dict"), ((256, 128), "model_state_dict")):
                with self.subTest(widths=widths):
                    model = StudentWithHead(5, pretrained=False, hidden_sizes=widths)
                    torch.save({state_field: model.state_dict()}, path)

                    loaded = load_student(path, num_classes=5)

                    torch.testing.assert_close(loaded.state_dict(), model.state_dict())

    def test_rejects_mismatched_class_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pth"
            model = StudentWithHead(5, pretrained=False)
            torch.save(model.state_dict(), path)

            with self.assertRaisesRegex(ValueError, "Dataset has 10 classes, but the checkpoint expects 5"):
                load_student(path, num_classes=10)
