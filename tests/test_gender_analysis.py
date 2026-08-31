import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from gender_analysis import analyse_gender


def detailed_test_image() -> Image.Image:
    checkerboard = np.indices((1_200, 1_200)).sum(axis=0) % 2
    grey = (checkerboard * 255).astype(np.uint8)
    rgb = np.repeat(grey[:, :, np.newaxis], 3, axis=2)
    return Image.fromarray(rgb, mode="RGB")


class StubDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, _image):
        return self.faces


class StubClassifier:
    def __init__(self, label):
        self.label = label

    def predict(self, _image, _face):
        return SimpleNamespace(sex=self.label)


def detected_face(bbox=(400, 400, 800, 800)):
    return SimpleNamespace(bbox=np.asarray(bbox, dtype=np.float32))


class AnalyseGenderTests(unittest.TestCase):
    def setUp(self):
        self.image = detailed_test_image()

    def test_returns_male_for_a_clear_face(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([detected_face()]),
            classifier=StubClassifier("Male"),
        )

        self.assertIsNotNone(outcome.prediction)
        self.assertEqual(outcome.prediction.label, "Male")

    def test_returns_female_for_a_clear_face(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([detected_face()]),
            classifier=StubClassifier("Female"),
        )

        self.assertIsNotNone(outcome.prediction)
        self.assertEqual(outcome.prediction.label, "Female")

    def test_declines_when_face_is_too_small(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([detected_face((20, 20, 70, 70))]),
            classifier=StubClassifier("Male"),
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("too far", outcome.message)

    def test_declines_when_multiple_faces_are_detected(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([detected_face(), detected_face()]),
            classifier=StubClassifier("Male"),
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("Multiple faces", outcome.message)

    def test_declines_when_no_face_is_detected(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([]),
            classifier=StubClassifier("Male"),
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("No clear face", outcome.message)

    def test_declines_an_unsupported_classifier_label(self):
        outcome = analyse_gender(
            self.image,
            detector=StubDetector([detected_face()]),
            classifier=StubClassifier("Unknown"),
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("could not be analysed", outcome.message)


if __name__ == "__main__":
    unittest.main()
