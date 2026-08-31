import unittest

import numpy as np

from gender_analysis import interpret_result


def detailed_test_image() -> np.ndarray:
    checkerboard = np.indices((200, 200)).sum(axis=0) % 2
    grey = (checkerboard * 255).astype(np.uint8)
    return np.repeat(grey[:, :, np.newaxis], 3, axis=2)


def result_with(**overrides):
    result = {
        "face_confidence": 0.99,
        "region": {"x": 40, "y": 40, "w": 100, "h": 100},
        "gender": {"Man": 90.0, "Woman": 10.0},
    }
    result.update(overrides)
    return result


class InterpretResultTests(unittest.TestCase):
    def setUp(self):
        self.image = detailed_test_image()

    def test_accepts_a_clear_high_confidence_result(self):
        outcome = interpret_result([result_with()], self.image, scale=1.0)

        self.assertIsNotNone(outcome.prediction)
        self.assertEqual(outcome.prediction.label, "Male")
        self.assertEqual(outcome.prediction.confidence, 90.0)

    def test_abstains_when_classification_scores_are_too_close(self):
        outcome = interpret_result(
            [result_with(gender={"Man": 55.0, "Woman": 45.0})],
            self.image,
            scale=1.0,
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("not confident enough", outcome.message)

    def test_abstains_when_face_was_too_small_in_original_image(self):
        outcome = interpret_result(
            [result_with(region={"x": 40, "y": 40, "w": 100, "h": 100})],
            self.image,
            scale=2.0,
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("too far", outcome.message)

    def test_abstains_when_multiple_faces_are_detected(self):
        outcome = interpret_result(
            [result_with(), result_with()],
            self.image,
            scale=1.0,
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("Multiple faces", outcome.message)

    def test_abstains_when_face_detection_is_uncertain(self):
        outcome = interpret_result(
            [result_with(face_confidence=0.70)],
            self.image,
            scale=1.0,
        )

        self.assertIsNone(outcome.prediction)
        self.assertIn("detected face is uncertain", outcome.message)


if __name__ == "__main__":
    unittest.main()
