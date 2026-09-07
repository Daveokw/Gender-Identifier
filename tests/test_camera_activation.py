import unittest
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


class CameraActivationTests(unittest.TestCase):
    def test_camera_widget_is_created_only_after_user_enables_it(self):
        pillow_heif_stub = SimpleNamespace(register_heif_opener=lambda: None)
        with patch.dict("sys.modules", {"pillow_heif": pillow_heif_stub}):
            app = AppTest.from_file("gender_identifier.py").run()

            self.assertEqual(len(app.get("camera_input")), 0)

            app.button(key="enable-camera").click().run()

            self.assertEqual(len(app.get("camera_input")), 1)

    def test_camera_uses_a_selfie_style_preview_without_altering_analysis(self):
        with open("gender_identifier.py", encoding="utf-8") as source_file:
            source = source_file.read()

        self.assertIn('transform: scaleX(-1) !important', source)
        self.assertIn(
            'render_analysis(camera_image, "Captured image", mirror_preview=True)',
            source,
        )
        self.assertIn("outcome = analyse_image_bytes(image_bytes)", source)


if __name__ == "__main__":
    unittest.main()
