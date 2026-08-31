import unittest

from streamlit.testing.v1 import AppTest


class CameraActivationTests(unittest.TestCase):
    def test_camera_widget_is_created_only_after_user_enables_it(self):
        app = AppTest.from_file("gender_identifier.py").run()

        self.assertEqual(len(app.get("camera_input")), 0)

        app.button(key="enable-camera").click().run()

        self.assertEqual(len(app.get("camera_input")), 1)


if __name__ == "__main__":
    unittest.main()
