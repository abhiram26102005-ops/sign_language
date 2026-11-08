import unittest
from unittest.mock import patch

import cv2
import numpy as np

from src.app import GestureService, app, service


class SentenceEventTests(unittest.TestCase):
    def setUp(self):
        self.service = GestureService()
        self.service.state.update(
            camera_status="running",
            prediction="hello",
            event_id=12,
        )

    def test_add_word_records_sentence_and_history_once(self):
        self.assertEqual(self.service.add_word(12), "hello")
        state = self.service.snapshot()
        self.assertEqual(state["sentence_text"], "hello")
        self.assertEqual(len(state["history"]), 1)

    def test_rejects_duplicate_prediction_event(self):
        self.service.add_word(12)
        with self.assertRaisesRegex(ValueError, "already added"):
            self.service.add_word(12)
        self.assertEqual(self.service.snapshot()["sentence"], ["hello"])

    def test_requires_running_recognition(self):
        self.service.state["camera_status"] = "stopped"
        with self.assertRaisesRegex(ValueError, "Hold a trained sign"):
            self.service.add_word(12)


class DashboardRouteTests(unittest.TestCase):
    def tearDown(self):
        with service.lock:
            service.sentence.clear()

    def test_dashboard_and_state_routes(self):
        client = app.test_client()
        self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(client.get("/api/state").status_code, 200)

    def test_empty_sentence_cannot_be_spoken(self):
        response = app.test_client().post("/api/speech/speak", json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Build a sentence", response.json["error"])


class BrowserCameraFrameTests(unittest.TestCase):
    def setUp(self):
        self.service = GestureService()
        self.assertEqual(self.service.start()["camera_status"], "running")
        self.service_patch = patch("src.app.service", self.service)
        self.service_patch.start()
        self.client = app.test_client()

    def tearDown(self):
        self.service_patch.stop()
        self.service.stop()

    def test_jpeg_frame_is_processed_without_server_webcam(self):
        ok, frame = cv2.imencode(".jpg", np.zeros((240, 320, 3), dtype=np.uint8))
        self.assertTrue(ok)
        response = self.client.post(
            "/api/frame",
            data=frame.tobytes(),
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["state"]["camera_status"], "running")
        self.assertIsNotNone(self.service.latest_frame)

    def test_rejects_non_image_frame(self):
        response = self.client.post(
            "/api/frame",
            data=b"not an image",
            content_type="image/jpeg",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("decode", response.json["error"])


if __name__ == "__main__":
    unittest.main()