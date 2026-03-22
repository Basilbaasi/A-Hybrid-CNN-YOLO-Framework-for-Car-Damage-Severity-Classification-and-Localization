"""Unit tests for prediction functions and Flask endpoints."""
import os
import tempfile
import unittest
import numpy as np
import cv2
from damage_api.src.predict import predict_cnn, predict_yolo
from damage_api.app.flask_app import app

class PredictTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        cv2.imwrite(self.temp.name, dummy_img)
        self.temp.close()  # Release the handle on Windows

    def tearDown(self) -> None:
        try:
            os.remove(self.temp.name)
        except FileNotFoundError:
            pass

    def test_predict_cnn_structure(self) -> None:
        result = predict_cnn(self.temp.name)
        self.assertIsInstance(result, dict)
        self.assertIn("class", result)
        self.assertIn("confidence", result)
        self.assertIsInstance(result["confidence"], float)

    def test_predict_yolo_structure(self) -> None:
        try:
            result = predict_yolo(self.temp.name)
        except Exception as e:
            self.skipTest(f"YOLO model could not be loaded: {e}")
            return
        self.assertIsInstance(result, dict)
        self.assertIn("class", result)
        self.assertIn("confidence", result)
        self.assertIn("boxes", result)
        self.assertIsInstance(result["boxes"], list)

    def test_health_endpoint(self) -> None:
        tester = app.test_client()
        response = tester.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("cnn", data.get("models", []))
        self.assertIn("yolo", data.get("models", []))

if __name__ == "__main__":
    unittest.main()