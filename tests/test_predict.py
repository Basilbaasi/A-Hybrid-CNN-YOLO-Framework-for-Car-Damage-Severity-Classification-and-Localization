"""Unit tests for prediction functions and FastAPI endpoint wiring."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from damage_api.app.main import app

class PredictTestCase(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        # Unit tests must not load the large, Git-ignored model artifacts.
        with patch("damage_api.app.main.models.load"):
            with TestClient(app) as tester:
                response = tester.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data.get("status"), {"ok", "degraded"})
        self.assertIn("cnn", data.get("models", {}))
        self.assertIn("yolo", data.get("models", {}))

if __name__ == "__main__":
    unittest.main()
