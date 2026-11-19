import unittest
from fastapi import HTTPException

from serving.app import bundle, health, predict, PredictRequest


class TestPredictionAPI(unittest.TestCase):
    def test_health_when_artifact_is_available(self):
        if bundle is None:
            self.skipTest("Model artifact is unavailable in this checkout")
        self.assertEqual(health(), {"status": "ok"})

    def test_missing_features_are_rejected(self):
        if bundle is None:
            self.skipTest("Model artifact is unavailable in this checkout")
        with self.assertRaises(HTTPException) as context:
            predict(PredictRequest(features={}))
        self.assertEqual(context.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
