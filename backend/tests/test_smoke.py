import unittest

from fastapi.testclient import TestClient

import main as backend_main


class BackendSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(backend_main.app)

    def test_cases_endpoint_returns_list(self) -> None:
        resp = self.client.get("/api/cases")
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()

        self.assertIsInstance(payload, list)
        self.assertGreater(len(payload), 0)

        first = payload[0]
        self.assertIn("id", first)
        self.assertIn("category_key", first)
        self.assertIn("methodology_id", first)

    def test_health_endpoint(self) -> None:
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
