import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main as backend_main


class ArticleSuiteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(backend_main.app)

    def setUp(self) -> None:
        backend_main.sessions.clear()

    def test_verify_turn_returns_proof_and_violations(self) -> None:
        resp = self.client.post(
            "/api/verify_turn",
            json={
                "case_id": "gtr_01",
                "user_message": "Вы должны просто перестать переживать и немедленно взять себя в руки.",
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("proof_object", data)
        self.assertIn("verifier_pass", data)
        self.assertIn("verifier_violations", data)
        self.assertIsInstance(data["verifier_violations"], list)
        self.assertGreater(len(data["verifier_violations"]), 0)

    def test_ravr_benchmark_shape(self) -> None:
        resp = self.client.post(
            "/api/ravr_benchmark",
            json={"n_per_case": 2, "random_seed": 7, "include_llm_eval": False},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("summary", data)
        self.assertIn("rows", data)
        self.assertIn("by_case", data)
        self.assertEqual(data["summary"]["cases_total"], len(backend_main.CASES_DATA))
        self.assertEqual(data["summary"]["turns_total"], len(backend_main.CASES_DATA) * 2)

    def test_ravr_multi_model_benchmark(self) -> None:
        with patch.object(backend_main, "evaluate_therapist_message", return_value=None):
            resp = self.client.post(
                "/api/ravr_multi_model_benchmark",
                json={
                    "providers": ["openrouter_gpt", "unknown_provider"],
                    "n_per_case": 1,
                    "random_seed": 1,
                    "include_llm_eval": False,
                },
            )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("results", data)
        self.assertIn("errors", data)
        self.assertIn("openrouter_gpt", data["results"])

    def test_ravr_benchmark_override_disables_repair(self) -> None:
        resp = self.client.post(
            "/api/ravr_benchmark",
            json={
                "n_per_case": 2,
                "random_seed": 42,
                "include_llm_eval": False,
                "override_enable_repair": False,
            },
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for row in data["rows"]:
            self.assertFalse(row["repair_triggered"])
            self.assertFalse(row["repair_success"])
        self.assertEqual(float(data["summary"]["metrics"]["repair_trigger_rate"]), 0.0)

    def test_ravr_benchmark_csv_export(self) -> None:
        resp = self.client.post(
            "/api/ravr_benchmark.csv",
            json={"n_per_case": 1, "random_seed": 1, "include_llm_eval": False},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers.get("content-type", ""))
        text = resp.text
        self.assertIn("case_id,category_key,methodology_id,prompt", text)
        self.assertIn("SUMMARY", text)

    def test_ravr_benchmark_jsonl_export(self) -> None:
        resp = self.client.post(
            "/api/ravr_benchmark.jsonl",
            json={"n_per_case": 1, "random_seed": 1, "include_llm_eval": False},
        )
        self.assertEqual(resp.status_code, 200)
        lines = [line for line in resp.text.splitlines() if line.strip()]
        self.assertGreater(len(lines), 1)
        last = json.loads(lines[-1])
        self.assertEqual(last.get("scope"), "summary")
        self.assertIn("summary", last)

    def test_chat_and_dataset_with_mocks(self) -> None:
        fake_eval = backend_main.TurnEvaluation(
            delta_trust=1,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.8,
            validation=0.7,
            directivity=0.2,
            open_question=0.8,
            safety=0.95,
            efficiency_index=0.4,
            comment="good turn",
        )
        with patch.object(backend_main, "evaluate_therapist_message", return_value=fake_eval), patch.object(
            backend_main, "call_llm_chat", return_value="Пациент отвечает нейтрально."
        ):
            chat_resp = self.client.post(
                "/api/chat",
                json={
                    "session_id": "article_suite_session",
                    "case_id": "gtr_01",
                    "user_message": "Что вы чувствуете в этой ситуации?",
                    "teacher_mode": True,
                },
            )
        self.assertEqual(chat_resp.status_code, 200)

        dataset_resp = self.client.get("/api/ravr_dataset.jsonl?session_id=article_suite_session")
        self.assertEqual(dataset_resp.status_code, 200)
        rows = [json.loads(x) for x in dataset_resp.text.splitlines() if x.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_id"], "article_suite_session")
        self.assertIn("proof_object", rows[0])

    def test_ravr_metrics_global_endpoint(self) -> None:
        resp = self.client.get("/api/ravr_metrics")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("metrics", data)
        self.assertIn("config", data)

    def test_core_functions_direct(self) -> None:
        ev = backend_main.TurnEvaluation(
            delta_trust=0,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.7,
            validation=0.7,
            directivity=0.3,
            open_question=0.8,
            safety=0.9,
            efficiency_index=0.2,
            comment="ok",
        )
        proof = backend_main.build_methodology_proof("gtr_01", "Что вы сейчас чувствуете?", ev)
        self.assertIn("methodology_id", proof.model_dump())
        self.assertTrue(len(proof.citations) >= 1)

        metrics = backend_main.build_ravr_metrics([proof.model_dump()])
        self.assertEqual(metrics.turns_total, 1)
        self.assertGreaterEqual(metrics.avg_adherence_score, 0.0)

        req = backend_main.RavrBenchmarkRequest(n_per_case=1, random_seed=1, include_llm_eval=False)
        bench = backend_main.run_ravr_benchmark(req)
        self.assertGreaterEqual(bench.summary.cases_total, 1)
        self.assertGreaterEqual(len(bench.rows), 1)

        session = {
            "case_id": "gtr_01",
            "history": [
                {"role": "user", "content": "Что вы чувствуете?"},
                {"role": "assistant", "content": "Мне тревожно."},
            ],
            "evals": [ev.model_dump()],
            "method_proofs": [proof.model_dump()],
        }
        rows = backend_main._session_turn_dataset_rows("direct_core_session", session)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["case_id"], "gtr_01")

    def test_methodology_proof_across_categories(self) -> None:
        ev = backend_main.TurnEvaluation(
            delta_trust=0,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.6,
            validation=0.6,
            directivity=0.2,
            open_question=0.7,
            safety=0.9,
            efficiency_index=0.2,
            comment="ok",
        )
        samples = [
            ("gtr_01", "Какая мысль появляется в этот момент?"),
            ("prl_01", "Что помогает вам стабилизироваться прямо сейчас?"),
            ("nrl_01", "Когда вспоминаете эпизод, что усиливает дистресс?"),
            ("ras_01", "Что было до реакции и что сразу после?"),
        ]
        for case_id, msg in samples:
            proof = backend_main.build_methodology_proof(case_id, msg, ev)
            self.assertIsNotNone(proof.methodology_id)
            self.assertGreaterEqual(proof.adherence_score, 0.0)
            self.assertTrue(isinstance(proof.citation_valid, bool))

    def test_chat_returns_502_on_llm_failure(self) -> None:
        fake_eval = backend_main.TurnEvaluation(
            delta_trust=0,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.5,
            validation=0.5,
            directivity=0.3,
            open_question=0.5,
            safety=0.9,
            efficiency_index=0.0,
            comment="neutral",
        )
        with patch.object(backend_main, "evaluate_therapist_message", return_value=fake_eval), patch.object(
            backend_main, "call_llm_chat", side_effect=RuntimeError("provider down")
        ):
            resp = self.client.post(
                "/api/chat",
                json={
                    "session_id": "error_session",
                    "case_id": "gtr_01",
                    "user_message": "Что происходит?",
                    "teacher_mode": True,
                },
            )
        self.assertEqual(resp.status_code, 502)

    def test_session_detail_and_progress(self) -> None:
        fake_eval = backend_main.TurnEvaluation(
            delta_trust=1,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.9,
            validation=0.8,
            directivity=0.2,
            open_question=0.9,
            safety=0.95,
            efficiency_index=0.5,
            comment="great",
        )
        with patch.object(backend_main, "evaluate_therapist_message", return_value=fake_eval), patch.object(
            backend_main, "call_llm_chat", return_value="Ответ пациента."
        ):
            self.client.post(
                "/api/chat",
                json={
                    "session_id": "progress_session",
                    "case_id": "gtr_01",
                    "user_message": "Что вы чувствуете?",
                    "teacher_mode": True,
                },
            )

        detail = self.client.get("/api/sessions/progress_session")
        self.assertEqual(detail.status_code, 200)
        self.assertIn("ravr_summary", detail.json())

        progress = self.client.get("/api/sessions/progress_session/progress")
        self.assertEqual(progress.status_code, 200)
        self.assertEqual(progress.json()["num_turns"], 1)

    def test_ravr_metrics_csv_and_dataset_global(self) -> None:
        resp_csv = self.client.get("/api/ravr_metrics.csv")
        self.assertEqual(resp_csv.status_code, 200)
        self.assertIn("text/csv", resp_csv.headers.get("content-type", ""))
        self.assertIn("verifier_pass_rate", resp_csv.text)

        resp_jsonl = self.client.get("/api/ravr_dataset.jsonl")
        self.assertEqual(resp_jsonl.status_code, 200)
        self.assertIn("application/jsonl", resp_jsonl.headers.get("content-type", ""))

    def test_session_report_success_and_fallback(self) -> None:
        session_id = "report_session"
        ev = backend_main.TurnEvaluation(
            delta_trust=1,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.8,
            validation=0.7,
            directivity=0.2,
            open_question=0.8,
            safety=0.95,
            efficiency_index=0.4,
            comment="solid",
        )
        proof = backend_main.build_methodology_proof("gtr_01", "Что вы чувствуете?", ev)
        backend_main.sessions[session_id] = {
            "case_id": "gtr_01",
            "history": [
                {"role": "user", "content": "Что вы чувствуете?"},
                {"role": "assistant", "content": "Я тревожусь."},
            ],
            "state": {"trust_level": 2, "emotional_intensity": 1, "fatigue": 0},
            "evals": [ev.model_dump()],
            "mistakes": [],
            "method_proofs": [proof.model_dump()],
        }

        with patch.object(
            backend_main,
            "generate_session_feedback_with_llm",
            return_value=("Хорошая динамика.", "Добавьте больше открытых вопросов.", []),
        ):
            ok_resp = self.client.get(f"/api/session_report?session_id={session_id}")
            self.assertEqual(ok_resp.status_code, 200)
            self.assertIn("overall_impression", ok_resp.json())

        with patch.object(backend_main, "generate_session_feedback_with_llm", side_effect=RuntimeError("llm failed")):
            fb_resp = self.client.get(f"/api/session_report?session_id={session_id}")
            self.assertEqual(fb_resp.status_code, 200)
            self.assertGreaterEqual(len(fb_resp.json().get("improved_examples", [])), 0)


if __name__ == "__main__":
    unittest.main()
