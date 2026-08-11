import unittest
from unittest.mock import patch

import main as backend_main


class MainUtilsTests(unittest.TestCase):
    def setUp(self) -> None:
        backend_main.sessions.clear()

    def test_normalize_case_id(self) -> None:
        self.assertEqual(backend_main.normalize_case_id("gtr_0002"), "gtr_02")
        self.assertEqual(backend_main.normalize_case_id("bad_format"), "bad_format")

    def test_extract_json_object(self) -> None:
        data = backend_main._extract_json_object('{"a": 1}')
        self.assertEqual(data["a"], 1)
        data2 = backend_main._extract_json_object("prefix {\"b\":2} suffix")
        self.assertEqual(data2["b"], 2)

    def test_truncate_text_and_trend(self) -> None:
        t = backend_main._truncate_text("x" * 400, max_len=20)
        self.assertTrue(len(t) <= 20)
        self.assertEqual(backend_main._calc_trend([1.0]), 0.0)
        self.assertNotEqual(backend_main._calc_trend([1.0, 2.0, 3.0]), 0.0)

    def test_methodology_mapping_and_chunks(self) -> None:
        m = backend_main.get_methodology_for_case("gtr_01")
        self.assertEqual(m["id"], "cbt")
        chunks = backend_main.build_ravr_chunks("gtr_01", None)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(c.chunk_id for c in chunks))

    def test_citations_helpers(self) -> None:
        chunks = backend_main.build_ravr_chunks("gtr_01", None)
        method_id = backend_main.get_methodology_for_case("gtr_01")["id"]
        cits = backend_main._pick_citations(
            method_id,
            violations=["избыточная директивность", "недостаток открытых вопросов"],
            satisfied=[],
            chunks=chunks,
        )
        self.assertTrue(backend_main._citations_valid(cits, chunks))
        self.assertFalse(backend_main._citations_valid(["bad_id"], chunks))

    def test_repair_message_builder(self) -> None:
        r = backend_main._build_repaired_message("cbt", "text", ["v1"])
        self.assertTrue(r.should_repair)
        self.assertIn("мысл", r.repaired_message.lower())
        r2 = backend_main._build_repaired_message("cbt", "text", [])
        self.assertFalse(r2.should_repair)

    def test_detect_mistake_reason_and_collect(self) -> None:
        ev = backend_main.TurnEvaluation(
            delta_trust=0,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.1,
            validation=0.2,
            directivity=0.9,
            open_question=0.1,
            safety=0.5,
            efficiency_index=-0.3,
            comment="bad",
        )
        reason = backend_main.detect_mistake_reason(ev)
        self.assertIsNotNone(reason)
        session = {
            "history": [{"role": "user", "content": "test user message"}],
            "evals": [ev.model_dump()],
            "mistakes": [],
        }
        mistakes = backend_main.collect_mistaken_replicas(session)
        self.assertEqual(len(mistakes), 1)
        examples = backend_main.fallback_improved_examples(mistakes)
        self.assertGreaterEqual(len(examples), 1)

    def test_build_session_progress(self) -> None:
        ev = backend_main.TurnEvaluation(
            delta_trust=1,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.9,
            validation=0.8,
            directivity=0.2,
            open_question=0.7,
            safety=0.95,
            efficiency_index=0.4,
            comment="ok",
        )
        session = {
            "case_id": "gtr_01",
            "state": backend_main.get_initial_state(),
            "evals": [ev.model_dump()],
            "history": [],
        }
        progress = backend_main.build_session_progress("s1", session)
        self.assertEqual(progress.num_turns, 1)
        self.assertEqual(len(progress.points), 1)

    def test_ravr_metrics_rows_and_benchmark_row_dict(self) -> None:
        ev = backend_main.TurnEvaluation(
            delta_trust=0,
            delta_emotional_intensity=0,
            delta_fatigue=0,
            empathy=0.7,
            validation=0.6,
            directivity=0.2,
            open_question=0.8,
            safety=0.9,
            efficiency_index=0.2,
            comment="ok",
        )
        proof = backend_main.build_methodology_proof("gtr_01", "Что вы чувствуете?", ev)
        m = backend_main.build_ravr_metrics([proof.model_dump()])
        row = backend_main._ravr_metrics_row("session", "s1", "gtr_01", m)
        self.assertIn("verifier_pass_rate", row)

        b_row = backend_main.RavrBenchmarkRow(
            case_id="gtr_01",
            category_key="gtr",
            methodology_id="cbt",
            prompt="prompt",
            verifier_pass=True,
            citation_valid=True,
            adherence_score=80.0,
            violations=[],
            repair_triggered=False,
            repair_success=False,
        )
        d = backend_main._benchmark_row_to_dict(b_row)
        self.assertEqual(d["case_id"], "gtr_01")

    def test_run_ravr_benchmark_invalid_case_selection(self) -> None:
        req = backend_main.RavrBenchmarkRequest(n_per_case=1, case_ids=["unknown_99"])
        with self.assertRaises(backend_main.HTTPException):
            backend_main.run_ravr_benchmark(req)

    def test_build_messages_and_apply_state_delta(self) -> None:
        case_profile = backend_main.CASES_BY_ID["gtr_01"]
        state = backend_main.get_initial_state()
        history = [{"role": "user", "content": "test"}]
        messages = backend_main.build_messages(case_profile, state, history)
        self.assertEqual(messages[0]["role"], "system")

        ev = backend_main.TurnEvaluation(
            delta_trust=1,
            delta_emotional_intensity=1,
            delta_fatigue=1,
            empathy=0.5,
            validation=0.5,
            directivity=0.5,
            open_question=0.5,
            safety=0.9,
            efficiency_index=0.1,
            comment="ok",
        )
        new_state = backend_main.apply_state_delta(state, ev)
        self.assertEqual(new_state["trust_level"], 2)

    def test_llm_provider_dispatch(self) -> None:
        msgs = [{"role": "user", "content": "test"}]
        with patch.object(backend_main, "gigachat_chat_completions", return_value="giga"):
            out = backend_main.llm_chat_completions(
                provider="gigachat",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "giga")
        with patch.object(backend_main, "openai_chat_completions", return_value="openai"):
            out = backend_main.llm_chat_completions(
                provider="openai",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "openai")
        with patch.object(backend_main, "openai_compatible_chat_completions", return_value="compat"):
            out = backend_main.llm_chat_completions(
                provider="openai_compatible",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "compat")
        with patch.object(backend_main, "openrouter_chat_completions", return_value="router"):
            out = backend_main.llm_chat_completions(
                provider="openrouter_gpt",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "router")
        with patch.object(backend_main, "openrouter_chat_completions", return_value="router"):
            out = backend_main.llm_chat_completions(
                provider="openrouter_deepseek",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "router")
        with patch.object(backend_main, "openrouter_chat_completions", return_value="router"):
            out = backend_main.llm_chat_completions(
                provider="openrouter_qwen",
                messages=msgs,
                temperature=0.1,
                max_tokens=8,
            )
            self.assertEqual(out, "router")
        with self.assertRaises(RuntimeError):
            backend_main.llm_chat_completions(provider="unknown_provider", messages=msgs, temperature=0.1, max_tokens=8)

    def test_patient_generation_uses_provider_default_model(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        with patch.object(backend_main, "llm_chat_completions", return_value="ok") as mocked:
            result = backend_main.call_llm_chat(messages, provider="openai")

        self.assertEqual(result, "ok")
        self.assertIsNone(mocked.call_args.kwargs["model"])


if __name__ == "__main__":
    unittest.main()
