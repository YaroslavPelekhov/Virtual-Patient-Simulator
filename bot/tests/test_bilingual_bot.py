import os
import re
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

from bot import bot


class BilingualBotTests(unittest.TestCase):
    def setUp(self):
        bot.CASES_CACHE["ru"] = []
        bot.CASES_CACHE["en"] = []

    def test_language_uses_separate_backend_and_session(self):
        self.assertEqual(bot.backend_url("ru"), bot.BACKEND_URL)
        self.assertEqual(bot.backend_url("en"), bot.BACKEND_URL_EN)
        self.assertEqual(bot.get_session_id(42, "ru"), "tg_ru_42")
        self.assertEqual(bot.get_session_id(42, "en"), "tg_en_42")

    def test_english_interface_has_no_cyrillic(self):
        bilingual_keys = {"choose_language", "language_button"}
        english_interface = " ".join(
            value for key, value in bot.TEXTS["en"].items() if key not in bilingual_keys
        )
        self.assertIsNone(re.search(r"[А-Яа-яЁё]", english_interface))

    def test_english_menu_is_fully_localized(self):
        state = bot.ensure_user(9001)
        state["language"] = "en"
        inline_labels = [
            button.text
            for row in bot.main_menu_keyboard(state).inline_keyboard
            for button in row
        ]
        reply_labels = [
            button.text
            for row in bot.bottom_reply_keyboard("en").keyboard
            for button in row
        ]
        self.assertIn("👥 Patient by diagnosis", inline_labels)
        self.assertIn("✅ Finish", reply_labels)
        self.assertNotIn("🏠 Меню", reply_labels)

    @patch("bot.bot.requests.get")
    def test_case_catalogues_are_cached_by_language(self, mock_get):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = [{"id": "en_case"}]
        mock_get.return_value = response

        cases = bot.fetch_cases("en")

        self.assertEqual(cases, [{"id": "en_case"}])
        self.assertEqual(bot.CASES_CACHE["en"], cases)
        self.assertEqual(bot.CASES_CACHE["ru"], [])
        mock_get.assert_called_once_with(f"{bot.BACKEND_URL_EN}/api/cases", timeout=10)

    @patch("bot.bot.requests.post")
    def test_english_chat_is_sent_to_english_backend(self, mock_post):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"assistant_message": "Tell me more."}
        mock_post.return_value = response

        result = bot.call_backend_chat("tg_en_7", "case-1", "Hello", "en")

        self.assertEqual(result["assistant_message"], "Tell me more.")
        self.assertEqual(mock_post.call_args.args[0], f"{bot.BACKEND_URL_EN}/api/chat")
        self.assertEqual(mock_post.call_args.kwargs["json"]["session_id"], "tg_en_7")

    def test_english_reports_are_localized(self):
        report = {
            "case_id": "case-1",
            "num_turns": 2,
            "avg_empathy": 1.5,
            "avg_validation": 1.0,
            "avg_directivity": 0.5,
            "avg_open_question": 1.0,
            "avg_safety": 2.0,
            "mean_efficiency_index": 0.7,
            "total_delta_trust": 1,
            "total_delta_emotional_intensity": -1,
            "total_delta_fatigue": 0,
            "overall_impression": "A careful consultation.",
            "recommendations": "Continue with one focused question.",
        }
        rendered = bot.format_session_report(report, "en")
        self.assertIn("Session report", rendered)
        self.assertIn("Overall assessment", rendered)
        self.assertIsNone(re.search(r"[А-Яа-яЁё]", rendered))


if __name__ == "__main__":
    unittest.main()
