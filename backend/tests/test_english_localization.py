import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest


BACKEND_DIR = Path(__file__).resolve().parents[1]


class EnglishLocalizationTests(unittest.TestCase):
    def test_english_case_dataset_matches_russian_dataset(self) -> None:
        source = json.loads((BACKEND_DIR / "virtual_patient_cases.json").read_text(encoding="utf-8"))
        english = json.loads((BACKEND_DIR / "virtual_patient_cases.en.json").read_text(encoding="utf-8"))

        source_cases = source["cases"]
        english_cases = english["cases"]
        self.assertEqual([item["id"] for item in english_cases], [item["id"] for item in source_cases])
        self.assertEqual(len(english_cases), 20)
        self.assertIsNone(re.search(r"[А-Яа-яЁё]", json.dumps(english, ensure_ascii=False)))

    def test_english_backend_uses_english_prompts_and_proof_text(self) -> None:
        check = """
import json
import re
import main

case = main.CASES_BY_ID['gtr_01']
messages = main.build_messages(case, main.get_initial_state(), [{'role': 'user', 'content': 'What thought appears first?'}])
proof = main.build_methodology_proof('gtr_01', 'What evidence supports this automatic thought?', None)
payload = {
    'is_english': main.IS_ENGLISH,
    'case_count': len(main.CASES_DATA),
    'system_prompt': messages[0]['content'],
    'proof': proof.model_dump(),
}
print(json.dumps(payload, ensure_ascii=False))
"""
        env = dict(os.environ)
        env["VP_LANGUAGE"] = "en"
        env["PYTHONPATH"] = str(BACKEND_DIR)
        result = subprocess.run(
            [sys.executable, "-c", check],
            env=env,
            cwd=BACKEND_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertTrue(payload["is_english"])
        self.assertEqual(payload["case_count"], 20)
        self.assertIn("VIRTUAL PATIENT", payload["system_prompt"])
        self.assertNotRegex(payload["system_prompt"], r"[А-Яа-яЁё]")
        self.assertNotRegex(json.dumps(payload["proof"], ensure_ascii=False), r"[А-Яа-яЁё]")
        self.assertTrue(any("cognitive" in item for item in payload["proof"]["satisfied_constraints"]))


if __name__ == "__main__":
    unittest.main()
