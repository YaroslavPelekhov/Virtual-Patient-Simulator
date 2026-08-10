# Contributing

Thank you for improving the Virtual Patient Simulator. This project combines software engineering, educational design, and safety-sensitive psychological content. Changes should be reviewable, reproducible, and explicit about their intended effect.

## Development Setup

```bash
git clone https://github.com/YaroslavPelekhov/Virtual-Patient-Simulator.git
cd Virtual-Patient-Simulator/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Use a test or mock provider configuration whenever possible. Never commit credentials or real dialogue records.

## Pull Requests

Keep each pull request focused. Include:

1. a concise description of the problem;
2. the behavior before and after the change;
3. tests or a reason tests are not applicable;
4. configuration or migration notes;
5. a safety rationale for changes to prompts, clinical cases, scoring, or repair rules.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
```

Provider integrations should be mocked in unit tests. A test must not incur paid API calls by default.

## Case and Methodology Changes

Changes to `backend/virtual_patient_cases.json` or methodology constraints require additional care:

- preserve the separation between student-visible and teacher-only information;
- use synthetic content and remove identifying details;
- avoid actionable self-harm, violence, or medication instructions;
- document the intended educational objective;
- add or update tests for the affected category;
- obtain domain review before treating a case as validated training material.

## Style

- Keep public documentation in English.
- Keep user-facing Russian text consistent with the existing interface unless localization is the purpose of the change.
- Prefer small functions, explicit types, and deterministic tests.
- Do not add generated experiment outputs or local databases to Git.

## Security

Report suspected credential exposure or a vulnerability privately to the repository owner. Do not open a public issue containing API keys, tokens, patient information, or exploitable deployment details.
