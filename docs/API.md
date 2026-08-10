# API Reference

The backend exposes an interactive OpenAPI specification at `/docs` and a machine-readable schema at `/openapi.json`. The summary below covers the project-specific endpoints.

Default development base URL:

```text
http://127.0.0.1:8000
```

## Cases

### `GET /api/cases`

Returns all student-visible case summaries. Teacher-only fields are excluded.

### `GET /api/cases/{case_id}/teacher`

Returns the complete case profile, including the provisional diagnosis, training goals, symptom profile, personality style, typical phrases, and triggers.

This endpoint exposes restricted educational information and should require authentication in a public deployment.

## Dialogue

### `POST /api/chat`

Evaluates a therapist turn, updates the patient state, generates the virtual patient's reply, and stores the session.

Request:

```json
{
  "session_id": "demo-session",
  "case_id": "gtr_01",
  "user_message": "What feels most difficult for you right now?",
  "teacher_mode": true,
  "llm_provider": "openrouter"
}
```

Fields:

| Field | Required | Description |
|---|---:|---|
| `session_id` | yes | Client-generated stable identifier for the dialogue |
| `case_id` | yes | Identifier returned by `/api/cases` |
| `user_message` | yes | Student's therapist turn |
| `teacher_mode` | no | Include evaluation and proof data; defaults to `true` |
| `llm_provider` | no | Override the configured provider for this request |

Response fields include `assistant_message`, `evaluation`, `proof_object`, `verifier_pass`, and `verifier_violations`. Evaluation and proof data are omitted when `teacher_mode` is false.

## Sessions and Feedback

### `GET /api/sessions/{session_id}`

Returns the current case, patient state, dialogue history, turn evaluations, detected mistakes, and aggregate RAVR metrics.

### `GET /api/sessions/{session_id}/progress`

Returns turn-level trajectories and trends for empathy, validation, directivity, open questions, safety, efficiency, trust, emotional intensity, and fatigue.

### `GET /api/session_report?session_id={id}`

Produces an aggregate teaching report for a completed session. An optional `llm_provider` query parameter selects the provider used for narrative feedback.

## Verification

### `POST /api/verify_turn`

Runs methodology verification and targeted repair without generating a virtual-patient response.

```json
{
  "case_id": "panic_01",
  "user_message": "Let us identify the thought that appeared when the episode started.",
  "llm_provider": "openrouter"
}
```

The response contains a structured proof object with satisfied constraints, violations, evidence, recommendations, citations, adherence score, and repair information.

## Metrics and Exports

### `GET /api/ravr_metrics`

Returns aggregate metrics over all stored sessions.

Use `?session_id={id}` to restrict the result to one session.

### `GET /api/ravr_metrics.csv`

Exports global and per-session RAVR metrics as CSV. The optional `session_id` query parameter restricts the export.

### `GET /api/ravr_dataset.jsonl`

Exports turn-level session records as JSONL. Records may contain dialogue text and must be handled as potentially sensitive data.

## Benchmarks

### `POST /api/ravr_benchmark`

Runs a controlled benchmark and returns JSON.

```json
{
  "n_per_case": 6,
  "random_seed": 42,
  "include_llm_eval": false,
  "llm_temperature": 0.2,
  "disable_eval_cache": false,
  "case_ids": ["gtr_01", "panic_01"]
}
```

Optional override fields:

- `override_enable_retrieval`
- `override_require_valid_citations`
- `override_enable_repair`
- `override_llm_provider`

### `POST /api/ravr_benchmark.csv`

Runs the same benchmark and returns CSV.

### `POST /api/ravr_benchmark.jsonl`

Runs the same benchmark and returns turn-level JSONL plus a summary record.

### `POST /api/ravr_multi_model_benchmark`

Runs one benchmark configuration across multiple providers.

```json
{
  "providers": ["openrouter_gpt", "openrouter_deepseek", "openrouter_qwen"],
  "n_per_case": 4,
  "random_seed": 42,
  "include_llm_eval": true
}
```

## Errors

- `400`: invalid case, missing evaluation data, or invalid request parameters.
- `404`: unknown case or session.
- `422`: request does not match the Pydantic schema.
- `502`: upstream LLM or speech-provider failure.

Do not expose raw upstream errors to untrusted clients in production without sanitization.
