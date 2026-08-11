# Configuration

The backend and Telegram bot load configuration from component-local `.env` files. Copy the corresponding `.env.example` file and set only the credentials required by the selected integrations.

```bash
cp backend/.env.example backend/.env
cp bot/.env.example bot/.env
```

Both `.env` files are ignored by Git.

For browser access, configure allowed origins as a comma-separated list:

```env
CORS_ALLOW_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
```

Use the exact HTTPS origin in production. The default `*` is retained only for backward-compatible local development.

## Backend Provider Selection

Set `LLM_PROVIDER` to one of the supported values:

| Value | Required configuration | Purpose |
|---|---|---|
| `gigachat` | `GIGACHAT_AUTHORIZATION_KEY` | GigaChat API |
| `openai` | `OPENAI_API_KEY` | OpenAI Chat Completions |
| `openai_compatible` | `OPENAI_COMPAT_BASE_URL` and optionally a key | vLLM or another compatible server |
| `openrouter` | `OPENROUTER_API_KEY` | Default OpenRouter model |
| `openrouter_gpt` | `OPENROUTER_API_KEY` | GPT route configured by `OPENROUTER_MODEL_GPT` |
| `openrouter_claude` | `OPENROUTER_API_KEY` | Claude route configured by `OPENROUTER_MODEL_CLAUDE` |
| `openrouter_gemini` | `OPENROUTER_API_KEY` | Gemini route configured by `OPENROUTER_MODEL_GEMINI` |
| `openrouter_deepseek` | `OPENROUTER_API_KEY` | DeepSeek route configured by `OPENROUTER_MODEL_DEEPSEEK` |
| `openrouter_qwen` | `OPENROUTER_API_KEY` | Qwen route configured by `OPENROUTER_MODEL_QWEN` |

### OpenRouter

```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_key
OPENROUTER_BASE_URL=https://openrouter.ai/api
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=virtual-patient-simulator
OPENROUTER_MODEL_DEFAULT=openai/gpt-4o-mini
```

Model identifiers can change. Verify current identifiers with the provider before a production or reproducibility run.

### OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key
OPENAI_CHAT_URL=https://api.openai.com/v1/chat/completions
OPENAI_MODEL=gpt-4o-mini
```

### OpenAI-Compatible Server

```env
LLM_PROVIDER=openai_compatible
OPENAI_COMPAT_BASE_URL=http://127.0.0.1:9000/v1
OPENAI_COMPAT_API_KEY=
OPENAI_COMPAT_MODEL=your-model-name
```

### GigaChat

```env
LLM_PROVIDER=gigachat
GIGACHAT_AUTHORIZATION_KEY=your_authorization_key
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_MODEL=GigaChat
GIGACHAT_VERIFY_SSL=1
```

Do not disable TLS verification in production. If a private certificate authority is required, configure `GIGACHAT_CA_BUNDLE` instead.

## RAVR Controls

| Variable | Default | Effect |
|---|---:|---|
| `RAVR_ENABLE_RETRIEVAL` | `1` | Retrieve methodology-specific constraint chunks |
| `RAVR_REQUIRE_VALID_CITATIONS` | `1` | Enforce citation validity in verification |
| `RAVR_ENABLE_REPAIR` | `1` | Generate and re-verify targeted repairs |

Boolean values accept `1`/`0`, `true`/`false`, or `yes`/`no`.

## RAVR-S Controls

RAVR-S is disabled by default and should be treated as an experimental extension.

| Variable | Default | Effect |
|---|---:|---|
| `RAVRS_ENABLE` | `0` | Enable state-sensitive candidate selection |
| `RAVRS_K` | `3` | Number of generated candidates |
| `RAVRS_MIN_SCORE` | `1.8` | Minimum state-aware candidate score |
| `RAVRS_CANDIDATE_TEMPERATURE` | `0.8` | Candidate-generation temperature |
| `RAVRS_EVAL_TEMPERATURE` | `0.2` | Candidate-evaluation temperature |
| `RAVRS_FORCE_PROTOCOL_PARITY` | `0` | Apply stricter protocol-parity behavior |
| `RAVRS_ALWAYS_REPAIR` | `0` | Repair even when the standard trigger is absent |

Record every value used in a research run. Comparisons are not reproducible if providers, model identifiers, prompts, or feature flags are changed without being logged.

## Telegram Bot

Required text-mode variables:

```env
TELEGRAM_BOT_TOKEN=your_bot_token
BACKEND_URL=http://127.0.0.1:8000
```

Optional SaluteSpeech variables:

```env
SALUTESPEECH_AUTH_KEY=your_key
SALUTESPEECH_SCOPE=SALUTE_SPEECH_PERS
SALUTESPEECH_VERIFY_SSL=1
SALUTESPEECH_STT_MODEL=general
SALUTESPEECH_STT_AUDIO_ENCODING=OGG_OPUS
SALUTESPEECH_STT_SAMPLE_RATE=48000
SALUTESPEECH_STT_CHANNELS=1
SALUTESPEECH_TTS_VOICE=Nec_24000
SALUTESPEECH_TTS_FORMAT=opus
```

## Secret Handling

- Never commit `.env` files.
- Use separate development and production credentials.
- Rotate any credential that has appeared in a shell transcript, issue, chat, or commit.
- Prefer a managed secret store for server deployments.
- Restrict provider keys by budget, project, or IP when supported.
- Do not include credentials in exported experiment metadata.
