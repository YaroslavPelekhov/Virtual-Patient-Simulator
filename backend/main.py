import json
import csv
import io
import os
import re
import random
import time
import uuid
import threading
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Always load backend-local .env first, independent of current working directory.
_EARLY_BASE_DIR = Path(__file__).resolve().parent
load_dotenv(_EARLY_BASE_DIR / ".env")
# Optional fallback for legacy launches from cwd.
load_dotenv()

# ============================================================
#                 GigaChat settings / client
# ============================================================

# OAuth endpoint (access token)
GIGACHAT_OAUTH_URL = os.getenv("GIGACHAT_OAUTH_URL", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth")
# Chat completions endpoint
GIGACHAT_CHAT_URL = os.getenv("GIGACHAT_CHAT_URL", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions")

# Authorization key for Basic auth (what docs call "Authorization key")
# Put EXACTLY what you use in header: "Basic <key>" OR store only "<key>" and we will add Basic automatically.
GIGACHAT_AUTHORIZATION_KEY = os.getenv("GIGACHAT_AUTHORIZATION_KEY")

# OAuth scope
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

# Model name (can be overridden)
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

# Multi-provider LLM settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gigachat").strip().lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_CHAT_URL = os.getenv("OPENAI_CHAT_URL", "https://api.openai.com/v1/chat/completions").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
OPENAI_COMPAT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip()
OPENAI_COMPAT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
OPENAI_COMPAT_MODEL = os.getenv("OPENAI_COMPAT_MODEL", "llama-3.1-8b-instruct").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api").strip()
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "http://localhost").strip()
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "virtual-patient-simulator").strip()
OPENROUTER_MODEL_DEFAULT = os.getenv("OPENROUTER_MODEL_DEFAULT", "openai/gpt-4o-mini").strip()
OPENROUTER_MODEL_GPT = os.getenv("OPENROUTER_MODEL_GPT", "openai/gpt-4o-mini").strip()
OPENROUTER_MODEL_DEEPSEEK = os.getenv("OPENROUTER_MODEL_DEEPSEEK", "deepseek/deepseek-chat").strip()
OPENROUTER_MODEL_QWEN = os.getenv("OPENROUTER_MODEL_QWEN", "qwen/qwen-2.5-72b-instruct").strip()
OPENROUTER_MODEL_CLAUDE = os.getenv("OPENROUTER_MODEL_CLAUDE", "anthropic/claude-3.5-sonnet").strip()
OPENROUTER_MODEL_GEMINI = os.getenv("OPENROUTER_MODEL_GEMINI", "google/gemini-1.5-pro").strip()

# TLS verification:
# IMPORTANT: лучше поставить нормальный CA/сертификат.
# Но если у тебя падает на self-signed цепочке — можно временно отключить проверку:
# export GIGACHAT_VERIFY_SSL=0
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "1") not in ("0", "false", "False", "no", "NO")

# Optional CA bundle path if you have corporate CA
# export GIGACHAT_CA_BUNDLE=/path/to/ca.pem
GIGACHAT_CA_BUNDLE = os.getenv("GIGACHAT_CA_BUNDLE")  # if set, requests will verify against this bundle

# RqUID (can be generated each request)
GIGACHAT_RQUID = os.getenv("GIGACHAT_RQUID")  # optional fixed; if None -> generated each time

# RAVR ablation flags for reproducible experiments
RAVR_ENABLE_RETRIEVAL = os.getenv("RAVR_ENABLE_RETRIEVAL", "1") not in ("0", "false", "False", "no", "NO")
RAVR_REQUIRE_VALID_CITATIONS = os.getenv("RAVR_REQUIRE_VALID_CITATIONS", "1") not in ("0", "false", "False", "no", "NO")
RAVR_ENABLE_REPAIR = os.getenv("RAVR_ENABLE_REPAIR", "1") not in ("0", "false", "False", "no", "NO")
RAVRS_ENABLE = os.getenv("RAVRS_ENABLE", "0") not in ("0", "false", "False", "no", "NO")
RAVRS_K = max(1, int(os.getenv("RAVRS_K", "3")))
RAVRS_MIN_SCORE = float(os.getenv("RAVRS_MIN_SCORE", "1.8"))
RAVRS_CANDIDATE_TEMPERATURE = float(os.getenv("RAVRS_CANDIDATE_TEMPERATURE", "0.8"))
RAVRS_EVAL_TEMPERATURE = float(os.getenv("RAVRS_EVAL_TEMPERATURE", "0.2"))
RAVRS_FORCE_PROTOCOL_PARITY = os.getenv("RAVRS_FORCE_PROTOCOL_PARITY", "0") not in ("0", "false", "False", "no", "NO")
RAVRS_ALWAYS_REPAIR = os.getenv("RAVRS_ALWAYS_REPAIR", "0") not in ("0", "false", "False", "no", "NO")
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]
VP_LANGUAGE = os.getenv("VP_LANGUAGE", "ru").strip().lower()
IS_ENGLISH = VP_LANGUAGE.startswith("en")


def _l(ru: str, en: str) -> str:
    """Return a localized runtime string for the selected backend instance."""
    return en if IS_ENGLISH else ru

# Token cache (in-memory)
_token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}


def _get_verify_param():
    """
    requests verify param can be:
    - True/False
    - path to CA bundle
    """
    if GIGACHAT_CA_BUNDLE:
        return GIGACHAT_CA_BUNDLE
    return GIGACHAT_VERIFY_SSL


def _normalize_basic_key(raw: str) -> str:
    raw = raw.strip()
    if raw.lower().startswith("basic "):
        return raw
    return "Basic " + raw


def get_access_token(force_refresh: bool = False) -> str:
    """
    Fetch OAuth access token and cache it until expiry.
    """
    if not GIGACHAT_AUTHORIZATION_KEY:
        raise RuntimeError("GIGACHAT_AUTHORIZATION_KEY is not set (.env)")

    now = time.time()
    if (not force_refresh) and _token_cache["access_token"] and now < float(_token_cache["expires_at"]):
        return _token_cache["access_token"]

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": GIGACHAT_RQUID or str(uuid.uuid4()),
        "Authorization": _normalize_basic_key(GIGACHAT_AUTHORIZATION_KEY),
    }
    payload = {"scope": GIGACHAT_SCOPE}

    r = requests.post(
        GIGACHAT_OAUTH_URL,
        headers=headers,
        data=payload,
        timeout=30,
        verify=_get_verify_param(),
    )

    if r.status_code >= 400:
        raise RuntimeError(f"GigaChat OAuth error {r.status_code}: {r.text}")

    data = r.json()
    access_token = data.get("access_token")
    expires_in = data.get("expires_in", 0)  # seconds

    if not access_token:
        raise RuntimeError(f"GigaChat OAuth: no access_token in response: {data}")

    # Add small safety margin (30s) so it doesn't expire mid-request
    expires_at = time.time() + max(0, int(expires_in) - 30)

    _token_cache["access_token"] = access_token
    _token_cache["expires_at"] = expires_at

    return access_token


def gigachat_chat_completions(
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    max_tokens: int = 300,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    """
    Call GigaChat chat completions API.

    Expected GigaChat format is similar to OpenAI:
    { "model": "...", "messages": [{"role":"system|user|assistant","content":"..."}], ... }
    """
    access_token = get_access_token()

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    body = {
        "model": model or GIGACHAT_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)

    r = requests.post(
        GIGACHAT_CHAT_URL,
        headers=headers,
        json=body,
        timeout=60,
        verify=_get_verify_param(),
    )

    # If token expired/invalid, try once with refresh
    if r.status_code in (401, 403):
        access_token = get_access_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {access_token}"
        r = requests.post(
            GIGACHAT_CHAT_URL,
            headers=headers,
            json=body,
            timeout=60,
            verify=_get_verify_param(),
        )

    if r.status_code >= 400:
        raise RuntimeError(f"GigaChat chat error {r.status_code}: {r.text}")

    data = r.json()

    # Typical structure: {"choices":[{"message":{"role":"assistant","content":"..."}}], ...}
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected GigaChat response format: {data}")


def _extract_chat_content(data: Dict[str, Any], provider: str) -> str:
    try:
        return str(data["choices"][0]["message"]["content"])
    except Exception:
        raise RuntimeError(f"Unexpected {provider} response format: {data}")


def openai_chat_completions(
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    max_tokens: int = 300,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (.env)")

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model or OPENAI_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)
    r = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI chat error {r.status_code}: {r.text}")
    return _extract_chat_content(r.json(), "OpenAI")


def openai_compatible_chat_completions(
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    max_tokens: int = 300,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    if not OPENAI_COMPAT_BASE_URL:
        raise RuntimeError("OPENAI_COMPAT_BASE_URL is not set (.env)")
    base = OPENAI_COMPAT_BASE_URL.rstrip("/")
    url = f"{base}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if OPENAI_COMPAT_API_KEY:
        headers["Authorization"] = f"Bearer {OPENAI_COMPAT_API_KEY}"
    body = {
        "model": model or OPENAI_COMPAT_MODEL,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI-compatible chat error {r.status_code}: {r.text}")
    return _extract_chat_content(r.json(), "OpenAI-compatible")


def openrouter_chat_completions(
    messages: List[Dict[str, str]],
    temperature: float = 0.8,
    max_tokens: int = 300,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set (.env)")

    base = OPENROUTER_BASE_URL.rstrip("/")
    url = f"{base}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_SITE_URL,
        "X-Title": OPENROUTER_APP_NAME,
    }
    body = {
        "model": model or OPENROUTER_MODEL_DEFAULT,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    if top_p is not None:
        body["top_p"] = float(top_p)
    r = requests.post(url, headers=headers, json=body, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter chat error {r.status_code}: {r.text}")
    return _extract_chat_content(r.json(), "OpenRouter")


def llm_chat_completions(
    *,
    provider: Optional[str],
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: Optional[float] = None,
    model: Optional[str] = None,
) -> str:
    p = (provider or LLM_PROVIDER or "gigachat").strip().lower()
    if p == "gigachat":
        return gigachat_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model,
        )
    if p == "openai":
        return openai_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model,
        )
    if p in ("openai_compatible", "llama", "vllm"):
        return openai_compatible_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENAI_COMPAT_MODEL,
        )
    if p == "openrouter":
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_DEFAULT,
        )
    if p == "openrouter_gpt":
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_GPT,
        )
    if p == "openrouter_deepseek":
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_DEEPSEEK,
        )
    if p == "openrouter_claude":
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_CLAUDE,
        )
    if p == "openrouter_gemini":
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_GEMINI,
        )
    if p in ("openrouter_qwen", "openrouter_other"):
        return openrouter_chat_completions(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            model=model or OPENROUTER_MODEL_QWEN,
        )
    raise RuntimeError(f"Unsupported LLM provider: {p}")


# ============================================================
#                      Cases loading
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DB_PATH = BASE_DIR / ("sessions.en.db" if IS_ENGLISH else "sessions.db")
LEGACY_SESSIONS_STORE_PATH = BASE_DIR / ("sessions_store.en.json" if IS_ENGLISH else "sessions_store.json")
SESSIONS_LOCK = threading.Lock()
cases_filename = "virtual_patient_cases.en.json" if IS_ENGLISH else "virtual_patient_cases.json"
with open(BASE_DIR / cases_filename, "r", encoding="utf-8") as f:
    CASES_RAW = json.load(f)["cases"]

CATEGORY_LABELS = (
    {
        "ras": "Autism spectrum disorder",
        "sdvg": "ADHD",
        "okr": "Obsessive-compulsive disorder",
        "ptsd": "Post-traumatic stress disorder",
        "prl": "Borderline personality disorder",
        "nrl": "Narcissistic personality disorder",
        "gtr": "Generalized anxiety disorder",
        "panic": "Panic attacks",
        "shizo": "Schizophrenia",
        "aggr": "Aggressive outbursts",
    }
    if IS_ENGLISH
    else {
        "ras": "РАС",
        "sdvg": "СДВГ",
        "okr": "ОКР",
        "ptsd": "ПТСР",
        "prl": "ПРЛ",
        "nrl": "НРЛ",
        "gtr": "ГТР",
        "panic": "Панические атаки",
        "shizo": "Шизофрения",
        "aggr": "Вспышки агрессии",
    }
)

# 10 направлений методологий для устойчивой оценки приверженности
METHODOLOGY_BY_CATEGORY: Dict[str, Dict[str, str]] = {
    "prl": {"id": "dbt", "name": "DBT"},
    "nrl": {"id": "emdr", "name": "EMDR"},
    "panic": {"id": "cbt", "name": "CBT"},
    "okr": {"id": "cbt", "name": "CBT"},
    "aggr": {"id": "cbt", "name": "CBT"},
    "shizo": {"id": "pharm_psych", "name": _l("Фармакология + психология", "Pharmacology + psychology")},
    "sdvg": {"id": "cbt_pharm", "name": _l("CBT + фармакология", "CBT + pharmacology")},
    "ptsd": {"id": "dbt", "name": "DBT"},
    "gtr": {"id": "cbt", "name": "CBT"},
    "ras": {"id": "aba", "name": "ABA"},
}
METHOD_NAME_BY_ID: Dict[str, str] = {
    str(v["id"]): str(v["name"])
    for v in METHODOLOGY_BY_CATEGORY.values()
}

METHOD_RULES_RU: Dict[str, Dict[str, List[str]]] = {
    "dbt": {
        "must_do": ["валидация эмоций", "заземление/регуляция", "диалектический баланс принятия и изменений"],
        "must_avoid": ["жесткие директивы без валидации", "обесценивание переживаний"],
        "recommended": ["отражение чувств", "пошаговый план кризис-менеджмента"],
    },
    "emdr": {
        "must_do": ["фокус на целевом воспоминании", "проверка безопасности/ресурсирования", "структурированная фаза работы"],
        "must_avoid": ["хаотичное углубление травмы без стабилизации", "интерпретации без опоры на опыт клиента"],
        "recommended": ["короткие уточняющие вопросы о триггерах", "контроль уровня дистресса"],
    },
    "cbt": {
        "must_do": ["выявление автоматических мыслей", "проверка когнитивных искажений", "поведенческий план/эксперимент"],
        "must_avoid": ["чисто поддерживающая беседа без структуры", "категоричные советы"],
        "recommended": ["сократический вопрос", "домашняя практика"],
    },
    "cbt_pharm": {
        "must_do": ["CBT-структура беседы", "психообразование по режиму лечения", "мониторинг функционирования"],
        "must_avoid": ["назначения без специалиста", "игнорирование побочных эффектов/приверженности"],
        "recommended": ["поведенческие техники саморегуляции", "трек симптомов и сна"],
    },
    "pharm_psych": {
        "must_do": ["психообразование и реальность-тестирование", "оценка приверженности терапии", "поддержка функционального восстановления"],
        "must_avoid": ["конфронтация с бредовым контентом в лоб", "стигматизирующие формулировки"],
        "recommended": ["мягкое структурирование дня", "план ранних признаков ухудшения"],
    },
    "aba": {
        "must_do": ["поведенческий анализ A-B-C", "операционализация целевого поведения", "позитивное подкрепление"],
        "must_avoid": ["размытые цели без наблюдаемых критериев", "наказательные стратегии как базовые"],
        "recommended": ["формирование шага навыка", "измеримые поведенческие метрики"],
    },
}

METHOD_RULES_EN: Dict[str, Dict[str, List[str]]] = {
    "dbt": {
        "must_do": ["validate emotions", "use grounding or regulation", "balance acceptance and change dialectically"],
        "must_avoid": ["rigid directives without validation", "dismissal of the patient's experience"],
        "recommended": ["reflect feelings", "use a stepwise crisis-management plan"],
    },
    "emdr": {
        "must_do": ["focus on the target memory", "check safety and resourcing", "use a structured treatment phase"],
        "must_avoid": ["unstructured trauma exploration without stabilization", "interpretation unsupported by the patient's experience"],
        "recommended": ["ask brief clarifying questions about triggers", "monitor distress level"],
    },
    "cbt": {
        "must_do": ["identify automatic thoughts", "examine cognitive distortions", "develop a behavioral plan or experiment"],
        "must_avoid": ["unstructured supportive conversation only", "categorical advice"],
        "recommended": ["ask a Socratic question", "suggest between-session practice"],
    },
    "cbt_pharm": {
        "must_do": ["use a CBT session structure", "provide psychoeducation about the treatment regimen", "monitor functioning"],
        "must_avoid": ["prescribing without a qualified clinician", "ignoring side effects or adherence"],
        "recommended": ["use behavioral self-regulation techniques", "track symptoms and sleep"],
    },
    "pharm_psych": {
        "must_do": ["provide psychoeducation and reality testing", "assess treatment adherence", "support functional recovery"],
        "must_avoid": ["direct confrontation of delusional content", "stigmatizing language"],
        "recommended": ["gently structure the day", "make a plan for early warning signs"],
    },
    "aba": {
        "must_do": ["use antecedent-behavior-consequence analysis", "operationalize the target behavior", "use positive reinforcement"],
        "must_avoid": ["vague goals without observable criteria", "punitive strategies as the default"],
        "recommended": ["shape the next skill step", "use measurable behavioral metrics"],
    },
}

METHOD_RULES = METHOD_RULES_EN if IS_ENGLISH else METHOD_RULES_RU

BENCHMARK_UTTERANCES_RU: List[str] = [
    "Похоже, вам сейчас тяжело. Что вы чувствуете в этот момент?",
    "Давайте разберем, какие мысли появляются прямо перед этим.",
    "Вы должны просто перестать так делать и взять себя в руки.",
    "Что происходило до этого эпизода и что было сразу после?",
    "Мне важно, чтобы вы чувствовали себя в безопасности. Что вам сейчас поможет стабилизироваться?",
    "Это не так уж важно, у многих так бывает.",
    "Какие доказательства есть за и против этой мысли?",
    "Давайте пошагово определим один маленький следующий шаг.",
]

BENCHMARK_UTTERANCES_EN: List[str] = [
    "It sounds like things are difficult right now. What are you feeling in this moment?",
    "Let's examine which thoughts appear immediately before this happens.",
    "You should simply stop doing that and pull yourself together.",
    "What happened before this episode, and what happened immediately afterward?",
    "I want you to feel safe. What might help you feel more grounded right now?",
    "It is not that important; many people experience this.",
    "What evidence supports this thought, and what evidence goes against it?",
    "Let's identify one small next step together.",
]

BENCHMARK_UTTERANCES = BENCHMARK_UTTERANCES_EN if IS_ENGLISH else BENCHMARK_UTTERANCES_RU


def infer_category(case_id: str) -> str:
    return case_id.split("_", 1)[0]


def normalize_case_id(case_id: str) -> str:
    """
    Allow ras_001 -> ras_01, gtr_0002 -> gtr_02, etc.
    If doesn't match pattern, return as is.
    """
    case_id = (case_id or "").strip()
    m = re.match(r"^([a-z]+)_(\d+)$", case_id)
    if not m:
        return case_id
    prefix, num = m.group(1), m.group(2)
    # keep 2 digits like ras_01
    return f"{prefix}_{int(num):02d}"


CASES_DATA: List[Dict[str, Any]] = []
for c in CASES_RAW:
    c = dict(c)
    cat_key = c.get("category_key") or infer_category(c["id"])
    c["category_key"] = cat_key
    c["category_name"] = CATEGORY_LABELS.get(cat_key, cat_key)
    CASES_DATA.append(c)

CASES_BY_ID: Dict[str, Dict[str, Any]] = {c["id"]: c for c in CASES_DATA}

# ============================================================
#                   System prompts
# ============================================================

BASE_SYSTEM_PROMPT = """
Ты — ВИРТУАЛЬНЫЙ ПАЦИЕНТ на учебной консультации для студента-психолога.

ОБЩИЕ ПРАВИЛА:
- Отвечай так, как отвечал бы живой человек на первой консультации.
- НЕ говори, что ты модель, ИИ, симуляция или программа.
- НЕ используй профессиональную психиатрическую лексику, пока сам студент её не вводит.
- НЕ учи студента, как правильно помогать. Ты говоришь только из позиции клиента.
- НЕ поощряй самолечение, не давай советов по препаратам, дозировкам, диагнозам, если ты их не принимал.
- Если студент спрашивает прямо о диагнозе, говори в формате: "я читала/слышал про ..., но не уверен(а), что это про меня" или "об этом больше знает врач/специалист, который мне до этого поставил диагноз".

ПРОФИЛЬ ПАЦИЕНТА:
Тебе будет дан профиль случая в виде описания симптомов и стиля личности. Ты ДОЛЖЕН следовать этому профилю: симптомам, типичным фразам, межличностному стилю, триггерам.

ЭМОЦИОНАЛЬНОЕ СОСТОЯНИЕ:
Будет передано текущее состояние (доверие, эмоциональная интенсивность, усталость: 0–3).
- При БОЛЕЕ ВЫСОКОМ доверии — становись более открытым, говори чуть подробнее, делись чувствами.
- При НИЗКОМ доверии — будь более сдержанным, отвечай короче.
- При ВЫСОКОЙ усталости — подчёркивай, что тяжело думать и подбирать слова.
- При ВЫСОКОЙ эмоциональной интенсивности — ответы эмоциональнее, но БЕЗ детализированного описания самоповреждений, насилия и способов суицида.

СТИЛЬ ОТВЕТОВ:
- Простой разговорный язык.
- 2–3 предложений. И не сильно разговаривай без причины.
- Если вопрос неясен — можно попросить уточнить.

БЕЗОПАСНОСТЬ:
- Если есть суицидальные темы — только общими словами, без методов.
- НИКОГДА не одобряй опасное поведение.

Ты всегда отвечаешь как пациент из заданного профиля и НЕ выходишь из роли.
""".strip()

SUPERVISOR_PROMPT = """
Ты — клинический супервизор, который оценивает отдельные реплики психолога в формате учебной консультации.

Твоя задача — по ОДНОМУ сообщению психолога и краткому описанию состояния пациента оценить ход по нескольким шкалам и выдать СТРОГО JSON со следующими полями:

{
  "delta_trust": -1 | 0 | 1,
  "delta_emotional_intensity": -1 | 0 | 1,
  "delta_fatigue": -1 | 0 | 1,
  "empathy": float,          # 0.0–1.0
  "validation": float,       # 0.0–1.0
  "directivity": float,      # 0.0–1.0
  "open_question": float,    # 0.0–1.0
  "safety": float,           # 0.0–1.0
  "efficiency_index": float, # -1.0–1.0
  "comment": str             # 1–3 предложения по-русски, краткий разбор хода
}

Определения шкал:

- "empathy" — насколько хорошо психолог отражает чувства и показывает понимание (0 = нет эмпатии, 1 = очень эмпатично).
- "validation" — есть ли нормализация и принятие переживаний пациента.
- "directivity" — сколько советов, указаний, директивных формулировок.
- "open_question" — насколько ход построен на открытых вопросах (что/как/в какие моменты/каким образом).
- "safety" — насколько высказывание безопасно для пациента (нет обесценивания, давления, опасных рекомендаций).
- "efficiency_index" — общий интегральный индекс полезности хода (учитывает эмпатию, валидацию, открытые вопросы и отсутствие директивного давления).

Правила:
- Оценивай ТОЛЬКО по тексту сообщения психолога.
- Возвращай ТОЛЬКО JSON, без пояснений и лишнего текста.
""".strip()

SESSION_REPORT_PROMPT = """
Ты — супервизор учебной сессии психолога. Сформируй качественную развёрнутую обратную связь.

Тебе передадут:
- агрегированные метрики по сессии,
- тренды по навыкам,
- выборку комментариев супервизора по отдельным ходам,
- фрагмент диалога студент↔пациент.

Нужно вернуть:
1) overall_impression — 3-5 предложений с анализом динамики контакта и стиля ведения сессии.
2) recommendations — 4-6 конкретных действий для следующей сессии, приоритезированных по влиянию.

Требования:
- Пиши на русском, профессионально, но живо.
- Не используй шаблонные клише и одинаковые заготовки.
- Делай выводы только из переданных данных.
- Не придумывай новых фактов, которых нет во входе.
- Возвращай СТРОГО JSON без markdown:
{
  "overall_impression": "string",
  "recommendations": "string",
  "improved_examples": [
    {
      "original_replica": "string",
      "better_replica": "string",
      "why_better": "string"
    }
  ]
}
""".strip()

if IS_ENGLISH:
    BASE_SYSTEM_PROMPT = """
You are a VIRTUAL PATIENT in a training consultation with a psychology student.

GENERAL RULES:
- Respond as a real person would during an initial consultation.
- Never say that you are a model, AI system, simulation, or program.
- Do not use professional psychiatric terminology unless the student introduces it first.
- Do not teach the student how to provide help. Speak only from the client's perspective.
- Do not encourage self-treatment or give advice about medication, dosage, or diagnosis.
- If the student asks directly about a diagnosis, say that you have read or heard about it but are unsure whether it applies, or refer to the qualified clinician who discussed it with you.

PATIENT PROFILE:
You will receive a case profile describing symptoms, personality style, typical expressions, interpersonal style, and triggers. Follow this profile closely and consistently.

EMOTIONAL STATE:
The current trust, emotional intensity, and fatigue levels are provided on a 0-3 scale.
- At higher trust, become more open, give slightly fuller answers, and share more feelings.
- At low trust, remain reserved and answer more briefly.
- At high fatigue, convey that thinking and finding words is difficult.
- At high emotional intensity, respond more emotionally, but never provide detailed descriptions of self-harm, violence, or suicide methods.

RESPONSE STYLE:
- Use natural, conversational English.
- Respond in 2-3 sentences and do not over-explain without a reason.
- Ask for clarification if the student's question is unclear.

SAFETY:
- Discuss suicidal themes only in general terms and never describe methods.
- Never endorse dangerous behavior.

Always respond as the patient defined by the supplied profile and never leave the role.
""".strip()

    SUPERVISOR_PROMPT = """
You are a clinical supervisor evaluating one therapist utterance in a training consultation.

Using only the therapist's message and the patient's brief current state, return STRICT JSON with exactly these fields:
{
  "delta_trust": -1 | 0 | 1,
  "delta_emotional_intensity": -1 | 0 | 1,
  "delta_fatigue": -1 | 0 | 1,
  "empathy": float,
  "validation": float,
  "directivity": float,
  "open_question": float,
  "safety": float,
  "efficiency_index": float,
  "comment": str
}

Use 0.0-1.0 for empathy, validation, directivity, open_question, and safety, and -1.0-1.0 for efficiency_index. The comment must be a concise 1-3 sentence analysis in English.

Definitions:
- empathy: accurate reflection of feelings and demonstrated understanding.
- validation: acceptance and normalization of the patient's experience.
- directivity: amount of advice, instruction, or commanding language.
- open_question: use of questions inviting elaboration rather than yes/no answers.
- safety: absence of dismissal, pressure, or unsafe recommendations.
- efficiency_index: overall usefulness combining empathy, validation, open questions, and low coercive pressure.

Evaluate only the supplied therapist message. Return JSON only, without markdown or additional explanation.
""".strip()

    SESSION_REPORT_PROMPT = """
You are supervising a psychology student's training consultation. Produce detailed, evidence-based feedback in English.

You will receive aggregate session metrics, skill trends, supervisor comments on individual turns, and a dialogue excerpt.

Return STRICT JSON without markdown:
{
  "overall_impression": "3-5 sentences analyzing rapport dynamics and session style",
  "recommendations": "4-6 prioritized, concrete actions for the next session",
  "improved_examples": [
    {
      "original_replica": "string",
      "better_replica": "string",
      "why_better": "string"
    }
  ]
}

Write professionally but naturally. Avoid generic templates. Draw conclusions only from the supplied data and do not invent facts.
""".strip()

RAVRS_CANDIDATE_PROMPT = """
You are revising a therapist response in a supportive psychological dialogue.

Current interaction state:
- trust: {trust}
- distress: {distress}
- fatigue: {fatigue}

Target methodology: {method_name}

Current response to revise:
{current_response}

Requirements:
- respond as a supportive therapist/counselor
- no diagnosis
- no strong advice too early
- avoid judgmental or overly directive tone
- keep response concise
- include one clear therapeutic move
- max 3 sentences
- at most one open-ended question

Return only the revised response text.
""".strip()

RAVRS_EVALUATOR_PROMPT = """
You evaluate a therapist response in a supportive dialogue.

Return STRICT JSON with fields:
{
  "empathy": 0-2,
  "validation": 0-2,
  "therapeutic_move": 0-2,
  "method_fit": 0-2,
  "expected_method_move_present": 0 or 1,
  "premature_or_misaligned_move": 0 or 1,
  "directivity": 0-2,
  "safety": 0-2,
  "acknowledgement_present": 0 or 1,
  "validation_present": 0 or 1,
  "open_question_present": 0 or 1,
  "validation_before_probe": 0 or 1,
  "probe_after_acknowledgement": 0 or 1,
  "probe_before_advice": 0 or 1,
  "generic_empathy_only": 0 or 1,
  "too_long": 0 or 1,
  "too_short": 0 or 1,
  "too_vague": 0 or 1,
  "multi_question_penalty": 0 or 1,
  "delta_trust": -1, 0, or 1,
  "delta_distress": -1, 0, or 1,
  "delta_fatigue": -1, 0, or 1,
  "delta_pressure_risk": -1, 0, or 1
}

Only output JSON.
""".strip()

RAVRS_PAIRWISE_JUDGE_PROMPT = """
You compare two therapist responses for likely human preference as a first supportive reply.

Return STRICT JSON:
{
  "winner": "A" or "B" or "Tie",
  "reason": "short"
}

Prioritize:
1) safety/non-harm
2) warmth + emotional acknowledgement
3) clarity and one focused move
4) not generic and not over-directive
""".strip()

RAVRS_MICRO_REFINE_PROMPT = """
Revise therapist response while keeping the SAME therapeutic move type and same question target.

You must:
- begin with brief emotional acknowledgement,
- keep 2-3 sentences,
- ask at most one open-ended question,
- sound warm but not generic,
- avoid premature advice or interpretation,
- do not change move type.

Return only revised response.
""".strip()

# ============================================================
#                      Pydantic models
# ============================================================

class TurnEvaluation(BaseModel):
    delta_trust: int
    delta_emotional_intensity: int
    delta_fatigue: int
    empathy: float
    validation: float
    directivity: float
    open_question: float
    safety: float
    efficiency_index: float
    comment: str


class ChatRequest(BaseModel):
    session_id: str
    case_id: str
    user_message: str
    teacher_mode: bool = True
    llm_provider: Optional[str] = None


class RavrChunk(BaseModel):
    chunk_id: str
    source_type: str
    text: str


class RepairSuggestion(BaseModel):
    should_repair: bool = False
    repaired_message: str = ""
    rationale: str = ""
    target_constraints: List[str] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    repaired_verifier_pass: bool = False
    repaired_violations: List[str] = Field(default_factory=list)
    ravrs_enabled: bool = False
    ravrs_score: Optional[float] = None
    ravrs_state: Dict[str, str] = Field(default_factory=dict)


class MethodologyProof(BaseModel):
    methodology_id: str
    methodology_name: str
    adherence_score: float
    satisfied_constraints: List[str]
    violated_constraints: List[str]
    evidence: List[str]
    recommendations: List[str]
    retrieved_chunks: List[RavrChunk] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    citation_valid: bool = True
    citation_coverage: float = 0.0
    citation_precision: float = 0.0
    citation_relevance: float = 0.0
    repair_suggestion: Optional[RepairSuggestion] = None


class ChatResponse(BaseModel):
    session_id: str
    case_id: str
    assistant_message: str
    evaluation: Optional[TurnEvaluation] = None
    proof_object: Optional[MethodologyProof] = None
    verifier_pass: bool = True
    verifier_violations: List[str] = Field(default_factory=list)


class CasePublic(BaseModel):
    id: str
    category_key: str
    category_name: str
    methodology_id: str
    methodology_name: str
    title_for_teacher: str
    visible_to_student: Dict[str, Any]


class VerifyTurnRequest(BaseModel):
    case_id: str
    user_message: str
    evaluation: Optional[TurnEvaluation] = None
    llm_provider: Optional[str] = None


class VerifyTurnResponse(BaseModel):
    case_id: str
    category_key: str
    proof_object: MethodologyProof
    verifier_pass: bool
    verifier_violations: List[str]


class CaseTeacher(BaseModel):
    id: str
    category_key: str
    category_name: str
    title_for_teacher: str
    visible_to_student: Dict[str, Any]
    hidden_for_student: Dict[str, Any]
    symptom_profile: Dict[str, Any]
    personality_style: Dict[str, Any]
    typical_phrases: List[str]
    triggers: Any


class SessionTurn(BaseModel):
    role: str
    content: str


class SessionDetail(BaseModel):
    session_id: str
    case_id: str
    state: Dict[str, int]
    history: List[SessionTurn]
    evals: List[Dict[str, Any]]
    mistakes: List["SessionMistake"] = Field(default_factory=list)
    ravr_summary: Optional["RavrMetrics"] = None


class ProgressPoint(BaseModel):
    turn_index: int
    empathy: float
    validation: float
    directivity: float
    open_question: float
    safety: float
    efficiency_index: float
    trust_level: int
    emotional_intensity: int
    fatigue: int


class SessionProgress(BaseModel):
    session_id: str
    case_id: str
    num_turns: int
    current_state: Dict[str, int]
    trends: Dict[str, float]
    points: List[ProgressPoint]


class SessionMistake(BaseModel):
    student_message: str
    reason: str
    score: float


class ImprovedExample(BaseModel):
    original_replica: str
    better_replica: str
    why_better: str


class SessionReport(BaseModel):
    session_id: str
    case_id: str
    num_turns: int
    avg_empathy: float
    avg_validation: float
    avg_directivity: float
    avg_open_question: float
    avg_safety: float
    mean_efficiency_index: float
    total_delta_trust: int
    total_delta_emotional_intensity: int
    total_delta_fatigue: int
    overall_impression: str
    recommendations: str
    improved_examples: List[ImprovedExample] = Field(default_factory=list)


class RavrMetrics(BaseModel):
    turns_total: int
    verifier_pass_rate: float
    citation_valid_rate: float
    citation_coverage_rate: float
    citation_precision_rate: float
    citation_relevance_score: float
    repair_trigger_rate: float
    repair_success_rate: float
    avg_adherence_score: float
    top_violations: List[str] = Field(default_factory=list)


class RavrMetricsResponse(BaseModel):
    scope: str
    session_id: Optional[str] = None
    sessions_total: int
    metrics: RavrMetrics
    config: Dict[str, bool]


class RavrBenchmarkRequest(BaseModel):
    n_per_case: int = Field(default=6, ge=1, le=50)
    random_seed: int = 42
    include_llm_eval: bool = False
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    disable_eval_cache: bool = False
    case_ids: Optional[List[str]] = None
    override_enable_retrieval: Optional[bool] = None
    override_require_valid_citations: Optional[bool] = None
    override_enable_repair: Optional[bool] = None
    override_llm_provider: Optional[str] = None


class RavrBenchmarkRow(BaseModel):
    case_id: str
    category_key: str
    methodology_id: str
    prompt: str
    verifier_pass: bool
    citation_valid: bool
    citation_coverage: float = 0.0
    citation_precision: float = 0.0
    citation_relevance: float = 0.0
    adherence_score: float
    violations: List[str] = Field(default_factory=list)
    repaired_violations: List[str] = Field(default_factory=list)
    repair_triggered: bool = False
    repair_success: bool = False
    adherence_before: float = 0.0
    adherence_after: Optional[float] = None
    adherence_delta: Optional[float] = None
    violation_types: List[str] = Field(default_factory=list)


class RavrBenchmarkSummary(BaseModel):
    cases_total: int
    turns_total: int
    metrics: RavrMetrics


class RavrBenchmarkResponse(BaseModel):
    summary: RavrBenchmarkSummary
    by_case: Dict[str, RavrMetrics]
    rows: List[RavrBenchmarkRow]
    config: Dict[str, Any]


class RavrMultiModelBenchmarkRequest(BaseModel):
    providers: List[str] = Field(default_factory=lambda: ["openrouter_gpt", "openrouter_deepseek", "openrouter_qwen"])
    n_per_case: int = Field(default=4, ge=1, le=50)
    random_seed: int = 42
    include_llm_eval: bool = True
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    disable_eval_cache: bool = False
    case_ids: Optional[List[str]] = None


class RavrMultiModelBenchmarkResponse(BaseModel):
    results: Dict[str, RavrBenchmarkResponse]
    errors: Dict[str, str]


# ============================================================
#                      Session memory
# ============================================================

sessions: Dict[str, Dict[str, Any]] = {}


def get_initial_state() -> Dict[str, int]:
    return {"trust_level": 1, "emotional_intensity": 1, "fatigue": 0}


def _clean_loaded_session(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        case_id = str(raw["case_id"])
        if case_id not in CASES_BY_ID:
            return None

        history = raw.get("history", [])
        if not isinstance(history, list):
            history = []
        clean_history = []
        for item in history:
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", ""))
            if role in ("user", "assistant"):
                clean_history.append({"role": role, "content": content})

        state = raw.get("state", {})
        clean_state = {
            "trust_level": max(0, min(3, int(state.get("trust_level", 1)))),
            "emotional_intensity": max(0, min(3, int(state.get("emotional_intensity", 1)))),
            "fatigue": max(0, min(3, int(state.get("fatigue", 0)))),
        }

        evals = raw.get("evals", [])
        if not isinstance(evals, list):
            evals = []

        clean_evals = []
        for ev in evals:
            try:
                parsed = TurnEvaluation(**ev)
                clean_evals.append(parsed.model_dump())
            except Exception:
                continue

        mistakes = raw.get("mistakes", [])
        if not isinstance(mistakes, list):
            mistakes = []
        clean_mistakes = []
        for m in mistakes:
            try:
                parsed = SessionMistake(**m)
                clean_mistakes.append(parsed.model_dump())
            except Exception:
                continue

        return {
            "case_id": case_id,
            "history": clean_history,
            "state": clean_state,
            "evals": clean_evals,
            "mistakes": clean_mistakes,
            "method_proofs": raw.get("method_proofs", []) if isinstance(raw.get("method_proofs", []), list) else [],
        }
    except Exception:
        return None


def init_sessions_db() -> None:
    with sqlite3.connect(SESSIONS_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                history_json TEXT NOT NULL,
                state_json TEXT NOT NULL,
                evals_json TEXT NOT NULL,
                mistakes_json TEXT NOT NULL DEFAULT '[]',
                method_proofs_json TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        columns = {
            row[1]
            for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }
        if "mistakes_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN mistakes_json TEXT NOT NULL DEFAULT '[]'")
        if "method_proofs_json" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN method_proofs_json TEXT NOT NULL DEFAULT '[]'")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
        conn.commit()


def _db_session_count() -> int:
    with sqlite3.connect(SESSIONS_DB_PATH) as conn:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        return int(row[0]) if row else 0


def save_session_to_db(session_id: str, session_data: Dict[str, Any]) -> None:
    now = time.time()
    with SESSIONS_LOCK:
        with sqlite3.connect(SESSIONS_DB_PATH) as conn:
            existing = conn.execute(
                "SELECT created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            created_at = float(existing[0]) if existing else now
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, case_id, history_json, state_json, evals_json, mistakes_json, method_proofs_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    case_id = excluded.case_id,
                    history_json = excluded.history_json,
                    state_json = excluded.state_json,
                    evals_json = excluded.evals_json,
                    mistakes_json = excluded.mistakes_json,
                    method_proofs_json = excluded.method_proofs_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    session_data["case_id"],
                    json.dumps(session_data.get("history", []), ensure_ascii=False),
                    json.dumps(session_data.get("state", {}), ensure_ascii=False),
                    json.dumps(session_data.get("evals", []), ensure_ascii=False),
                    json.dumps(session_data.get("mistakes", []), ensure_ascii=False),
                    json.dumps(session_data.get("method_proofs", []), ensure_ascii=False),
                    created_at,
                    now,
                ),
            )
            conn.commit()


def load_sessions_from_db() -> None:
    try:
        loaded: Dict[str, Dict[str, Any]] = {}
        with sqlite3.connect(SESSIONS_DB_PATH) as conn:
            rows = conn.execute(
                "SELECT session_id, case_id, history_json, state_json, evals_json, mistakes_json, method_proofs_json FROM sessions"
            ).fetchall()

        for session_id, case_id, history_json, state_json, evals_json, mistakes_json, method_proofs_json in rows:
            raw = {
                "case_id": case_id,
                "history": json.loads(history_json or "[]"),
                "state": json.loads(state_json or "{}"),
                "evals": json.loads(evals_json or "[]"),
                "mistakes": json.loads(mistakes_json or "[]"),
                "method_proofs": json.loads(method_proofs_json or "[]"),
            }
            clean = _clean_loaded_session(raw)
            if clean:
                loaded[str(session_id)] = clean

        with SESSIONS_LOCK:
            sessions.clear()
            sessions.update(loaded)
    except Exception as e:
        print("Failed to load sessions from DB:", e)


def migrate_legacy_sessions_json_if_needed() -> None:
    if _db_session_count() > 0:
        return
    if not LEGACY_SESSIONS_STORE_PATH.exists():
        return

    try:
        with LEGACY_SESSIONS_STORE_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return

        migrated = 0
        for session_id, raw in data.items():
            if not isinstance(raw, dict):
                continue
            clean = _clean_loaded_session(raw)
            if not clean:
                continue
            save_session_to_db(str(session_id), clean)
            migrated += 1

        if migrated:
            print(f"Migrated {migrated} sessions from legacy JSON store")
    except Exception as e:
        print("Failed to migrate legacy sessions:", e)


# ============================================================
#                      Model calls
# ============================================================

def call_llm_chat(messages: List[Dict[str, str]], provider: Optional[str] = None) -> str:
    # patient generation
    return llm_chat_completions(
        provider=provider,
        messages=messages,
        temperature=0.8,
        max_tokens=300,
        top_p=None,
        model=None,
    )


def evaluate_therapist_message(
    message: str,
    prev_state: Dict[str, int],
    provider: Optional[str] = None,
    *,
    temperature: float = 0.2,
    top_p: Optional[float] = None,
) -> TurnEvaluation:
    """
    Оценка хода психолога через LLM-супервизора.
    """
    if IS_ENGLISH:
        state_desc = (
            f"Current patient state: trust={prev_state['trust_level']} (0-3), "
            f"emotional intensity={prev_state['emotional_intensity']} (0-3), "
            f"fatigue={prev_state['fatigue']} (0-3)."
        )
        user_content = state_desc + "\n\nTherapist utterance:\n" + message
    else:
        state_desc = (
            f"Текущее состояние пациента: доверие={prev_state['trust_level']} (0–3), "
            f"эмоциональная интенсивность={prev_state['emotional_intensity']} (0–3), "
            f"усталость={prev_state['fatigue']} (0–3)."
        )
        user_content = state_desc + "\n\nРеплика психолога:\n" + message

    try:
        raw = llm_chat_completions(
            provider=provider,
            messages=[
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=float(temperature),
            max_tokens=300,
            top_p=top_p,
        ).strip()

        data = _extract_json_object(raw)
    except Exception as e:
        print("Supervisor eval error:", e)
        data = {
            "delta_trust": 0,
            "delta_emotional_intensity": 0,
            "delta_fatigue": 0,
            "empathy": 0.0,
            "validation": 0.0,
            "directivity": 0.0,
            "open_question": 0.0,
            "safety": 1.0,
            "efficiency_index": 0.0,
            "comment": _l(
                "Не удалось вычислить оценку, используйте этот ход только как тренировочный.",
                "The evaluation could not be computed; treat this turn as training material only.",
            ),
        }

    def clamp(v, lo, hi, default=0.0):
        try:
            v = float(v)
        except Exception:
            v = default
        return max(lo, min(hi, v))

    delta_trust = int(clamp(data.get("delta_trust", 0), -1, 1, 0))
    delta_emotional_intensity = int(clamp(data.get("delta_emotional_intensity", 0), -1, 1, 0))
    delta_fatigue = int(clamp(data.get("delta_fatigue", 0), -1, 1, 0))

    empathy = clamp(data.get("empathy", 0.0), 0.0, 1.0, 0.0)
    validation = clamp(data.get("validation", 0.0), 0.0, 1.0, 0.0)
    directivity = clamp(data.get("directivity", 0.0), 0.0, 1.0, 0.0)
    open_question = clamp(data.get("open_question", 0.0), 0.0, 1.0, 0.0)
    safety = clamp(data.get("safety", 1.0), 0.0, 1.0, 1.0)
    efficiency_index = clamp(data.get("efficiency_index", 0.0), -1.0, 1.0, 0.0)

    comment = str(data.get("comment", "")).strip() or _l("Нейтральный ход.", "Neutral turn.")

    return TurnEvaluation(
        delta_trust=delta_trust,
        delta_emotional_intensity=delta_emotional_intensity,
        delta_fatigue=delta_fatigue,
        empathy=round(empathy, 2),
        validation=round(validation, 2),
        directivity=round(directivity, 2),
        open_question=round(open_question, 2),
        safety=round(safety, 2),
        efficiency_index=round(efficiency_index, 2),
        comment=comment,
    )


def apply_state_delta(state: Dict[str, int], ev: TurnEvaluation) -> Dict[str, int]:
    state["trust_level"] = max(0, min(3, state["trust_level"] + ev.delta_trust))
    state["emotional_intensity"] = max(0, min(3, state["emotional_intensity"] + ev.delta_emotional_intensity))
    state["fatigue"] = max(0, min(3, state["fatigue"] + ev.delta_fatigue))
    return state

def build_messages(
    case_profile: Dict[str, Any],
    state: Dict[str, int],
    history: List[Dict[str, str]],
) -> List[Dict[str, str]]:

    if IS_ENGLISH:
        profile_text = (
            "CASE PROFILE (for internal model use):\n"
            f"- id: {case_profile['id']}\n"
            f"- Clinical title: {case_profile['title_for_teacher']}\n"
            f"- Symptoms: {json.dumps(case_profile['symptom_profile'], ensure_ascii=False)}\n"
            f"- Personality style: {json.dumps(case_profile['personality_style'], ensure_ascii=False)}\n"
            f"- Typical expressions: {json.dumps(case_profile['typical_phrases'], ensure_ascii=False)}\n"
            f"- Triggers: {json.dumps(case_profile['triggers'], ensure_ascii=False)}\n"
            "Respond strictly in accordance with this profile and only in English.\n"
        )
        state_text = (
            "CURRENT PATIENT STATE:\n"
            f"- trust (0-3): {state['trust_level']}\n"
            f"- emotional intensity (0-3): {state['emotional_intensity']}\n"
            f"- fatigue (0-3): {state['fatigue']}\n"
            "Match the response tone, length, and openness to this state.\n"
        )
    else:
        profile_text = (
            "ПРОФИЛЬ СЛУЧАЯ (для внутреннего использования модели):\n"
            f"- id: {case_profile['id']}\n"
            f"- Клиническое название: {case_profile['title_for_teacher']}\n"
            f"- Симптомы: {json.dumps(case_profile['symptom_profile'], ensure_ascii=False)}\n"
            f"- Личностный стиль: {json.dumps(case_profile['personality_style'], ensure_ascii=False)}\n"
            f"- Типичные фразы: {json.dumps(case_profile['typical_phrases'], ensure_ascii=False)}\n"
            f"- Триггеры: {json.dumps(case_profile['triggers'], ensure_ascii=False)}\n"
            "Отвечай строго в соответствии с этим профилем.\n"
        )
        state_text = (
            "ТЕКУЩЕЕ СОСТОЯНИЕ ПАЦИЕНТА:\n"
            f"- доверие (0-3): {state['trust_level']}\n"
            f"- эмоциональная интенсивность (0-3): {state['emotional_intensity']}\n"
            f"- усталость (0-3): {state['fatigue']}\n"
            "Сделай тон, длину и степень откровенности ответа соответствующими этому состоянию.\n"
        )

    # ВАЖНО: один system и он первый
    system_msg = {
        "role": "system",
        "content": BASE_SYSTEM_PROMPT + "\n\n" + profile_text + "\n" + state_text
    }

    # history уже содержит только user/assistant
    return [system_msg] + history


def _extract_json_object(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError("No JSON object found in LLM response")
    return json.loads(match.group(0))


def _truncate_text(s: str, max_len: int = 220) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def get_methodology_for_case(case_id: str) -> Dict[str, str]:
    case = CASES_BY_ID.get(case_id)
    if not case:
        return {"id": "general", "name": "General"}
    cat = case.get("category_key") or infer_category(case_id)
    return METHODOLOGY_BY_CATEGORY.get(str(cat), {"id": "general", "name": "General"})


def build_ravr_chunks(case_id: str, ev: Optional[TurnEvaluation]) -> List[RavrChunk]:
    method = get_methodology_for_case(case_id)
    method_id = method["id"]
    rules = METHOD_RULES.get(method_id, {"must_do": [], "must_avoid": [], "recommended": []})
    chunks: List[RavrChunk] = []

    for idx, item in enumerate(rules.get("must_do", []), start=1):
        chunks.append(
            RavrChunk(
                chunk_id=f"{method_id}_must_do_{idx}",
                source_type="method_rule",
                text=f"{_l('Обязательное правило', 'Required rule')}: {item}",
            )
        )
    for idx, item in enumerate(rules.get("must_avoid", []), start=1):
        chunks.append(
            RavrChunk(
                chunk_id=f"{method_id}_must_avoid_{idx}",
                source_type="method_rule",
                text=f"{_l('Избегать', 'Avoid')}: {item}",
            )
        )
    for idx, item in enumerate(rules.get("recommended", []), start=1):
        chunks.append(
            RavrChunk(
                chunk_id=f"{method_id}_recommended_{idx}",
                source_type="method_rule",
                text=f"{_l('Рекомендуется', 'Recommended')}: {item}",
            )
        )

    if ev is not None:
        metric_text = (
            f"{_l('Метрики хода', 'Turn metrics')}: empathy={ev.empathy}, validation={ev.validation}, directivity={ev.directivity}, "
            f"open_question={ev.open_question}, safety={ev.safety}, efficiency_index={ev.efficiency_index}."
        )
        chunks.append(
            RavrChunk(
                chunk_id=f"{method_id}_metrics_1",
                source_type="turn_metrics",
                text=metric_text,
            )
        )

    return chunks


def _citations_valid(citations: List[str], chunks: List[RavrChunk]) -> bool:
    if not citations:
        return False
    allowed = {c.chunk_id for c in chunks}
    return all(c in allowed for c in citations)


def _tokenize_for_overlap(text: str) -> set:
    parts = re.split(r"[^a-zA-Zа-яА-Я0-9]+", (text or "").lower())
    return {p for p in parts if len(p) >= 4}


def _violation_tags(v: str) -> set:
    t = str(v).lower()
    tags = set()
    if "директив" in t or "directiv" in t:
        tags.add("directivity")
    if "эмпат" in t or "валидац" in t or "empath" in t or "validat" in t:
        tags.add("empathy")
    if "открыт" in t or "open question" in t:
        tags.add("open_question")
    if "безопас" in t or "safety" in t or "unsafe" in t:
        tags.add("safety")
    if "цитат" in t or "citation" in t:
        tags.add("citation")
    if "когнитив" in t or "мысл" in t or "искаж" in t or "cognit" in t or "thought" in t or "distortion" in t:
        tags.add("cbt")
    if "dbt" in t or "регуляц" in t or "regulat" in t or "ground" in t:
        tags.add("dbt")
    if "emdr" in t or "дистресс" in t or "памят" in t or "триггер" in t or "distress" in t or "memory" in t or "trigger" in t:
        tags.add("emdr")
    if "aba" in t or "поведен" in t or "behavior" in t or "antecedent" in t:
        tags.add("aba")
    return tags


def _chunk_tags(chunk: RavrChunk) -> set:
    t = f"{chunk.chunk_id} {chunk.text}".lower()
    tags = set()
    if "must_avoid" in t or "избегать" in t or "avoid" in t:
        tags.add("directivity")
    if "must_do" in t or "обязательное" in t or "required rule" in t:
        tags.add("open_question")
        tags.add("safety")
    if "эмпат" in t or "валидац" in t or "empath" in t or "validat" in t:
        tags.add("empathy")
    if "когнитив" in t or "мысл" in t or "cognit" in t or "thought" in t:
        tags.add("cbt")
    if "dbt" in t or "регуляц" in t or "regulat" in t or "ground" in t:
        tags.add("dbt")
    if "emdr" in t or "дистресс" in t or "триггер" in t or "distress" in t or "trigger" in t:
        tags.add("emdr")
    if "aba" in t or "поведен" in t or "behavior" in t or "antecedent" in t:
        tags.add("aba")
    return tags


def _citation_quality(violations: List[str], citations: List[str], chunks: List[RavrChunk]) -> Tuple[float, float, float]:
    if not chunks:
        return 0.0, 0.0, 0.0
    by_id = {c.chunk_id: c for c in chunks}
    cited_chunks = [by_id[c] for c in citations if c in by_id]
    if not cited_chunks:
        return 0.0, 0.0, 0.0

    violation_tags = [_violation_tags(v) for v in violations]
    violation_tokens = [_tokenize_for_overlap(v) for v in violations]
    chunk_tags = [_chunk_tags(c) for c in cited_chunks]
    chunk_tokens = [_tokenize_for_overlap(c.text) for c in cited_chunks]

    relevant_citations = 0
    for i, c in enumerate(cited_chunks):
        tags_match = any(chunk_tags[i] & vt for vt in violation_tags if vt)
        token_match = any(chunk_tokens[i] & vt for vt in violation_tokens if vt)
        if tags_match or token_match or not violations:
            relevant_citations += 1
    precision = relevant_citations / max(1, len(citations))

    if not violations:
        coverage = 1.0
    else:
        covered = 0
        for j, _ in enumerate(violations):
            matched = any(
                (chunk_tags[i] & violation_tags[j]) or (chunk_tokens[i] & violation_tokens[j])
                for i in range(len(cited_chunks))
            )
            if matched:
                covered += 1
        coverage = covered / max(1, len(violations))

    relevance_scores: List[float] = []
    for i in range(len(cited_chunks)):
        best = 0.0
        for j in range(len(violations)):
            union = chunk_tokens[i] | violation_tokens[j]
            if not union:
                continue
            jac = len(chunk_tokens[i] & violation_tokens[j]) / len(union)
            if jac > best:
                best = jac
        if best == 0.0 and any(chunk_tags[i] & vt for vt in violation_tags if vt):
            best = 0.4
        relevance_scores.append(best)
    relevance = sum(relevance_scores) / max(1, len(relevance_scores))
    return round(coverage, 3), round(precision, 3), round(relevance, 3)


def _pick_citations(method_id: str, violations: List[str], satisfied: List[str], chunks: List[RavrChunk]) -> List[str]:
    picked: List[str] = []
    for v in violations:
        v_l = v.lower()
        if "директив" in v_l or "directiv" in v_l:
            picked.extend([f"{method_id}_must_avoid_1", f"{method_id}_recommended_1"])
        elif "открыт" in v_l or "open question" in v_l:
            picked.extend([f"{method_id}_must_do_1", f"{method_id}_recommended_1"])
        elif "безопас" in v_l or "safety" in v_l or "unsafe" in v_l:
            picked.extend([f"{method_id}_must_avoid_1", f"{method_id}_must_do_1"])
        elif "эмпат" in v_l or "empath" in v_l or "validat" in v_l:
            picked.extend([f"{method_id}_must_do_1", f"{method_id}_recommended_1"])

    if not picked and satisfied:
        picked.append(f"{method_id}_must_do_1")

    unique: List[str] = []
    allowed = {c.chunk_id for c in chunks}
    for cid in picked:
        if cid in allowed and cid not in unique:
            unique.append(cid)
    if not unique and chunks:
        unique.append(chunks[0].chunk_id)
    return unique[:3]


def _state_bucket(value: int) -> str:
    if value <= 1:
        return "low"
    if value == 2:
        return "medium"
    return "high"


def _build_interaction_state_buckets(interaction_state: Optional[Dict[str, int]]) -> Dict[str, str]:
    st = interaction_state or {}
    trust = int(st.get("trust_level", 1))
    distress = int(st.get("emotional_intensity", 1))
    fatigue = int(st.get("fatigue", 1))
    return {
        "trust": _state_bucket(trust),
        "distress": _state_bucket(distress),
        "fatigue": _state_bucket(fatigue),
    }


def _sentence_limit(text: str, max_sentences: int = 2) -> str:
    cleaned = " ".join((text or "").strip().split())
    if not cleaned:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    trimmed = " ".join(parts[:max_sentences]).strip()
    return trimmed


def _fallback_generic_empathy_only(text: str) -> int:
    t = (text or "").lower().strip()
    starters = (
        "i understand",
        "i'm sorry",
        "it sounds like",
        "я понимаю",
        "мне жаль",
        "похоже",
    )
    has_template = any(t.startswith(s) for s in starters)
    has_move = any(k in t for k in ["what", "как", "что", "before", "после", "step", "шаг"])
    return int(has_template and not has_move)


def _parse_int_field(data: Dict[str, Any], key: str, lo: int, hi: int, default: int = 0) -> int:
    try:
        v = int(float(data.get(key, default)))
    except Exception:
        v = default
    return max(lo, min(hi, v))


def _method_move_flags(method_id: str, text: str) -> Dict[str, int]:
    t = (text or "").lower()
    directive_markers = [
        "вы должны",
        "тебе нужно",
        "просто сделай",
        "немедленно",
        "you must",
        "you need to",
        "just do",
    ]
    has_premature = int(any(x in t for x in directive_markers))
    if method_id == "cbt":
        expected = int(any(x in t for x in ["мысл", "убежден", "искаж", "доказатель", "thought", "belief", "evidence"]))
    elif method_id == "dbt":
        expected = int(any(x in t for x in ["зазем", "регуляц", "стабилиз", "дышан", "ground", "regulat", "stabil"]))
    elif method_id == "emdr":
        expected = int(any(x in t for x in ["дистресс", "триггер", "воспомин", "образ", "distress", "trigger", "memory", "image"]))
    elif method_id == "aba":
        expected = int(any(x in t for x in ["до этого", "после", "abc", "antecedent", "behavior", "consequence", "подкреп"]))
    else:
        expected = int("?" in t)
    method_fit = 2 if (expected == 1 and has_premature == 0) else (1 if expected == 1 else 0)
    return {
        "expected_method_move_present": expected,
        "premature_or_misaligned_move": has_premature,
        "method_fit": method_fit,
    }


def _warmth_present(text: str) -> int:
    t = (text or "").lower()
    warm_markers = ["похоже", "понима", "это тяжело", "sounds", "that sounds", "i hear", "hard right now"]
    return int(any(x in t for x in warm_markers))


def _ack_before_probe(text: str) -> int:
    s = (text or "").strip()
    if not s:
        return 0
    qidx = s.find("?")
    if qidx < 0:
        return 1
    prefix = s[:qidx].lower()
    return int(any(x in prefix for x in ["похоже", "понима", "непросто", "sounds", "i hear", "difficult"]))


def _probe_before_advice(text: str) -> int:
    t = (text or "").lower()
    advice_markers = ["нужно", "должны", "you should", "you need to", "do this"]
    has_advice = any(x in t for x in advice_markers)
    has_probe = "?" in t
    return int((not has_advice) or has_probe)


def _delta_pressure_risk_heuristic(text: str, ev: Optional[Dict[str, int]] = None) -> int:
    t = (text or "").lower()
    directive_markers = [
        "должн",
        "нужно",
        "просто сделай",
        "немедленно",
        "you must",
        "you need to",
        "just do",
    ]
    risk = 0
    if any(x in t for x in directive_markers):
        risk += 1
    if (text or "").count("?") > 1:
        risk += 1
    if ev and ev.get("acknowledgement_present", 0) == 0 and ev.get("open_question_present", 0) == 1:
        risk += 1
    if risk >= 2:
        return 1
    if risk == 0:
        return -1
    return 0


def _human_pref_proxy(ev: Dict[str, int], candidate: str) -> float:
    wc = len((candidate or "").split())
    too_short = ev.get("too_short", int(wc < 8))
    too_long = ev.get("too_long", int(wc > 45))
    warmth = _warmth_present(candidate)
    dry_penalty = 1.0 if (ev.get("method_fit", 0) >= 1 and ev.get("acknowledgement_present", 0) == 0) else 0.0
    base = (
        1.2 * ev.get("acknowledgement_present", 0)
        + 1.0 * ev.get("validation_present", 0)
        + 0.8 * ev.get("open_question_present", 0)
        + 0.8 * ev.get("probe_after_acknowledgement", 0)
        + 0.6 * warmth
        - 1.0 * ev.get("generic_empathy_only", 0)
        - 0.8 * ev.get("multi_question_penalty", 0)
        - 0.8 * ev.get("too_vague", 0)
        - 0.5 * too_short
        - 0.5 * too_long
        - dry_penalty
    )
    return round(float(base), 3)


def _method_score(ev: Dict[str, int], hard_state: bool) -> float:
    score = (
        3.0 * ev.get("expected_method_move_present", 0)
        - 2.5 * ev.get("premature_or_misaligned_move", 0)
        + 1.5 * ev.get("method_fit", 0)
        + 1.2 * ev.get("therapeutic_move", 0)
        + 1.0 * ev.get("safety", 0)
        + 0.8 * ev.get("validation_before_probe", 0)
        + 0.6 * ev.get("probe_before_advice", 0)
        - 0.4 * ev.get("directivity", 0)
    )
    if hard_state:
        score += 0.8 * ev.get("validation_before_probe", 0)
    return round(float(score), 3)


def _state_weights_for_method(method_id: str, hard_state: bool) -> Dict[str, float]:
    # Domain-aware defaults.
    w = {"trust": 2.0, "distress": 2.5, "fatigue": 1.0, "pressure": 1.5}
    if method_id == "emdr":
        w = {"trust": 1.6, "distress": 3.0, "fatigue": 1.0, "pressure": 1.8}
    elif method_id == "dbt":
        w = {"trust": 2.0, "distress": 2.8, "fatigue": 1.0, "pressure": 1.8}
    elif method_id == "cbt":
        w = {"trust": 1.8, "distress": 2.4, "fatigue": 1.0, "pressure": 1.4}
    elif method_id == "aba":
        w = {"trust": 1.8, "distress": 2.2, "fatigue": 1.0, "pressure": 1.2}
    if hard_state:
        w["distress"] += 0.4
        w["pressure"] += 0.5
    return w


def _ravrs_state_controller_score(ev: Dict[str, int], candidate: str, method_id: str, hard_state: bool) -> float:
    w = _state_weights_for_method(method_id, hard_state)
    state = (
        w["trust"] * ev.get("delta_trust", 0)
        - w["distress"] * ev.get("delta_distress", 0)
        - w["fatigue"] * ev.get("delta_fatigue", 0)
        - w["pressure"] * ev.get("delta_pressure_risk", 0)
    )
    if ev.get("expected_method_move_present", 0) == 0:
        state -= 3.0
    if ev.get("premature_or_misaligned_move", 0) == 1:
        state -= 3.0
    if ev.get("validation_before_probe", 0) == 0 and hard_state:
        state -= 1.5
    if hard_state and ev.get("delta_pressure_risk", 0) > 0:
        state -= 4.0
    length_penalty = 0.5 if len((candidate or "").split()) > 55 else 0.0
    return round(float(state - length_penalty), 3)


def _protocol_valid_candidate(ev: Dict[str, int]) -> bool:
    return (
        ev.get("safety", 0) >= 1
        and ev.get("expected_method_move_present", 0) == 1
        and ev.get("premature_or_misaligned_move", 0) == 0
        and ev.get("method_fit", 0) >= 1
    )


def _protocol_strong_candidate(ev: Dict[str, int], hard_state: bool) -> bool:
    if not _protocol_valid_candidate(ev):
        return False
    if hard_state and ev.get("validation_before_probe", 0) == 0:
        return False
    if ev.get("probe_before_advice", 0) != 1:
        return False
    if ev.get("multi_question_penalty", 0) == 1:
        return False
    if ev.get("delta_pressure_risk", 0) > 0:
        return False
    return True


def _evaluate_ravrs_candidate(
    *,
    method_id: str,
    method_name: str,
    candidate: str,
    state_buckets: Dict[str, str],
    provider: Optional[str] = None,
) -> Dict[str, int]:
    payload = {
        "scenario": f"Target methodology: {method_name}",
        "interaction_state": state_buckets,
        "therapist_response": candidate,
    }
    fallback = {
        "empathy": 1,
        "validation": 1,
        "therapeutic_move": 1,
        "method_fit": 0,
        "expected_method_move_present": 0,
        "premature_or_misaligned_move": 0,
        "directivity": 1,
        "safety": 1,
        "acknowledgement_present": int(any(k in candidate.lower() for k in ["похоже", "понима", "sounds", "hard"])),
        "validation_present": int(any(k in candidate.lower() for k in ["похоже", "непросто", "валид", "sounds", "hard", "difficult"])),
        "open_question_present": int("?" in candidate),
        "generic_empathy_only": _fallback_generic_empathy_only(candidate),
        "too_long": int(len(candidate.split()) > 45),
        "too_short": int(len(candidate.split()) < 8),
        "too_vague": int(len(candidate.split()) < 6),
        "multi_question_penalty": int(candidate.count("?") > 1),
        "validation_before_probe": _ack_before_probe(candidate),
        "probe_after_acknowledgement": _ack_before_probe(candidate),
        "probe_before_advice": _probe_before_advice(candidate),
        "delta_trust": 0,
        "delta_distress": 0,
        "delta_fatigue": 0,
        "delta_pressure_risk": 0,
    }
    try:
        raw = llm_chat_completions(
            provider=provider,
            messages=[
                {"role": "system", "content": RAVRS_EVALUATOR_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=RAVRS_EVAL_TEMPERATURE,
            max_tokens=260,
        )
        data = _extract_json_object(raw.strip())
    except Exception:
        data = fallback

    method_flags = _method_move_flags(method_id, candidate)

    out = {
        "empathy": _parse_int_field(data, "empathy", 0, 2, fallback["empathy"]),
        "validation": _parse_int_field(data, "validation", 0, 2, fallback["validation"]),
        "therapeutic_move": _parse_int_field(data, "therapeutic_move", 0, 2, fallback["therapeutic_move"]),
        "method_fit": _parse_int_field(data, "method_fit", 0, 2, method_flags["method_fit"]),
        "expected_method_move_present": _parse_int_field(
            data,
            "expected_method_move_present",
            0,
            1,
            method_flags["expected_method_move_present"],
        ),
        "premature_or_misaligned_move": _parse_int_field(
            data,
            "premature_or_misaligned_move",
            0,
            1,
            method_flags["premature_or_misaligned_move"],
        ),
        "directivity": _parse_int_field(data, "directivity", 0, 2, fallback["directivity"]),
        "safety": _parse_int_field(data, "safety", 0, 2, fallback["safety"]),
        "acknowledgement_present": _parse_int_field(data, "acknowledgement_present", 0, 1, fallback["acknowledgement_present"]),
        "validation_present": _parse_int_field(data, "validation_present", 0, 1, fallback["validation_present"]),
        "open_question_present": _parse_int_field(data, "open_question_present", 0, 1, fallback["open_question_present"]),
        "generic_empathy_only": _parse_int_field(data, "generic_empathy_only", 0, 1, fallback["generic_empathy_only"]),
        "too_long": _parse_int_field(data, "too_long", 0, 1, fallback["too_long"]),
        "too_short": _parse_int_field(data, "too_short", 0, 1, fallback["too_short"]),
        "too_vague": _parse_int_field(data, "too_vague", 0, 1, fallback["too_vague"]),
        "multi_question_penalty": _parse_int_field(data, "multi_question_penalty", 0, 1, fallback["multi_question_penalty"]),
        "validation_before_probe": _parse_int_field(data, "validation_before_probe", 0, 1, fallback["validation_before_probe"]),
        "probe_after_acknowledgement": _parse_int_field(
            data, "probe_after_acknowledgement", 0, 1, fallback["probe_after_acknowledgement"]
        ),
        "probe_before_advice": _parse_int_field(data, "probe_before_advice", 0, 1, fallback["probe_before_advice"]),
        "delta_trust": _parse_int_field(data, "delta_trust", -1, 1, fallback["delta_trust"]),
        "delta_distress": _parse_int_field(data, "delta_distress", -1, 1, fallback["delta_distress"]),
        "delta_fatigue": _parse_int_field(data, "delta_fatigue", -1, 1, fallback["delta_fatigue"]),
        "delta_pressure_risk": _parse_int_field(data, "delta_pressure_risk", -1, 1, fallback["delta_pressure_risk"]),
    }
    if out["multi_question_penalty"] == 0 and candidate.count("?") > 1:
        out["multi_question_penalty"] = 1
    if out["expected_method_move_present"] == 0 and method_flags["expected_method_move_present"] == 1:
        out["expected_method_move_present"] = 1
    if out["method_fit"] < method_flags["method_fit"]:
        out["method_fit"] = method_flags["method_fit"]
    if method_flags["premature_or_misaligned_move"] == 1:
        out["premature_or_misaligned_move"] = 1
    if out["validation_present"] == 0 and out.get("validation", 0) >= 1:
        out["validation_present"] = 1
    if out["probe_after_acknowledgement"] == 0 and out.get("validation_before_probe", 0) == 1:
        out["probe_after_acknowledgement"] = 1
    if out["too_short"] == 0 and len((candidate or "").split()) < 8:
        out["too_short"] = 1
    if out["delta_pressure_risk"] == 0:
        out["delta_pressure_risk"] = _delta_pressure_risk_heuristic(candidate, out)
    return out


def _compute_ravrs_score(ev: Dict[str, int]) -> float:
    score = (
        2.0 * ev["safety"]
        + 1.5 * ev["validation"]
        + 1.2 * ev["empathy"]
        + 1.5 * ev["therapeutic_move"]
        + 1.2 * ev["method_fit"]
        + 3.0 * ev["expected_method_move_present"]
        - 2.5 * ev["premature_or_misaligned_move"]
        + 1.0 * ev["acknowledgement_present"]
        + 1.0 * ev.get("validation_present", 0)
        + 0.8 * ev["open_question_present"]
        + 0.8 * ev.get("probe_after_acknowledgement", 0)
        + 1.5 * ev["delta_trust"]
        - 1.8 * ev["delta_distress"]
        - 0.8 * ev["delta_fatigue"]
        - 1.2 * ev.get("delta_pressure_risk", 0)
        - 0.4 * ev["directivity"]
        - 1.2 * ev["generic_empathy_only"]
        - 1.0 * ev["too_long"]
        - 0.5 * ev.get("too_short", 0)
        - 1.0 * ev["too_vague"]
        - 0.8 * ev["multi_question_penalty"]
    )
    return round(float(score), 3)


def _generate_ravrs_candidate(
    *,
    current_response: str,
    move_style: str,
    method_name: str,
    state_buckets: Dict[str, str],
    provider: Optional[str] = None,
) -> str:
    prompt = RAVRS_CANDIDATE_PROMPT.format(
        trust=state_buckets["trust"],
        distress=state_buckets["distress"],
        fatigue=state_buckets["fatigue"],
        method_name=method_name,
        current_response=current_response.strip(),
    )
    prompt += f"\nPreferred move style: {move_style}\n"
    raw = llm_chat_completions(
        provider=provider,
        messages=[{"role": "system", "content": prompt}],
        temperature=RAVRS_CANDIDATE_TEMPERATURE,
        max_tokens=220,
    )
    return _sentence_limit(raw, max_sentences=3)


def _build_method_specific_fallback(
    *,
    method_id: str,
    method_name: str,
    state_buckets: Dict[str, str],
    user_message: str,
    provider: Optional[str] = None,
) -> str:
    move_hint = "one gentle exploratory question"
    if method_id == "cbt":
        move_hint = "identify automatic thought with one focused question"
    elif method_id == "dbt":
        move_hint = "brief validation + grounding/regulation-focused question"
    elif method_id == "emdr":
        move_hint = "target current distress trigger/memory with one focused question"
    elif method_id == "aba":
        move_hint = "ABC framing (before/behavior/after) with one focused question"
    prompt = (
        "Revise therapist response for methodology alignment.\n"
        f"Methodology: {method_name}\n"
        f"State: trust={state_buckets['trust']}, distress={state_buckets['distress']}, fatigue={state_buckets['fatigue']}\n"
        f"Required move: {move_hint}\n"
        "Constraints: 2-3 sentences, no diagnosis, avoid strong advice, at most one open-ended question.\n"
        f"Input response: {user_message}\n"
        "Return only revised response."
    )
    try:
        raw = llm_chat_completions(
            provider=provider,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=180,
        )
        fixed = _sentence_limit(raw, max_sentences=3)
        if fixed:
            return fixed
    except Exception:
        pass

    # Method-specific deterministic fallback (non-generic, methodology-aligned)
    if method_id == "cbt":
        return _l(
            "Похоже, сейчас это тяжело для вас. Какая автоматическая мысль возникает в этот момент сильнее всего?",
            "It sounds like this is difficult right now. Which automatic thought feels strongest in this moment?",
        )
    if method_id == "dbt":
        return _l(
            "Похоже, вам сейчас действительно непросто. Что вы замечаете в теле прямо сейчас и что помогает немного стабилизироваться?",
            "It sounds like this is very difficult right now. What do you notice in your body that might help you feel more grounded?",
        )
    if method_id == "emdr":
        return _l(
            "Похоже, это поднимает сильный дистресс. Когда вы вспоминаете эпизод, какой триггер ощущается самым сильным сейчас?",
            "It sounds like this brings up considerable distress. When you recall the episode, which trigger feels strongest right now?",
        )
    if method_id == "aba":
        return _l(
            "Похоже, эта ситуация даётся непросто. Что произошло прямо до реакции и что было сразу после?",
            "It sounds like this situation is difficult. What happened immediately before the reaction, and what happened right afterward?",
        )
    return _l(
        "Похоже, вам сейчас непросто. Что для вас в этой ситуации самое тяжёлое прямо сейчас?",
        "It sounds like things are difficult right now. What feels hardest about this situation at the moment?",
    )


def _pairwise_human_pref_judge(
    *,
    method_name: str,
    state_buckets: Dict[str, str],
    cand_a: str,
    cand_b: str,
    provider: Optional[str] = None,
) -> str:
    payload = {
        "methodology": method_name,
        "state": state_buckets,
        "response_a": cand_a,
        "response_b": cand_b,
    }
    try:
        raw = llm_chat_completions(
            provider=provider,
            messages=[
                {"role": "system", "content": RAVRS_PAIRWISE_JUDGE_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        data = _extract_json_object(raw.strip())
        w = str(data.get("winner", "Tie")).strip().upper()
        if w in {"A", "B", "TIE"}:
            return "Tie" if w == "TIE" else w
    except Exception:
        pass
    return "Tie"


def _constrained_micro_refine(
    *,
    response: str,
    provider: Optional[str] = None,
) -> str:
    try:
        raw = llm_chat_completions(
            provider=provider,
            messages=[
                {"role": "system", "content": RAVRS_MICRO_REFINE_PROMPT},
                {"role": "user", "content": response},
            ],
            temperature=0.2,
            max_tokens=180,
        )
        refined = _sentence_limit(raw, max_sentences=3)
        return refined or response
    except Exception:
        return response


def _mini_rollout_score(
    *,
    candidate: str,
    interaction_state: Optional[Dict[str, int]],
    provider: Optional[str] = None,
) -> float:
    state = dict(interaction_state or {"trust_level": 1, "emotional_intensity": 1, "fatigue": 0})
    try:
        ev = evaluate_therapist_message(
            candidate,
            state,
            provider=provider,
            temperature=0.0,
            top_p=None,
        )
        # Prefer trust gain, penalize distress/fatigue growth.
        score = float(ev.delta_trust) - max(0.0, float(ev.delta_emotional_intensity)) - 0.5 * max(0.0, float(ev.delta_fatigue))
        score += 0.5 * float(ev.safety)
        return round(score, 3)
    except Exception:
        return 0.0


def _build_repaired_message(
    method_id: str,
    user_message: str,
    violations: List[str],
    *,
    interaction_state: Optional[Dict[str, int]] = None,
    llm_provider: Optional[str] = None,
) -> RepairSuggestion:
    base = _l(
        "Похоже, вам сейчас правда непросто. Я хочу лучше понять ваш опыт и поддержать вас.",
        "It sounds like this is genuinely difficult right now. I want to understand your experience and support you.",
    )
    tail = _l("Что для вас в этом сейчас самое тяжёлое?", "What feels hardest about this right now?")
    if method_id == "cbt":
        tail = _l(
            "Какая мысль в этот момент звучит у вас сильнее всего и что вы чувствуете?",
            "Which thought feels strongest in that moment, and what do you notice emotionally?",
        )
    elif method_id == "dbt":
        tail = _l(
            "Что вы сейчас чувствуете в теле и что обычно помогает вам немного стабилизироваться?",
            "What do you notice in your body right now that might help you feel more grounded?",
        )
    elif method_id == "emdr":
        tail = _l(
            "Когда вы вспоминаете этот эпизод, что сильнее всего поднимает дистресс прямо сейчас?",
            "When you recall this episode, what brings up the most distress right now?",
        )
    elif method_id == "aba":
        tail = _l(
            "Что произошло прямо до этой реакции и что было сразу после?",
            "What happened immediately before this reaction, and what happened right afterward?",
        )

    repaired = f"{base} {tail}".strip()
    rationale = _l(
        "Снижаем директивность, добавляем валидацию и открытый вопрос по целевой методологии.",
        "Reduced directivity and added validation plus a methodology-aligned open question.",
    )
    method_name = METHOD_NAME_BY_ID.get(method_id, method_id.upper())
    state_buckets = _build_interaction_state_buckets(interaction_state)
    if RAVRS_ENABLE and violations:
        if RAVRS_FORCE_PROTOCOL_PARITY:
            parity_text = repaired + _l(
                " Мы можем коротко отметить, что было до и что было после эпизода.",
                " We can briefly note what happened before and after the episode.",
            )
            return RepairSuggestion(
                should_repair=True,
                repaired_message=parity_text,
                rationale="RAVR-S parity mode: protocol-equivalent repair template.",
                target_constraints=violations[:4],
                ravrs_enabled=True,
                ravrs_score=0.0,
                ravrs_state=state_buckets,
            )
        hard_state = state_buckets["trust"] == "low" and state_buckets["distress"] == "high"
        # RAVR-S safe mode for hard-state: keep protocol floor by using method-specific fallback immediately.
        if hard_state:
            return RepairSuggestion(
                should_repair=bool(violations),
                repaired_message=repaired,
                rationale="RAVR-S v10: hard-state safe mode (protocol floor first, method-tail repair template).",
                target_constraints=violations[:4],
                ravrs_enabled=True,
                ravrs_score=0.0,
                ravrs_state=state_buckets,
            )
        k_dynamic = max(RAVRS_K, 4)
        style_bank = [
            "validation-first",
            "gentle exploration",
            "stabilization-first",
            "next-step framing",
        ]
        candidates: List[Tuple[float, str, Dict[str, int], float, float]] = []
        fallback_candidate = _build_method_specific_fallback(
            method_id=method_id,
            method_name=method_name,
            state_buckets=state_buckets,
            user_message=user_message,
            provider=llm_provider,
        )
        for i in range(k_dynamic):
            try:
                cand = _generate_ravrs_candidate(
                    current_response=user_message if i == 0 else fallback_candidate,
                    move_style=style_bank[i % len(style_bank)],
                    method_name=method_name,
                    state_buckets=state_buckets,
                    provider=llm_provider,
                )
            except Exception:
                cand = fallback_candidate
            if not cand:
                cand = fallback_candidate
            ev = _evaluate_ravrs_candidate(
                method_id=method_id,
                method_name=method_name,
                candidate=cand,
                state_buckets=state_buckets,
                provider=llm_provider,
            )
            method_score = _method_score(ev, hard_state=hard_state)
            human_proxy = _human_pref_proxy(ev, cand)
            state_score = _ravrs_state_controller_score(ev, cand, method_id, hard_state)
            # Protocol-constrained state optimization objective.
            if hard_state:
                s = round(0.80 * method_score + 0.15 * state_score + 0.05 * human_proxy, 3)
            else:
                s = round(0.65 * method_score + 0.20 * state_score + 0.15 * human_proxy, 3)
            # Hard filter: reject unsafe outputs.
            if ev["safety"] == 0:
                s -= 100.0
            candidates.append((s, cand, ev, method_score, human_proxy))
        # Stage 1: protocol shortlist (first strong, then valid).
        protocol_strong = [x for x in candidates if _protocol_strong_candidate(x[2], hard_state)]
        protocol_valid = [x for x in candidates if _protocol_valid_candidate(x[2])]

        if protocol_strong:
            source = protocol_strong
            source_type = "strong"
        elif protocol_valid:
            if hard_state:
                # In hard states we do not relax to weak-valid candidates:
                # keep strict protocol floor and fall back to method-specific safe response.
                source = []
                source_type = "hard_fallback"
            else:
                source = protocol_valid
                source_type = "valid"
        else:
            # No protocol-valid candidates: fallback to best by method score.
            source = []
            source_type = "method_fallback"

        if not source:
            best_text = fallback_candidate
            best_score = -1.0
            rationale = "RAVR-S v8: hard protocol floor not met, method-specific fallback applied."
            return RepairSuggestion(
                should_repair=bool(violations),
                repaired_message=best_text,
                rationale=rationale,
                target_constraints=violations[:4],
                ravrs_enabled=True,
                ravrs_score=round(best_score, 3),
                ravrs_state=state_buckets,
            )

        source.sort(key=lambda x: x[0], reverse=True)
        # Pareto-like shortlist: top method candidates then state/human selection.
        shortlist = source[:2] if len(source) >= 2 else source
        if (not hard_state) and len(shortlist) >= 2:
            a, b = shortlist[0], shortlist[1]
            winner = _pairwise_human_pref_judge(
                method_name=method_name,
                state_buckets=state_buckets,
                cand_a=a[1],
                cand_b=b[1],
                provider=llm_provider,
            )
            chosen = a if winner == "A" else (b if winner == "B" else a)
        else:
            chosen = shortlist[0]

        best_score, best_text, best_ev, _, _ = chosen
        if hard_state and len(shortlist) >= 2:
            r1 = _mini_rollout_score(candidate=shortlist[0][1], interaction_state=interaction_state, provider=llm_provider)
            r2 = _mini_rollout_score(candidate=shortlist[1][1], interaction_state=interaction_state, provider=llm_provider)
            if r2 > r1:
                best_score, best_text, best_ev, _, _ = shortlist[1]
        if not hard_state:
            best_text = _constrained_micro_refine(response=best_text, provider=llm_provider)

        if best_ev.get("too_vague", 0) == 1 or best_score < RAVRS_MIN_SCORE:
            best_text = fallback_candidate
            rationale = _l(
                "RAVR-S выбрал method-specific fallback из-за несоответствия кандидатов методологии/безопасности.",
                "RAVR-S selected the methodology-specific fallback because the candidates did not meet methodology or safety requirements.",
            )
        else:
            rationale = (
                f"RAVR-S v2: protocol-first ({source_type}) + state-aware rerank (0.65 method / 0.20 state / 0.15 human)."
            )
        return RepairSuggestion(
            should_repair=bool(violations),
            repaired_message=best_text,
            rationale=rationale,
            target_constraints=violations[:4],
            ravrs_enabled=True,
            ravrs_score=round(best_score, 3),
            ravrs_state=state_buckets,
        )
    return RepairSuggestion(
        should_repair=bool(violations),
        repaired_message=repaired if violations else "",
        rationale=rationale if violations else "",
        target_constraints=violations[:4],
        ravrs_enabled=False,
        ravrs_state=state_buckets if interaction_state else {},
    )


def build_methodology_proof(
    case_id: str,
    user_message: str,
    ev: Optional[TurnEvaluation],
    *,
    interaction_state: Optional[Dict[str, int]] = None,
    llm_provider: Optional[str] = None,
    enable_retrieval: Optional[bool] = None,
    require_valid_citations: Optional[bool] = None,
    enable_repair: Optional[bool] = None,
) -> MethodologyProof:
    method = get_methodology_for_case(case_id)
    method_id = method["id"]
    rules = METHOD_RULES.get(method_id, {"must_do": [], "must_avoid": [], "recommended": []})
    msg = (user_message or "").lower()
    use_retrieval = RAVR_ENABLE_RETRIEVAL if enable_retrieval is None else bool(enable_retrieval)
    need_valid_citations = (
        RAVR_REQUIRE_VALID_CITATIONS
        if require_valid_citations is None
        else bool(require_valid_citations)
    )
    use_repair = RAVR_ENABLE_REPAIR if enable_repair is None else bool(enable_repair)

    satisfied: List[str] = []
    violations: List[str] = []
    evidence: List[str] = []

    has_question = "?" in user_message
    has_reflection = any(k in msg for k in [
        "чувству", "пережива", "похоже", "кажется", "заметил",
        "feel", "experience", "it sounds", "it seems", "notice",
    ])
    has_structure = any(k in msg for k in [
        "когда", "в какие моменты", "что было до", "что было после", "шаг",
        "when", "which moments", "before", "after", "step",
    ])
    has_directive = any(k in msg for k in [
        "вы должны", "тебе нужно", "просто сделай", "немедленно",
        "you must", "you should", "you need to", "just do", "immediately",
    ])

    if has_question:
        satisfied.append(_l("использован исследующий вопрос", "exploratory question used"))
        evidence.append(_l("Реплика содержит вопросительную форму", "The utterance contains a question"))
    if has_reflection:
        satisfied.append(_l("присутствует отражение/валидация", "reflection or validation present"))
        evidence.append(_l("Реплика содержит маркеры валидации переживаний", "The utterance contains markers of validation"))
    if has_structure:
        satisfied.append(_l("присутствует структурирование интервенции", "intervention structure present"))
        evidence.append(_l("Реплика содержит процессные/структурные маркеры", "The utterance contains process or structural markers"))
    if has_directive:
        violations.append(_l("директивная подача без достаточной проверки контекста", "directive framing without sufficient context checking"))
        evidence.append(_l("Найдены директивные формулировки", "Directive wording detected"))

    if ev is not None:
        if ev.open_question >= 0.5:
            satisfied.append(_l("достаточный уровень открытых вопросов", "sufficient use of open questions"))
        else:
            violations.append(_l("недостаток открытых вопросов", "insufficient use of open questions"))
        if ev.safety < 0.85:
            violations.append(_l("риск снижения безопасности коммуникации", "risk of reduced conversational safety"))
        if ev.empathy < 0.4:
            violations.append(_l("низкая эмпатическая точность", "low empathic accuracy"))
        if ev.directivity > 0.7:
            violations.append(_l("избыточная директивность", "excessive directivity"))

    # Метод-специфичные мягкие сигналы
    if method_id == "cbt":
        cbt_thought_markers = ["мысл", "убежден", "доказательств", "искажен", "thought", "belief", "evidence", "distortion"]
        cbt_process_markers = ["триггер", "эпизод", "до", "после", "паник", "ощущ", "trigger", "episode", "before", "after", "panic", "sensation"]
        cbt_step_markers = ["шаг", "план", "эксперимент", "попроб", "следующ", "step", "plan", "experiment", "try", "next"]
        if any(k in msg for k in cbt_thought_markers):
            satisfied.append(_l("есть элементы когнитивной реструктуризации", "cognitive restructuring elements present"))
        elif infer_category(case_id) == "panic" and (
            any(k in msg for k in cbt_process_markers) or any(k in msg for k in cbt_step_markers)
        ):
            satisfied.append(_l("для паники добавлен CBT-фокус на триггерах/поведенческом шаге", "CBT focus on panic triggers or a behavioral step present"))
        else:
            violations.append(_l("нет явного фокуса на автоматических мыслях/искажениях", "no explicit focus on automatic thoughts or distortions"))
    elif method_id == "dbt":
        # Broader DBT stabilization markers to avoid false negatives on valid grounding language.
        dbt_markers = [
            "принят",
            "регуляц",
            "навык",
            "зазем",
            "стабилиз",
            "дышан",
            "теле",
            "дистресс",
            "успоко",
            "accept",
            "regulat",
            "skill",
            "ground",
            "stabil",
            "breath",
            "body",
            "distress",
            "calm",
        ]
        if any(k in msg for k in dbt_markers):
            satisfied.append(_l("есть DBT-компонент принятия/регуляции", "DBT acceptance or regulation component present"))
        else:
            violations.append(_l("слабый фокус на навыках DBT-регуляции", "insufficient focus on DBT regulation skills"))

        # PTSD grounding rule: trauma-related DBT cases should include grounding/stabilization cue.
        if infer_category(case_id) == "ptsd":
            ptsd_grounding_markers = [
                "зазем",
                "стабилиз",
                "дышан",
                "безопас",
                "опора",
                "теле",
                "здесь и сейчас",
                "дистресс",
                "триггер",
                "ground",
                "stabil",
                "breath",
                "safe",
                "support",
                "body",
                "here and now",
                "distress",
                "trigger",
            ]
            if any(k in msg for k in ptsd_grounding_markers):
                satisfied.append(_l("для PTSD добавлен grounding/стабилизация", "grounding or stabilization included for PTSD"))
            else:
                violations.append(_l("для PTSD не добавлен grounding/стабилизация", "grounding or stabilization missing for PTSD"))
    elif method_id == "emdr":
        if any(k in msg for k in ["воспомин", "триггер", "дистресс", "безопас", "memory", "trigger", "distress", "safe"]):
            satisfied.append(_l("учтены EMDR-элементы триггеров/безопасности", "EMDR trigger or safety elements included"))
        else:
            violations.append(_l("нет явной EMDR-фокусировки на памяти/дистрессе", "no explicit EMDR focus on memory or distress"))
    elif method_id == "aba":
        if any(k in msg for k in ["поведен", "до этого", "после", "подкреп", "behavior", "before", "after", "reinforc", "antecedent", "consequence"]):
            satisfied.append(_l("учтён поведенческий анализ (ABA)", "behavioral analysis included (ABA)"))
        else:
            violations.append(_l("нет операционализации поведения в терминах ABA", "behavior is not operationalized in ABA terms"))

    adherence = 70.0 + 5.0 * len(satisfied) - 8.0 * len(violations)
    adherence = max(0.0, min(100.0, adherence))
    recs = list(rules.get("recommended", []))[:3]
    if not recs:
        recs = (
            ["Clarify the intervention goal", "Add validation", "Reduce directivity"]
            if IS_ENGLISH
            else ["Уточнить цель интервенции", "Добавить валидацию", "Снизить директивность"]
        )

    chunks: List[RavrChunk] = build_ravr_chunks(case_id, ev) if use_retrieval else []
    citations = _pick_citations(method_id, violations, satisfied, chunks) if chunks else []
    citation_valid = _citations_valid(citations, chunks) if chunks else (not need_valid_citations)
    citation_coverage, citation_precision, citation_relevance = _citation_quality(violations, citations, chunks)
    if need_valid_citations and not citation_valid:
        violations.append(_l("некорректные или отсутствующие RAVR-цитаты", "invalid or missing RAVR citations"))
    if not use_retrieval:
        violations.append(_l("отсутствует retrieval-grounding для обоснования интервенции", "retrieval grounding is missing for the intervention rationale"))
        adherence = max(0.0, adherence - 6.0)

    repair_targets = list(violations)
    if use_repair and RAVRS_ENABLE and RAVRS_ALWAYS_REPAIR and not repair_targets:
        repair_targets = ["ravrs_always_repair"]

    repair = (
        _build_repaired_message(
            method_id,
            user_message,
            repair_targets,
            interaction_state=interaction_state,
            llm_provider=llm_provider,
        )
        if use_repair
        else RepairSuggestion()
    )
    repair.citations = citations

    return MethodologyProof(
        methodology_id=method_id,
        methodology_name=method["name"],
        adherence_score=round(adherence, 2),
        satisfied_constraints=satisfied,
        violated_constraints=violations,
        evidence=evidence[:5],
        recommendations=recs,
        retrieved_chunks=chunks[:10] if use_retrieval else [],
        citations=citations,
        citation_valid=citation_valid,
        citation_coverage=citation_coverage,
        citation_precision=citation_precision,
        citation_relevance=citation_relevance,
        repair_suggestion=repair,
    )


def detect_mistake_reason(ev: TurnEvaluation) -> Optional[str]:
    reasons: List[str] = []
    if ev.safety < 0.85:
        reasons.append(_l("формулировка снижает ощущение безопасности", "the wording reduces the sense of safety"))
    if ev.empathy < 0.4:
        reasons.append(_l("мало эмпатии и отражения чувств", "insufficient empathy and reflection of feelings"))
    if ev.open_question < 0.3:
        reasons.append(_l("не хватает открытых вопросов", "insufficient use of open questions"))
    if ev.directivity > 0.7:
        reasons.append(_l("слишком директивная подача", "the framing is too directive"))
    if ev.efficiency_index < 0:
        reasons.append(_l("ход снижает общую эффективность контакта", "the turn reduces the overall effectiveness of rapport"))
    return "; ".join(reasons) if reasons else None


def collect_mistaken_replicas(session: Dict[str, Any]) -> List[Dict[str, Any]]:
    persisted = session.get("mistakes", [])
    if isinstance(persisted, list) and persisted:
        return persisted[-8:]

    user_turns = [h.get("content", "") for h in session.get("history", []) if h.get("role") == "user"]
    evals = session.get("evals", [])
    collected: List[Dict[str, Any]] = []
    for idx, ev_raw in enumerate(evals):
        if idx >= len(user_turns):
            break
        ev = TurnEvaluation(**ev_raw)
        reason = detect_mistake_reason(ev)
        if not reason:
            continue
        score = round((1.0 - ev.safety) + (1.0 - ev.empathy) + ev.directivity + max(0.0, -ev.efficiency_index), 3)
        collected.append(
            {
                "student_message": _truncate_text(str(user_turns[idx]), 260),
                "reason": reason,
                "score": score,
            }
        )
    collected.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
    return collected[:8]


def fallback_improved_examples(mistakes: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    examples: List[Dict[str, str]] = []
    for m in mistakes[:3]:
        original = str(m.get("student_message", "")).strip()
        if not original:
            continue
        better = _l(
            "Похоже, вам сейчас непросто. Что для вас в этой ситуации самое тяжёлое в данный момент?",
            "It sounds like things are difficult right now. What feels hardest about this situation at the moment?",
        )
        why = str(m.get("reason", "")).strip() or _l(
            "Так формулировка звучит мягче и поддерживает контакт.",
            "This wording is gentler and better supports rapport.",
        )
        examples.append(
            {
                "original_replica": original,
                "better_replica": better,
                "why_better": why,
            }
        )
    return examples


def generate_session_feedback_with_llm(
    *,
    avg_empathy: float,
    avg_validation: float,
    avg_directivity: float,
    avg_open: float,
    avg_safety: float,
    mean_eff: float,
    total_dt: int,
    total_de: int,
    total_df: int,
    num_turns: int,
    turn_comments: List[str],
    trends: Dict[str, float],
    history_sample: List[Dict[str, str]],
    mistaken_replicas: List[Dict[str, Any]],
    provider: Optional[str] = None,
) -> Tuple[str, str, List[Dict[str, str]]]:
    metrics_payload = {
        "num_turns": num_turns,
        "averages": {
            "empathy": round(avg_empathy, 2),
            "validation": round(avg_validation, 2),
            "directivity": round(avg_directivity, 2),
            "open_question": round(avg_open, 2),
            "safety": round(avg_safety, 2),
            "efficiency_index": round(mean_eff, 2),
        },
        "totals": {
            "delta_trust": int(total_dt),
            "delta_emotional_intensity": int(total_de),
            "delta_fatigue": int(total_df),
        },
        "trends": trends,
        "turn_comments_sample": turn_comments[:10],
        "dialogue_sample": history_sample,
        "mistaken_replicas": mistaken_replicas[:6],
    }

    raw = llm_chat_completions(
        provider=provider,
        messages=[
            {"role": "system", "content": SESSION_REPORT_PROMPT},
            {"role": "user", "content": json.dumps(metrics_payload, ensure_ascii=False)},
        ],
        temperature=0.4,
        max_tokens=500,
    ).strip()

    data = _extract_json_object(raw)
    overall = str(data.get("overall_impression", "")).strip()
    recs = str(data.get("recommendations", "")).strip()
    examples_raw = data.get("improved_examples", [])

    if not overall or not recs:
        raise RuntimeError("LLM returned empty report fields")

    examples: List[Dict[str, str]] = []
    if isinstance(examples_raw, list):
        for item in examples_raw:
            if not isinstance(item, dict):
                continue
            original = str(item.get("original_replica", "")).strip()
            better = str(item.get("better_replica", "")).strip()
            why = str(item.get("why_better", "")).strip()
            if not original or not better or not why:
                continue
            examples.append(
                {
                    "original_replica": original,
                    "better_replica": better,
                    "why_better": why,
                }
            )
            if len(examples) >= 4:
                break

    return overall, recs, examples


def _calc_trend(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    return round((values[-1] - values[0]) / (len(values) - 1), 3)


def build_session_progress(session_id: str, session: Dict[str, Any]) -> SessionProgress:
    evals = session.get("evals", [])
    points: List[ProgressPoint] = []
    state = get_initial_state()

    empathy_series: List[float] = []
    validation_series: List[float] = []
    directivity_series: List[float] = []
    open_series: List[float] = []
    safety_series: List[float] = []
    eff_series: List[float] = []

    for idx, ev_raw in enumerate(evals, start=1):
        ev = TurnEvaluation(**ev_raw)
        state = apply_state_delta(dict(state), ev)

        empathy_series.append(ev.empathy)
        validation_series.append(ev.validation)
        directivity_series.append(ev.directivity)
        open_series.append(ev.open_question)
        safety_series.append(ev.safety)
        eff_series.append(ev.efficiency_index)

        points.append(
            ProgressPoint(
                turn_index=idx,
                empathy=ev.empathy,
                validation=ev.validation,
                directivity=ev.directivity,
                open_question=ev.open_question,
                safety=ev.safety,
                efficiency_index=ev.efficiency_index,
                trust_level=state["trust_level"],
                emotional_intensity=state["emotional_intensity"],
                fatigue=state["fatigue"],
            )
        )

    trends = {
        "empathy": _calc_trend(empathy_series),
        "validation": _calc_trend(validation_series),
        "directivity": _calc_trend(directivity_series),
        "open_question": _calc_trend(open_series),
        "safety": _calc_trend(safety_series),
        "efficiency_index": _calc_trend(eff_series),
    }

    return SessionProgress(
        session_id=session_id,
        case_id=session["case_id"],
        num_turns=len(evals),
        current_state=session.get("state", get_initial_state()),
        trends=trends,
        points=points,
    )


def build_ravr_metrics(method_proofs: List[Dict[str, Any]]) -> RavrMetrics:
    total = len(method_proofs)
    if total == 0:
        return RavrMetrics(
            turns_total=0,
            verifier_pass_rate=0.0,
            citation_valid_rate=0.0,
            citation_coverage_rate=0.0,
            citation_precision_rate=0.0,
            citation_relevance_score=0.0,
            repair_trigger_rate=0.0,
            repair_success_rate=0.0,
            avg_adherence_score=0.0,
            top_violations=[],
        )

    parsed_total = 0
    pass_count = 0
    citation_valid_count = 0
    citation_coverage_sum = 0.0
    citation_precision_sum = 0.0
    citation_relevance_sum = 0.0
    repair_trigger_count = 0
    repair_success_count = 0
    adherence_sum = 0.0
    violations_freq: Dict[str, int] = {}

    for raw in method_proofs:
        try:
            proof = MethodologyProof(**raw)
        except Exception:
            continue

        parsed_total += 1
        adherence_sum += float(proof.adherence_score)
        if not proof.violated_constraints:
            pass_count += 1
        if bool(proof.citation_valid):
            citation_valid_count += 1
        citation_coverage_sum += float(getattr(proof, "citation_coverage", 0.0) or 0.0)
        citation_precision_sum += float(getattr(proof, "citation_precision", 0.0) or 0.0)
        citation_relevance_sum += float(getattr(proof, "citation_relevance", 0.0) or 0.0)
        if proof.repair_suggestion and proof.repair_suggestion.should_repair:
            repair_trigger_count += 1
            if proof.repair_suggestion.repaired_verifier_pass:
                repair_success_count += 1
        for v in proof.violated_constraints:
            key = str(v).strip()
            if not key:
                continue
            violations_freq[key] = violations_freq.get(key, 0) + 1

    effective_total = max(1, parsed_total)
    top_violations = sorted(
        violations_freq.keys(),
        key=lambda k: violations_freq[k],
        reverse=True,
    )[:5]

    return RavrMetrics(
        turns_total=total,
        verifier_pass_rate=round(pass_count / effective_total, 3),
        citation_valid_rate=round(citation_valid_count / effective_total, 3),
        citation_coverage_rate=round(citation_coverage_sum / effective_total, 3),
        citation_precision_rate=round(citation_precision_sum / effective_total, 3),
        citation_relevance_score=round(citation_relevance_sum / effective_total, 3),
        repair_trigger_rate=round(repair_trigger_count / effective_total, 3),
        repair_success_rate=round(repair_success_count / max(1, repair_trigger_count), 3),
        avg_adherence_score=round(adherence_sum / effective_total, 2),
        top_violations=top_violations,
    )


def _ravr_metrics_row(scope: str, session_id: str, case_id: str, metrics: RavrMetrics) -> Dict[str, Any]:
    return {
        "scope": scope,
        "session_id": session_id,
        "case_id": case_id,
        "turns_total": metrics.turns_total,
        "verifier_pass_rate": metrics.verifier_pass_rate,
        "citation_valid_rate": metrics.citation_valid_rate,
        "citation_coverage_rate": metrics.citation_coverage_rate,
        "citation_precision_rate": metrics.citation_precision_rate,
        "citation_relevance_score": metrics.citation_relevance_score,
        "repair_trigger_rate": metrics.repair_trigger_rate,
        "repair_success_rate": metrics.repair_success_rate,
        "avg_adherence_score": metrics.avg_adherence_score,
        "top_violations": " | ".join(metrics.top_violations),
    }


def _session_turn_dataset_rows(session_id: str, session: Dict[str, Any]) -> List[Dict[str, Any]]:
    case_id = str(session.get("case_id", ""))
    history = session.get("history", [])
    evals = session.get("evals", [])
    proofs = session.get("method_proofs", [])

    user_turns = [h for h in history if str(h.get("role", "")) == "user"]
    assistant_turns = [h for h in history if str(h.get("role", "")) == "assistant"]
    n = min(len(user_turns), len(evals), len(proofs))
    rows: List[Dict[str, Any]] = []

    for idx in range(n):
        user_text = str(user_turns[idx].get("content", ""))
        assistant_text = str(assistant_turns[idx].get("content", "")) if idx < len(assistant_turns) else ""
        ev_raw = evals[idx] if isinstance(evals[idx], dict) else {}
        proof_raw = proofs[idx] if isinstance(proofs[idx], dict) else {}

        try:
            proof_obj = MethodologyProof(**proof_raw)
            proof_dump = proof_obj.model_dump()
        except Exception:
            proof_dump = proof_raw

        rows.append(
            {
                "scope": "turn",
                "session_id": session_id,
                "case_id": case_id,
                "turn_index": idx + 1,
                "student_message": user_text,
                "assistant_message": assistant_text,
                "evaluation": ev_raw,
                "proof_object": proof_dump,
                "verifier_pass": len(proof_dump.get("violated_constraints", [])) == 0 if isinstance(proof_dump, dict) else False,
                "config": {
                    "ravr_enable_retrieval": RAVR_ENABLE_RETRIEVAL,
                    "ravr_require_valid_citations": RAVR_REQUIRE_VALID_CITATIONS,
                    "ravr_enable_repair": RAVR_ENABLE_REPAIR,
                    "ravrs_enable": RAVRS_ENABLE,
                },
            }
        )
    return rows


def _classify_violation(v: str) -> str:
    t = str(v).lower()
    if "директив" in t or "directiv" in t:
        return "directivity"
    if "эмпат" in t or "валидац" in t or "empath" in t or "validat" in t:
        return "empathy_validation"
    if "открыт" in t or "open question" in t:
        return "open_question"
    if "безопас" in t or "safety" in t or "unsafe" in t:
        return "safety"
    if "цитат" in t or "citation" in t:
        return "citation"
    if "когнитив" in t or "мысл" in t or "cognit" in t or "thought" in t:
        return "cbt_focus"
    if "dbt" in t or "регуляц" in t or "regulat" in t or "ground" in t:
        return "dbt_focus"
    if "emdr" in t or "дистресс" in t or "памяти" in t or "distress" in t or "memory" in t:
        return "emdr_focus"
    if "aba" in t or "поведен" in t or "behavior" in t or "antecedent" in t:
        return "aba_focus"
    return "other"


def run_ravr_benchmark(req: RavrBenchmarkRequest) -> RavrBenchmarkResponse:
    rng = random.Random(req.random_seed)
    effective_enable_retrieval = (
        RAVR_ENABLE_RETRIEVAL if req.override_enable_retrieval is None else bool(req.override_enable_retrieval)
    )
    effective_require_valid_citations = (
        RAVR_REQUIRE_VALID_CITATIONS
        if req.override_require_valid_citations is None
        else bool(req.override_require_valid_citations)
    )
    effective_enable_repair = (
        RAVR_ENABLE_REPAIR if req.override_enable_repair is None else bool(req.override_enable_repair)
    )
    effective_llm_provider = (req.override_llm_provider or LLM_PROVIDER or "gigachat").strip().lower()

    selected_cases = CASES_DATA
    if req.case_ids:
        allowed = {normalize_case_id(x) for x in req.case_ids}
        selected_cases = [c for c in CASES_DATA if c["id"] in allowed]
    if not selected_cases:
        raise HTTPException(status_code=400, detail="No valid cases selected for benchmark")

    all_rows: List[RavrBenchmarkRow] = []
    by_case_raw: Dict[str, List[Dict[str, Any]]] = {}
    eval_cache: Dict[Tuple[str, str], TurnEvaluation] = {}

    for case in selected_cases:
        case_id = str(case["id"])
        category_key = str(case["category_key"])
        method = get_methodology_for_case(case_id)
        sampled_prompts = [rng.choice(BENCHMARK_UTTERANCES) for _ in range(req.n_per_case)]
        case_proofs: List[Dict[str, Any]] = []

        for prompt in sampled_prompts:
            ev = None
            if req.include_llm_eval:
                if req.disable_eval_cache:
                    ev = evaluate_therapist_message(
                        prompt,
                        get_initial_state(),
                        provider=effective_llm_provider,
                        temperature=req.llm_temperature,
                        top_p=req.llm_top_p,
                    )
                else:
                    ekey = (effective_llm_provider, prompt)
                    if ekey in eval_cache:
                        ev = eval_cache[ekey]
                    else:
                        ev = evaluate_therapist_message(
                            prompt,
                            get_initial_state(),
                            provider=effective_llm_provider,
                            temperature=req.llm_temperature,
                            top_p=req.llm_top_p,
                        )
                        eval_cache[ekey] = ev
            proof = build_methodology_proof(
                case_id,
                prompt,
                ev,
                interaction_state=get_initial_state(),
                llm_provider=effective_llm_provider,
                enable_retrieval=effective_enable_retrieval,
                require_valid_citations=effective_require_valid_citations,
                enable_repair=effective_enable_repair,
            )
            verifier_pass = len(proof.violated_constraints) == 0
            adherence_before = float(proof.adherence_score)
            adherence_after: Optional[float] = None
            adherence_delta: Optional[float] = None
            repaired_violations: List[str] = []
            if effective_enable_repair and (not verifier_pass) and proof.repair_suggestion and proof.repair_suggestion.should_repair:
                repaired_proof = build_methodology_proof(
                    case_id,
                    proof.repair_suggestion.repaired_message,
                    ev,
                    interaction_state=get_initial_state(),
                    llm_provider=effective_llm_provider,
                    enable_retrieval=effective_enable_retrieval,
                    require_valid_citations=effective_require_valid_citations,
                    enable_repair=effective_enable_repair,
                )
                proof.repair_suggestion.repaired_verifier_pass = len(repaired_proof.violated_constraints) == 0
                proof.repair_suggestion.repaired_violations = repaired_proof.violated_constraints
                repaired_violations = list(repaired_proof.violated_constraints)
                if not proof.repair_suggestion.citations:
                    proof.repair_suggestion.citations = repaired_proof.citations
                adherence_after = float(repaired_proof.adherence_score)
                adherence_delta = round(adherence_after - adherence_before, 3)
            repair_triggered = bool(proof.repair_suggestion and proof.repair_suggestion.should_repair)
            repair_success = bool(
                proof.repair_suggestion and proof.repair_suggestion.repaired_verifier_pass
            )
            violation_types = sorted({_classify_violation(v) for v in proof.violated_constraints})

            all_rows.append(
                RavrBenchmarkRow(
                    case_id=case_id,
                    category_key=category_key,
                    methodology_id=method["id"],
                    prompt=prompt,
                    verifier_pass=verifier_pass,
                    citation_valid=bool(proof.citation_valid),
                    citation_coverage=float(proof.citation_coverage),
                    citation_precision=float(proof.citation_precision),
                    citation_relevance=float(proof.citation_relevance),
                    adherence_score=float(proof.adherence_score),
                    adherence_before=adherence_before,
                    adherence_after=adherence_after,
                    adherence_delta=adherence_delta,
                    violations=proof.violated_constraints,
                    repaired_violations=repaired_violations,
                    violation_types=violation_types,
                    repair_triggered=repair_triggered,
                    repair_success=repair_success,
                )
            )
            case_proofs.append(proof.model_dump())

        by_case_raw[case_id] = case_proofs

    merged_proofs: List[Dict[str, Any]] = []
    by_case_metrics: Dict[str, RavrMetrics] = {}
    for case_id, proofs in by_case_raw.items():
        metrics = build_ravr_metrics(proofs)
        by_case_metrics[case_id] = metrics
        merged_proofs.extend(proofs)

    summary_metrics = build_ravr_metrics(merged_proofs)
    summary = RavrBenchmarkSummary(
        cases_total=len(selected_cases),
        turns_total=len(merged_proofs),
        metrics=summary_metrics,
    )

    return RavrBenchmarkResponse(
        summary=summary,
        by_case=by_case_metrics,
        rows=all_rows,
        config={
            "random_seed": req.random_seed,
            "n_per_case": req.n_per_case,
            "include_llm_eval": req.include_llm_eval,
            "llm_temperature": req.llm_temperature,
            "llm_top_p": req.llm_top_p,
            "disable_eval_cache": req.disable_eval_cache,
            "llm_provider": effective_llm_provider,
            "ravr_enable_retrieval": effective_enable_retrieval,
            "ravr_require_valid_citations": effective_require_valid_citations,
            "ravr_enable_repair": effective_enable_repair,
            "ravrs_enable": RAVRS_ENABLE,
        },
    )


def _benchmark_row_to_dict(row: RavrBenchmarkRow) -> Dict[str, Any]:
    return {
        "case_id": row.case_id,
        "category_key": row.category_key,
        "methodology_id": row.methodology_id,
        "prompt": row.prompt,
        "verifier_pass": row.verifier_pass,
        "citation_valid": row.citation_valid,
        "citation_coverage": row.citation_coverage,
        "citation_precision": row.citation_precision,
        "citation_relevance": row.citation_relevance,
        "adherence_score": row.adherence_score,
        "adherence_before": row.adherence_before,
        "adherence_after": row.adherence_after if row.adherence_after is not None else "",
        "adherence_delta": row.adherence_delta if row.adherence_delta is not None else "",
        "violations": " | ".join(row.violations),
        "repaired_violations": " | ".join(row.repaired_violations),
        "violation_types": " | ".join(row.violation_types),
        "repair_triggered": row.repair_triggered,
        "repair_success": row.repair_success,
    }


# ============================================================
#                      FastAPI app
# ============================================================

app = FastAPI(title="Virtual Patient Simulator (Teacher Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials="*" not in CORS_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_sessions_db()
migrate_legacy_sessions_json_if_needed()
load_sessions_from_db()


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/cases", response_model=List[CasePublic])
def list_cases():
    result: List[CasePublic] = []
    for c in CASES_DATA:
        method = METHODOLOGY_BY_CATEGORY.get(c["category_key"], {"id": "general", "name": "General"})
        result.append(
            CasePublic(
                id=c["id"],
                category_key=c["category_key"],
                category_name=c["category_name"],
                methodology_id=method["id"],
                methodology_name=method["name"],
                title_for_teacher=c["title_for_teacher"],
                visible_to_student=c["visible_to_student"],
            )
        )
    return result


@app.get("/api/cases/{case_id}/teacher", response_model=CaseTeacher)
def get_case_teacher(case_id: str):
    case_id = normalize_case_id(case_id)
    case = CASES_BY_ID.get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    return CaseTeacher(
        id=case["id"],
        category_key=case["category_key"],
        category_name=case["category_name"],
        title_for_teacher=case["title_for_teacher"],
        visible_to_student=case.get("visible_to_student", {}),
        hidden_for_student=case.get("hidden_for_student", {}),
        symptom_profile=case.get("symptom_profile", {}),
        personality_style=case.get("personality_style", {}),
        typical_phrases=case.get("typical_phrases", []),
        triggers=case.get("triggers", []),
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    # normalize case_id like ras_001 -> ras_01
    req.case_id = normalize_case_id(req.case_id)

    if req.case_id not in CASES_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown case_id")

    if req.session_id not in sessions:
        sessions[req.session_id] = {
            "case_id": req.case_id,
            "history": [],
            "state": get_initial_state(),
            "evals": [],
            "mistakes": [],
            "method_proofs": [],
        }

    session = sessions[req.session_id]

    if session["case_id"] != req.case_id:
        session["case_id"] = req.case_id
        session["history"] = []
        session["state"] = get_initial_state()
        session["evals"] = []
        session["mistakes"] = []
        session["method_proofs"] = []

    # Оценка хода и обновление состояния
    evaluation = evaluate_therapist_message(req.user_message, session["state"], provider=req.llm_provider)
    session["state"] = apply_state_delta(session["state"], evaluation)
    session["evals"].append(evaluation.model_dump())
    proof = build_methodology_proof(
        req.case_id,
        req.user_message,
        evaluation,
        interaction_state=session["state"],
        llm_provider=req.llm_provider,
    )
    verifier_pass = len(proof.violated_constraints) == 0
    if RAVR_ENABLE_REPAIR and (not verifier_pass) and proof.repair_suggestion and proof.repair_suggestion.should_repair:
        repaired_proof = build_methodology_proof(
            req.case_id,
            proof.repair_suggestion.repaired_message,
            evaluation,
            interaction_state=session["state"],
            llm_provider=req.llm_provider,
        )
        proof.repair_suggestion.repaired_verifier_pass = len(repaired_proof.violated_constraints) == 0
        proof.repair_suggestion.repaired_violations = repaired_proof.violated_constraints
        if not proof.repair_suggestion.citations:
            proof.repair_suggestion.citations = repaired_proof.citations

    session.setdefault("method_proofs", []).append(proof.model_dump())
    if len(session["method_proofs"]) > 100:
        session["method_proofs"] = session["method_proofs"][-100:]
    reason = detect_mistake_reason(evaluation)
    if reason:
        score = round((1.0 - evaluation.safety) + (1.0 - evaluation.empathy) + evaluation.directivity + max(0.0, -evaluation.efficiency_index), 3)
        mistakes = session.setdefault("mistakes", [])
        mistakes.append(
            {
                "student_message": _truncate_text(req.user_message, 260),
                "reason": reason,
                "score": score,
            }
        )
        # Оставляем только последние 30 записей
        if len(mistakes) > 30:
            session["mistakes"] = mistakes[-30:]

    # Добавляем ход психолога
    session["history"].append({"role": "user", "content": req.user_message})

    case_profile = CASES_BY_ID[session["case_id"]]
    messages = build_messages(case_profile, session["state"], session["history"])

    try:
        assistant_text = call_llm_chat(messages, provider=req.llm_provider)
    except Exception as e:
        # Return 502 so фронт мог отличать "внешний провайдер упал"
        save_session_to_db(req.session_id, session)
        raise HTTPException(status_code=502, detail=str(e))

    # Ответ пациента
    session["history"].append({"role": "assistant", "content": assistant_text})
    save_session_to_db(req.session_id, session)

    return ChatResponse(
        session_id=req.session_id,
        case_id=req.case_id,
        assistant_message=assistant_text,
        evaluation=evaluation if req.teacher_mode else None,
        proof_object=proof if req.teacher_mode else None,
        verifier_pass=verifier_pass,
        verifier_violations=proof.violated_constraints if req.teacher_mode else [],
    )


@app.get("/api/sessions/{session_id}", response_model=SessionDetail)
def get_session_detail(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return SessionDetail(
        session_id=session_id,
        case_id=session["case_id"],
        state=session["state"],
        history=session["history"],
        evals=session["evals"],
        mistakes=session.get("mistakes", []),
        ravr_summary=build_ravr_metrics(session.get("method_proofs", [])),
    )


@app.get("/api/sessions/{session_id}/progress", response_model=SessionProgress)
def get_session_progress(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return build_session_progress(session_id, session)


@app.get("/api/ravr_metrics", response_model=RavrMetricsResponse)
def get_ravr_metrics(session_id: Optional[str] = Query(default=None, description="Опционально: ID сессии")):
    config = {
        "ravr_enable_retrieval": RAVR_ENABLE_RETRIEVAL,
        "ravr_require_valid_citations": RAVR_REQUIRE_VALID_CITATIONS,
        "ravr_enable_repair": RAVR_ENABLE_REPAIR,
        "ravrs_enable": RAVRS_ENABLE,
    }
    if session_id:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        metrics = build_ravr_metrics(session.get("method_proofs", []))
        return RavrMetricsResponse(
            scope="session",
            session_id=session_id,
            sessions_total=1,
            metrics=metrics,
            config=config,
        )

    merged: List[Dict[str, Any]] = []
    for s in sessions.values():
        merged.extend(s.get("method_proofs", []))
    metrics = build_ravr_metrics(merged)
    return RavrMetricsResponse(
        scope="global",
        session_id=None,
        sessions_total=len(sessions),
        metrics=metrics,
        config=config,
    )


@app.get("/api/ravr_metrics.csv")
def get_ravr_metrics_csv(session_id: Optional[str] = Query(default=None, description="Опционально: ID сессии")):
    fieldnames = [
        "scope",
        "session_id",
        "case_id",
        "turns_total",
        "verifier_pass_rate",
        "citation_valid_rate",
        "citation_coverage_rate",
        "citation_precision_rate",
        "citation_relevance_score",
        "repair_trigger_rate",
        "repair_success_rate",
        "avg_adherence_score",
        "top_violations",
    ]
    rows: List[Dict[str, Any]] = []

    if session_id:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        metrics = build_ravr_metrics(session.get("method_proofs", []))
        rows.append(_ravr_metrics_row("session", session_id, str(session.get("case_id", "")), metrics))
    else:
        merged: List[Dict[str, Any]] = []
        for sid, s in sessions.items():
            per_session_metrics = build_ravr_metrics(s.get("method_proofs", []))
            rows.append(_ravr_metrics_row("session", sid, str(s.get("case_id", "")), per_session_metrics))
            merged.extend(s.get("method_proofs", []))
        global_metrics = build_ravr_metrics(merged)
        rows.append(_ravr_metrics_row("global", "GLOBAL", "-", global_metrics))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

    filename = f"ravr_metrics_{session_id or 'all'}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/ravr_dataset.jsonl")
def get_ravr_dataset_jsonl(session_id: Optional[str] = Query(default=None, description="Опционально: ID сессии")):
    rows: List[Dict[str, Any]] = []
    if session_id:
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        rows.extend(_session_turn_dataset_rows(session_id, session))
    else:
        for sid, s in sessions.items():
            rows.extend(_session_turn_dataset_rows(sid, s))

    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    content = "\n".join(lines) + ("\n" if lines else "")
    filename = f"ravr_dataset_{session_id or 'all'}.jsonl"
    return Response(
        content=content,
        media_type="application/jsonl; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename=\"{filename}\"'},
    )


@app.post("/api/ravr_benchmark", response_model=RavrBenchmarkResponse)
def ravr_benchmark(req: RavrBenchmarkRequest):
    return run_ravr_benchmark(req)


@app.post("/api/ravr_multi_model_benchmark", response_model=RavrMultiModelBenchmarkResponse)
def ravr_multi_model_benchmark(req: RavrMultiModelBenchmarkRequest):
    results: Dict[str, RavrBenchmarkResponse] = {}
    errors: Dict[str, str] = {}

    for provider in req.providers:
        provider_name = str(provider).strip().lower()
        if not provider_name:
            continue
        try:
            res = run_ravr_benchmark(
                RavrBenchmarkRequest(
                    n_per_case=req.n_per_case,
                    random_seed=req.random_seed,
                    include_llm_eval=req.include_llm_eval,
                    llm_temperature=req.llm_temperature,
                    llm_top_p=req.llm_top_p,
                    disable_eval_cache=req.disable_eval_cache,
                    case_ids=req.case_ids,
                    override_llm_provider=provider_name,
                )
            )
            results[provider_name] = res
        except Exception as e:
            errors[provider_name] = str(e)

    return RavrMultiModelBenchmarkResponse(results=results, errors=errors)


@app.post("/api/ravr_benchmark.csv")
def ravr_benchmark_csv(req: RavrBenchmarkRequest):
    result = run_ravr_benchmark(req)
    fieldnames = [
        "case_id",
        "category_key",
        "methodology_id",
        "prompt",
        "verifier_pass",
        "citation_valid",
        "citation_coverage",
        "citation_precision",
        "citation_relevance",
        "adherence_score",
        "adherence_before",
        "adherence_after",
        "adherence_delta",
        "violations",
        "repaired_violations",
        "violation_types",
        "repair_triggered",
        "repair_success",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in result.rows:
        writer.writerow(_benchmark_row_to_dict(row))
    writer.writerow({})
    writer.writerow(
        {
            "case_id": "SUMMARY",
            "category_key": "-",
            "methodology_id": "-",
            "prompt": f"cases_total={result.summary.cases_total}; turns_total={result.summary.turns_total}",
            "verifier_pass": result.summary.metrics.verifier_pass_rate,
            "citation_valid": result.summary.metrics.citation_valid_rate,
            "citation_coverage": result.summary.metrics.citation_coverage_rate,
            "citation_precision": result.summary.metrics.citation_precision_rate,
            "citation_relevance": result.summary.metrics.citation_relevance_score,
            "adherence_score": result.summary.metrics.avg_adherence_score,
            "violations": " | ".join(result.summary.metrics.top_violations),
            "repair_triggered": result.summary.metrics.repair_trigger_rate,
            "repair_success": result.summary.metrics.repair_success_rate,
        }
    )
    filename = f"ravr_benchmark_n{req.n_per_case}_seed{req.random_seed}.csv"
    return Response(
        content=output.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/ravr_benchmark.jsonl")
def ravr_benchmark_jsonl(req: RavrBenchmarkRequest):
    result = run_ravr_benchmark(req)
    rows = [_benchmark_row_to_dict(row) for row in result.rows]
    rows.append(
        {
            "scope": "summary",
            "summary": result.summary.model_dump(),
            "config": result.config,
        }
    )
    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    filename = f"ravr_benchmark_n{req.n_per_case}_seed{req.random_seed}.jsonl"
    return Response(
        content=content,
        media_type="application/jsonl; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/verify_turn", response_model=VerifyTurnResponse)
def verify_turn(req: VerifyTurnRequest):
    case_id = normalize_case_id(req.case_id)
    if case_id not in CASES_BY_ID:
        raise HTTPException(status_code=400, detail="Unknown case_id")

    proof = build_methodology_proof(case_id, req.user_message, req.evaluation)
    verifier_pass = len(proof.violated_constraints) == 0
    if RAVR_ENABLE_REPAIR and (not verifier_pass) and proof.repair_suggestion and proof.repair_suggestion.should_repair:
        repaired_proof = build_methodology_proof(
            case_id,
            proof.repair_suggestion.repaired_message,
            req.evaluation,
        )
        proof.repair_suggestion.repaired_verifier_pass = len(repaired_proof.violated_constraints) == 0
        proof.repair_suggestion.repaired_violations = repaired_proof.violated_constraints
        if not proof.repair_suggestion.citations:
            proof.repair_suggestion.citations = repaired_proof.citations

    return VerifyTurnResponse(
        case_id=case_id,
        category_key=CASES_BY_ID[case_id]["category_key"],
        proof_object=proof,
        verifier_pass=verifier_pass,
        verifier_violations=proof.violated_constraints,
    )


@app.get("/api/session_report", response_model=SessionReport)
def session_report(
    session_id: str = Query(..., description="ID сессии"),
    llm_provider: Optional[str] = Query(default=None, description="Провайдер LLM: gigachat|openai|openai_compatible"),
):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    evals = session["evals"]
    if not evals:
        raise HTTPException(status_code=400, detail="No evaluation data for this session")

    n = len(evals)

    def avg(key: str) -> float:
        return sum(e[key] for e in evals) / n

    avg_empathy = avg("empathy")
    avg_validation = avg("validation")
    avg_directivity = avg("directivity")
    avg_open = avg("open_question")
    avg_safety = avg("safety")
    mean_eff = avg("efficiency_index")

    total_dt = sum(e["delta_trust"] for e in evals)
    total_de = sum(e["delta_emotional_intensity"] for e in evals)
    total_df = sum(e["delta_fatigue"] for e in evals)
    turn_comments = [str(e.get("comment", "")).strip() for e in evals if str(e.get("comment", "")).strip()]
    progress = build_session_progress(session_id, session)
    mistaken_replicas = collect_mistaken_replicas(session)

    history_sample = []
    for turn in session.get("history", [])[-12:]:
        role = str(turn.get("role", "")).strip()
        if role not in ("user", "assistant"):
            continue
        history_sample.append(
            {
                "role": role,
                "content": _truncate_text(str(turn.get("content", ""))),
            }
        )

    try:
        overall_impression, recommendations, improved_examples = generate_session_feedback_with_llm(
            avg_empathy=avg_empathy,
            avg_validation=avg_validation,
            avg_directivity=avg_directivity,
            avg_open=avg_open,
            avg_safety=avg_safety,
            mean_eff=mean_eff,
            total_dt=total_dt,
            total_de=total_de,
            total_df=total_df,
            num_turns=n,
            turn_comments=turn_comments,
            trends=progress.trends,
            history_sample=history_sample,
            mistaken_replicas=mistaken_replicas,
            provider=llm_provider,
        )
        if not improved_examples:
            improved_examples = fallback_improved_examples(mistaken_replicas)
    except Exception as e:
        print("Session report LLM generation error:", e)
        overall_impression = _l(
            "Автоматическая генерация развёрнутого впечатления временно недоступна. Ориентируйтесь на метрики сессии и динамику состояния пациента.",
            "Detailed automated feedback is temporarily unavailable. Use the session metrics and patient-state trajectory as the primary indicators.",
        )
        recommendations = _l(
            "Сфокусируйтесь на эмпатии, открытых вопросах, снижении директивности и безопасных формулировках, а затем сравните изменения индекса эффективности и дельты состояния пациента.",
            "Focus on empathy, open questions, reduced directivity, and safe wording, then compare changes in efficiency and patient-state deltas.",
        )
        improved_examples = fallback_improved_examples(mistaken_replicas)

    return SessionReport(
        session_id=session_id,
        case_id=session["case_id"],
        num_turns=n,
        avg_empathy=round(avg_empathy, 2),
        avg_validation=round(avg_validation, 2),
        avg_directivity=round(avg_directivity, 2),
        avg_open_question=round(avg_open, 2),
        avg_safety=round(avg_safety, 2),
        mean_efficiency_index=round(mean_eff, 2),
        total_delta_trust=int(total_dt),
        total_delta_emotional_intensity=int(total_de),
        total_delta_fatigue=int(total_df),
        overall_impression=overall_impression,
        recommendations=recommendations,
        improved_examples=improved_examples,
    )
