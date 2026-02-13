import json
import os
import re
import time
import uuid
import threading
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()  # загрузит .env из текущей папки

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
if not GIGACHAT_AUTHORIZATION_KEY:
    raise RuntimeError("GIGACHAT_AUTHORIZATION_KEY is not set (.env)")

# OAuth scope
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")

# Model name (can be overridden)
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")

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


# ============================================================
#                      Cases loading
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DB_PATH = BASE_DIR / "sessions.db"
LEGACY_SESSIONS_STORE_PATH = BASE_DIR / "sessions_store.json"
SESSIONS_LOCK = threading.Lock()
with open(BASE_DIR / "virtual_patient_cases.json", "r", encoding="utf-8") as f:
    CASES_RAW = json.load(f)["cases"]

CATEGORY_LABELS = {
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


class ChatResponse(BaseModel):
    session_id: str
    case_id: str
    assistant_message: str
    evaluation: Optional[TurnEvaluation] = None


class CasePublic(BaseModel):
    id: str
    category_key: str
    category_name: str
    title_for_teacher: str
    visible_to_student: Dict[str, Any]


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
    triggers: List[str]


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
                    session_id, case_id, history_json, state_json, evals_json, mistakes_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    case_id = excluded.case_id,
                    history_json = excluded.history_json,
                    state_json = excluded.state_json,
                    evals_json = excluded.evals_json,
                    mistakes_json = excluded.mistakes_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    session_data["case_id"],
                    json.dumps(session_data.get("history", []), ensure_ascii=False),
                    json.dumps(session_data.get("state", {}), ensure_ascii=False),
                    json.dumps(session_data.get("evals", []), ensure_ascii=False),
                    json.dumps(session_data.get("mistakes", []), ensure_ascii=False),
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
                "SELECT session_id, case_id, history_json, state_json, evals_json, mistakes_json FROM sessions"
            ).fetchall()

        for session_id, case_id, history_json, state_json, evals_json, mistakes_json in rows:
            raw = {
                "case_id": case_id,
                "history": json.loads(history_json or "[]"),
                "state": json.loads(state_json or "{}"),
                "evals": json.loads(evals_json or "[]"),
                "mistakes": json.loads(mistakes_json or "[]"),
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

def call_llm_chat(messages: List[Dict[str, str]]) -> str:
    # patient generation
    return gigachat_chat_completions(
        messages=messages,
        temperature=0.8,
        max_tokens=300,
        model=GIGACHAT_MODEL,
    )


def evaluate_therapist_message(message: str, prev_state: Dict[str, int]) -> TurnEvaluation:
    """
    Оценка хода психолога через LLM-супервизора.
    """
    state_desc = (
        f"Текущее состояние пациента: доверие={prev_state['trust_level']} (0–3), "
        f"эмоциональная интенсивность={prev_state['emotional_intensity']} (0–3), "
        f"усталость={prev_state['fatigue']} (0–3)."
    )

    user_content = state_desc + "\n\nРеплика психолога:\n" + message

    try:
        raw = gigachat_chat_completions(
            model=GIGACHAT_MODEL,
            messages=[
                {"role": "system", "content": SUPERVISOR_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
            max_tokens=300,
        ).strip()

        data = json.loads(raw)
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
            "comment": "Не удалось вычислить оценку, используйте этот ход только как тренировочный.",
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

    comment = str(data.get("comment", "")).strip() or "Нейтральный ход."

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


def detect_mistake_reason(ev: TurnEvaluation) -> Optional[str]:
    reasons: List[str] = []
    if ev.safety < 0.85:
        reasons.append("формулировка снижает ощущение безопасности")
    if ev.empathy < 0.4:
        reasons.append("мало эмпатии и отражения чувств")
    if ev.open_question < 0.3:
        reasons.append("не хватает открытых вопросов")
    if ev.directivity > 0.7:
        reasons.append("слишком директивная подача")
    if ev.efficiency_index < 0:
        reasons.append("ход снижает общую эффективность контакта")
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
        better = (
            "Похоже, вам сейчас непросто. "
            "Что для вас в этой ситуации самое тяжёлое в данный момент?"
        )
        why = str(m.get("reason", "")).strip() or "Так формулировка звучит мягче и поддерживает контакт."
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

    raw = gigachat_chat_completions(
        model=GIGACHAT_MODEL,
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


# ============================================================
#                      FastAPI app
# ============================================================

app = FastAPI(title="Virtual Patient Simulator (Teacher Mode)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_sessions_db()
migrate_legacy_sessions_json_if_needed()
load_sessions_from_db()


@app.get("/api/cases", response_model=List[CasePublic])
def list_cases():
    result: List[CasePublic] = []
    for c in CASES_DATA:
        result.append(
            CasePublic(
                id=c["id"],
                category_key=c["category_key"],
                category_name=c["category_name"],
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
        }

    session = sessions[req.session_id]

    if session["case_id"] != req.case_id:
        session["case_id"] = req.case_id
        session["history"] = []
        session["state"] = get_initial_state()
        session["evals"] = []
        session["mistakes"] = []

    # Оценка хода и обновление состояния
    evaluation = evaluate_therapist_message(req.user_message, session["state"])
    session["state"] = apply_state_delta(session["state"], evaluation)
    session["evals"].append(evaluation.model_dump())
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
        assistant_text = call_llm_chat(messages)
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
    )


@app.get("/api/sessions/{session_id}/progress", response_model=SessionProgress)
def get_session_progress(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return build_session_progress(session_id, session)


@app.get("/api/session_report", response_model=SessionReport)
def session_report(session_id: str = Query(..., description="ID сессии")):
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
        )
        if not improved_examples:
            improved_examples = fallback_improved_examples(mistaken_replicas)
    except Exception as e:
        print("Session report LLM generation error:", e)
        overall_impression = (
            "Автоматическая генерация развёрнутого впечатления временно недоступна. "
            "Ориентируйтесь на метрики сессии и динамику состояния пациента."
        )
        recommendations = (
            "Сфокусируйтесь на эмпатии, открытых вопросах, снижении директивности и безопасных формулировках, "
            "а затем сравните изменения индекса эффективности и дельты состояния пациента."
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
