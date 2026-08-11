import os
import random
import re
import time
import uuid
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
BACKEND_URL_EN = os.getenv("BACKEND_URL_EN", "http://localhost:8001")
SALUTESPEECH_AUTH_KEY = os.getenv("SALUTESPEECH_AUTH_KEY")
SALUTESPEECH_SCOPE = os.getenv("SALUTESPEECH_SCOPE", "SALUTE_SPEECH_PERS")
SALUTESPEECH_VERIFY_SSL = os.getenv("SALUTESPEECH_VERIFY_SSL", "1") not in (
    "0",
    "false",
    "False",
    "no",
    "NO",
)
SALUTESPEECH_STT_MODEL = os.getenv("SALUTESPEECH_STT_MODEL", "general")
SALUTESPEECH_STT_MODEL_EN = os.getenv("SALUTESPEECH_STT_MODEL_EN", SALUTESPEECH_STT_MODEL)
SALUTESPEECH_STT_AUDIO_ENCODING = os.getenv("SALUTESPEECH_STT_AUDIO_ENCODING", "OGG_OPUS")
SALUTESPEECH_STT_SAMPLE_RATE = int(os.getenv("SALUTESPEECH_STT_SAMPLE_RATE", "48000"))
SALUTESPEECH_STT_CHANNELS = int(os.getenv("SALUTESPEECH_STT_CHANNELS", "1"))
SALUTESPEECH_TTS_VOICE = os.getenv("SALUTESPEECH_TTS_VOICE", "Nec_24000")
SALUTESPEECH_TTS_VOICE_EN = os.getenv("SALUTESPEECH_TTS_VOICE_EN", SALUTESPEECH_TTS_VOICE)
SALUTESPEECH_TTS_FORMAT = os.getenv("SALUTESPEECH_TTS_FORMAT", "opus")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

TEXTS: Dict[str, Dict[str, str]] = {
    "ru": {
        "menu_button": "🏠 Меню",
        "finish_button": "✅ Завершить",
        "report_button": "📊 Отчёт",
        "progress_button": "📈 Прогресс",
        "toggle_button": "🔄 Сменить режим",
        "start_button": "✅ Начать",
        "language_button": "🌐 Language / Язык",
        "choose_language": "🌐 Выберите язык / Choose your language:",
        "welcome": (
            "👋 Добро пожаловать в симулятор виртуальных пациентов.\n\n"
            "Как работает:\n"
            "• Вы выбираете пациента или берёте случайного.\n"
            "• Общаетесь текстом или голосом.\n"
            "• В конце получаете итоговый отчёт.\n\n"
            "Нажмите «✅ Начать»."
        ),
        "menu": (
            "🏠 Меню\n\n"
            "• 👥 Пациент по диагнозу: вы знаете тему.\n"
            "• 🎲 Случайный пациент: диагноз нужно определить в конце.\n"
            "• ✅ Завершить: закончить сессию.\n"
            "• 📊 Отчёт: получить итоговый разбор."
        ),
        "choose_action": "Выберите действие:",
        "select_case": "👥 Пациент по диагнозу",
        "random_case": "🎲 Случайный пациент",
        "voice": "🎤 Голос",
        "text": "💬 Текст",
        "finish_session": "✅ Завершить сессию",
        "help": "ℹ️ Помощь",
        "back": "⬅️ Назад",
        "load_cases_error": "Не удалось загрузить кейсы: {error}",
        "voice_on": "🎤 Включён голосовой режим. Отправляйте голосовые сообщения.",
        "text_on": "💬 Включён текстовый режим. Отправляйте текст.",
        "report_error": "Не удалось получить отчёт: {error}",
        "progress_error": "Не удалось получить прогресс: {error}",
        "select_patient_first": "Сначала выберите пациента.",
        "guess_prompt": (
            "✅ Сессия завершена.\n\n"
            "Теперь напишите предполагаемый диагноз кратко.\n"
            "Например: «депрессия», «паническая атака» или «ОКР»."
        ),
        "session_finished": "✅ Сессия завершена.\n\n",
        "choose_diagnosis": "Выберите диагноз:",
        "choose_patient": "Выберите пациента:",
        "patient_selected": (
            "✅ Пациент выбран: {case_id}\n\nТеперь можете задавать вопросы пациенту.\n\n"
            "Чтобы открыть меню, отправьте /start или нажмите «🏠 Меню»."
        ),
        "no_cases": "Нет доступных кейсов.",
        "random_selected": (
            "🎲 Случайный пациент выбран.\n"
            "Диагноз скрыт. Общайтесь, как на приёме.\n\n"
            "Чтобы завершить сессию или открыть меню, используйте кнопки внизу."
        ),
        "help_text": (
            "ℹ️ Помощь\n\n"
            "• /start или «🏠 Меню»: открыть меню.\n"
            "• 🎤/💬: переключить голосовой и текстовый режим.\n"
            "• 🎲 Случайный пациент: диагноз открывается после вашей догадки.\n"
            "• ✅ Завершить: закончить сессию.\n"
            "• 📊 Отчёт: получить итоговый разбор."
        ),
        "correct": "✅ Верно!",
        "incorrect": "❌ Неверно.",
        "your_diagnosis": "Ваш диагноз",
        "correct_diagnosis": "Правильный диагноз",
        "voice_expected": "Сейчас включён голосовой режим. Отправьте голосовое сообщение или переключитесь на текст.",
        "text_expected": "Сейчас включён текстовый режим. Отправьте текст или переключитесь на голос.",
        "select_via_menu": "Сначала выберите пациента через меню (/start).",
        "server_error": "Ошибка при обращении к серверу: {error}",
        "voice_recognition_error": "Не удалось распознать голос: {error}",
        "voice_synthesis_error": "Ошибка синтеза голоса: {error}\n\nОтвет пациента:\n{answer}",
    },
    "en": {
        "menu_button": "🏠 Menu",
        "finish_button": "✅ Finish",
        "report_button": "📊 Report",
        "progress_button": "📈 Progress",
        "toggle_button": "🔄 Switch mode",
        "start_button": "✅ Start",
        "language_button": "🌐 Language / Язык",
        "choose_language": "🌐 Choose your language / Выберите язык:",
        "welcome": (
            "👋 Welcome to the Virtual Patient Simulator.\n\n"
            "How it works:\n"
            "• Choose a patient or start a random case.\n"
            "• Conduct the consultation by text or voice.\n"
            "• Receive a structured report at the end.\n\n"
            "Tap “✅ Start” to continue."
        ),
        "menu": (
            "🏠 Menu\n\n"
            "• 👥 Patient by diagnosis: the clinical topic is shown.\n"
            "• 🎲 Random patient: identify the diagnosis at the end.\n"
            "• ✅ Finish: end the current session.\n"
            "• 📊 Report: receive structured feedback."
        ),
        "choose_action": "Choose an action:",
        "select_case": "👥 Patient by diagnosis",
        "random_case": "🎲 Random patient",
        "voice": "🎤 Voice",
        "text": "💬 Text",
        "finish_session": "✅ Finish session",
        "help": "ℹ️ Help",
        "back": "⬅️ Back",
        "load_cases_error": "Could not load cases: {error}",
        "voice_on": "🎤 Voice mode is on. Send a voice message.",
        "text_on": "💬 Text mode is on. Send a text message.",
        "report_error": "Could not generate the report: {error}",
        "progress_error": "Could not load session progress: {error}",
        "select_patient_first": "Choose a patient first.",
        "guess_prompt": (
            "✅ Session finished.\n\n"
            "Now enter the diagnosis you consider most likely. Keep it brief.\n"
            "For example: depression, panic disorder, or OCD."
        ),
        "session_finished": "✅ Session finished.\n\n",
        "choose_diagnosis": "Choose a diagnosis:",
        "choose_patient": "Choose a patient:",
        "patient_selected": (
            "✅ Patient selected: {case_id}\n\nYou can now begin the consultation.\n\n"
            "Use /start or tap “🏠 Menu” to open the menu again."
        ),
        "no_cases": "No cases are available.",
        "random_selected": (
            "🎲 A random patient has been selected.\n"
            "The diagnosis is hidden. Conduct the consultation as you normally would.\n\n"
            "Use the buttons below to finish the session or open the menu."
        ),
        "help_text": (
            "ℹ️ Help\n\n"
            "• /start or “🏠 Menu”: open the menu.\n"
            "• 🎤/💬: switch between voice and text.\n"
            "• 🎲 Random patient: the diagnosis is revealed after your guess.\n"
            "• ✅ Finish: end the current session.\n"
            "• 📊 Report: receive structured session feedback."
        ),
        "correct": "✅ Correct!",
        "incorrect": "❌ Not quite.",
        "your_diagnosis": "Your diagnosis",
        "correct_diagnosis": "Reference diagnosis",
        "voice_expected": "Voice mode is on. Send a voice message or switch to text mode.",
        "text_expected": "Text mode is on. Send a text message or switch to voice mode.",
        "select_via_menu": "Choose a patient from the menu first (/start).",
        "server_error": "The server request failed: {error}",
        "voice_recognition_error": "Could not recognize the voice message: {error}",
        "voice_synthesis_error": "Voice synthesis failed: {error}\n\nPatient response:\n{answer}",
    },
}

user_state: Dict[int, Dict[str, Any]] = {}
CASES_CACHE: Dict[str, List[Dict[str, Any]]] = {"ru": [], "en": []}

SALUTE_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SALUTE_STT_URL = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
SALUTE_TTS_URL = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
_salute_token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}


def ensure_user(chat_id: int) -> Dict[str, Any]:
    if chat_id not in user_state:
        user_state[chat_id] = {
            "language": None,
            "case_id": None,
            "comm_mode": "text",
            "welcome_seen": False,
            "random_mode": False,
            "pending_guess": False,
            "hidden_diagnosis": None,
        }
    return user_state[chat_id]


def user_language(state: Dict[str, Any]) -> str:
    language = state.get("language")
    return language if language in TEXTS else "en"


def tr(language: str, key: str, **values: Any) -> str:
    text = TEXTS[language][key]
    return text.format(**values) if values else text


def backend_url(language: str) -> str:
    return BACKEND_URL_EN if language == "en" else BACKEND_URL


def get_session_id(chat_id: int, language: str) -> str:
    return f"tg_{language}_{chat_id}"


def fetch_cases(language: str) -> List[Dict[str, Any]]:
    resp = requests.get(f"{backend_url(language)}/api/cases", timeout=10)
    resp.raise_for_status()
    CASES_CACHE[language] = resp.json()
    return CASES_CACHE[language]


def group_cases_by_category(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        key = case["category_key"]
        if key not in grouped:
            grouped[key] = {"name": case["category_name"], "cases": []}
        grouped[key]["cases"].append(case)
    return grouped


def short_label(value: str, max_len: int = 22) -> str:
    value = (value or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 1].rstrip() + "…"


def normalize_diag(value: str) -> str:
    return "".join(
        char.lower() for char in (value or "").strip() if char.isalnum() or char.isspace()
    ).strip()


def normalize_button_text(value: str) -> str:
    normalized = (value or "").lower().strip()
    normalized = re.sub(r"[^\w\sа-яё]", " ", normalized, flags=re.IGNORECASE)
    return " ".join(normalized.split())


def get_case_diagnosis_label(case: Dict[str, Any]) -> str:
    for key in ("diagnosis_short", "diagnosis_name", "category_name", "category_key"):
        if case.get(key):
            return str(case[key])
    return "unknown"


def get_salutespeech_token(force_refresh: bool = False) -> str:
    if not SALUTESPEECH_AUTH_KEY:
        raise RuntimeError("SALUTESPEECH_AUTH_KEY is not set")

    now = time.time()
    token = _salute_token_cache.get("access_token")
    expires_at = float(_salute_token_cache.get("expires_at") or 0.0)
    if not force_refresh and token and now < expires_at:
        return str(token)

    response = requests.post(
        SALUTE_OAUTH_URL,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": f"Basic {SALUTESPEECH_AUTH_KEY}",
        },
        data={"scope": SALUTESPEECH_SCOPE},
        timeout=20,
        verify=SALUTESPEECH_VERIFY_SSL,
    )
    response.raise_for_status()
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in SaluteSpeech OAuth response: {data}")

    expires_ms = int(data.get("expires_at", 0) or 0)
    expires_at = (expires_ms / 1000.0) - 60 if expires_ms else time.time() + 25 * 60
    _salute_token_cache["access_token"] = token
    _salute_token_cache["expires_at"] = max(time.time() + 60, expires_at)
    return str(token)


async def transcribe_voice(file_bytes: bytes, language: str) -> str:
    token = get_salutespeech_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/ogg;codecs=opus",
    }
    params = {
        "model": SALUTESPEECH_STT_MODEL_EN if language == "en" else SALUTESPEECH_STT_MODEL,
        "audio_encoding": SALUTESPEECH_STT_AUDIO_ENCODING,
        "sample_rate": SALUTESPEECH_STT_SAMPLE_RATE,
        "channels_count": SALUTESPEECH_STT_CHANNELS,
    }

    response = requests.post(
        SALUTE_STT_URL,
        headers=headers,
        params=params,
        data=file_bytes,
        timeout=60,
        verify=SALUTESPEECH_VERIFY_SSL,
    )
    if response.status_code in (401, 403):
        headers["Authorization"] = f"Bearer {get_salutespeech_token(force_refresh=True)}"
        response = requests.post(
            SALUTE_STT_URL,
            headers=headers,
            params=params,
            data=file_bytes,
            timeout=60,
            verify=SALUTESPEECH_VERIFY_SSL,
        )
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        for key in ("text", "result", "transcript"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, list):
                parts: List[str] = []
                for item in value:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"].strip())
                if parts:
                    return " ".join(part for part in parts if part)
        hypotheses = data.get("hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            first = hypotheses[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                return first["text"].strip()
    raise RuntimeError("Speech recognition returned no transcript.")


async def tts_to_bytes(text: str, language: str) -> bytes:
    token = get_salutespeech_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/text",
    }
    voice = SALUTESPEECH_TTS_VOICE_EN if language == "en" else SALUTESPEECH_TTS_VOICE
    payload = (text or "").encode("utf-8")
    candidates = [
        {"voice": voice, "format": SALUTESPEECH_TTS_FORMAT},
        {"voice": voice, "audio_encoding": SALUTESPEECH_TTS_FORMAT},
        {"voice": voice, "audio_encoding": "opus"},
        {"voice": voice, "format": "oggopus"},
    ]

    last_error = ""
    for params in candidates:
        response = requests.post(
            SALUTE_TTS_URL,
            headers=headers,
            params=params,
            data=payload,
            timeout=60,
            verify=SALUTESPEECH_VERIFY_SSL,
        )
        if response.status_code in (401, 403):
            headers["Authorization"] = f"Bearer {get_salutespeech_token(force_refresh=True)}"
            response = requests.post(
                SALUTE_TTS_URL,
                headers=headers,
                params=params,
                data=payload,
                timeout=60,
                verify=SALUTESPEECH_VERIFY_SSL,
            )
        if response.ok:
            return response.content
        last_error = response.text[:500] if response.text else f"HTTP {response.status_code}"
        if response.status_code != 400:
            response.raise_for_status()

    raise RuntimeError(
        f"SaluteSpeech TTS does not support voice={voice}, format={SALUTESPEECH_TTS_FORMAT}. "
        f"Details: {last_error}"
    )


def call_backend_chat(
    session_id: str,
    case_id: str,
    user_message: str,
    language: str,
) -> Dict[str, Any]:
    response = requests.post(
        f"{backend_url(language)}/api/chat",
        json={
            "session_id": session_id,
            "case_id": case_id,
            "user_message": user_message,
            "teacher_mode": False,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def call_backend_report(session_id: str, language: str) -> Dict[str, Any]:
    response = requests.get(
        f"{backend_url(language)}/api/session_report",
        params={"session_id": session_id},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def call_backend_progress(session_id: str, language: str) -> Dict[str, Any]:
    response = requests.get(
        f"{backend_url(language)}/api/sessions/{session_id}/progress",
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def language_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[
            InlineKeyboardButton("🇬🇧 English", callback_data="lang:en"),
            InlineKeyboardButton("🇷🇺 Русский", callback_data="lang:ru"),
        ]]
    )


def welcome_keyboard(language: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [[InlineKeyboardButton(tr(language, "start_button"), callback_data="welcome:start")]]
    )


def bottom_reply_keyboard(language: str) -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [
                tr(language, "menu_button"),
                tr(language, "finish_button"),
                tr(language, "report_button"),
                tr(language, "progress_button"),
            ],
            [tr(language, "toggle_button")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def main_menu_keyboard(state: Dict[str, Any]) -> InlineKeyboardMarkup:
    language = user_language(state)
    mode_label = tr(language, "voice") if state["comm_mode"] == "text" else tr(language, "text")
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton(tr(language, "select_case"), callback_data="menu:select_case")],
            [InlineKeyboardButton(tr(language, "random_case"), callback_data="menu:random_case")],
            [InlineKeyboardButton(mode_label, callback_data="menu:toggle_comm")],
            [InlineKeyboardButton(tr(language, "finish_session"), callback_data="menu:finish")],
            [InlineKeyboardButton(tr(language, "report_button"), callback_data="menu:report")],
            [InlineKeyboardButton(tr(language, "help"), callback_data="menu:help")],
            [InlineKeyboardButton(tr(language, "language_button"), callback_data="menu:language")],
        ]
    )


def back_to_main_button(language: str) -> List[InlineKeyboardButton]:
    return [InlineKeyboardButton(tr(language, "menu_button"), callback_data="menu:main")]


def diagnosis_keyboard(cases: List[Dict[str, Any]], language: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(short_label(data["name"], 24), callback_data=f"diag:{key}")]
        for key, data in group_cases_by_category(cases).items()
    ]
    rows.append(back_to_main_button(language))
    return InlineKeyboardMarkup(rows)


def patients_keyboard(
    cases: List[Dict[str, Any]],
    category_key: str,
    language: str,
) -> InlineKeyboardMarkup:
    grouped = group_cases_by_category(cases)
    if category_key not in grouped:
        return InlineKeyboardMarkup([back_to_main_button(language)])

    rows: List[List[InlineKeyboardButton]] = []
    for case in grouped[category_key]["cases"]:
        title = (
            case.get("title_short")
            or case.get("title_for_teacher")
            or case.get("title")
            or f"case {case.get('id')}"
        )
        rows.append(
            [InlineKeyboardButton(short_label(str(title), 26), callback_data=f"case:{case['id']}")]
        )
    rows.append([InlineKeyboardButton(tr(language, "back"), callback_data="menu:select_case")])
    rows.append(back_to_main_button(language))
    return InlineKeyboardMarkup(rows)


def format_session_report(report: Dict[str, Any], language: str) -> str:
    examples = report.get("improved_examples") or []
    example_blocks: List[str] = []
    if isinstance(examples, list):
        for index, example in enumerate(examples[:3], start=1):
            if not isinstance(example, dict):
                continue
            original = str(example.get("original_replica", "")).strip()
            better = str(example.get("better_replica", "")).strip()
            why = str(example.get("why_better", "")).strip() or "-"
            if original and better:
                if language == "en":
                    example_blocks.append(
                        f"{index}. Original: {original}\n   Improved: {better}\n   Why: {why}"
                    )
                else:
                    example_blocks.append(
                        f"{index}. Было: {original}\n   Лучше: {better}\n   Почему: {why}"
                    )

    if language == "en":
        examples_text = (
            "\n\n🧠 Suggested reformulations:\n" + "\n\n".join(example_blocks)
            if example_blocks
            else ""
        )
        return (
            "📊 Session report\n"
            f"• Case: {report.get('case_id')}\n"
            f"• Turns: {report.get('num_turns')}\n\n"
            "🎯 Mean scores\n"
            f"• Empathy: {report.get('avg_empathy', 0):.2f}\n"
            f"• Validation: {report.get('avg_validation', 0):.2f}\n"
            f"• Directivity: {report.get('avg_directivity', 0):.2f}\n"
            f"• Open questions: {report.get('avg_open_question', 0):.2f}\n"
            f"• Safety: {report.get('avg_safety', 0):.2f}\n"
            f"• Efficiency index: {report.get('mean_efficiency_index', 0):.2f}\n\n"
            "📉 Cumulative patient-state changes\n"
            f"• Δ trust: {report.get('total_delta_trust')}\n"
            f"• Δ emotional intensity: {report.get('total_delta_emotional_intensity')}\n"
            f"• Δ fatigue: {report.get('total_delta_fatigue')}\n\n"
            "🧾 Overall assessment:\n"
            f"{report.get('overall_impression', '-')}\n\n"
            "🛠 Recommendations:\n"
            f"{report.get('recommendations', '-')}"
            f"{examples_text}"
        )

    examples_text = (
        "\n\n🧠 Примеры, как лучше переформулировать:\n" + "\n\n".join(example_blocks)
        if example_blocks
        else ""
    )
    return (
        "📊 Итоговый отчёт по сессии\n"
        f"• Кейс: {report.get('case_id')}\n"
        f"• Кол-во ходов: {report.get('num_turns')}\n\n"
        "🎯 Средние показатели\n"
        f"• Эмпатия: {report.get('avg_empathy', 0):.2f}\n"
        f"• Валидация: {report.get('avg_validation', 0):.2f}\n"
        f"• Директивность: {report.get('avg_directivity', 0):.2f}\n"
        f"• Открытые вопросы: {report.get('avg_open_question', 0):.2f}\n"
        f"• Безопасность: {report.get('avg_safety', 0):.2f}\n"
        f"• Индекс эффективности: {report.get('mean_efficiency_index', 0):.2f}\n\n"
        "📉 Суммарные изменения состояния пациента\n"
        f"• Δ доверия: {report.get('total_delta_trust')}\n"
        f"• Δ эмоц. интенсивности: {report.get('total_delta_emotional_intensity')}\n"
        f"• Δ усталости: {report.get('total_delta_fatigue')}\n\n"
        "🧾 Общее впечатление:\n"
        f"{report.get('overall_impression', '-')}\n\n"
        "🛠 Рекомендации:\n"
        f"{report.get('recommendations', '-')}"
        f"{examples_text}"
    )


def trend_arrow(value: float) -> str:
    if value > 0:
        return "↑"
    if value < 0:
        return "↓"
    return "→"


def format_progress_report(progress_data: Dict[str, Any], language: str) -> str:
    trends = progress_data.get("trends", {}) or {}
    empathy = float(trends.get("empathy", 0) or 0)
    safety = float(trends.get("safety", 0) or 0)
    directivity = float(trends.get("directivity", 0) or 0)
    if language == "en":
        return (
            "📈 Session progress\n"
            f"• Case: {progress_data.get('case_id')}\n"
            f"• Turns: {progress_data.get('num_turns')}\n\n"
            "Skill trends:\n"
            f"• Empathy: {trend_arrow(empathy)} ({empathy:+.3f})\n"
            f"• Safety: {trend_arrow(safety)} ({safety:+.3f})\n"
            f"• Directivity: {trend_arrow(directivity)} ({directivity:+.3f})\n"
            "  (lower directivity is often preferable)\n\n"
            "Command: /progress"
        )
    return (
        "📈 Динамика сессии\n"
        f"• Кейс: {progress_data.get('case_id')}\n"
        f"• Ходов: {progress_data.get('num_turns')}\n\n"
        "Тренды по навыкам:\n"
        f"• Эмпатия: {trend_arrow(empathy)} ({empathy:+.3f})\n"
        f"• Безопасность: {trend_arrow(safety)} ({safety:+.3f})\n"
        f"• Директивность: {trend_arrow(directivity)} ({directivity:+.3f})\n"
        "  (для директивности чаще лучше снижение)\n\n"
        "Команда: /progress"
    )


async def send_language_selector(message) -> None:
    await message.reply_text(TEXTS["en"]["choose_language"], reply_markup=language_keyboard())


async def send_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_user(update.effective_chat.id)
    language = user_language(state)
    message = update.message or update.callback_query.message
    await message.reply_text(tr(language, "welcome"), reply_markup=welcome_keyboard(language))


async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_user(update.effective_chat.id)
    if state.get("language") not in TEXTS:
        await send_language_selector(update.message or update.callback_query.message)
        return
    language = user_language(state)
    message = update.message or update.callback_query.message
    await message.reply_text(tr(language, "menu"), reply_markup=bottom_reply_keyboard(language))
    await message.reply_text(tr(language, "choose_action"), reply_markup=main_menu_keyboard(state))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ensure_user(update.effective_chat.id)
    await send_language_selector(update.message)


async def do_toggle_comm(chat_id: int, message, use_inline_menu: bool = False) -> None:
    state = ensure_user(chat_id)
    language = user_language(state)
    state["comm_mode"] = "voice" if state["comm_mode"] == "text" else "text"
    mode_text = tr(language, "voice_on" if state["comm_mode"] == "voice" else "text_on")
    reply_markup = main_menu_keyboard(state) if use_inline_menu else bottom_reply_keyboard(language)
    await message.reply_text(mode_text, reply_markup=reply_markup)


async def do_report(chat_id: int, message) -> None:
    state = ensure_user(chat_id)
    language = user_language(state)
    try:
        report = call_backend_report(get_session_id(chat_id, language), language)
    except Exception as error:
        await message.reply_text(
            tr(language, "report_error", error=error),
            reply_markup=bottom_reply_keyboard(language),
        )
        return
    await message.reply_text(
        format_session_report(report, language),
        reply_markup=bottom_reply_keyboard(language),
    )


async def do_progress(chat_id: int, message) -> None:
    state = ensure_user(chat_id)
    language = user_language(state)
    try:
        progress_data = call_backend_progress(get_session_id(chat_id, language), language)
    except Exception as error:
        await message.reply_text(
            tr(language, "progress_error", error=error),
            reply_markup=bottom_reply_keyboard(language),
        )
        return
    await message.reply_text(
        format_progress_report(progress_data, language),
        reply_markup=bottom_reply_keyboard(language),
    )


async def do_finish(chat_id: int, state: Dict[str, Any], message) -> None:
    language = user_language(state)
    if not state["case_id"]:
        await message.reply_text(
            tr(language, "select_patient_first"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    if state["random_mode"]:
        state["pending_guess"] = True
        await message.reply_text(
            tr(language, "guess_prompt"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    try:
        report = call_backend_report(get_session_id(chat_id, language), language)
    except Exception as error:
        await message.reply_text(
            tr(language, "report_error", error=error),
            reply_markup=bottom_reply_keyboard(language),
        )
        return
    await message.reply_text(
        tr(language, "session_finished") + format_session_report(report, language),
        reply_markup=bottom_reply_keyboard(language),
    )


def reset_case_state(state: Dict[str, Any]) -> None:
    state["case_id"] = None
    state["random_mode"] = False
    state["pending_guess"] = False
    state["hidden_diagnosis"] = None


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    state = ensure_user(chat_id)
    data = query.data

    if data.startswith("lang:"):
        language = data.split(":", 1)[1]
        if language not in TEXTS:
            await send_language_selector(query.message)
            return
        language_changed = state.get("language") != language
        state["language"] = language
        if language_changed:
            reset_case_state(state)
        try:
            fetch_cases(language)
        except Exception as error:
            await query.message.reply_text(
                tr(language, "load_cases_error", error=error),
                reply_markup=language_keyboard(),
            )
            return
        await send_welcome(update, context)
        return

    if state.get("language") not in TEXTS:
        await send_language_selector(query.message)
        return
    language = user_language(state)

    if data == "welcome:start" or data == "menu:main":
        await send_main_menu(update, context)
        return

    if data == "menu:language":
        await send_language_selector(query.message)
        return

    if data == "menu:select_case":
        try:
            cases = fetch_cases(language)
        except Exception as error:
            await query.message.reply_text(
                tr(language, "load_cases_error", error=error),
                reply_markup=bottom_reply_keyboard(language),
            )
            return
        state["random_mode"] = False
        state["pending_guess"] = False
        state["hidden_diagnosis"] = None
        await query.message.reply_text(
            tr(language, "choose_diagnosis"),
            reply_markup=diagnosis_keyboard(cases, language),
        )
        return

    if data.startswith("diag:"):
        category_key = data.split(":", 1)[1]
        cases = CASES_CACHE[language] or fetch_cases(language)
        await query.message.reply_text(
            tr(language, "choose_patient"),
            reply_markup=patients_keyboard(cases, category_key, language),
        )
        return

    if data.startswith("case:"):
        case_id = data.split(":", 1)[1]
        state["case_id"] = case_id
        state["random_mode"] = False
        state["pending_guess"] = False
        state["hidden_diagnosis"] = None
        await query.message.reply_text(
            tr(language, "patient_selected", case_id=case_id),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    if data == "menu:random_case":
        cases = CASES_CACHE[language] or fetch_cases(language)
        if not cases:
            await query.message.reply_text(
                tr(language, "no_cases"),
                reply_markup=main_menu_keyboard(state),
            )
            return
        case = random.choice(cases)
        state["case_id"] = str(case["id"])
        state["random_mode"] = True
        state["pending_guess"] = False
        state["hidden_diagnosis"] = get_case_diagnosis_label(case)
        await query.message.reply_text(
            tr(language, "random_selected"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    if data == "menu:toggle_comm":
        await do_toggle_comm(chat_id, query.message, use_inline_menu=True)
        return
    if data == "menu:help":
        await query.message.reply_text(
            tr(language, "help_text"),
            reply_markup=main_menu_keyboard(state),
        )
        return
    if data == "menu:report":
        await do_report(chat_id, query.message)
        return
    if data == "menu:finish":
        await do_finish(chat_id, state, query.message)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)
    text = (update.message.text or "").strip()
    normalized = normalize_button_text(text)

    if state.get("language") not in TEXTS:
        await send_language_selector(update.message)
        return
    language = user_language(state)

    if normalized in ("меню", "menu"):
        await send_main_menu(update, context)
        return
    if normalized in ("сменить режим", "режим", "switch mode", "mode"):
        await do_toggle_comm(chat_id, update.message)
        return
    if normalized in ("отчёт", "отчет", "report"):
        await do_report(chat_id, update.message)
        return
    if normalized in ("прогресс", "progress"):
        await do_progress(chat_id, update.message)
        return
    if normalized in ("завершить", "завершить сессию", "finish", "finish session"):
        await do_finish(chat_id, state, update.message)
        return

    if state.get("pending_guess"):
        state["pending_guess"] = False
        guess = normalize_diag(text)
        correct = normalize_diag(state.get("hidden_diagnosis") or "")
        is_correct = bool(guess and correct) and (
            guess == correct or guess in correct or correct in guess
        )
        try:
            report = call_backend_report(get_session_id(chat_id, language), language)
            report_text = format_session_report(report, language)
        except Exception as error:
            report_text = tr(language, "report_error", error=error)
        verdict = tr(language, "correct" if is_correct else "incorrect")
        await update.message.reply_text(
            f"{verdict}\n"
            f"{tr(language, 'your_diagnosis')}: {text}\n"
            f"{tr(language, 'correct_diagnosis')}: {state.get('hidden_diagnosis')}\n\n"
            f"{report_text}",
            reply_markup=bottom_reply_keyboard(language),
        )
        state["random_mode"] = False
        state["hidden_diagnosis"] = None
        return

    if state["comm_mode"] == "voice":
        await update.message.reply_text(
            tr(language, "voice_expected"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return
    if not state["case_id"]:
        await update.message.reply_text(
            tr(language, "select_via_menu"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    try:
        data = call_backend_chat(
            get_session_id(chat_id, language),
            state["case_id"],
            text,
            language,
        )
    except Exception as error:
        await update.message.reply_text(tr(language, "server_error", error=error))
        return
    await update.message.reply_text(data.get("assistant_message", ""))


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)
    if state.get("language") not in TEXTS:
        await send_language_selector(update.message)
        return
    language = user_language(state)

    if state["comm_mode"] == "text":
        await update.message.reply_text(
            tr(language, "text_expected"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return
    if not state["case_id"]:
        await update.message.reply_text(
            tr(language, "select_via_menu"),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    voice_file = await update.message.voice.get_file()
    file_bytes = await voice_file.download_as_bytearray()
    try:
        text = await transcribe_voice(file_bytes, language)
    except Exception as error:
        await update.message.reply_text(
            tr(language, "voice_recognition_error", error=error),
            reply_markup=bottom_reply_keyboard(language),
        )
        return

    try:
        data = call_backend_chat(
            get_session_id(chat_id, language),
            state["case_id"],
            text,
            language,
        )
    except Exception as error:
        await update.message.reply_text(tr(language, "server_error", error=error))
        return

    answer = data.get("assistant_message", "")
    try:
        audio_bytes = await tts_to_bytes(answer, language)
        await update.message.reply_voice(voice=audio_bytes)
    except Exception as error:
        await update.message.reply_text(
            tr(language, "voice_synthesis_error", error=error, answer=answer)
        )


async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = ensure_user(update.effective_chat.id)
    if state.get("language") not in TEXTS:
        await send_language_selector(update.message)
        return
    await do_progress(update.effective_chat.id, update.message)


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("language", start))
    app.add_handler(CommandHandler("progress", progress))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    print("Bilingual Telegram bot started...")
    app.run_polling()


if __name__ == "__main__":
    main()
