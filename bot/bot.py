import os
import random
import re
import time
import uuid
from typing import Dict, Any, List, Optional

import requests
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
SALUTESPEECH_AUTH_KEY = os.getenv("SALUTESPEECH_AUTH_KEY")
SALUTESPEECH_SCOPE = os.getenv("SALUTESPEECH_SCOPE", "SALUTE_SPEECH_PERS")
SALUTESPEECH_VERIFY_SSL = os.getenv("SALUTESPEECH_VERIFY_SSL", "1") not in ("0", "false", "False", "no", "NO")
SALUTESPEECH_STT_MODEL = os.getenv("SALUTESPEECH_STT_MODEL", "general")
SALUTESPEECH_STT_AUDIO_ENCODING = os.getenv("SALUTESPEECH_STT_AUDIO_ENCODING", "OGG_OPUS")
SALUTESPEECH_STT_SAMPLE_RATE = int(os.getenv("SALUTESPEECH_STT_SAMPLE_RATE", "48000"))
SALUTESPEECH_STT_CHANNELS = int(os.getenv("SALUTESPEECH_STT_CHANNELS", "1"))
SALUTESPEECH_TTS_VOICE = os.getenv("SALUTESPEECH_TTS_VOICE", "Nec_24000")
SALUTESPEECH_TTS_FORMAT = os.getenv("SALUTESPEECH_TTS_FORMAT", "opus")

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

# ===== глобальное состояние пользователей =====
# user_state[chat_id] = {
#   "case_id": str | None,
#   "comm_mode": "text" | "voice",
#   "welcome_seen": bool,
#   "random_mode": bool,           # инкогнито режим
#   "pending_guess": bool,         # ожидаем ввод диагноза в конце
#   "hidden_diagnosis": str | None,# правильный диагноз/категория для проверки
# }
user_state: Dict[int, Dict[str, Any]] = {}

CASES_CACHE: List[Dict[str, Any]] = []

BTN_MENU = "🏠 Меню"
BTN_FINISH = "✅ Завершить"
BTN_REPORT = "📊 Отчёт"
BTN_PROGRESS = "📈 Прогресс"
BTN_TOGGLE_MODE = "🔄 Сменить режим"

SALUTE_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SALUTE_STT_URL = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
SALUTE_TTS_URL = "https://smartspeech.sber.ru/rest/v1/text:synthesize"

_salute_token_cache: Dict[str, Any] = {"access_token": None, "expires_at": 0.0}

# ===== Утилиты =====

def ensure_user(chat_id: int) -> Dict[str, Any]:
    if chat_id not in user_state:
        user_state[chat_id] = {
            "case_id": None,
            "comm_mode": "text",     # text | voice
            "welcome_seen": False,
            "random_mode": False,
            "pending_guess": False,
            "hidden_diagnosis": None,
        }
    return user_state[chat_id]


def get_session_id(chat_id: int) -> str:
    return f"tg_{chat_id}"


def fetch_cases() -> List[Dict[str, Any]]:
    global CASES_CACHE
    resp = requests.get(f"{BACKEND_URL}/api/cases", timeout=10)
    resp.raise_for_status()
    CASES_CACHE = resp.json()
    return CASES_CACHE


def group_cases_by_category(cases: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        key = c["category_key"]
        if key not in grouped:
            grouped[key] = {"name": c["category_name"], "cases": []}
        grouped[key]["cases"].append(c)
    return grouped


def short_label(s: str, max_len: int = 22) -> str:
    s = (s or "").strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1].rstrip() + "…"


def normalize_diag(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "").strip() if ch.isalnum() or ch.isspace()).strip()


def normalize_button_text(s: str) -> str:
    s = (s or "").lower().strip()
    # Оставляем только буквы/цифры/пробелы: убираем эмодзи и служебные символы
    s = re.sub(r"[^\w\sа-яё]", " ", s, flags=re.IGNORECASE)
    return " ".join(s.split())


def get_case_by_id(case_id: str) -> Optional[Dict[str, Any]]:
    for c in CASES_CACHE:
        if str(c.get("id")) == str(case_id):
            return c
    return None


def get_case_diagnosis_label(case: Dict[str, Any]) -> str:
    """
    Что считаем 'правильным диагнозом' в инкогнито режиме.
    Можно заменить на поле backend, если оно есть.
    """
    # Предпочтение: краткое поле если есть
    for key in ("diagnosis_short", "diagnosis_name", "category_name", "category_key"):
        if case.get(key):
            return str(case.get(key))
    return str(case.get("category_key", "unknown"))


# ===== Voice / TTS =====

def get_salutespeech_token(force_refresh: bool = False) -> str:
    if not SALUTESPEECH_AUTH_KEY:
        raise RuntimeError("SALUTESPEECH_AUTH_KEY is not set")

    now = time.time()
    token = _salute_token_cache.get("access_token")
    expires_at = float(_salute_token_cache.get("expires_at") or 0.0)
    if (not force_refresh) and token and now < expires_at:
        return str(token)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": str(uuid.uuid4()),
        "Authorization": f"Basic {SALUTESPEECH_AUTH_KEY}",
    }
    payload = {"scope": SALUTESPEECH_SCOPE}
    resp = requests.post(
        SALUTE_OAUTH_URL,
        headers=headers,
        data=payload,
        timeout=20,
        verify=SALUTESPEECH_VERIFY_SSL,
    )
    resp.raise_for_status()
    data = resp.json()
    token = data.get("access_token")
    exp_ms = int(data.get("expires_at", 0) or 0)
    if not token:
        raise RuntimeError(f"No access_token in SaluteSpeech OAuth response: {data}")

    exp_ts = (exp_ms / 1000.0) - 60 if exp_ms > 0 else (time.time() + 25 * 60)
    _salute_token_cache["access_token"] = token
    _salute_token_cache["expires_at"] = max(time.time() + 60, exp_ts)
    return str(token)


async def transcribe_voice(file_bytes: bytes) -> str:
    token = get_salutespeech_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "audio/ogg;codecs=opus",
    }
    params = {
        "model": SALUTESPEECH_STT_MODEL,
        "audio_encoding": SALUTESPEECH_STT_AUDIO_ENCODING,
        "sample_rate": SALUTESPEECH_STT_SAMPLE_RATE,
        "channels_count": SALUTESPEECH_STT_CHANNELS,
    }
    resp = requests.post(
        SALUTE_STT_URL,
        headers=headers,
        params=params,
        data=file_bytes,
        timeout=60,
        verify=SALUTESPEECH_VERIFY_SSL,
    )
    if resp.status_code in (401, 403):
        token = get_salutespeech_token(force_refresh=True)
        headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(
            SALUTE_STT_URL,
            headers=headers,
            params=params,
            data=file_bytes,
            timeout=60,
            verify=SALUTESPEECH_VERIFY_SSL,
        )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):
        for key in ("text", "result", "transcript"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
            if isinstance(val, list):
                parts: List[str] = []
                for item in val:
                    if isinstance(item, str) and item.strip():
                        parts.append(item.strip())
                    elif isinstance(item, dict):
                        txt = item.get("text")
                        if isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                if parts:
                    return " ".join(parts).strip()
        hypotheses = data.get("hypotheses")
        if isinstance(hypotheses, list) and hypotheses:
            first = hypotheses[0]
            if isinstance(first, dict):
                txt = first.get("text")
                if isinstance(txt, str) and txt.strip():
                    return txt.strip()
    raise RuntimeError("Не удалось распознать речь. Попробуйте говорить чуть громче и без паузы в начале.")


async def tts_to_bytes(text: str) -> bytes:
    token = get_salutespeech_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/text",
    }
    payload = (text or "").encode("utf-8")

    # У разных версий SaluteSpeech могут отличаться имена query-параметров.
    # Пробуем несколько совместимых вариантов.
    candidates = [
        {"voice": SALUTESPEECH_TTS_VOICE, "format": SALUTESPEECH_TTS_FORMAT},
        {"voice": SALUTESPEECH_TTS_VOICE, "audio_encoding": SALUTESPEECH_TTS_FORMAT},
        {"voice": SALUTESPEECH_TTS_VOICE, "audio_encoding": "opus"},
        {"voice": SALUTESPEECH_TTS_VOICE, "format": "oggopus"},
    ]

    last_error_text = ""
    for params in candidates:
        resp = requests.post(
            SALUTE_TTS_URL,
            headers=headers,
            params=params,
            data=payload,
            timeout=60,
            verify=SALUTESPEECH_VERIFY_SSL,
        )
        if resp.status_code in (401, 403):
            token = get_salutespeech_token(force_refresh=True)
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.post(
                SALUTE_TTS_URL,
                headers=headers,
                params=params,
                data=payload,
                timeout=60,
                verify=SALUTESPEECH_VERIFY_SSL,
            )

        if resp.ok:
            return resp.content

        # Для совместимости пробуем следующий формат только на 400.
        # Остальные статусы считаем финальной ошибкой.
        try:
            last_error_text = resp.text[:500]
        except Exception:
            last_error_text = f"HTTP {resp.status_code}"
        if resp.status_code != 400:
            resp.raise_for_status()

    raise RuntimeError(
        f"SaluteSpeech TTS error: unsupported params for voice={SALUTESPEECH_TTS_VOICE}, "
        f"format={SALUTESPEECH_TTS_FORMAT}. Details: {last_error_text}"
    )


# ===== Backend calls =====

def call_backend_chat(session_id: str, case_id: str, user_message: str) -> Dict[str, Any]:
    payload = {
        "session_id": session_id,
        "case_id": case_id,
        "user_message": user_message,
        "teacher_mode": False,  # всегда без покадрового разбора
    }
    resp = requests.post(f"{BACKEND_URL}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_backend_report(session_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{BACKEND_URL}/api/session_report", params={"session_id": session_id}, timeout=20)
    resp.raise_for_status()
    return resp.json()


def call_backend_progress(session_id: str) -> Dict[str, Any]:
    resp = requests.get(f"{BACKEND_URL}/api/sessions/{session_id}/progress", timeout=20)
    resp.raise_for_status()
    return resp.json()


# ===== Клавиатуры =====

def welcome_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("✅ Начать", callback_data="welcome:start")],
    ])


def bottom_reply_keyboard() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [
            [BTN_MENU, BTN_FINISH, BTN_REPORT, BTN_PROGRESS],
            [BTN_TOGGLE_MODE],
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
    )


def main_menu_keyboard(state: Dict[str, Any]) -> InlineKeyboardMarkup:
    comm = state["comm_mode"]
    comm_label = "🎤 Голос" if comm == "text" else "💬 Текст"

    buttons = [
        [InlineKeyboardButton("👥 Пациент (по диагнозу)", callback_data="menu:select_case")],
        [InlineKeyboardButton("🎲 Случайный пациент", callback_data="menu:random_case")],
        [InlineKeyboardButton(comm_label, callback_data="menu:toggle_comm")],
        [InlineKeyboardButton("✅ Завершить сессию", callback_data="menu:finish")],
        [InlineKeyboardButton("📊 Отчёт", callback_data="menu:report")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="menu:help")],
    ]
    return InlineKeyboardMarkup(buttons)


def back_to_main_button() -> List[InlineKeyboardButton]:
    return [InlineKeyboardButton("🏠 Меню", callback_data="menu:main")]


def diagnosis_keyboard(cases: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    grouped = group_cases_by_category(cases)
    rows: List[List[InlineKeyboardButton]] = []
    for key, data in grouped.items():
        rows.append([InlineKeyboardButton(short_label(data["name"], 24), callback_data=f"diag:{key}")])
    rows.append(back_to_main_button())
    return InlineKeyboardMarkup(rows)


def patients_keyboard(cases: List[Dict[str, Any]], category_key: str) -> InlineKeyboardMarkup:
    grouped = group_cases_by_category(cases)
    if category_key not in grouped:
        return InlineKeyboardMarkup([back_to_main_button()])

    rows: List[List[InlineKeyboardButton]] = []
    for c in grouped[category_key]["cases"]:
        # Стараемся брать краткое поле, иначе teacher title, иначе title
        title = c.get("title_short") or c.get("title_for_teacher") or c.get("title") or f"case {c.get('id')}"
        rows.append([InlineKeyboardButton(short_label(str(title), 26), callback_data=f"case:{c['id']}")])
    rows.append([InlineKeyboardButton("⬅️ Назад", callback_data="menu:select_case")])
    rows.append(back_to_main_button())
    return InlineKeyboardMarkup(rows)


# ===== Форматирование отчёта =====

def format_session_report(rep: Dict[str, Any]) -> str:
    examples = rep.get("improved_examples") or []
    examples_txt = ""
    if isinstance(examples, list) and examples:
        parts: List[str] = []
        for i, ex in enumerate(examples[:3], start=1):
            if not isinstance(ex, dict):
                continue
            original = str(ex.get("original_replica", "")).strip()
            better = str(ex.get("better_replica", "")).strip()
            why = str(ex.get("why_better", "")).strip()
            if not original or not better:
                continue
            parts.append(
                f"{i}. Было: {original}\n"
                f"   Лучше: {better}\n"
                f"   Почему: {why or '-'}"
            )
        if parts:
            examples_txt = "\n\n🧠 Примеры, как лучше переформулировать:\n" + "\n\n".join(parts)

    return (
        "📊 Итоговый отчёт по сессии\n"
        f"• Кейс: {rep.get('case_id')}\n"
        f"• Кол-во ходов: {rep.get('num_turns')}\n\n"
        "🎯 Средние показатели\n"
        f"• Эмпатия: {rep.get('avg_empathy', 0):.2f}\n"
        f"• Валидация: {rep.get('avg_validation', 0):.2f}\n"
        f"• Директивность: {rep.get('avg_directivity', 0):.2f}\n"
        f"• Открытые вопросы: {rep.get('avg_open_question', 0):.2f}\n"
        f"• Безопасность: {rep.get('avg_safety', 0):.2f}\n"
        f"• Индекс эффективности: {rep.get('mean_efficiency_index', 0):.2f}\n\n"
        "📉 Суммарные изменения состояния пациента\n"
        f"• Δ доверия: {rep.get('total_delta_trust')}\n"
        f"• Δ эмоц. интенсивности: {rep.get('total_delta_emotional_intensity')}\n"
        f"• Δ усталости: {rep.get('total_delta_fatigue')}\n\n"
        "🧾 Общее впечатление:\n"
        f"{rep.get('overall_impression', '-')}\n\n"
        "🛠 Рекомендации:\n"
        f"{rep.get('recommendations', '-')}"
        f"{examples_txt}"
    )


def trend_arrow(v: float) -> str:
    if v > 0:
        return "↑"
    if v < 0:
        return "↓"
    return "→"


def format_progress_report(progress: Dict[str, Any]) -> str:
    trends = progress.get("trends", {}) or {}
    em = float(trends.get("empathy", 0) or 0)
    sf = float(trends.get("safety", 0) or 0)
    dr = float(trends.get("directivity", 0) or 0)

    return (
        "📈 Динамика сессии\n"
        f"• Кейс: {progress.get('case_id')}\n"
        f"• Ходов: {progress.get('num_turns')}\n\n"
        "Тренды по навыкам:\n"
        f"• Эмпатия: {trend_arrow(em)} ({em:+.3f})\n"
        f"• Безопасность: {trend_arrow(sf)} ({sf:+.3f})\n"
        f"• Директивность: {trend_arrow(dr)} ({dr:+.3f})\n"
        "  (для директивности чаще лучше снижение)\n\n"
        "Команда: /progress"
    )


# ===== Handlers =====

async def send_welcome(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "👋 Добро пожаловать в симулятор виртуальных пациентов.\n\n"
        "Как работает:\n"
        "• Вы выбираете пациента (или берёте случайного).\n"
        "• Общаетесь (текст/голос).\n"
        "• В конце — итоговый отчёт.\n\n"
        "Нажмите «✅ Начать»."
    )
    if update.message:
        await update.message.reply_text(text, reply_markup=welcome_keyboard())
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=welcome_keyboard())


async def send_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)

    text = (
        "🏠 Меню\n\n"
        "• 👥 Пациент (по диагнозу) — вы знаете тему.\n"
        "• 🎲 Случайный пациент — инкогнито режим (диагноз угадывается в конце).\n"
        "• ✅ Завершить — закончить сессию (в инкогнито попросит диагноз).\n"
        "• 📊 Отчёт — итоговый отчёт по сессии."
    )

    if update.message:
        await update.message.reply_text(text, reply_markup=bottom_reply_keyboard())
        await update.message.reply_text("Выберите действие:", reply_markup=main_menu_keyboard(state))
    elif update.callback_query:
        await update.callback_query.message.reply_text(text, reply_markup=bottom_reply_keyboard())
        await update.callback_query.message.reply_text("Выберите действие:", reply_markup=main_menu_keyboard(state))


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)

    # Подтягиваем кейсы сразу
    try:
        fetch_cases()
    except Exception as e:
        await update.message.reply_text(f"Не удалось загрузить кейсы: {e}", reply_markup=bottom_reply_keyboard())
        return

    # Приветственный экран только при первом входе
    if not state["welcome_seen"]:
        state["welcome_seen"] = True
        await send_welcome(update, context)
        return

    await send_main_menu(update, context)


async def do_toggle_comm(chat_id: int, message, use_inline_menu: bool = False):
    state = ensure_user(chat_id)
    state["comm_mode"] = "voice" if state["comm_mode"] == "text" else "text"
    mode_txt = (
        "🎤 Включён голосовой режим. Отправляйте голосовые."
        if state["comm_mode"] == "voice"
        else "💬 Включён текстовый режим. Отправляйте текст."
    )
    reply_markup = main_menu_keyboard(state) if use_inline_menu else bottom_reply_keyboard()
    await message.reply_text(mode_txt, reply_markup=reply_markup)


async def do_report(chat_id: int, message):
    session_id = get_session_id(chat_id)
    try:
        rep = call_backend_report(session_id)
    except Exception as e:
        await message.reply_text(f"Не удалось получить отчёт: {e}", reply_markup=bottom_reply_keyboard())
        return
    await message.reply_text(format_session_report(rep), reply_markup=bottom_reply_keyboard())


async def do_progress(chat_id: int, message):
    session_id = get_session_id(chat_id)
    try:
        progress = call_backend_progress(session_id)
    except Exception as e:
        await message.reply_text(f"Не удалось получить прогресс: {e}", reply_markup=bottom_reply_keyboard())
        return
    await message.reply_text(format_progress_report(progress), reply_markup=bottom_reply_keyboard())


async def do_finish(chat_id: int, state: Dict[str, Any], message):
    if not state["case_id"]:
        await message.reply_text("Сначала выберите пациента.", reply_markup=bottom_reply_keyboard())
        return

    if state["random_mode"]:
        state["pending_guess"] = True
        await message.reply_text(
            "✅ Сессия завершена.\n\n"
            "Теперь напишите *диагноз*, который вы предполагаете (кратко).\n"
            "Например: «депрессия», «паническая атака», «ОКР» и т.д.",
            reply_markup=bottom_reply_keyboard(),
            parse_mode="Markdown",
        )
        return

    session_id = get_session_id(chat_id)
    try:
        rep = call_backend_report(session_id)
    except Exception as e:
        await message.reply_text(f"Не удалось получить отчёт: {e}", reply_markup=bottom_reply_keyboard())
        return
    await message.reply_text("✅ Сессия завершена.\n\n" + format_session_report(rep), reply_markup=bottom_reply_keyboard())


async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    state = ensure_user(chat_id)
    data = query.data

    if data == "welcome:start":
        await send_main_menu(update, context)
        return

    if data == "menu:main":
        await send_main_menu(update, context)
        return

    if data == "menu:select_case":
        try:
            cases = fetch_cases()
        except Exception as e:
            await query.message.reply_text(f"Ошибка загрузки кейсов: {e}", reply_markup=bottom_reply_keyboard())
            return
        state["random_mode"] = False
        state["pending_guess"] = False
        state["hidden_diagnosis"] = None
        await query.message.reply_text("Выберите диагноз:", reply_markup=diagnosis_keyboard(cases))
        return

    if data.startswith("diag:"):
        _, diag_key = data.split(":", 1)
        if not CASES_CACHE:
            fetch_cases()
        await query.message.reply_text("Выберите пациента:", reply_markup=patients_keyboard(CASES_CACHE, diag_key))
        return

    if data.startswith("case:"):
        case_id = data.split(":", 1)[1]
        state["case_id"] = case_id
        state["random_mode"] = False
        state["pending_guess"] = False
        state["hidden_diagnosis"] = None

        await query.message.reply_text(
            f"✅ Пациент выбран: {case_id}\n\nТеперь можете задавать вопросы пациенту.\n\n"
            "Чтобы снова открыть меню, отправьте /start или «🏠 Меню».",
            reply_markup=bottom_reply_keyboard(),
        )
        return

    if data == "menu:random_case":
        if not CASES_CACHE:
            fetch_cases()
        if not CASES_CACHE:
            await query.message.reply_text("Нет доступных кейсов.", reply_markup=main_menu_keyboard(state))
            return

        c = random.choice(CASES_CACHE)
        state["case_id"] = str(c["id"])
        state["random_mode"] = True
        state["pending_guess"] = False
        state["hidden_diagnosis"] = get_case_diagnosis_label(c)

        await query.message.reply_text(
            "🎲 Случайный пациент выбран (инкогнито).\n"
            "Диагноз скрыт. Общайтесь, как на приёме.\n\n"
            "Чтобы завершить сессию или открыть меню, отправьте /start или «🏠 Меню».",
            reply_markup=bottom_reply_keyboard(),
        )
        return

    if data == "menu:toggle_comm":
        await do_toggle_comm(chat_id, query.message, use_inline_menu=True)
        return

    if data == "menu:help":
        txt = (
            "ℹ️ Помощь\n\n"
            "• /start или «🏠 Меню» — открыть меню.\n"
            "• 🎤/💬 — переключение голос/текст.\n"
            "• 🎲 Случайный пациент — диагноз узнаёте только в конце.\n"
            "• ✅ Завершить — завершает сессию; в инкогнито попросит ваш диагноз.\n"
            "• 📊 Отчёт — итоговый отчёт по сессии."
        )
        await query.message.reply_text(txt, reply_markup=main_menu_keyboard(state))
        return

    if data == "menu:report":
        await do_report(chat_id, query.message)
        return

    if data == "menu:finish":
        await do_finish(chat_id, state, query.message)
        return


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)
    text = (update.message.text or "").strip()
    normalized_text = normalize_button_text(text)

    # Быстрые кнопки нижней клавиатуры
    if text in ("/start",) or normalized_text == "меню":
        await send_main_menu(update, context)
        return
    if normalized_text in ("сменить режим", "режим"):
        await do_toggle_comm(chat_id, update.message)
        return
    if normalized_text in ("отчёт", "отчет"):
        await do_report(chat_id, update.message)
        return
    if normalized_text == "прогресс":
        await do_progress(chat_id, update.message)
        return
    if normalized_text in ("завершить", "завершить сессию"):
        await do_finish(chat_id, state, update.message)
        return

    # Если ждём диагноз в конце инкогнито-режима
    if state.get("pending_guess"):
        state["pending_guess"] = False

        guess = normalize_diag(text)
        correct = normalize_diag(state.get("hidden_diagnosis") or "")
        ok = bool(guess) and bool(correct) and (guess == correct or guess in correct or correct in guess)

        # Отчёт
        session_id = get_session_id(chat_id)
        try:
            rep = call_backend_report(session_id)
            rep_txt = format_session_report(rep)
        except Exception as e:
            rep_txt = f"(Не удалось получить отчёт: {e})"

        verdict = "✅ Верно!" if ok else "❌ Неверно."
        await update.message.reply_text(
            f"{verdict}\n"
            f"Ваш диагноз: {text}\n"
            f"Правильный диагноз: {state.get('hidden_diagnosis')}\n\n"
            f"{rep_txt}",
            reply_markup=bottom_reply_keyboard(),
        )

        # Сбрасываем инкогнито-режим (чтобы следующая сессия начиналась чисто)
        state["random_mode"] = False
        state["hidden_diagnosis"] = None
        return

    if state["comm_mode"] == "voice":
        await update.message.reply_text(
            "Сейчас включён голосовой режим. Отправьте голосовое или переключитесь на текст в меню.",
            reply_markup=bottom_reply_keyboard(),
        )
        return

    if not state["case_id"]:
        await update.message.reply_text("Сначала выберите пациента через меню (/start).", reply_markup=bottom_reply_keyboard())
        return

    session_id = get_session_id(chat_id)

    try:
        data = call_backend_chat(session_id, state["case_id"], text)
    except Exception as e:
        await update.message.reply_text(f"Ошибка при обращении к серверу: {e}")
        return

    await update.message.reply_text(data.get("assistant_message", ""))


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    state = ensure_user(chat_id)

    if state["comm_mode"] == "text":
        await update.message.reply_text(
            "Сейчас включён текстовый режим. Отправьте текст или переключитесь на голос в меню.",
            reply_markup=bottom_reply_keyboard(),
        )
        return

    if not state["case_id"]:
        await update.message.reply_text("Сначала выберите пациента через меню (/start).", reply_markup=bottom_reply_keyboard())
        return

    voice = update.message.voice
    file = await voice.get_file()
    file_bytes = await file.download_as_bytearray()

    try:
        text = await transcribe_voice(file_bytes)
    except Exception as e:
        await update.message.reply_text(f"Не удалось распознать голос: {e}", reply_markup=bottom_reply_keyboard())
        return

    session_id = get_session_id(chat_id)

    try:
        data = call_backend_chat(session_id, state["case_id"], text)
    except Exception as e:
        await update.message.reply_text(f"Ошибка при обращении к серверу: {e}")
        return

    # Пациент отвечает голосом
    try:
        audio_bytes = await tts_to_bytes(data.get("assistant_message", ""))
        await update.message.reply_voice(voice=audio_bytes)
    except Exception as e:
        await update.message.reply_text(
            f"(Ошибка синтеза голоса: {e})\n\nОтвет пациента:\n{data.get('assistant_message','')}",
        )


async def progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    ensure_user(chat_id)
    await do_progress(chat_id, update.message)


def main():
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("progress", progress))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))

    print("Telegram bot started...")
    app.run_polling()


if __name__ == "__main__":
    main()
