const apiBase = "http://localhost:8000";

let sessionId = localStorage.getItem("vp_session_id") || null;
let currentCaseId = null;
let casesIndex = {};
let teacherMode = false;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatMultilineText(value) {
  return escapeHtml(value).replaceAll("\n", "<br>");
}

function trendArrow(value) {
  if (value > 0) return "↑";
  if (value < 0) return "↓";
  return "→";
}

function generateSessionId() {
  return "sess_" + Math.random().toString(36).slice(2);
}

function setSessionId(id) {
  sessionId = id;
  localStorage.setItem("vp_session_id", id);
}

async function loadCases() {
  const res = await fetch(apiBase + "/api/cases");
  if (!res.ok) {
    throw new Error(`Failed to load cases: HTTP ${res.status}`);
  }
  const data = await res.json();
  const select = document.getElementById("caseSelect");
  select.innerHTML = "";
  casesIndex = {};
  data.forEach((c) => {
    casesIndex[c.id] = c;
    const opt = document.createElement("option");
    opt.value = c.id;
    opt.textContent = c.title_for_teacher;
    select.appendChild(opt);
  });
  if (data.length > 0) {
    currentCaseId = data[0].id;
    select.value = currentCaseId;
    renderCaseInfo();
    if (teacherMode) {
      loadTeacherCase();
      loadTeacherSession();
      loadTeacherProgress();
    }
  }
}

function renderCaseInfo() {
  const infoDiv = document.getElementById("caseInfo");
  const c = casesIndex[currentCaseId];
  if (!c) {
    infoDiv.innerHTML = "";
    return;
  }
  const v = c.visible_to_student;
  infoDiv.innerHTML = `
    <h2>${escapeHtml(c.title_for_teacher)}</h2>
    <p><strong>Возраст:</strong> ${escapeHtml(v.age)}, <strong>пол:</strong> ${escapeHtml(v.gender)}</p>
    <p><strong>Контекст обращения:</strong> ${escapeHtml(v.context)}</p>
    <p><strong>Формат:</strong> ${escapeHtml(v.setting)}</p>
  `;
}

async function loadTeacherCase() {
  if (!teacherMode || !currentCaseId) return;
  const panel = document.getElementById("teacherCase");
  panel.innerHTML = "Загрузка данных кейса...";
  try {
    const res = await fetch(`${apiBase}/api/cases/${currentCaseId}/teacher`);
    if (!res.ok) {
      throw new Error(`Failed to load teacher case: HTTP ${res.status}`);
    }
    const data = await res.json();
    const hidden = data.hidden_for_student || {};
    const goals = Array.isArray(hidden.goals_for_training) ? hidden.goals_for_training : [];
    panel.innerHTML = `
      <p><strong>Предполагаемый диагноз:</strong> ${escapeHtml(hidden.provisional_diagnosis)}</p>
      <p><strong>Цели обучения:</strong></p>
      <ul>
        ${goals.map(g => `<li>${escapeHtml(g)}</li>`).join("")}
      </ul>
      <details>
        <summary>Симптомы, стиль, триггеры (подробно)</summary>
        <pre>${escapeHtml(JSON.stringify({
          symptom_profile: data.symptom_profile,
          personality_style: data.personality_style,
          typical_phrases: data.typical_phrases,
          triggers: data.triggers
        }, null, 2))}</pre>
      </details>
    `;
  } catch (e) {
    console.error(e);
    panel.innerHTML = "Ошибка загрузки данных кейса.";
  }
}

async function loadTeacherSession() {
  if (!teacherMode || !sessionId) return;
  const panel = document.getElementById("teacherSession");
  panel.innerHTML = "Загрузка информации о сессии...";
  try {
    const res = await fetch(`${apiBase}/api/sessions/${sessionId}`);
    if (!res.ok) {
      panel.innerHTML = "Сессия пока не начата или информация недоступна.";
      return;
    }
    const data = await res.json();
    const state = data.state || {};
    const history = data.history || [];
    const historyHtml = history.map(h => {
      const who = h.role === "user" ? "Студент" : "Пациент";
      return `<li><strong>${who}:</strong> ${formatMultilineText(h.content)}</li>`;
    }).join("");

    panel.innerHTML = `
      <p><strong>Сессия:</strong> ${escapeHtml(data.session_id)}</p>
      <p><strong>Кейс:</strong> ${escapeHtml(data.case_id)}</p>
      <p><strong>Состояние пациента:</strong> доверие=${escapeHtml(state.trust_level)}, эмоции=${escapeHtml(state.emotional_intensity)}, усталость=${escapeHtml(state.fatigue)}</p>
      <p><strong>Ход сессии:</strong></p>
      <ol>${historyHtml}</ol>
    `;
  } catch (e) {
    console.error(e);
    panel.innerHTML = "Ошибка загрузки информации о сессии.";
  }
}

async function loadTeacherProgress() {
  if (!teacherMode || !sessionId) return;
  const panel = document.getElementById("teacherProgress");
  panel.innerHTML = "Загрузка динамики...";
  try {
    const res = await fetch(`${apiBase}/api/sessions/${sessionId}/progress`);
    if (!res.ok) {
      panel.innerHTML = "Динамика пока недоступна.";
      return;
    }
    const data = await res.json();
    const trends = data.trends || {};
    const empathy = Number(trends.empathy || 0);
    const safety = Number(trends.safety || 0);
    const directivity = Number(trends.directivity || 0);

    panel.innerHTML = `
      <h3>Динамика</h3>
      <p><strong>Эмпатия:</strong> ${trendArrow(empathy)} (${empathy.toFixed(3)})</p>
      <p><strong>Безопасность:</strong> ${trendArrow(safety)} (${safety.toFixed(3)})</p>
      <p><strong>Директивность:</strong> ${trendArrow(directivity)} (${directivity.toFixed(3)})</p>
      <p class="teacher-note">Для директивности чаще лучше снижение.</p>
    `;
  } catch (e) {
    console.error(e);
    panel.innerHTML = "Ошибка загрузки динамики.";
  }
}

function appendMessage(role, text) {
  const chat = document.getElementById("chatWindow");
  const div = document.createElement("div");
  div.className = "message " + (role === "user" ? "user" : "assistant");

  const who = role === "user" ? "Вы" : "Пациент";
  div.innerHTML = `
    <div class="meta">${who}</div>
    <div class="body">${formatMultilineText(text)}</div>
  `;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("messageInput");
  const text = input.value.trim();
  if (!text || !currentCaseId) return;

  if (!sessionId) {
    setSessionId(generateSessionId());
  }

  appendMessage("user", text);
  input.value = "";

  const payload = {
    session_id: sessionId,
    case_id: currentCaseId,
    user_message: text,
  };

  appendMessage("assistant", "…печатает ответ…");

  try {
    const res = await fetch(apiBase + "/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const errBody = await res.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();

    const chat = document.getElementById("chatWindow");
    chat.removeChild(chat.lastChild);

    appendMessage("assistant", data.assistant_message);

    if (teacherMode) {
      loadTeacherSession();
      loadTeacherProgress();
    }
  } catch (err) {
    console.error(err);
    const chat = document.getElementById("chatWindow");
    chat.removeChild(chat.lastChild);
    appendMessage("assistant", "Ошибка при обращении к серверу.");
  }
}

function newSession() {
  setSessionId(generateSessionId());
  document.getElementById("chatWindow").innerHTML = "";
  appendMessage("assistant", "Сессия сброшена. Можете начать новую беседу с этим пациентом.");
  if (teacherMode) {
    loadTeacherSession();
    loadTeacherProgress();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadCases().catch((e) => {
    console.error(e);
    appendMessage("assistant", "Не удалось загрузить кейсы. Проверьте backend и обновите страницу.");
  });

  const select = document.getElementById("caseSelect");
  select.addEventListener("change", (e) => {
    currentCaseId = e.target.value;
    document.getElementById("chatWindow").innerHTML = "";
    renderCaseInfo();
    if (teacherMode) {
      loadTeacherCase();
      loadTeacherSession();
      loadTeacherProgress();
    }
  });

  document.getElementById("sendBtn").addEventListener("click", sendMessage);
  document.getElementById("messageInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  document.getElementById("newSessionBtn").addEventListener("click", newSession);

  const tToggle = document.getElementById("teacherModeToggle");
  tToggle.addEventListener("change", (e) => {
    teacherMode = e.target.checked;
    const panel = document.getElementById("teacherPanel");
    panel.style.display = teacherMode ? "block" : "none";
    if (teacherMode) {
      loadTeacherCase();
      loadTeacherSession();
      loadTeacherProgress();
    }
  });

  document.getElementById("refreshTeacherSessionBtn").addEventListener("click", () => {
    if (teacherMode) {
      loadTeacherSession();
      loadTeacherProgress();
    }
  });
});
