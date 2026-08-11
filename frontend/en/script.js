const isLocalDevelopment = ["localhost", "127.0.0.1"].includes(window.location.hostname);
const apiBase = window.VP_API_BASE || (isLocalDevelopment ? "http://localhost:8001" : `${window.location.origin}/en`);

let sessionId = localStorage.getItem("vp_en_session_id") || null;
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
  return "sess_en_" + Math.random().toString(36).slice(2);
}

function setSessionId(id) {
  sessionId = id;
  localStorage.setItem("vp_en_session_id", id);
}

async function loadCases() {
  const res = await fetch(`${apiBase}/api/cases`);
  if (!res.ok) throw new Error(`Failed to load cases: HTTP ${res.status}`);

  const data = await res.json();
  const select = document.getElementById("caseSelect");
  select.innerHTML = "";
  casesIndex = {};
  data.forEach((item) => {
    casesIndex[item.id] = item;
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = `${item.title_for_teacher} · ${item.category_name}`;
    select.appendChild(option);
  });

  if (data.length > 0) {
    currentCaseId = data[0].id;
    select.value = currentCaseId;
    renderCaseInfo();
    if (teacherMode) await refreshTeacherData();
  }
}

function renderCaseInfo() {
  const info = document.getElementById("caseInfo");
  const current = casesIndex[currentCaseId];
  if (!current) {
    info.innerHTML = "";
    return;
  }
  const visible = current.visible_to_student;
  info.innerHTML = `
    <h2>${escapeHtml(current.title_for_teacher)}</h2>
    <p><strong>Age:</strong> ${escapeHtml(visible.age)} · <strong>Gender:</strong> ${escapeHtml(visible.gender)}</p>
    <p><strong>Presenting context:</strong> ${escapeHtml(visible.context)}</p>
    <p><strong>Setting:</strong> ${escapeHtml(visible.setting)}</p>
    <p><strong>Initial concern:</strong> ${escapeHtml(visible.initial_request)}</p>
    <p><strong>Clinical note:</strong> ${escapeHtml(visible.warning_to_student)}</p>
  `;
}

async function loadTeacherCase() {
  if (!teacherMode || !currentCaseId) return;
  const panel = document.getElementById("teacherCase");
  panel.textContent = "Loading case details...";
  try {
    const res = await fetch(`${apiBase}/api/cases/${currentCaseId}/teacher`);
    if (!res.ok) throw new Error(`Failed to load instructor case: HTTP ${res.status}`);
    const data = await res.json();
    const hidden = data.hidden_for_student || {};
    const goals = Array.isArray(hidden.goals_for_training) ? hidden.goals_for_training : [];
    panel.innerHTML = `
      <p><strong>Provisional diagnosis:</strong> ${escapeHtml(hidden.provisional_diagnosis)}</p>
      <p><strong>Training goals:</strong></p>
      <ul>${goals.map((goal) => `<li>${escapeHtml(goal)}</li>`).join("")}</ul>
      <details>
        <summary>Symptoms, personality style, and triggers</summary>
        <pre>${escapeHtml(JSON.stringify({
          symptom_profile: data.symptom_profile,
          personality_style: data.personality_style,
          typical_phrases: data.typical_phrases,
          triggers: data.triggers,
        }, null, 2))}</pre>
      </details>
    `;
  } catch (error) {
    console.error(error);
    panel.textContent = "The case details could not be loaded.";
  }
}

async function loadTeacherSession() {
  if (!teacherMode || !sessionId) return;
  const panel = document.getElementById("teacherSession");
  panel.textContent = "Loading session details...";
  try {
    const res = await fetch(`${apiBase}/api/sessions/${sessionId}`);
    if (!res.ok) {
      panel.textContent = "The session has not started yet, or its details are unavailable.";
      return;
    }
    const data = await res.json();
    const state = data.state || {};
    const history = data.history || [];
    const historyHtml = history.map((turn) => {
      const speaker = turn.role === "user" ? "Student" : "Patient";
      return `<li><strong>${speaker}:</strong> ${formatMultilineText(turn.content)}</li>`;
    }).join("");

    panel.innerHTML = `
      <p><strong>Session:</strong> ${escapeHtml(data.session_id)}</p>
      <p><strong>Case:</strong> ${escapeHtml(data.case_id)}</p>
      <p><strong>Patient state:</strong> trust=${escapeHtml(state.trust_level)}, emotional intensity=${escapeHtml(state.emotional_intensity)}, fatigue=${escapeHtml(state.fatigue)}</p>
      <p><strong>Session history:</strong></p>
      <ol>${historyHtml}</ol>
    `;
  } catch (error) {
    console.error(error);
    panel.textContent = "The session details could not be loaded.";
  }
}

async function loadTeacherProgress() {
  if (!teacherMode || !sessionId) return;
  const panel = document.getElementById("teacherProgress");
  panel.textContent = "Loading skill trends...";
  try {
    const res = await fetch(`${apiBase}/api/sessions/${sessionId}/progress`);
    if (!res.ok) {
      panel.textContent = "Skill trends are not available yet.";
      return;
    }
    const data = await res.json();
    const trends = data.trends || {};
    const empathy = Number(trends.empathy || 0);
    const safety = Number(trends.safety || 0);
    const directivity = Number(trends.directivity || 0);
    panel.innerHTML = `
      <h3>Skill trends</h3>
      <p><strong>Empathy:</strong> ${trendArrow(empathy)} (${empathy.toFixed(3)})</p>
      <p><strong>Safety:</strong> ${trendArrow(safety)} (${safety.toFixed(3)})</p>
      <p><strong>Directivity:</strong> ${trendArrow(directivity)} (${directivity.toFixed(3)})</p>
      <p class="teacher-note">A downward trend is usually preferable for directivity.</p>
    `;
  } catch (error) {
    console.error(error);
    panel.textContent = "Skill trends could not be loaded.";
  }
}

async function refreshTeacherData() {
  await Promise.all([loadTeacherCase(), loadTeacherSession(), loadTeacherProgress()]);
}

function appendMessage(role, text) {
  const chat = document.getElementById("chatWindow");
  const message = document.createElement("div");
  message.className = `message ${role === "user" ? "user" : "assistant"}`;
  const speaker = role === "user" ? "You" : "Patient";
  message.innerHTML = `<div class="meta">${speaker}</div><div class="body">${formatMultilineText(text)}</div>`;
  chat.appendChild(message);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById("messageInput");
  const text = input.value.trim();
  if (!text || !currentCaseId) return;
  if (!sessionId) setSessionId(generateSessionId());

  appendMessage("user", text);
  input.value = "";
  appendMessage("assistant", "Typing...");

  try {
    const res = await fetch(`${apiBase}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        case_id: currentCaseId,
        user_message: text,
      }),
    });
    if (!res.ok) {
      const errorBody = await res.json().catch(() => ({}));
      throw new Error(errorBody.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    const chat = document.getElementById("chatWindow");
    chat.removeChild(chat.lastChild);
    appendMessage("assistant", data.assistant_message);
    if (teacherMode) await Promise.all([loadTeacherSession(), loadTeacherProgress()]);
  } catch (error) {
    console.error(error);
    const chat = document.getElementById("chatWindow");
    chat.removeChild(chat.lastChild);
    appendMessage("assistant", "The server could not process the request. Please try again.");
  }
}

function newSession() {
  setSessionId(generateSessionId());
  document.getElementById("chatWindow").innerHTML = "";
  appendMessage("assistant", "The session has been reset. You can begin a new conversation with this patient.");
  if (teacherMode) refreshTeacherData();
}

document.addEventListener("DOMContentLoaded", () => {
  loadCases().catch((error) => {
    console.error(error);
    appendMessage("assistant", "The clinical cases could not be loaded. Check the backend and refresh the page.");
  });

  document.getElementById("caseSelect").addEventListener("change", (event) => {
    currentCaseId = event.target.value;
    document.getElementById("chatWindow").innerHTML = "";
    renderCaseInfo();
    if (teacherMode) refreshTeacherData();
  });

  document.getElementById("sendBtn").addEventListener("click", sendMessage);
  document.getElementById("messageInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  });
  document.getElementById("newSessionBtn").addEventListener("click", newSession);
  document.getElementById("teacherModeToggle").addEventListener("change", (event) => {
    teacherMode = event.target.checked;
    document.getElementById("teacherPanel").style.display = teacherMode ? "block" : "none";
    if (teacherMode) refreshTeacherData();
  });
  document.getElementById("refreshTeacherSessionBtn").addEventListener("click", refreshTeacherData);
});
