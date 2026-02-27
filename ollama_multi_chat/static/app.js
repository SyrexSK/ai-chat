const statusEl = document.getElementById("status");
const hostEl = document.getElementById("host");
const timeoutEl = document.getElementById("timeout");
const delayEl = document.getElementById("delay");
const agentsEl = document.getElementById("agents");
const agentNameEl = document.getElementById("agentName");
const agentModelEl = document.getElementById("agentModel");
const promptPresetEl = document.getElementById("promptPreset");
const systemPromptEl = document.getElementById("systemPrompt");
const historyEl = document.getElementById("history");
const userTextEl = document.getElementById("userText");

let selectedAgent = "";
let state = null;
let configDirty = false;

async function api(path, method = "GET", body = null) {
  const response = await fetch(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : null,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail || payload.message || `HTTP ${response.status}`);
  }
  return payload;
}

function setStatus(text) {
  statusEl.textContent = text || "";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMessageContent(value) {
  const escaped = escapeHtml(value);
  return escaped
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`(.+?)`/g, "<code>$1</code>")
    .replace(/\n/g, "<br>");
}

function renderState(nextState) {
  state = nextState;
  if (!configDirty) {
    hostEl.value = state.host || hostEl.value;
    timeoutEl.value = state.timeout;
    delayEl.value = state.delay;
  }
  setStatus(state.status || "");

  const models = state.models || [];
  const currentModel = agentModelEl.value;
  agentModelEl.innerHTML = models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  if (currentModel && models.includes(currentModel)) {
    agentModelEl.value = currentModel;
  }

  const prompts = state.prompts || {};
  const promptKeys = Object.keys(prompts);
  promptPresetEl.innerHTML = promptKeys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  if (!systemPromptEl.value && promptKeys.length > 0) {
    systemPromptEl.value = prompts[promptKeys[0]];
  }

  agentsEl.innerHTML = (state.agents || [])
    .map((agent) => {
      const active = agent.name === selectedAgent ? "active" : "";
      return `<li data-name="${escapeHtml(agent.name)}" class="${active}">${escapeHtml(agent.name)} &lt;${escapeHtml(agent.model)}&gt;</li>`;
    })
    .join("");

  if (selectedAgent && !state.agents.some((agent) => agent.name === selectedAgent)) {
    selectedAgent = "";
  }

  const prevScrollTop = historyEl.scrollTop;
  const prevScrollHeight = historyEl.scrollHeight;
  const wasNearBottom = prevScrollHeight - (prevScrollTop + historyEl.clientHeight) < 40;

  historyEl.innerHTML = (state.history || [])
    .map((msg) => {
      const role = msg.role || "assistant";
      return `<div class="msg ${escapeHtml(role)}"><div class="head">${escapeHtml(msg.speaker)}</div><div>${renderMessageContent(msg.content || "")}</div></div>`;
    })
    .join("");

  if (wasNearBottom) {
    historyEl.scrollTop = historyEl.scrollHeight;
  } else {
    const delta = historyEl.scrollHeight - prevScrollHeight;
    historyEl.scrollTop = Math.max(0, prevScrollTop + delta);
  }

  if (state.last_error) {
    setStatus(`${state.status} | ${state.last_error}`);
  }
}

async function loadState() {
  const payload = await api("/api/state");
  renderState(payload);
}

async function safeAction(handler) {
  try {
    await handler();
  } catch (error) {
    setStatus(error.message || String(error));
  }
}

function setupEvents() {
  document.getElementById("saveConfig").addEventListener("click", () =>
    safeAction(async () => {
      const payload = await api("/api/config", "POST", {
        host: hostEl.value,
        timeout: Number(timeoutEl.value),
        delay: Number(delayEl.value),
      });
      configDirty = false;
      renderState(payload);
    }),
  );

  document.getElementById("refreshModels").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/models");
      await loadState();
    }),
  );

  document.getElementById("addAgent").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/agents", "POST", {
        name: agentNameEl.value.trim(),
        model: agentModelEl.value,
        system_prompt: systemPromptEl.value,
      });
      agentNameEl.value = "";
      await loadState();
    }),
  );

  document.getElementById("removeAgent").addEventListener("click", () =>
    safeAction(async () => {
      if (!selectedAgent) {
        throw new Error("Выберите агента в списке");
      }
      await api(`/api/agents/${encodeURIComponent(selectedAgent)}`, "DELETE");
      selectedAgent = "";
      await loadState();
    }),
  );

  document.getElementById("sendMessage").addEventListener("click", () =>
    safeAction(async () => {
      const text = userTextEl.value.trim();
      if (!text) {
        return;
      }
      await api("/api/messages", "POST", { text });
      userTextEl.value = "";
      await loadState();
    }),
  );

  document.getElementById("startChat").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/chat/start", "POST");
      await loadState();
    }),
  );

  document.getElementById("stopChat").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/chat/stop", "POST");
      await loadState();
    }),
  );

  document.getElementById("resetChat").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/chat/reset", "POST");
      await loadState();
    }),
  );

  promptPresetEl.addEventListener("change", () => {
    const key = promptPresetEl.value;
    if (!state?.prompts || !state.prompts[key]) {
      return;
    }
    systemPromptEl.value = state.prompts[key];
  });

  agentsEl.addEventListener("click", (event) => {
    const li = event.target.closest("li[data-name]");
    if (!li) {
      return;
    }
    selectedAgent = li.dataset.name;
    renderState(state);
  });

  userTextEl.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      document.getElementById("sendMessage").click();
    }
  });

  hostEl.addEventListener("input", () => {
    configDirty = true;
  });
  timeoutEl.addEventListener("input", () => {
    configDirty = true;
  });
  delayEl.addEventListener("input", () => {
    configDirty = true;
  });
}

setupEvents();
loadState();
setInterval(() => {
  loadState().catch((error) => setStatus(error.message || String(error)));
}, 2000);
