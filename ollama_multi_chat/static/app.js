const statusEl = document.getElementById("status");
const providerEl = document.getElementById("provider");
const hostEl = document.getElementById("host");
const apiKeyEl = document.getElementById("apiKey");
const tokenEl = document.getElementById("token");
const timeoutEl = document.getElementById("timeout");
const delayEl = document.getElementById("delay");
const agentsEl = document.getElementById("agents");
const agentNameEl = document.getElementById("agentName");
const agentModelEl = document.getElementById("agentModel");
const promptPresetEl = document.getElementById("promptPreset");
const systemPromptEl = document.getElementById("systemPrompt");
const historyEl = document.getElementById("history");
const commentatorHistoryEl = document.getElementById("commentatorHistory");
const userTextEl = document.getElementById("userText");
const commentatorStatusEl = document.getElementById("commentatorStatus");
const commentatorNameEl = document.getElementById("commentatorName");
const commentatorModelEl = document.getElementById("commentatorModel");
const commentatorPresetEl = document.getElementById("commentatorPreset");
const commentatorPromptEl = document.getElementById("commentatorPrompt");

let selectedAgent = "";
let state = null;
let configDirty = false;
let systemPromptDirty = false;
let commentatorDirty = false;

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

function renderHistory(container, messages) {
  const prevScrollTop = container.scrollTop;
  const prevScrollHeight = container.scrollHeight;
  const wasNearBottom = prevScrollHeight - (prevScrollTop + container.clientHeight) < 40;

  container.innerHTML = (messages || [])
    .map((msg) => {
      const role = msg.role || "assistant";
      return `<div class="msg ${escapeHtml(role)}"><div class="head">${escapeHtml(msg.speaker)}</div><div>${renderMessageContent(msg.content || "")}</div></div>`;
    })
    .join("");

  if (wasNearBottom) {
    container.scrollTop = container.scrollHeight;
  } else {
    const delta = container.scrollHeight - prevScrollHeight;
    container.scrollTop = Math.max(0, prevScrollTop + delta);
  }
}

function renderState(nextState) {
  state = nextState;
  if (!configDirty) {
    providerEl.value = state.provider || "ollama";
    hostEl.value = state.host || hostEl.value;
    apiKeyEl.value = state.api_key || "";
    tokenEl.value = state.token || "";
    timeoutEl.value = state.timeout;
    delayEl.value = state.delay;
  }
  setStatus(state.status || "");

  const models = state.models || [];
  const currentModel = agentModelEl.value;
  agentModelEl.innerHTML = models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  if (currentModel && models.includes(currentModel)) {
    agentModelEl.value = currentModel;
  } else if (models.length > 0) {
    agentModelEl.value = models[0];
  }

  const prompts = state.prompts || {};
  const promptKeys = Object.keys(prompts);
  const currentPreset = promptPresetEl.value;
  promptPresetEl.innerHTML = promptKeys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  if (currentPreset && promptKeys.includes(currentPreset)) {
    promptPresetEl.value = currentPreset;
  } else if (promptKeys.length > 0) {
    promptPresetEl.value = promptKeys[0];
  }
  if (!systemPromptDirty && promptPresetEl.value) {
    systemPromptEl.value = prompts[promptPresetEl.value] || "";
  }
  const currentCommentatorModel = commentatorModelEl.value;
  commentatorModelEl.innerHTML = models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  if (currentCommentatorModel && models.includes(currentCommentatorModel)) {
    commentatorModelEl.value = currentCommentatorModel;
  } else if (models.length > 0) {
    commentatorModelEl.value = models[0];
  }
  const currentCommentatorPreset = commentatorPresetEl.value;
  commentatorPresetEl.innerHTML = promptKeys.map((key) => `<option value="${escapeHtml(key)}">${escapeHtml(key)}</option>`).join("");
  if (currentCommentatorPreset && promptKeys.includes(currentCommentatorPreset)) {
    commentatorPresetEl.value = currentCommentatorPreset;
  } else if (promptKeys.length > 0) {
    commentatorPresetEl.value = promptKeys[0];
  }
  if (!commentatorDirty && commentatorPresetEl.value) {
    commentatorPromptEl.value = prompts[commentatorPresetEl.value] || "";
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

  renderHistory(historyEl, state.history || []);
  renderHistory(commentatorHistoryEl, state.commentator_history || []);

  if (state.commentator) {
    commentatorStatusEl.textContent = `Подключен: ${state.commentator.name} <${state.commentator.model}>`;
    if (!commentatorDirty) {
      commentatorNameEl.value = state.commentator.name || commentatorNameEl.value;
      commentatorModelEl.value = state.commentator.model || commentatorModelEl.value;
      commentatorPromptEl.value = state.commentator.system_prompt || commentatorPromptEl.value;
    }
  } else {
    commentatorStatusEl.textContent = "Не подключен";
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
        provider: providerEl.value,
        host: hostEl.value,
        api_key: apiKeyEl.value,
        token: tokenEl.value,
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
      systemPromptDirty = false;
      await loadState();
    }),
  );

  document.getElementById("setCommentator").addEventListener("click", () =>
    safeAction(async () => {
      const payload = await api("/api/commentator", "POST", {
        name: commentatorNameEl.value.trim(),
        model: commentatorModelEl.value,
        system_prompt: commentatorPromptEl.value,
      });
      commentatorDirty = false;
      renderState(payload);
    }),
  );

  document.getElementById("clearCommentator").addEventListener("click", () =>
    safeAction(async () => {
      const payload = await api("/api/commentator", "DELETE");
      commentatorDirty = false;
      renderState(payload);
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
    systemPromptDirty = false;
  });
  commentatorPresetEl.addEventListener("change", () => {
    const key = commentatorPresetEl.value;
    if (!state?.prompts || !state.prompts[key]) {
      return;
    }
    commentatorPromptEl.value = state.prompts[key];
    commentatorDirty = false;
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

  providerEl.addEventListener("change", () => {
    configDirty = true;
  });
  hostEl.addEventListener("input", () => {
    configDirty = true;
  });
  apiKeyEl.addEventListener("input", () => {
    configDirty = true;
  });
  tokenEl.addEventListener("input", () => {
    configDirty = true;
  });
  timeoutEl.addEventListener("input", () => {
    configDirty = true;
  });
  delayEl.addEventListener("input", () => {
    configDirty = true;
  });
  systemPromptEl.addEventListener("input", () => {
    systemPromptDirty = true;
  });
  commentatorNameEl.addEventListener("input", () => {
    commentatorDirty = true;
  });
  commentatorModelEl.addEventListener("change", () => {
    commentatorDirty = true;
  });
  commentatorPromptEl.addEventListener("input", () => {
    commentatorDirty = true;
  });
}

setupEvents();
loadState();
setInterval(() => {
  loadState().catch((error) => setStatus(error.message || String(error)));
}, 2000);
