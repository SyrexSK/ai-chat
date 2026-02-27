const statusEl = document.getElementById("status");

const ollamaHostEl = document.getElementById("ollamaHost");
const ollamaTimeoutEl = document.getElementById("ollamaTimeout");
const openaiHostEl = document.getElementById("openaiHost");
const openaiApiKeyEl = document.getElementById("openaiApiKey");
const openaiTokenEl = document.getElementById("openaiToken");
const openaiTimeoutEl = document.getElementById("openaiTimeout");
const delayEl = document.getElementById("delay");

const agentsEl = document.getElementById("agents");
const agentNameEl = document.getElementById("agentName");
const agentConnectionEl = document.getElementById("agentConnection");
const agentModelEl = document.getElementById("agentModel");
const promptPresetEl = document.getElementById("promptPreset");
const systemPromptEl = document.getElementById("systemPrompt");

const commentatorStatusEl = document.getElementById("commentatorStatus");
const commentatorNameEl = document.getElementById("commentatorName");
const commentatorConnectionEl = document.getElementById("commentatorConnection");
const commentatorModelEl = document.getElementById("commentatorModel");
const commentatorPresetEl = document.getElementById("commentatorPreset");
const commentatorPromptEl = document.getElementById("commentatorPrompt");

const historyEl = document.getElementById("history");
const commentatorHistoryEl = document.getElementById("commentatorHistory");
const userTextEl = document.getElementById("userText");

let selectedAgent = "";
let state = null;
let systemPromptDirty = false;
let commentatorDirty = false;
let ollamaDirty = false;
let openaiDirty = false;
let delayDirty = false;

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
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderMessageContent(value) {
  const escaped = escapeHtml(value || "");
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

function findConnection(connectionId) {
  return (state?.connections || []).find((c) => c.connection_id === connectionId) || null;
}

function connectionModels(connectionId) {
  if (!state?.models_by_connection) {
    return [];
  }
  return state.models_by_connection[connectionId] || [];
}

function renderModelsForSelect(selectEl, connectionId) {
  const models = connectionModels(connectionId);
  const current = selectEl.value;
  selectEl.innerHTML = models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  if (current && models.includes(current)) {
    selectEl.value = current;
  } else if (models.length > 0) {
    selectEl.value = models[0];
  }
}

function renderConnectionSelects() {
  const options = (state?.connections || [])
    .map((conn) => `<option value="${escapeHtml(conn.connection_id)}">${escapeHtml(conn.connection_id)} (${escapeHtml(conn.provider)})</option>`)
    .join("");

  const prevAgentConn = agentConnectionEl.value;
  const prevCommentConn = commentatorConnectionEl.value;

  agentConnectionEl.innerHTML = options;
  commentatorConnectionEl.innerHTML = options;

  const connIds = (state?.connections || []).map((c) => c.connection_id);
  agentConnectionEl.value = connIds.includes(prevAgentConn) ? prevAgentConn : (connIds[0] || "default");
  commentatorConnectionEl.value = connIds.includes(prevCommentConn) ? prevCommentConn : (connIds[0] || "default");
}

function syncConnectionEditors() {
  const ollama = findConnection("ollama") || findConnection("default");
  const openai = findConnection("openai");

  if (ollama && !ollamaDirty) {
    ollamaHostEl.value = ollama.host || "";
    ollamaTimeoutEl.value = ollama.timeout || 45;
  }

  if (openai && !openaiDirty) {
    openaiHostEl.value = openai.host || "";
    openaiApiKeyEl.value = openai.api_key || "";
    openaiTokenEl.value = openai.token || "";
    openaiTimeoutEl.value = openai.timeout || 45;
  }
}

function renderState(nextState) {
  state = nextState;
  setStatus(state.status || "");

  if (!delayDirty) {
    delayEl.value = state.delay;
  }

  renderConnectionSelects();
  syncConnectionEditors();

  renderModelsForSelect(agentModelEl, agentConnectionEl.value || "default");
  renderModelsForSelect(commentatorModelEl, commentatorConnectionEl.value || "default");

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
      const suffix = agent.connection_id ? ` @${agent.connection_id}` : "";
      return `<li data-name="${escapeHtml(agent.name)}" class="${active}">${escapeHtml(agent.name)} &lt;${escapeHtml(agent.model)}&gt;${escapeHtml(suffix)}</li>`;
    })
    .join("");

  if (selectedAgent && !state.agents.some((agent) => agent.name === selectedAgent)) {
    selectedAgent = "";
  }

  renderHistory(historyEl, state.history || []);
  renderHistory(commentatorHistoryEl, state.commentator_history || []);

  if (state.commentator) {
    commentatorStatusEl.textContent = `Подключен: ${state.commentator.name} <${state.commentator.model}> @${state.commentator.connection_id}`;
    if (!commentatorDirty) {
      commentatorNameEl.value = state.commentator.name || commentatorNameEl.value;
      commentatorConnectionEl.value = state.commentator.connection_id || commentatorConnectionEl.value;
      renderModelsForSelect(commentatorModelEl, commentatorConnectionEl.value || "default");
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

async function saveConnection(connectionId, provider, host, apiKey, token, timeout) {
  const payload = await api("/api/config", "POST", {
    connection_id: connectionId,
    provider,
    host,
    api_key: apiKey,
    token,
    timeout: Number(timeout),
    delay: Number(delayEl.value),
  });
  renderState(payload);
}

function setupEvents() {
  document.getElementById("saveOllama").addEventListener("click", () =>
    safeAction(async () => {
      await saveConnection("ollama", "ollama", ollamaHostEl.value, "", "", ollamaTimeoutEl.value);
      ollamaDirty = false;
      delayDirty = false;
      await loadState();
    }),
  );

  document.getElementById("refreshOllama").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/models/ollama");
      await loadState();
    }),
  );

  document.getElementById("saveOpenai").addEventListener("click", () =>
    safeAction(async () => {
      await saveConnection("openai", "openai", openaiHostEl.value, openaiApiKeyEl.value, openaiTokenEl.value, openaiTimeoutEl.value);
      openaiDirty = false;
      delayDirty = false;
      await loadState();
    }),
  );

  document.getElementById("refreshOpenai").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/models/openai");
      await loadState();
    }),
  );

  document.getElementById("deleteOpenai").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/connections/openai", "DELETE");
      await loadState();
    }),
  );

  document.getElementById("addAgent").addEventListener("click", () =>
    safeAction(async () => {
      await api("/api/agents", "POST", {
        name: agentNameEl.value.trim(),
        connection_id: agentConnectionEl.value,
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
        connection_id: commentatorConnectionEl.value,
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
    commentatorDirty = true;
  });

  agentConnectionEl.addEventListener("change", () => {
    renderModelsForSelect(agentModelEl, agentConnectionEl.value || "default");
  });

  commentatorConnectionEl.addEventListener("change", () => {
    commentatorDirty = true;
    renderModelsForSelect(commentatorModelEl, commentatorConnectionEl.value || "default");
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

  ollamaHostEl.addEventListener("input", () => {
    ollamaDirty = true;
  });
  ollamaTimeoutEl.addEventListener("input", () => {
    ollamaDirty = true;
  });
  openaiHostEl.addEventListener("input", () => {
    openaiDirty = true;
  });
  openaiApiKeyEl.addEventListener("input", () => {
    openaiDirty = true;
  });
  openaiTokenEl.addEventListener("input", () => {
    openaiDirty = true;
  });
  openaiTimeoutEl.addEventListener("input", () => {
    openaiDirty = true;
  });
  delayEl.addEventListener("input", () => {
    delayDirty = true;
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
