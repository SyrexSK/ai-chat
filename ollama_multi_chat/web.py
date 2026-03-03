from __future__ import annotations

import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from litestar import Litestar, Request, delete, get, post
from litestar.exceptions import ClientException
from litestar.response import File
from litestar.static_files import StaticFilesConfig

from ollama_multi_chat.core import Agent, Message, OllamaClient, OpenAICompatibleClient, load_system_prompts


STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ConnectionConfig:
    connection_id: str
    provider: str
    host: str
    api_key: str = ""
    token: str = ""
    timeout: float = 100.0


@dataclass
class ChatServerState:
    prompts_path: Path
    provider: str = "ollama"
    host: str = "http://127.0.0.1:11434"
    api_key: str = ""
    token: str = ""
    timeout: float = 100.0
    delay: float = 8.0

    def __post_init__(self) -> None:
        self.prompts = load_system_prompts(self.prompts_path)
        self.connections: dict[str, ConnectionConfig] = {
            "default": ConnectionConfig(
                connection_id="default",
                provider=self.provider,
                host=self.host,
                api_key=self.api_key,
                token=self.token,
                timeout=self.timeout,
            )
        }
        self.clients: dict[str, OllamaClient | OpenAICompatibleClient] = {}
        self.models_by_connection: dict[str, list[str]] = {"default": []}
        self.agents: list[Agent] = []
        self.agent_connections: dict[str, str] = {}
        self.history: list[Message] = []
        self.commentator: Agent | None = None
        self.commentator_connection_id: str = "default"
        self.commentator_history: list[Message] = []

        self.running = False
        self.stop_requested = False
        self.status = "Готово"
        self.last_error = ""

        self.lock = threading.RLock()
        self.commentator_step_lock = threading.Lock()
        self.worker_thread: threading.Thread | None = None

        self.agent_contexts: dict[str, list[dict[str, str]]] = {}
        self.agent_seen_count: dict[str, int] = {}
        self.current_agent_idx = 0
        self.max_context_messages = 60
        self.commentator_context: list[dict[str, str]] = []
        self.commentator_seen_count = 0

        self.refresh_models(connection_id="default", raise_error=False)

    def _normalize_text(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", normalized)
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()

    def _sanitize_agent_response(self, text: str, agent_name: str) -> str:
        cleaned = self._normalize_text(text)
        patterns = [
            rf"^\s*{re.escape(agent_name)}\s*[:：-]\s*",
            r"^\s*assistant\s*[:：-]\s*",
            r"^\s*user\s*[:：-]\s*",
            r"^\s*system\s*[:：-]\s*",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Some models wrap full answers as `markdown ... ` or ```markdown ... ```
        cleaned = re.sub(r"^\s*```(?:markdown|md)?\s*\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n```\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*`(?:markdown|md)?\s*\n", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n`\s*$", "", cleaned, flags=re.IGNORECASE)

        # Remove stray wrapper lines left at boundaries.
        cleaned = re.sub(r"^\s*`(?:markdown|md)?\s*$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*```\s*$", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _looks_truncated(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 60:
            return False
        return not stripped.endswith((".", "!", "?", "…", '"', "'", "»", ")", "]"))

    def _chat_with_auto_continue(
        self,
        *,
        client: OllamaClient | OpenAICompatibleClient,
        agent_name: str,
        model: str,
        context: list[dict[str, str]],
        options: dict[str, Any],
        max_continuations: int = 2,
    ) -> str:
        chunks: list[str] = []

        first = client.chat(model, context, options=options)
        first = self._sanitize_agent_response(first, agent_name)
        if not first:
            return ""
        chunks.append(first)

        for _ in range(max_continuations):
            merged = "\n".join(chunks).strip()
            if not self._looks_truncated(merged):
                break
            continuation_messages = context + [
                {"role": "assistant", "content": merged},
                {"role": "user", "content": "Продолжи с места остановки, без повторов."},
            ]
            cont = client.chat(
                model,
                continuation_messages,
                options={
                    "num_predict": min(600, int(options.get("num_predict", 512))),
                    "temperature": max(0.2, float(options.get("temperature", 0.7)) - 0.2),
                    "repeat_penalty": float(options.get("repeat_penalty", 1.2)),
                },
            )
            cont = self._sanitize_agent_response(cont, agent_name)
            if not cont:
                break
            chunks.append(cont)

        return "\n".join(chunks).strip()

    def _ensure_agent_context(self, agent: Agent) -> None:
        if agent.name in self.agent_contexts:
            return
        self.agent_contexts[agent.name] = [
            {
                "role": "system",
                "content": (
                    f"Твоя роль: {agent.name}. "
                    "Ты участвуешь в мультиагентном диалоге с человеком и другими моделями. "
                    f"Следуй этой инструкции:\n{agent.system_prompt.strip()}"
                ),
            }
        ]
        self.agent_seen_count[agent.name] = 0

    def _ensure_commentator_context(self) -> None:
        if not self.commentator:
            return
        if self.commentator_context:
            return
        self.commentator_context = [
            {
                "role": "system",
                "content": (
                    f"Твоя роль: {self.commentator.name}. "
                    "Ты наблюдатель в отдельном канале комментариев. "
                    "Читай основной диалог и давай короткие аналитические комментарии по сути, "
                    "без участия в основном споре. Пиши на русском языке. "
                    f"Следуй этой инструкции:\n{self.commentator.system_prompt.strip()}"
                ),
            }
        ]
        self.commentator_seen_count = 0

    def _build_client(self, config: ConnectionConfig) -> OllamaClient | OpenAICompatibleClient:
        if config.provider == "openai":
            return OpenAICompatibleClient(
                config.host.strip(),
                api_key=config.api_key,
                token=config.token,
                timeout=float(config.timeout),
            )
        return OllamaClient(config.host.strip(), timeout=float(config.timeout))

    def upsert_connection(
        self,
        connection_id: str,
        provider: str,
        host: str,
        api_key: str,
        token: str,
        timeout: float,
    ) -> None:
        if not connection_id:
            raise ValueError("Connection id is required")
        if provider not in {"ollama", "openai"}:
            raise ValueError("Provider must be one of: ollama, openai")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")

        with self.lock:
            self.connections[connection_id] = ConnectionConfig(
                connection_id=connection_id,
                provider=provider,
                host=host.strip(),
                api_key=api_key,
                token=token,
                timeout=timeout,
            )
            self.models_by_connection.setdefault(connection_id, [])
            self.clients.pop(connection_id, None)

    def remove_connection(self, connection_id: str) -> None:
        if connection_id == "default":
            raise ValueError("Default connection cannot be removed")
        with self.lock:
            if connection_id not in self.connections:
                raise ValueError(f"Connection '{connection_id}' not found")
            if any(conn_id == connection_id for conn_id in self.agent_connections.values()):
                raise ValueError("Connection is used by agents")
            if self.commentator and self.commentator_connection_id == connection_id:
                raise ValueError("Connection is used by commentator")
            self.connections.pop(connection_id, None)
            self.models_by_connection.pop(connection_id, None)
            self.clients.pop(connection_id, None)

    def refresh_models(self, connection_id: str, raise_error: bool = True) -> list[str]:
        with self.lock:
            config = self.connections.get(connection_id)
        if not config:
            raise ValueError(f"Connection '{connection_id}' not found")

        client = self._build_client(config)
        try:
            models = client.list_models()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.models_by_connection[connection_id] = []
                self.last_error = str(exc)
                self.status = f"Ошибка подключения: {connection_id}"
            if raise_error:
                raise
            return []

        with self.lock:
            self.clients[connection_id] = client
            self.models_by_connection[connection_id] = models
            self.last_error = ""
            total_models = sum(len(items) for items in self.models_by_connection.values())
            self.status = f"Подключено. Моделей: {total_models}"
        return models

    def _client_for_connection(self, connection_id: str) -> OllamaClient | OpenAICompatibleClient:
        with self.lock:
            config = self.connections.get(connection_id)
            cached = self.clients.get(connection_id)
        if not config:
            raise RuntimeError(f"Connection '{connection_id}' not found")
        if cached is not None:
            return cached
        client = self._build_client(config)
        with self.lock:
            self.clients[connection_id] = client
        return client

    def add_agent(self, name: str, model: str, system_prompt: str, connection_id: str) -> None:
        with self.lock:
            if not name:
                raise ValueError("Agent name is required")
            if any(agent.name == name for agent in self.agents):
                raise ValueError(f"Agent '{name}' already exists")
            if connection_id not in self.connections:
                raise ValueError("Unknown connection")
            available = self.models_by_connection.get(connection_id, [])
            if model not in available:
                raise ValueError("Choose a model from the selected connection")
            if not system_prompt.strip():
                raise ValueError("System prompt cannot be empty")

            agent = Agent(name=name, model=model, system_prompt=system_prompt.strip())
            self.agents.append(agent)
            self.agent_connections[name] = connection_id
            self.history.append(Message(speaker="Система", role="system", content=f"Подключен агент {name} <{model}>"))
            if self.running:
                self._ensure_agent_context(agent)

    def remove_agent(self, name: str) -> None:
        with self.lock:
            before = len(self.agents)
            self.agents = [agent for agent in self.agents if agent.name != name]
            if len(self.agents) == before:
                raise ValueError(f"Agent '{name}' not found")
            self.agent_contexts.pop(name, None)
            self.agent_seen_count.pop(name, None)
            self.agent_connections.pop(name, None)
            self.history.append(Message(speaker="Система", role="system", content=f"Агент удален: {name}"))

    def add_user_message(self, text: str) -> None:
        msg = self._normalize_text(text)
        if not msg:
            raise ValueError("Message cannot be empty")
        with self.lock:
            self.history.append(Message(speaker="Вы", role="user", content=msg))
            should_comment = self.commentator is not None
        if should_comment:
            threading.Thread(target=self._run_commentator_step, daemon=True).start()

    def set_commentator(self, name: str, model: str, system_prompt: str, connection_id: str) -> None:
        with self.lock:
            if not name:
                raise ValueError("Commentator name is required")
            if connection_id not in self.connections:
                raise ValueError("Unknown connection")
            available = self.models_by_connection.get(connection_id, [])
            if model not in available:
                raise ValueError("Choose a model from the selected connection")
            if not system_prompt.strip():
                raise ValueError("Commentator prompt cannot be empty")
            self.commentator = Agent(name=name, model=model, system_prompt=system_prompt.strip())
            self.commentator_connection_id = connection_id
            self.commentator_context = []
            self.commentator_seen_count = 0
            self.commentator_history.append(
                Message(speaker="Система", role="system", content=f"Подключен комментатор {name} <{model}>")
            )
            self._ensure_commentator_context()

    def clear_commentator(self) -> None:
        with self.lock:
            if not self.commentator:
                raise ValueError("Commentator is not configured")
            old_name = self.commentator.name
            self.commentator = None
            self.commentator_connection_id = "default"
            self.commentator_context = []
            self.commentator_seen_count = 0
            self.commentator_history.append(
                Message(speaker="Система", role="system", content=f"Комментатор отключен: {old_name}")
            )

    def reset_chat(self) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("Сначала остановите чат")
            self.history.clear()
            self.commentator_history.clear()
            self.agent_contexts.clear()
            self.agent_seen_count.clear()
            self.commentator_context.clear()
            self.commentator_seen_count = 0
            self.status = "Чат сброшен"
            self.last_error = ""

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            if not self.agents:
                raise RuntimeError("Подключите хотя бы одного агента")

            self.running = True
            self.stop_requested = False
            self.status = "Чат запущен"
            self.current_agent_idx = 0
            self.agent_contexts.clear()
            self.agent_seen_count.clear()
            for agent in self.agents:
                self._ensure_agent_context(agent)
            self._ensure_commentator_context()

            if not self.history:
                self.history.append(
                    Message(
                        speaker="Вы",
                        role="user",
                        content="Начните разговор на русском языке по теме последних сообщений пользователя.",
                    )
                )

            self.worker_thread = threading.Thread(target=self._chat_loop_worker, daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        with self.lock:
            self.stop_requested = True
            self.status = "Остановка..."

    def _chat_loop_worker(self) -> None:
        try:
            while True:
                with self.lock:
                    if self.stop_requested:
                        break
                    agents_snapshot = list(self.agents)
                    delay = float(self.delay)

                if not agents_snapshot:
                    with self.lock:
                        self.history.append(
                            Message(
                                speaker="Система",
                                role="system",
                                content="Нет подключенных агентов. Добавьте модель и нажмите Старт снова.",
                            )
                        )
                        self.stop_requested = True
                    break

                with self.lock:
                    agent = agents_snapshot[self.current_agent_idx % len(agents_snapshot)]
                    self.current_agent_idx = (self.current_agent_idx + 1) % len(agents_snapshot)
                    self._ensure_agent_context(agent)
                    connection_id = self.agent_connections.get(agent.name, "default")
                    self.status = f"Отвечает: {agent.name}<{agent.model}>"

                try:
                    client = self._client_for_connection(connection_id)
                    with self.lock:
                        context = self.agent_contexts[agent.name]
                        start_idx = self.agent_seen_count.get(agent.name, 0)
                        unseen = self.history[start_idx:]
                        self.agent_seen_count[agent.name] = len(self.history)

                    for item in unseen:
                        role = "assistant" if item.speaker == agent.name else "user"
                        context.append({"role": role, "content": f"{item.speaker}: {item.content}"})

                    if len(context) > self.max_context_messages + 1:
                        context[:] = [context[0]] + context[-self.max_context_messages :]

                    response = self._chat_with_auto_continue(
                        client=client,
                        agent_name=agent.name,
                        model=agent.model,
                        context=context,
                        options={"num_predict": 900, "temperature": 0.7, "repeat_penalty": 1.2},
                        max_continuations=2,
                    )

                    if not response:
                        with self.lock:
                            self.history.append(
                                Message(
                                    speaker="Система",
                                    role="system",
                                    content=f"{agent.name} не дал ответ. Пропускаю ход.",
                                )
                            )
                        continue

                    with self.lock:
                        self.history.append(Message(speaker=agent.name, role="assistant", content=response))
                        self.status = "Чат запущен"
                except Exception as exc:  # noqa: BLE001
                    with self.lock:
                        self.history.append(
                            Message(speaker=f"{agent.name}<{agent.model}>", role="error", content=str(exc))
                        )
                        self.status = "Ошибка в ходе ответа агента"
                        self.last_error = str(exc)

                self._run_commentator_step()

                for _ in range(max(1, int(delay / 0.1))):
                    with self.lock:
                        if self.stop_requested:
                            break
                    time.sleep(0.1)
        finally:
            with self.lock:
                self.running = False
                if self.status.startswith("Остановка"):
                    self.status = "Остановлено"

    def _run_commentator_step(self) -> None:
        if not self.commentator_step_lock.acquire(blocking=False):
            return
        with self.lock:
            commentator = self.commentator
            if not commentator:
                self.commentator_step_lock.release()
                return
            connection_id = self.commentator_connection_id
            self._ensure_commentator_context()
            source_unseen = self.history[self.commentator_seen_count :]
            self.commentator_seen_count = len(self.history)
            context = self.commentator_context

        if not source_unseen:
            self.commentator_step_lock.release()
            return

        summary_lines = [f"{item.speaker}: {item.content}" for item in source_unseen[-8:]]
        user_turn = (
            "Новые сообщения в основном чате:\n"
            + "\n".join(summary_lines)
            + "\n\nДай короткий комментарий (1-4 абзаца), что важно в этом фрагменте."
        )
        context.append({"role": "user", "content": user_turn})
        if len(context) > self.max_context_messages + 1:
            context[:] = [context[0]] + context[-self.max_context_messages :]

        try:
            client = self._client_for_connection(connection_id)
            final_response = self._chat_with_auto_continue(
                client=client,
                agent_name=commentator.name,
                model=commentator.model,
                context=context,
                options={"num_predict": 600, "temperature": 0.5, "repeat_penalty": 1.1},
                max_continuations=2,
            )
            if not final_response:
                return
            context.append({"role": "assistant", "content": final_response})
            with self.lock:
                self.commentator_history.append(
                    Message(speaker=commentator.name, role="assistant", content=final_response)
                )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.commentator_history.append(
                    Message(speaker=f"{commentator.name}<{commentator.model}>", role="error", content=str(exc))
                )
        finally:
            self.commentator_step_lock.release()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            default = self.connections.get("default")
            total_models = sum(len(items) for items in self.models_by_connection.values())
            return {
                "host": default.host if default else self.host,
                "provider": default.provider if default else self.provider,
                "api_key": default.api_key if default else self.api_key,
                "token": default.token if default else self.token,
                "timeout": default.timeout if default else self.timeout,
                "delay": self.delay,
                "running": self.running,
                "status": self.status,
                "last_error": self.last_error,
                "models": list(self.models_by_connection.get("default", [])),
                "models_by_connection": {k: list(v) for k, v in self.models_by_connection.items()},
                "connections": [
                    {
                        "connection_id": cfg.connection_id,
                        "provider": cfg.provider,
                        "host": cfg.host,
                        "api_key": cfg.api_key,
                        "token": cfg.token,
                        "timeout": cfg.timeout,
                        "models_count": len(self.models_by_connection.get(cfg.connection_id, [])),
                    }
                    for cfg in self.connections.values()
                ],
                "models_total": total_models,
                "prompts": dict(self.prompts),
                "agents": [
                    {
                        "name": agent.name,
                        "model": agent.model,
                        "system_prompt": agent.system_prompt,
                        "connection_id": self.agent_connections.get(agent.name, "default"),
                    }
                    for agent in self.agents
                ],
                "history": [asdict(message) for message in self.history[-300:]],
                "commentator": (
                    {
                        "name": self.commentator.name,
                        "model": self.commentator.model,
                        "system_prompt": self.commentator.system_prompt,
                        "connection_id": self.commentator_connection_id,
                    }
                    if self.commentator
                    else None
                ),
                "commentator_history": [asdict(message) for message in self.commentator_history[-300:]],
            }


def _state(request: Request[Any, Any, Any]) -> ChatServerState:
    return request.app.state.chat_state


def _bad_request(message: str) -> ClientException:
    return ClientException(detail=message, status_code=400)


@get("/", sync_to_thread=False)
def index() -> File:
    return File(
        path=str(STATIC_DIR / "index.html"),
        content_disposition_type="inline",
        media_type="text/html; charset=utf-8",
    )


@get("/api/state", sync_to_thread=True)
def get_state(request: Request[Any, Any, Any]) -> dict[str, Any]:
    return _state(request).snapshot()


@get("/api/prompts", sync_to_thread=True)
def get_prompts(request: Request[Any, Any, Any]) -> dict[str, str]:
    return _state(request).prompts


@post("/api/config", sync_to_thread=True)
def update_config(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)

    connection_id = str(data.get("connection_id", "default")).strip() or "default"
    host = str(data.get("host", state.host)).strip()
    provider = str(data.get("provider", state.provider)).strip().lower()
    api_key = str(data.get("api_key", state.api_key))
    token = str(data.get("token", state.token))
    timeout_raw = data.get("timeout", state.timeout)
    delay_raw = data.get("delay", state.delay)

    try:
        timeout = float(timeout_raw)
        delay = max(0.0, min(30.0, float(delay_raw)))
        if timeout <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise _bad_request("Timeout must be positive, delay must be a number") from exc

    try:
        state.upsert_connection(
            connection_id=connection_id,
            provider=provider,
            host=host,
            api_key=api_key,
            token=token,
            timeout=timeout,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc

    with state.lock:
        state.delay = delay

    state.refresh_models(connection_id=connection_id, raise_error=True)
    return state.snapshot()


@get("/api/models", sync_to_thread=True)
def refresh_models(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    models = state.refresh_models(connection_id="default", raise_error=True)
    return {"models": models, "connection_id": "default"}


@get("/api/models/{connection_id:str}", sync_to_thread=True)
def refresh_models_for_connection(connection_id: str, request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        models = state.refresh_models(connection_id=connection_id, raise_error=True)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return {"models": models, "connection_id": connection_id}


@post("/api/connections", sync_to_thread=True)
def upsert_connection(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    connection_id = str(data.get("connection_id", "")).strip()
    provider = str(data.get("provider", "ollama")).strip().lower()
    host = str(data.get("host", "")).strip()
    api_key = str(data.get("api_key", ""))
    token = str(data.get("token", ""))
    try:
        timeout = float(data.get("timeout", 45))
        if timeout <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise _bad_request("Timeout must be positive") from exc

    try:
        state.upsert_connection(
            connection_id=connection_id,
            provider=provider,
            host=host,
            api_key=api_key,
            token=token,
            timeout=timeout,
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    state.refresh_models(connection_id=connection_id, raise_error=True)
    return state.snapshot()


@delete("/api/connections/{connection_id:str}", status_code=200, sync_to_thread=True)
def delete_connection(connection_id: str, request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.remove_connection(connection_id)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@post("/api/agents", sync_to_thread=True)
def add_agent(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.add_agent(
            name=str(data.get("name", "")).strip(),
            model=str(data.get("model", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
            connection_id=str(data.get("connection_id", "default")).strip() or "default",
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@delete("/api/agents/{name:str}", status_code=200, sync_to_thread=True)
def remove_agent(name: str, request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.remove_agent(name)
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@post("/api/messages", sync_to_thread=True)
def add_message(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.add_user_message(str(data.get("text", "")))
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@post("/api/commentator", sync_to_thread=True)
def set_commentator(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.set_commentator(
            name=str(data.get("name", "")).strip(),
            model=str(data.get("model", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
            connection_id=str(data.get("connection_id", "default")).strip() or "default",
        )
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@delete("/api/commentator", status_code=200, sync_to_thread=True)
def clear_commentator(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.clear_commentator()
    except ValueError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@post("/api/chat/start", sync_to_thread=True)
def start_chat(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.start()
    except RuntimeError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


@post("/api/chat/stop", sync_to_thread=True)
def stop_chat(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    state.stop()
    return state.snapshot()


@post("/api/chat/reset", sync_to_thread=True)
def reset_chat(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.reset_chat()
    except RuntimeError as exc:
        raise _bad_request(str(exc)) from exc
    return state.snapshot()


async def on_startup(app: Litestar) -> None:
    app.state.chat_state = ChatServerState(prompts_path=Path.cwd() / "system_prompts.json")


app = Litestar(
    route_handlers=[
        index,
        get_state,
        get_prompts,
        update_config,
        refresh_models,
        refresh_models_for_connection,
        upsert_connection,
        delete_connection,
        add_agent,
        remove_agent,
        add_message,
        set_commentator,
        clear_commentator,
        start_chat,
        stop_chat,
        reset_chat,
    ],
    on_startup=[on_startup],
    static_files_config=[StaticFilesConfig(path="/static", directories=[str(STATIC_DIR)])],
)


def main() -> None:
    """Local helper for direct run without CLI wrappers."""
    from granian import Granian

    server = Granian(
        target="ollama_multi_chat.web:app",
        address="127.0.0.1",
        port=8000,
        interface="asgi",
    )
    server.serve()


if __name__ == "__main__":
    main()
