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

from ollama_multi_chat.core import Agent, Message, OllamaClient, load_system_prompts


STATIC_DIR = Path(__file__).parent / "static"


@dataclass
class ChatServerState:
    prompts_path: Path
    host: str = "http://127.0.0.1:11434"
    timeout: float = 45.0
    delay: float = 2.0

    def __post_init__(self) -> None:
        self.prompts = load_system_prompts(self.prompts_path)
        self.client = OllamaClient(self.host, timeout=self.timeout)
        self.available_models: list[str] = []
        self.agents: list[Agent] = []
        self.history: list[Message] = []

        self.running = False
        self.stop_requested = False
        self.status = "Готово"
        self.last_error = ""

        self.lock = threading.RLock()
        self.worker_thread: threading.Thread | None = None

        self.agent_contexts: dict[str, list[dict[str, str]]] = {}
        self.agent_seen_count: dict[str, int] = {}
        self.current_agent_idx = 0
        self.max_context_messages = 60

        self.refresh_models(raise_error=False)

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

    def rebuild_client(self) -> None:
        self.client = OllamaClient(self.host.strip(), timeout=float(self.timeout))

    def refresh_models(self, raise_error: bool = True) -> list[str]:
        self.rebuild_client()
        try:
            models = self.client.list_models()
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.available_models = []
                self.last_error = str(exc)
                self.status = "Ошибка подключения"
            if raise_error:
                raise
            return []

        with self.lock:
            self.available_models = models
            self.last_error = ""
            self.status = f"Подключено. Моделей: {len(models)}"
        return models

    def add_agent(self, name: str, model: str, system_prompt: str) -> None:
        with self.lock:
            if not name:
                raise ValueError("Agent name is required")
            if any(agent.name == name for agent in self.agents):
                raise ValueError(f"Agent '{name}' already exists")
            if model not in self.available_models:
                raise ValueError("Choose a local model from the list")
            if not system_prompt.strip():
                raise ValueError("System prompt cannot be empty")

            agent = Agent(name=name, model=model, system_prompt=system_prompt.strip())
            self.agents.append(agent)
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
            self.history.append(Message(speaker="Система", role="system", content=f"Агент удален: {name}"))

    def add_user_message(self, text: str) -> None:
        msg = self._normalize_text(text)
        if not msg:
            raise ValueError("Message cannot be empty")
        with self.lock:
            self.history.append(Message(speaker="Вы", role="user", content=msg))

    def reset_chat(self) -> None:
        with self.lock:
            if self.running:
                raise RuntimeError("Сначала остановите чат")
            self.history.clear()
            self.agent_contexts.clear()
            self.agent_seen_count.clear()
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
                    self.status = f"Отвечает: {agent.name}<{agent.model}>"

                try:
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

                    response = self.client.chat(
                        agent.model,
                        context,
                        options={"num_predict": 512, "temperature": 0.7, "repeat_penalty": 1.2},
                    )
                    response = self._sanitize_agent_response(response, agent.name)

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

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "host": self.host,
                "timeout": self.timeout,
                "delay": self.delay,
                "running": self.running,
                "status": self.status,
                "last_error": self.last_error,
                "models": list(self.available_models),
                "prompts": dict(self.prompts),
                "agents": [asdict(agent) for agent in self.agents],
                "history": [asdict(message) for message in self.history[-300:]],
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

    host = str(data.get("host", state.host)).strip()
    timeout_raw = data.get("timeout", state.timeout)
    delay_raw = data.get("delay", state.delay)

    try:
        timeout = float(timeout_raw)
        delay = max(0.0, min(30.0, float(delay_raw)))
        if timeout <= 0:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise _bad_request("Timeout must be positive, delay must be a number") from exc

    with state.lock:
        state.host = host
        state.timeout = timeout
        state.delay = delay

    state.refresh_models(raise_error=True)
    return state.snapshot()


@get("/api/models", sync_to_thread=True)
def refresh_models(request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    models = state.refresh_models(raise_error=True)
    return {"models": models}


@post("/api/agents", sync_to_thread=True)
def add_agent(data: dict[str, Any], request: Request[Any, Any, Any]) -> dict[str, Any]:
    state = _state(request)
    try:
        state.add_agent(
            name=str(data.get("name", "")).strip(),
            model=str(data.get("model", "")).strip(),
            system_prompt=str(data.get("system_prompt", "")).strip(),
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
        add_agent,
        remove_agent,
        add_message,
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
