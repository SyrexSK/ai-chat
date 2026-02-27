from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_PROMPTS: dict[str, str] = {
    "default": "Всегда отвечай на русском языке. Ты участвуешь в беседе. Общайся вежливо, спокойно и по делу.",
    "analyst": "Всегда отвечай на русском языке. Ты аналитик: разбирай предположения, выделяй риски и объясняй ход мысли по шагам. Пиши вежливо и конструктивно.",
    "critic": "Всегда отвечай на русском языке. Ты критик: находи слабые места, предлагай альтернативы и аргументируй. Пиши вежливо и конструктивно.",
    "coder": "Всегда отвечай на русском языке. Ты инженер-программист: предлагай практичные технические решения и конкретные шаги реализации. Пиши вежливо и понятно.",
}



@dataclass
class Agent:
    name: str
    model: str
    system_prompt: str


@dataclass
class Message:
    speaker: str
    role: str
    content: str


class OllamaClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. Is `ollama serve` running?"
            ) from exc

    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags")
        models = data.get("models", [])
        return [item.get("name", "") for item in models if item.get("name")]

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> str:
        data = self._request(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": options or {},
            },
        )
        msg = data.get("message", {})
        content = msg.get("content", "")
        return content.strip()


class OpenAICompatibleClient:
    def __init__(self, base_url: str, api_key: str = "", token: str = "", timeout: float = 120.0) -> None:
        normalized = base_url.rstrip("/")
        for suffix in ("/v1/chat/completions", "/chat/completions", "/v1"):
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]
                break
        self.base_url = normalized.rstrip("/")
        self.api_key = api_key.strip()
        self.token = token.strip()
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        auth_value = self.token or self.api_key
        if auth_value:
            headers["Authorization"] = f"Bearer {auth_value}"
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot connect to OpenAI-compatible endpoint at {self.base_url}"
            ) from exc

    def list_models(self) -> list[str]:
        data = self._request("GET", "/v1/models")
        raw = data.get("data", [])
        models: list[str] = []
        for item in raw:
            model_id = item.get("id") if isinstance(item, dict) else None
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
        return models

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "num_predict" in options:
                payload["max_tokens"] = options["num_predict"]

        data = self._request("POST", "/v1/chat/completions", payload)
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            return "\n".join(text_parts).strip()
        if isinstance(content, str):
            return content.strip()
        return ""


def load_system_prompts(path: Path) -> dict[str, str]:
    if not path.exists():
        path.write_text(json.dumps(DEFAULT_PROMPTS, indent=2), encoding="utf-8")
        return dict(DEFAULT_PROMPTS)

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in prompt file: {path}") from exc

    if not isinstance(raw, dict):
        raise RuntimeError(f"Prompt file must contain an object at top-level: {path}")

    prompts: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, str):
            prompts[key] = value

    if not prompts:
        prompts = dict(DEFAULT_PROMPTS)
    return prompts


def transcript(history: list[Message], limit: int = 20) -> str:
    items = history[-limit:]
    if not items:
        return "(empty)"
    return "\n".join(f"{m.speaker}: {m.content}" for m in items)


def build_prompt_messages(agent: Agent, history: list[Message]) -> list[dict[str, str]]:
    system = (
        f"Your identity is '{agent.name}'. "
        "You are in a multi-agent conversation with other models and a human user. "
        "Follow your role instruction below.\n\n"
        f"ROLE INSTRUCTION:\n{agent.system_prompt.strip()}"
    )
    user = (
        "Conversation transcript:\n"
        f"{transcript(history)}\n\n"
        f"Now reply as {agent.name}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
