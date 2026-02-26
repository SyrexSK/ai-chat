from __future__ import annotations

import threading
import time
import tkinter as tk
import locale
import os
import queue
import re
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any
import tkinter.font as tkfont

from ollama_multi_chat.core import Agent, Message, OllamaClient, load_system_prompts


class MultiChatApp(tk.Tk):
    def __init__(self, prompts_path: Path) -> None:
        os.environ.setdefault("XMODIFIERS", "@im=none")
        super().__init__()
        try:
            locale.setlocale(locale.LC_ALL, "")
        except locale.Error:
            pass
        self.title("Ollama Multi Chat")
        self.geometry("1100x720")
        self.minsize(900, 600)

        try:
            self.tk.call("encoding", "system", "utf-8")
        except tk.TclError:
            pass
        try:
            self.tk.call("tk", "useinputmethods", True)
        except tk.TclError:
            pass

        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.ui_font_family = self._pick_font_family()
        self.mono_font_family = self._pick_mono_family()
        self.style.configure(".", font=(self.ui_font_family, 11))
        self.style.configure("Card.TLabelframe", padding=8)
        self.style.configure("Card.TLabelframe.Label", font=(self.ui_font_family, 11, "bold"))
        self.style.configure("Primary.TButton", font=(self.ui_font_family, 11, "bold"))

        self.prompts_path = prompts_path
        self.prompts = load_system_prompts(prompts_path)

        self.host_var = tk.StringVar(value="http://127.0.0.1:11434")
        self.timeout_var = tk.StringVar(value="45")
        self.delay_var = tk.StringVar(value="2.0")

        self.client = OllamaClient(self.host_var.get(), timeout=120.0)
        self.available_models: list[str] = []
        self.agents: list[Agent] = []
        self.history: list[Message] = []

        self.running = False
        self.stop_requested = False
        self.history_lock = threading.Lock()
        self.ui_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.debug_log_path = Path.cwd() / "app_debug.log"
        self.agent_contexts: dict[str, list[dict[str, str]]] = {}
        self.agent_seen_count: dict[str, int] = {}
        self.current_agent_idx = 0
        self.max_context_messages = 60
        self.inter_message_delay_sec = 2.0

        self._build_ui()
        self.refresh_models(show_error=False)
        self.after(80, self._drain_ui_queue)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=10)
        toolbar.pack(fill="x")

        self.status_label = ttk.Label(toolbar, text="Готово", font=(self.ui_font_family, 11, "bold"))
        self.status_label.pack(side="left")

        ttk.Button(toolbar, text="Настройки подключения", command=self.open_connection_window).pack(
            side="right", padx=(6, 0)
        )
        ttk.Button(toolbar, text="Добавить модель", command=self.open_add_model_window).pack(
            side="right", padx=(6, 0)
        )
        ttk.Button(toolbar, text="Обновить модели", command=self.refresh_models).pack(
            side="right", padx=(6, 0)
        )

        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        left = ttk.LabelFrame(body, text="Агенты", style="Card.TLabelframe")
        left.pack(side="left", fill="y", padx=(0, 8))

        self.agents_list = tk.Listbox(
            left,
            height=15,
            width=38,
            font=(self.ui_font_family, 11),
            activestyle="none",
            selectmode="browse",
        )
        self.agents_list.pack(fill="y", expand=False, pady=(4, 6))

        ttk.Button(left, text="Удалить выбранного", command=self.remove_selected_agent).pack(fill="x")

        controls = ttk.Frame(left)
        controls.pack(fill="x", pady=(10, 0))
        self.start_btn = ttk.Button(controls, text="Старт", command=self.start_chat, style="Primary.TButton")
        self.start_btn.pack(fill="x")
        self.stop_btn = ttk.Button(controls, text="Стоп", command=self.stop_chat, state="disabled")
        self.stop_btn.pack(fill="x", pady=(6, 0))
        self.reset_btn = ttk.Button(controls, text="Сброс чата", command=self.reset_chat)
        self.reset_btn.pack(fill="x", pady=(6, 0))

        delay_box = ttk.Frame(left)
        delay_box.pack(fill="x", pady=(10, 0))
        ttk.Label(delay_box, text="Пауза между ответами (сек)").pack(anchor="w")
        ttk.Entry(delay_box, textvariable=self.delay_var, width=10).pack(anchor="w", pady=(2, 0))

        right = ttk.LabelFrame(body, text="Диалог", style="Card.TLabelframe")
        right.pack(side="left", fill="both", expand=True)

        self.chat_box = ScrolledText(right, wrap="word", state="disabled", font=(self.ui_font_family, 13))
        self.chat_box.pack(fill="both", expand=True, pady=(4, 8))
        self.chat_box.configure(padx=8, pady=8)
        self.chat_box.tag_configure("meta", foreground="#6b7280", font=(self.mono_font_family, 10))
        self.chat_box.tag_configure("user", foreground="#1d4ed8", font=(self.ui_font_family, 12, "bold"))
        self.chat_box.tag_configure("system", foreground="#4b5563", font=(self.ui_font_family, 12, "italic"))
        self.chat_box.tag_configure("agent", foreground="#065f46", font=(self.ui_font_family, 12, "bold"))
        self.chat_box.tag_configure("error", foreground="#b91c1c", font=(self.ui_font_family, 12, "bold"))
        self.chat_box.tag_configure("text", foreground="#111827", font=(self.ui_font_family, 13))

        ttk.Label(right, text="Ваше сообщение (Enter = отправить, Shift+Enter = новая строка)").pack(anchor="w")
        self.input_box = tk.Text(right, height=3, wrap="word", font=(self.ui_font_family, 13), undo=True)
        self.input_box.pack(fill="x")
        self.input_box.bind("<Return>", self._on_input_return)
        self.input_box.bind("<Shift-Return>", self._on_shift_return)
        self.input_box.bind("<Control-v>", lambda _event: self._paste_event())
        self.input_box.bind("<Control-V>", lambda _event: self._paste_event())

        send_row = ttk.Frame(right)
        send_row.pack(fill="x", pady=(6, 0))
        ttk.Button(send_row, text="Вставить", command=self.paste_from_clipboard).pack(side="left")
        ttk.Button(send_row, text="Отправить", command=self.send_user_message, style="Primary.TButton").pack(side="right")

    def _pick_font_family(self) -> str:
        candidates = [
            "Noto Sans",
            "Noto Sans Display",
            "Inter",
            "Ubuntu",
            "Segoe UI",
            "DejaVu Sans",
        ]
        try:
            installed = {name.lower(): name for name in tkfont.families(self)}
        except tk.TclError:
            return "DejaVu Sans"
        for candidate in candidates:
            match = installed.get(candidate.lower())
            if match:
                return match
        return "DejaVu Sans"

    def _pick_mono_family(self) -> str:
        candidates = [
            "JetBrains Mono",
            "Fira Code",
            "Noto Sans Mono",
            "Ubuntu Mono",
            "DejaVu Sans Mono",
        ]
        try:
            installed = {name.lower(): name for name in tkfont.families(self)}
        except tk.TclError:
            return "DejaVu Sans Mono"
        for candidate in candidates:
            match = installed.get(candidate.lower())
            if match:
                return match
        return "DejaVu Sans Mono"

    def _on_input_return(self, event: tk.Event) -> str:
        self.send_user_message()
        return "break"

    def _on_shift_return(self, event: tk.Event) -> str:
        self.input_box.insert("insert", "\n")
        return "break"

    def _paste_event(self) -> str:
        self.paste_from_clipboard()
        return "break"

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def set_running(self, value: bool) -> None:
        self.running = value
        if value:
            self.start_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
            self.reset_btn.configure(state="disabled")
        else:
            self.start_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")
            self.reset_btn.configure(state="normal")

    def _parse_delay(self) -> float:
        try:
            parsed = float(self.delay_var.get().strip())
        except ValueError:
            return 2.0
        if parsed < 0.0:
            return 0.0
        if parsed > 30.0:
            return 30.0
        return parsed

    def append_chat(self, line: str) -> None:
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"{line}\n")
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def append_chat_entry(
        self,
        role: str,
        speaker: str,
        text: str,
        model: str = "",
    ) -> None:
        timestamp = time.strftime("%H:%M:%S")
        text = self._normalize_text_for_display(text)
        self.chat_box.configure(state="normal")
        self.chat_box.insert("end", f"[{timestamp}] ", ("meta",))
        if role == "user":
            self.chat_box.insert("end", f"{speaker}: ", ("user",))
            self.chat_box.insert("end", f"{text}\n\n", ("text",))
        elif role == "system":
            self.chat_box.insert("end", f"{speaker}: ", ("system",))
            self.chat_box.insert("end", f"{text}\n\n", ("system",))
        elif role == "error":
            self.chat_box.insert("end", f"{speaker}: ", ("error",))
            self.chat_box.insert("end", f"{text}\n\n", ("error",))
        else:
            label = f"{speaker}<{model}>: " if model else f"{speaker}: "
            self.chat_box.insert("end", label, ("agent",))
            self.chat_box.insert("end", f"{text}\n\n", ("text",))
        self.chat_box.see("end")
        self.chat_box.configure(state="disabled")

    def _normalize_text_for_display(self, text: str) -> str:
        # Normalize line endings and remove non-printable control chars except newlines/tabs.
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"[\x00-\x08\x0B-\x1F\x7F]", "", normalized)
        # Collapse excessive empty lines for more consistent rendering.
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        lines = [line.rstrip() for line in normalized.split("\n")]
        return "\n".join(lines).strip()

    def _sanitize_agent_response(self, text: str, agent_name: str) -> str:
        cleaned = self._normalize_text_for_display(text)
        # Remove model-added role labels that duplicate app formatting.
        patterns = [
            rf"^\s*{re.escape(agent_name)}\s*[:：-]\s*",
            r"^\s*assistant\s*[:：-]\s*",
            r"^\s*user\s*[:：-]\s*",
            r"^\s*system\s*[:：-]\s*",
        ]
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _log(self, text: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with self.debug_log_path.open("a", encoding="utf-8") as fp:
                fp.write(f"[{timestamp}] {text}\n")
        except OSError:
            pass

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

    def _drain_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                self._log(f"ui_event kind={kind} payload={str(payload)[:160]}")
                if kind == "chat":
                    self.append_chat_entry(
                        role=payload.get("role", "agent"),
                        speaker=payload.get("speaker", "agent"),
                        text=payload.get("text", ""),
                        model=payload.get("model", ""),
                    )
                elif kind == "status":
                    self.set_status(payload.get("text", ""))
                elif kind == "finalize":
                    self.set_running(False)
                    self.set_status(payload.get("text") or "Остановлено")
        except queue.Empty:
            pass
        self.after(80, self._drain_ui_queue)

    def refresh_agents_list(self) -> None:
        self.agents_list.delete(0, "end")
        for agent in self.agents:
            preview = agent.system_prompt.replace("\n", " ")[:55]
            self.agents_list.insert("end", f"{agent.name} <{agent.model}> | {preview}")

    def rebuild_client(self) -> bool:
        timeout_text = self.timeout_var.get().strip()
        try:
            timeout = float(timeout_text)
            if timeout <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid timeout", "Timeout must be a positive number")
            return False
        self.client = OllamaClient(self.host_var.get().strip(), timeout=timeout)
        return True

    def refresh_models(self, show_error: bool = True) -> None:
        if not self.rebuild_client():
            return
        try:
            self.available_models = self.client.list_models()
            self.set_status(f"Подключено. Моделей: {len(self.available_models)}")
        except Exception as exc:  # noqa: BLE001
            self.available_models = []
            self.set_status("Ошибка подключения")
            if show_error:
                messagebox.showerror("Ollama connection error", str(exc))

    def open_connection_window(self) -> None:
        win = tk.Toplevel(self)
        win.title("Connection Settings")
        win.geometry("500x180")
        win.resizable(False, False)

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        ttk.Label(frame, text="Ollama Host").grid(row=0, column=0, sticky="w")
        host_entry = ttk.Entry(frame, textvariable=self.host_var, width=50)
        host_entry.grid(row=1, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(frame, text="Timeout (seconds)").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=self.timeout_var, width=16).grid(row=3, column=0, sticky="w", pady=(2, 12))

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=4, column=0, sticky="e")

        def save_and_test() -> None:
            self.refresh_models(show_error=True)
            if self.available_models:
                win.destroy()

        ttk.Button(btn_row, text="Save & Test", command=save_and_test).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Close", command=win.destroy).pack(side="left")

        frame.columnconfigure(0, weight=1)
        host_entry.focus_set()

    def open_add_model_window(self) -> None:
        if not self.available_models:
            self.refresh_models(show_error=True)
            if not self.available_models:
                return

        win = tk.Toplevel(self)
        win.title("Connect Model")
        win.geometry("620x450")

        frame = ttk.Frame(win, padding=12)
        frame.pack(fill="both", expand=True)

        name_var = tk.StringVar(value=f"agent{len(self.agents) + 1}")
        model_var = tk.StringVar(value=self.available_models[0])
        keys = list(self.prompts.keys()) or ["default"]
        prompt_key_var = tk.StringVar(value=keys[0])

        ttk.Label(frame, text="Agent Name").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=name_var).grid(row=1, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(frame, text="Model").grid(row=2, column=0, sticky="w")
        model_combo = ttk.Combobox(frame, textvariable=model_var, values=self.available_models, state="readonly")
        model_combo.grid(row=3, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(frame, text="Prompt Preset (from system_prompts.json)").grid(row=4, column=0, sticky="w")
        prompt_combo = ttk.Combobox(frame, textvariable=prompt_key_var, values=keys, state="readonly")
        prompt_combo.grid(row=5, column=0, sticky="ew", pady=(2, 10))

        ttk.Label(frame, text="System Prompt").grid(row=6, column=0, sticky="w")
        prompt_box = ScrolledText(frame, height=10, wrap="word", font=(self.ui_font_family, 12))
        prompt_box.grid(row=7, column=0, sticky="nsew", pady=(2, 12))

        def fill_prompt() -> None:
            key = prompt_key_var.get()
            text = self.prompts.get(key, "")
            prompt_box.delete("1.0", "end")
            prompt_box.insert("1.0", text)

        prompt_combo.bind("<<ComboboxSelected>>", lambda _event: fill_prompt())
        fill_prompt()

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=8, column=0, sticky="e")

        def add_agent() -> None:
            name = name_var.get().strip()
            model = model_var.get().strip()
            system_prompt = prompt_box.get("1.0", "end").strip()

            if not name:
                messagebox.showerror("Invalid name", "Agent name is required")
                return
            if any(a.name == name for a in self.agents):
                messagebox.showerror("Duplicate name", f"Agent '{name}' already exists")
                return
            if model not in self.available_models:
                messagebox.showerror("Invalid model", "Choose a local model from the list")
                return
            if not system_prompt:
                messagebox.showerror("Empty prompt", "System prompt cannot be empty")
                return

            self.agents.append(Agent(name=name, model=model, system_prompt=system_prompt))
            self.refresh_agents_list()
            self.append_chat_entry("system", "Система", f"Подключен агент {name} <{model}>")
            if self.running:
                self._ensure_agent_context(self.agents[-1])
            win.destroy()

        ttk.Button(btn_row, text="Connect", command=add_agent).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="left")

        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(7, weight=1)
        model_combo.focus_set()

    def remove_selected_agent(self) -> None:
        sel = self.agents_list.curselection()
        if not sel:
            return
        idx = sel[0]
        agent = self.agents[idx]
        del self.agents[idx]
        self.agent_contexts.pop(agent.name, None)
        self.agent_seen_count.pop(agent.name, None)
        self.refresh_agents_list()
        self.append_chat_entry("system", "Система", f"Агент удален: {agent.name}")

    def send_user_message(self) -> None:
        text = self.input_box.get("1.0", "end-1c").strip()
        if not text:
            return
        self.input_box.delete("1.0", "end")
        with self.history_lock:
            self.history.append(Message(speaker="User", role="user", content=text))
        self.append_chat_entry("user", "Вы", text)

    def paste_from_clipboard(self) -> None:
        try:
            text = self.clipboard_get()
        except tk.TclError:
            return
        self.input_box.insert("insert", text)

    def start_chat(self) -> None:
        if self.running:
            return
        if not self.agents:
            messagebox.showwarning("No agents", "Connect at least one model first")
            return

        self.stop_requested = False
        self.set_running(True)
        self.set_status("Чат запущен")
        self._log("start_chat")
        self.inter_message_delay_sec = self._parse_delay()
        self.current_agent_idx = 0
        self.agent_contexts.clear()
        self.agent_seen_count.clear()
        for agent in self.agents:
            self._ensure_agent_context(agent)

        with self.history_lock:
            if not self.history:
                self.history.append(
                    Message(
                        speaker="User",
                        role="user",
                        content="Начните разговор на русском языке по теме последних сообщений пользователя.",
                    )
                )

        threading.Thread(target=self._chat_loop_worker, daemon=True).start()

    def stop_chat(self) -> None:
        self.stop_requested = True
        self.set_status("Остановка...")
        self._log("stop_chat")

    def reset_chat(self) -> None:
        if self.running:
            messagebox.showwarning("Chat running", "Сначала нажмите Стоп")
            return
        with self.history_lock:
            self.history.clear()
        self.agent_contexts.clear()
        self.agent_seen_count.clear()
        self.chat_box.configure(state="normal")
        self.chat_box.delete("1.0", "end")
        self.chat_box.configure(state="disabled")
        self.set_status("Чат сброшен")

    def _chat_loop_worker(self) -> None:
        try:
            self._log("worker_started")
            while not self.stop_requested:
                agents_snapshot = list(self.agents)
                if not agents_snapshot:
                    self.ui_queue.put(
                        (
                            "chat",
                            {
                                "role": "system",
                                "speaker": "Система",
                                "text": "Нет подключенных агентов. Добавьте модель и нажмите Старт снова.",
                            },
                        )
                    )
                    self.stop_requested = True
                    break

                agent = agents_snapshot[self.current_agent_idx % len(agents_snapshot)]
                self.current_agent_idx = (self.current_agent_idx + 1) % len(agents_snapshot)

                self._ensure_agent_context(agent)
                self.ui_queue.put(("status", {"text": f"Отвечает: {agent.name}<{agent.model}>"}))
                self._log(f"agent_request {agent.name}<{agent.model}>")
                try:
                    context = self.agent_contexts[agent.name]
                    with self.history_lock:
                        start_idx = self.agent_seen_count.get(agent.name, 0)
                        unseen = self.history[start_idx:]
                        self.agent_seen_count[agent.name] = len(self.history)

                    for item in unseen:
                        # For each agent:
                        # - its own previous messages are assistant turns
                        # - all other participants are user turns
                        role = "assistant" if item.speaker == agent.name else "user"
                        context.append(
                            {
                                "role": role,
                                "content": f"{item.speaker}: {item.content}",
                            }
                        )

                    # Keep context bounded so models don't degrade into empty/garbled replies.
                    if len(context) > self.max_context_messages + 1:
                        context[:] = [context[0]] + context[-self.max_context_messages:]

                    response = self.client.chat(
                        agent.model,
                        context,
                        options={
                            "num_predict": 80,
                            "temperature": 0.7,
                            "repeat_penalty": 1.2,
                        },
                    )
                    response = self._sanitize_agent_response(response, agent.name)
                    if not response:
                        self._log(f"agent_response_empty {agent.name}")
                        self.ui_queue.put(
                            (
                                "chat",
                                {
                                    "role": "system",
                                    "speaker": "Система",
                                    "text": f"{agent.name} не дал ответ. Пропускаю ход.",
                                },
                            )
                        )
                        continue

                    msg = Message(speaker=agent.name, role="assistant", content=response)
                    with self.history_lock:
                        self.history.append(msg)
                    self._log(f"agent_response_ok {agent.name} len={len(response)}")
                    self.ui_queue.put(
                        (
                            "chat",
                            {
                                "role": "agent",
                                "speaker": agent.name,
                                "model": agent.model,
                                "text": response,
                            },
                        )
                    )
                    self.ui_queue.put(("status", {"text": "Чат запущен"}))
                except Exception as exc:  # noqa: BLE001
                    self._log(f"agent_response_err {agent.name} err={exc}")
                    err_text = str(exc)
                    self.ui_queue.put(
                        (
                            "chat",
                            {
                                "role": "error",
                                "speaker": f"{agent.name}<{agent.model}>",
                                "text": err_text,
                            },
                        )
                    )

                # Artificial pacing: let the conversation breathe between agents.
                for _ in range(max(1, int(self.inter_message_delay_sec / 0.1))):
                    if self.stop_requested:
                        break
                    time.sleep(0.1)
        except Exception as exc:  # noqa: BLE001
            self._log(f"worker_crash err={exc}")
            self.ui_queue.put(
                (
                    "chat",
                    {
                        "role": "error",
                        "speaker": "Система",
                        "text": f"Внутренняя ошибка worker: {exc}",
                    },
                )
            )
        finally:
            self._log("worker_finalize")
            self.ui_queue.put(("finalize", {"text": "Остановлено"}))


def main() -> None:
    app = MultiChatApp(prompts_path=Path.cwd() / "system_prompts.json")
    app.mainloop()


if __name__ == "__main__":
    main()
