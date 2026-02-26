from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ollama_multi_chat.core import Agent, Message, OllamaClient, build_prompt_messages, load_system_prompts


def parse_agent_spec(spec: str, index: int, system_prompt: str) -> Agent:
    if "=" in spec:
        name, model = spec.split("=", 1)
        name = name.strip()
        model = model.strip()
        if not name or not model:
            raise ValueError(f"Invalid --agent spec: {spec}")
        return Agent(name=name, model=model, system_prompt=system_prompt)
    return Agent(name=f"agent{index}", model=spec.strip(), system_prompt=system_prompt)


def print_help() -> None:
    print("Commands:")
    print("  /help                 Show this help")
    print("  /models               Show available local Ollama models")
    print("  /agents               Show connected agents")
    print("  /add [name=]model     Connect another model")
    print("  /remove <name>        Remove agent by name")
    print("  /next                 Run one full round (all agents respond)")
    print("  /auto <n>             Run n rounds")
    print("  /history [n]          Show last n messages (default 15)")
    print("  /quit                 Exit")
    print("  <text>                Send a human message into chat")


def run_round(client: OllamaClient, agents: list[Agent], history: list[Message]) -> None:
    if not agents:
        print("No agents connected. Use /add [name=]model")
        return
    for agent in agents:
        try:
            reply = client.chat(agent.model, build_prompt_messages(agent, history))
        except Exception as exc:  # noqa: BLE001
            print(f"[{agent.name}] error: {exc}")
            continue
        history.append(Message(speaker=agent.name, role="assistant", content=reply))
        print(f"[{agent.name}<{agent.model}>] {reply}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLI multi-model chat over local Ollama")
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--agent", action="append", default=[])
    parser.add_argument("--prompt-file", default="system_prompts.json")
    parser.add_argument("--prompt-key", default="default")
    args = parser.parse_args()

    prompts = load_system_prompts(Path(args.prompt_file))
    default_prompt = prompts.get(args.prompt_key) or prompts.get("default") or next(iter(prompts.values()))

    client = OllamaClient(args.host)
    try:
        available = client.list_models()
    except Exception as exc:  # noqa: BLE001
        print(exc)
        sys.exit(1)

    if not available:
        print("No models found in Ollama. Pull one first, e.g.: ollama pull llama3.2")
        sys.exit(1)

    agents: list[Agent] = []
    if args.agent:
        for idx, spec in enumerate(args.agent, 1):
            agents.append(parse_agent_spec(spec, idx, default_prompt))
    else:
        seed = available[:2] if len(available) >= 2 else available[:1]
        for idx, model in enumerate(seed, 1):
            agents.append(Agent(name=f"agent{idx}", model=model, system_prompt=default_prompt))

    history: list[Message] = []

    print("Multi-agent Ollama CLI")
    print(f"Ollama: {args.host}")
    print("Connected agents:", ", ".join(f"{a.name}<{a.model}>" for a in agents) if agents else "none")
    print_help()

    while True:
        try:
            raw = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            break

        if not raw:
            continue
        if raw == "/quit":
            print("Bye")
            break
        if raw == "/help":
            print_help()
            continue
        if raw == "/models":
            print("Models:", ", ".join(client.list_models()))
            continue
        if raw == "/agents":
            for agent in agents:
                print(f"- {agent.name}<{agent.model}>")
            continue
        if raw.startswith("/add "):
            spec = raw.removeprefix("/add ").strip()
            if not spec:
                print("Usage: /add [name=]model")
                continue
            agent = parse_agent_spec(spec, len(agents) + 1, default_prompt)
            agents.append(agent)
            print(f"Added {agent.name}<{agent.model}>")
            continue
        if raw.startswith("/remove "):
            name = raw.removeprefix("/remove ").strip()
            agents = [a for a in agents if a.name != name]
            print(f"Removed `{name}`")
            continue
        if raw == "/next":
            run_round(client, agents, history)
            continue
        if raw.startswith("/auto "):
            try:
                count = int(raw.removeprefix("/auto ").strip())
            except ValueError:
                print("Usage: /auto <n>")
                continue
            for _ in range(max(0, count)):
                run_round(client, agents, history)
            continue
        if raw.startswith("/history"):
            limit = 15
            parts = raw.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    limit = int(parts[1])
                except ValueError:
                    print("Usage: /history [n]")
                    continue
            for msg in history[-limit:]:
                print(f"[{msg.speaker}] {msg.content}")
            continue

        history.append(Message(speaker="User", role="user", content=raw))
        print("(message added, use /next or /auto N)")


if __name__ == "__main__":
    main()
