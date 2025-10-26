from __future__ import annotations

import os
from typing import Dict, List, Sequence

import httpx

from .game import AGENT_NAMES, GameState
from .tommind import evaluate_tom_hierarchy

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekLLMClient:
    """Minimal DeepSeek API client used by the :class:`LLMAgentController`."""

    def __init__(self, api_key: str | None = None, *, model: str = DEFAULT_MODEL, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is not configured. Set DEEPSEEK_API_KEY or pass api_key explicitly.")
        self.model = model
        self._client = httpx.Client(base_url=DEEPSEEK_API_BASE, timeout=timeout)

    def generate(self, messages: Sequence[Dict[str, str]], *, temperature: float = 0.3, max_tokens: int = 512) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._client.post("/chat/completions", json=payload, headers={"Authorization": f"Bearer {self.api_key}"})
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def _summarise_state(state: GameState) -> str:
    room_lines = []
    for room, neighbours in state.rooms.items():
        room_lines.append(f"{room} -> {', '.join(neighbours)}")
    bomb_lines = []
    for bomb in state.bombs.values():
        status = "defused" if bomb.defused else f"phase {bomb.current_phase + 1}/{bomb.phase_count}"
        bomb_lines.append(f"{bomb.bomb_id} in {bomb.room}: {status} via {' '.join(bomb.phases)}")
    agent_lines = []
    for agent in state.agents.values():
        agent_lines.append(
            f"{agent.name} at {agent.room} with cutters {', '.join(agent.cutters)} knows {list(agent.known_bombs.keys()) or 'nothing'}"
        )
    return "\n".join(["Rooms:"] + room_lines + ["Bombs:"] + bomb_lines + ["Agents:"] + agent_lines)


class LLMAgentController:
    """Coordinates DeepSeek responses for every agent with hierarchical ToM prompts."""

    def __init__(self, client: DeepSeekLLMClient) -> None:
        self.client = client

    def build_agent_prompt(self, state: GameState, agent: str) -> List[Dict[str, str]]:
        probes = evaluate_tom_hierarchy(state)[agent]
        tom_prompt = "\n".join(
            f"- {probe.question}\n  Expected answer: {probe.answer}" for probe in probes
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a cooperative bomb-defusal specialist. Use theory of mind reasoning to plan joint actions."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Current world state:\n"
                    f"{_summarise_state(state)}\n\n"
                    "Your task:\n"
                    f"{tom_prompt}\n\n"
                    "Provide a JSON object with fields 'introspection', 'first_order', 'second_order', and 'next_actions'."
                ),
            },
        ]
        return messages

    def propose_joint_actions(self, state: GameState) -> Dict[str, Dict[str, str]]:
        plans: Dict[str, Dict[str, str]] = {}
        for agent in AGENT_NAMES:
            messages = self.build_agent_prompt(state, agent)
            response_text = self.client.generate(messages)
            plans[agent] = {"raw_response": response_text}
        return plans


__all__ = ["DeepSeekLLMClient", "LLMAgentController"]
