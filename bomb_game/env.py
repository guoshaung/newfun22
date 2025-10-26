from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, MutableMapping, Tuple

import numpy as np

from .game import AGENT_NAMES, GameManager, GameState, create_manager

BOMB_FEATURES = 4


def _normalize(value: float, maximum: float) -> float:
    if maximum == 0:
        return 0.0
    return min(max(value / maximum, 0.0), 1.0)


@dataclass
class MultiAgentObservation:
    """Container returned by :class:`BombDefusalEnv` to describe a joint state."""

    observation: MutableMapping[str, np.ndarray]
    action_masks: MutableMapping[str, np.ndarray]


class BombDefusalEnv:
    """A lightweight multi-agent environment suitable for MAPPO style algorithms."""

    def __init__(self, manager: GameManager | None = None) -> None:
        self.manager = manager or create_manager()
        self.state: GameState | None = None
        sample_state = create_manager().create_game()
        self.max_actions = 1 + len(sample_state.rooms) + len(sample_state.bombs)
        self.observation_size = self._compute_observation_size(sample_state)

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None) -> MultiAgentObservation:
        if seed is not None:
            np.random.seed(seed)
        self.state = self.manager.create_game()
        return self._collect_observation(self.state)

    def step(self, actions: Mapping[str, int]) -> Tuple[MultiAgentObservation, Dict[str, float], bool, Dict[str, Dict]]:
        if self.state is None:
            raise RuntimeError("Environment has not been reset")

        decoded_actions = [self._decode_action(agent_name, index) for agent_name, index in actions.items()]
        result = self.manager.apply_actions(self.state, decoded_actions)
        rewards = self._compute_rewards(self.state, decoded_actions)
        obs = self._collect_observation(self.state)
        done = self.state.game_over
        info: Dict[str, Dict] = {agent: {"messages": result["messages"]} for agent in AGENT_NAMES}
        return obs, rewards, done, info

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------
    def _compute_observation_size(self, state: GameState) -> int:
        room_count = len(state.rooms)
        bomb_count = len(state.bombs)
        return room_count + bomb_count * BOMB_FEATURES

    def _collect_observation(self, state: GameState) -> MultiAgentObservation:
        observations: Dict[str, np.ndarray] = {}
        masks: Dict[str, np.ndarray] = {}
        for agent_name in AGENT_NAMES:
            observations[agent_name] = self._build_agent_observation(agent_name, state)
            masks[agent_name] = self._build_agent_action_mask(agent_name, state)
        return MultiAgentObservation(observation=observations, action_masks=masks)

    def _build_agent_observation(self, agent_name: str, state: GameState) -> np.ndarray:
        agent = state.agents[agent_name]
        rooms = list(state.rooms.keys())
        room_one_hot = np.zeros(len(rooms), dtype=np.float32)
        room_one_hot[rooms.index(agent.room)] = 1.0

        bomb_features: List[float] = []
        for bomb in state.bombs.values():
            bomb_features.append(1.0 if bomb.room == agent.room else 0.0)
            bomb_features.append(1.0 if bomb.defused else 0.0)
            bomb_features.append(_normalize(bomb.current_phase, bomb.phase_count))
            bomb_features.append(_normalize(len(agent.known_bombs.get(bomb.bomb_id, [])), bomb.phase_count))
        return np.concatenate([room_one_hot, np.array(bomb_features, dtype=np.float32)])

    def _build_agent_action_mask(self, agent_name: str, state: GameState) -> np.ndarray:
        mask = np.zeros(self.max_actions, dtype=np.float32)
        agent = state.agents[agent_name]
        rooms = list(state.rooms.keys())
        bombs = list(state.bombs.values())

        # index 0 -> inspect
        mask[0] = 1.0

        # Moves occupy [1, room_count]
        for idx, room in enumerate(rooms, start=1):
            if room in state.rooms[agent.room]:
                mask[idx] = 1.0

        # Cuts occupy the remainder
        offset = 1 + len(rooms)
        for idx, bomb in enumerate(bombs):
            if bomb.room == agent.room and not bomb.defused:
                expected_color = bomb.phases[bomb.current_phase]
                if expected_color in agent.cutters:
                    mask[offset + idx] = 1.0
        return mask

    # ------------------------------------------------------------------
    # Action decoding and reward shaping
    # ------------------------------------------------------------------
    def _decode_action(self, agent_name: str, action_index: int) -> Dict:
        if self.state is None:
            raise RuntimeError("Environment has not been reset")
        rooms = list(self.state.rooms.keys())
        bombs = list(self.state.bombs.values())
        if action_index == 0:
            return {"agent": agent_name, "action": "inspect"}
        if 1 <= action_index <= len(rooms):
            return {"agent": agent_name, "action": "move", "target": rooms[action_index - 1]}
        offset = 1 + len(rooms)
        bomb_idx = action_index - offset
        if 0 <= bomb_idx < len(bombs):
            bomb = bombs[bomb_idx]
            return {
                "agent": agent_name,
                "action": "cut",
                "bomb_id": bomb.bomb_id,
                "color": bomb.phases[bomb.current_phase],
            }
        raise ValueError(f"Invalid action index {action_index} for agent {agent_name}")

    def _compute_rewards(self, state: GameState, actions: List[Dict]) -> Dict[str, float]:
        rewards: Dict[str, float] = {agent: 0.0 for agent in AGENT_NAMES}
        for action in actions:
            if action["action"] == "cut":
                bomb = state.bombs[action["bomb_id"]]
                if bomb.defused:
                    rewards[action["agent"]] += 1.0
                else:
                    rewards[action["agent"]] += 0.1
            elif action["action"] == "inspect":
                rewards[action["agent"]] += 0.05
        if state.game_over and state.termination_reason == "All bombs defused":
            for agent in rewards:
                rewards[agent] += 10.0
        elif state.game_over and state.termination_reason == "Deadlock detected (repeated actions)":
            for agent in rewards:
                rewards[agent] -= 1.0
        return rewards

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def observation_space(self) -> int:
        return self.observation_size

    def action_space(self) -> int:
        return self.max_actions


__all__ = ["BombDefusalEnv", "MultiAgentObservation"]
