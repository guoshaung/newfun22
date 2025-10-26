from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence

Color = Literal["red", "green", "blue"]


BOMB_ID_PREFIX = "B"
AGENT_NAMES = ["Alpha", "Bravo", "Charlie"]
ROOM_COUNT = 5
ROUND_LIMIT = 30


def _generate_game_id(length: int = 8) -> str:
    """Generate a random identifier for a new game instance."""

    return "".join(random.choices(string.ascii_uppercase + string.digits, k=length))


@dataclass
class Bomb:
    """Simple representation of a bomb with an ordered defusal sequence."""

    bomb_id: str
    room: str
    phases: List[Color]
    current_phase: int = 0
    defused: bool = False

    def process_cut(self, color: Color) -> bool:
        """Process a wire cut and advance the state if it matches the expected colour."""

        if self.defused:
            raise ValueError("Bomb already defused")
        expected = self.phases[self.current_phase]
        if expected != color:
            raise ValueError(f"Incorrect color: expected {expected}")
        self.current_phase += 1
        if self.current_phase == len(self.phases):
            self.defused = True
        return self.defused

    @property
    def phase_count(self) -> int:
        return len(self.phases)


@dataclass
class Agent:
    """Agent encapsulating local knowledge of bombs and available cutters."""

    name: str
    room: str
    cutters: Sequence[Color]
    known_bombs: Dict[str, List[Color]] = field(default_factory=dict)

    def observe_bombs(self, bombs: Sequence[Bomb]) -> None:
        for bomb in bombs:
            self.known_bombs.setdefault(bomb.bomb_id, list(bomb.phases))


@dataclass
class GameState:
    """Holds all mutable state for a single game run."""

    game_id: str
    rooms: Dict[str, List[str]]
    bombs: Dict[str, Bomb]
    agents: Dict[str, Agent]
    score: int = 0
    round_count: int = 0
    history: List[List[Dict]] = field(default_factory=list)
    game_over: bool = False
    termination_reason: Optional[str] = None

    @property
    def remaining_bombs(self) -> List[Bomb]:
        return [bomb for bomb in self.bombs.values() if not bomb.defused]

    def advance_round(self, actions: List[Dict]) -> None:
        self.round_count += 1
        self.history.append(actions)
        if len(self.history) > 3:
            self.history.pop(0)

    def check_completion(self) -> None:
        if all(bomb.defused for bomb in self.bombs.values()):
            self.game_over = True
            self.termination_reason = "All bombs defused"
        elif self.round_count >= ROUND_LIMIT:
            self.game_over = True
            self.termination_reason = "Round limit exceeded"
        elif len(self.history) == 3 and self.history.count(self.history[-1]) == 3:
            self.game_over = True
            self.termination_reason = "Deadlock detected (repeated actions)"


class GameManager:
    """Factory/registry class used by the FastAPI app and reinforcement-learning env."""

    def __init__(self) -> None:
        self._games: Dict[str, GameState] = {}

    def create_game(self) -> GameState:
        game_id = _generate_game_id()
        rooms = self._generate_rooms()
        bombs = self._generate_bombs(rooms)
        agents = self._generate_agents(rooms)
        state = GameState(game_id=game_id, rooms=rooms, bombs=bombs, agents=agents)
        self._games[game_id] = state
        return state

    def get_game(self, game_id: str) -> GameState:
        try:
            return self._games[game_id]
        except KeyError as exc:
            raise KeyError(f"Unknown game id: {game_id}") from exc

    def _generate_rooms(self) -> Dict[str, List[str]]:
        rooms = {f"Room-{i}": [] for i in range(1, ROOM_COUNT + 1)}
        room_names = list(rooms.keys())
        # Ensure connectivity with a simple chain
        for idx in range(len(room_names) - 1):
            a, b = room_names[idx], room_names[idx + 1]
            rooms[a].append(b)
            rooms[b].append(a)
        # Add extra random edges
        extra_edges = random.randint(1, 3)
        for _ in range(extra_edges):
            a, b = random.sample(room_names, 2)
            if b not in rooms[a]:
                rooms[a].append(b)
                rooms[b].append(a)
        return rooms

    def _generate_bombs(self, rooms: Dict[str, List[str]]) -> Dict[str, Bomb]:
        phase_lengths = [1, 1, 2, 2, 3]
        bombs: Dict[str, Bomb] = {}
        room_names = list(rooms.keys())
        for idx, phase_count in enumerate(phase_lengths, start=1):
            bomb_id = f"{BOMB_ID_PREFIX}{idx}"
            phases = [random.choice(["red", "green", "blue"]) for _ in range(phase_count)]
            room = random.choice(room_names)
            bombs[bomb_id] = Bomb(bomb_id=bomb_id, room=room, phases=phases)
        return bombs

    def _generate_agents(self, rooms: Dict[str, List[str]]) -> Dict[str, Agent]:
        room_names = list(rooms.keys())
        cutters = {"Alpha": ("red",), "Bravo": ("green",), "Charlie": ("blue",)}
        agents: Dict[str, Agent] = {}
        for name in AGENT_NAMES:
            room = random.choice(room_names)
            agents[name] = Agent(name=name, room=room, cutters=cutters[name])
        return agents

    def serialize_state(self, state: GameState) -> Dict:
        return {
            "game_id": state.game_id,
            "score": state.score,
            "round": state.round_count,
            "game_over": state.game_over,
            "termination_reason": state.termination_reason,
            "rooms": state.rooms,
            "bombs": {
                bomb_id: {
                    "room": bomb.room,
                    "phase_count": bomb.phase_count,
                    "current_phase": bomb.current_phase,
                    "defused": bomb.defused,
                }
                for bomb_id, bomb in state.bombs.items()
            },
            "agents": {
                agent.name: {
                    "room": agent.room,
                    "cutters": list(agent.cutters),
                    "known_bombs": agent.known_bombs,
                }
                for agent in state.agents.values()
            },
        }

    def apply_actions(self, state: GameState, actions: List[Dict]) -> Dict:
        if state.game_over:
            raise ValueError("Game is already complete")

        action_log: List[Dict] = []
        round_messages: List[str] = []

        for action in actions:
            agent_name = action.get("agent")
            if agent_name not in state.agents:
                raise ValueError(f"Unknown agent: {agent_name}")
            agent = state.agents[agent_name]
            action_type = action.get("action")
            if action_type == "move":
                target = action.get("target")
                if not target:
                    raise ValueError("Move action requires target room")
                if target not in state.rooms[agent.room]:
                    raise ValueError(f"Room {target} not adjacent to {agent.room}")
                agent.room = target
                action_log.append({"agent": agent_name, "action": "move", "target": target})
            elif action_type == "inspect":
                bombs_in_room = [bomb for bomb in state.remaining_bombs if bomb.room == agent.room]
                if not bombs_in_room:
                    round_messages.append(f"{agent_name} found no bombs in {agent.room}")
                else:
                    agent.observe_bombs(bombs_in_room)
                    round_messages.append(
                        f"{agent_name} inspected {', '.join(b.bomb_id for b in bombs_in_room)} in {agent.room}"
                    )
                action_log.append({"agent": agent_name, "action": "inspect"})
            elif action_type == "cut":
                bomb_id = action.get("bomb_id")
                color = action.get("color")
                if bomb_id is None or color is None:
                    raise ValueError("Cut action requires bomb_id and color")
                if color not in agent.cutters:
                    raise ValueError(f"{agent_name} does not have a {color} cutter")
                try:
                    bomb = state.bombs[bomb_id]
                except KeyError as exc:
                    raise ValueError(f"Unknown bomb: {bomb_id}") from exc
                if bomb.room != agent.room:
                    raise ValueError(f"{bomb_id} is not in {agent.room}")
                if bomb.defused:
                    raise ValueError(f"{bomb_id} is already defused")
                try:
                    completed = bomb.process_cut(color)
                except ValueError as exc:
                    raise ValueError(str(exc)) from exc
                if completed:
                    state.score += 10 * bomb.phase_count
                    round_messages.append(f"{agent_name} completed defusal of {bomb_id}")
                else:
                    round_messages.append(
                        f"{agent_name} progressed {bomb_id} to phase {bomb.current_phase}/{bomb.phase_count}"
                    )
                action_log.append({"agent": agent_name, "action": "cut", "bomb_id": bomb_id, "color": color})
            else:
                raise ValueError(f"Unknown action type: {action_type}")

        state.advance_round(action_log)
        state.check_completion()

        return {
            "state": self.serialize_state(state),
            "messages": round_messages,
        }


def create_manager() -> GameManager:
    """Helper used by FastAPI application and tests."""

    return GameManager()
