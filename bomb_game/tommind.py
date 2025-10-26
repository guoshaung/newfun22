from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .game import AGENT_NAMES, GameState


@dataclass
class ToMProbe:
    """Base class describing a question/answer pair for theory-of-mind evaluation."""

    agent: str
    target: str
    question: str
    answer: str


@dataclass
class IntrospectionProbe(ToMProbe):
    """Probe checking whether an agent can report its own mental state."""

    @staticmethod
    def from_state(state: GameState, agent: str) -> "IntrospectionProbe":
        known = state.agents[agent].known_bombs
        bomb_summary = ", ".join(sorted(f"{bomb_id}:{' '.join(phases)}" for bomb_id, phases in known.items()))
        bomb_summary = bomb_summary or "no bombs"
        question = f"What wires does {agent} believe are needed for bombs they currently know about?"
        answer = bomb_summary
        return IntrospectionProbe(agent=agent, target=agent, question=question, answer=answer)


@dataclass
class FirstOrderToMProbe(ToMProbe):
    """Probe checking whether an agent can reason about another agent's beliefs."""

    @staticmethod
    def from_state(state: GameState, agent: str, target: str) -> "FirstOrderToMProbe":
        known = state.agents[target].known_bombs
        bomb_summary = ", ".join(sorted(f"{bomb_id}:{' '.join(phases)}" for bomb_id, phases in known.items()))
        bomb_summary = bomb_summary or "no bombs"
        question = f"What bombs does {agent} think {target} currently knows how to defuse?"
        answer = bomb_summary
        return FirstOrderToMProbe(agent=agent, target=target, question=question, answer=answer)


def _observed_joint_actions(state: GameState, subject: str, observer: str) -> Iterable[str]:
    """Return bomb ids that the observer has evidence the subject interacted with."""

    for round_actions in state.history:
        subject_action = next((action for action in round_actions if action["agent"] == subject), None)
        observer_action = next((action for action in round_actions if action["agent"] == observer), None)
        if subject_action is None or observer_action is None:
            continue
        if subject_action["action"] == "cut":
            yield subject_action["bomb_id"]
        if subject_action["action"] == "inspect" and observer_action["action"] == "inspect":
            # Shared inspection grants mutual belief of knowledge about bombs in the room
            for bomb in state.bombs.values():
                if bomb.room == state.agents[subject].room:
                    yield bomb.bomb_id


@dataclass
class SecondOrderToMProbe(ToMProbe):
    """Probe checking whether an agent can reason about another agent's belief of their knowledge."""

    evidence: List[str]

    @staticmethod
    def from_state(state: GameState, agent: str, observer: str) -> "SecondOrderToMProbe":
        evidence = sorted(set(_observed_joint_actions(state, agent, observer)))
        if evidence:
            answer = f"{observer} believes {agent} knows about {', '.join(evidence)}"
        else:
            answer = f"{observer} believes {agent} knows about no bombs"
        question = (
            f"Based on recent rounds, what does {agent} think {observer} believes about {agent}'s bomb knowledge?"
        )
        return SecondOrderToMProbe(agent=agent, target=observer, question=question, answer=answer, evidence=list(evidence))


def evaluate_tom_hierarchy(state: GameState) -> Dict[str, List[ToMProbe]]:
    """Generate introspection, first-order and second-order probes for every agent."""

    probes: Dict[str, List[ToMProbe]] = {agent: [] for agent in AGENT_NAMES}
    for agent in AGENT_NAMES:
        probes[agent].append(IntrospectionProbe.from_state(state, agent))
        others = [other for other in AGENT_NAMES if other != agent]
        for other in others:
            probes[agent].append(FirstOrderToMProbe.from_state(state, agent, other))
            probes[agent].append(SecondOrderToMProbe.from_state(state, agent, other))
    return probes


__all__ = [
    "ToMProbe",
    "IntrospectionProbe",
    "FirstOrderToMProbe",
    "SecondOrderToMProbe",
    "evaluate_tom_hierarchy",
]
