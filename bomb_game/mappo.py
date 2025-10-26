from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import BombDefusalEnv, MultiAgentObservation
from .game import AGENT_NAMES


def _mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Replace invalid action logits with a very negative number
    invalid = mask <= 0
    return logits.masked_fill(invalid, -1e9)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        value = self.critic(obs)
        return logits, value

    def act(self, obs: torch.Tensor, mask: torch.Tensor) -> Tuple[int, float, float]:
        logits, value = self.forward(obs)
        masked_logits = _mask_logits(logits, mask)
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def evaluate_actions(
        self, obs: torch.Tensor, mask: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.forward(obs)
        masked_logits = _mask_logits(logits, mask)
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze(-1)


@dataclass
class Transition:
    observations: Dict[str, np.ndarray]
    masks: Dict[str, np.ndarray]
    actions: Dict[str, int]
    log_probs: Dict[str, float]
    values: Dict[str, float]
    rewards: Dict[str, float]
    done: bool


class MAPPOBuffer:
    def __init__(self) -> None:
        self.storage: List[Transition] = []

    def add(self, transition: Transition) -> None:
        self.storage.append(transition)

    def clear(self) -> None:
        self.storage.clear()


class MAPPOTrainer:
    def __init__(
        self,
        env: BombDefusalEnv,
        *,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_param: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        epochs: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device)

        observation_dim = env.observation_space()
        action_dim = env.action_space()
        self.model = ActorCritic(observation_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.buffer = MAPPOBuffer()

    # ------------------------------------------------------------------
    def collect_rollout(self, rollout_length: int) -> MultiAgentObservation:
        obs = self.env.reset()
        for _ in range(rollout_length):
            actions: Dict[str, int] = {}
            log_probs: Dict[str, float] = {}
            values: Dict[str, float] = {}
            for agent in AGENT_NAMES:
                obs_tensor = torch.from_numpy(obs.observation[agent]).float().to(self.device)
                mask_tensor = torch.from_numpy(obs.action_masks[agent]).float().to(self.device)
                action, log_prob, value = self.model.act(obs_tensor, mask_tensor)
                actions[agent] = action
                log_probs[agent] = log_prob
                values[agent] = value
            next_obs, rewards, done, _ = self.env.step(actions)
            transition = Transition(
                observations={agent: obs.observation[agent] for agent in AGENT_NAMES},
                masks={agent: obs.action_masks[agent] for agent in AGENT_NAMES},
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                done=done,
            )
            self.buffer.add(transition)
            obs = next_obs
            if done:
                break
        return obs

    # ------------------------------------------------------------------
    def update(self, last_obs: MultiAgentObservation) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        returns: Dict[str, List[float]] = {agent: [] for agent in AGENT_NAMES}
        advantages: Dict[str, List[float]] = {agent: [] for agent in AGENT_NAMES}

        next_values = {
            agent: float(
                self.model.forward(torch.from_numpy(last_obs.observation[agent]).float().to(self.device))[1].item()
            )
            for agent in AGENT_NAMES
        }

        gae: Dict[str, float] = {agent: 0.0 for agent in AGENT_NAMES}
        for step in reversed(self.buffer.storage):
            for agent in AGENT_NAMES:
                mask = 1.0 - float(step.done)
                delta = step.rewards[agent] + self.gamma * next_values[agent] * mask - step.values[agent]
                gae[agent] = delta + self.gamma * self.gae_lambda * mask * gae[agent]
                advantages[agent].insert(0, gae[agent])
                returns[agent].insert(0, gae[agent] + step.values[agent])
                next_values[agent] = step.values[agent]

        # Flatten data per agent
        obs_batch: List[torch.Tensor] = []
        mask_batch: List[torch.Tensor] = []
        action_batch: List[int] = []
        log_prob_batch: List[float] = []
        advantage_batch: List[float] = []
        return_batch: List[float] = []

        for idx, transition in enumerate(self.buffer.storage):
            for agent in AGENT_NAMES:
                obs_batch.append(torch.from_numpy(transition.observations[agent]).float())
                mask_batch.append(torch.from_numpy(transition.masks[agent]).float())
                action_batch.append(transition.actions[agent])
                log_prob_batch.append(transition.log_probs[agent])
                advantage_batch.append(advantages[agent][idx])
                return_batch.append(returns[agent][idx])

        obs_tensor = torch.stack(obs_batch).to(self.device)
        mask_tensor = torch.stack(mask_batch).to(self.device)
        action_tensor = torch.tensor(action_batch, dtype=torch.long, device=self.device)
        old_log_probs_tensor = torch.tensor(log_prob_batch, dtype=torch.float32, device=self.device)
        advantage_tensor = torch.tensor(advantage_batch, dtype=torch.float32, device=self.device)
        return_tensor = torch.tensor(return_batch, dtype=torch.float32, device=self.device)

        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (advantage_tensor.std() + 1e-8)

        dataset_size = obs_tensor.size(0)
        indices = np.arange(dataset_size)

        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_obs = obs_tensor[batch_idx]
                batch_mask = mask_tensor[batch_idx]
                batch_actions = action_tensor[batch_idx]
                batch_old_log_probs = old_log_probs_tensor[batch_idx]
                batch_advantages = advantage_tensor[batch_idx]
                batch_returns = return_tensor[batch_idx]

                log_probs, entropy, values = self.model.evaluate_actions(batch_obs, batch_mask, batch_actions)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        metrics["actor_loss"] = float(actor_loss.item())
        metrics["critic_loss"] = float(critic_loss.item())
        metrics["entropy"] = float(entropy_loss.item())
        self.buffer.clear()
        return metrics


__all__ = ["MAPPOTrainer"]
