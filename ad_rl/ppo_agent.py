from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


@dataclass
class PPORollout:
    observations: list
    actions: list[int]
    log_probs: list[float]
    rewards: list[float]
    dones: list[bool]
    values: list[float]

    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def push(self, obs, action: int, log_prob: float, reward: float, done: bool, value: float) -> None:
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.values.append(float(value))

    def __len__(self) -> int:
        return len(self.rewards)


class PPOActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(obs)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        return logits, value

    def distribution(self, obs: torch.Tensor) -> tuple[Categorical, torch.Tensor]:
        logits, value = self.forward(obs)
        return Categorical(logits=logits), value


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_coef: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        minibatch_size: int = 64,
    ):
        # ADDED: PPO actor-critic for the PyADM1 setpoint controller.
        # Reason: compare DQN/QMIX with an on-policy policy-gradient method
        # under the same Gym environment, action set, reward, and PI loop.
        # Role: learn a stochastic discrete policy over T_setpoint increments.
        # Reference: user-requested PPO addition after DQN and QMIX.
        self.network = PPOActorCritic(obs_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_coef = float(clip_coef)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.update_epochs = int(update_epochs)
        self.minibatch_size = int(minibatch_size)

    def select_action(self, obs) -> tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.network.distribution(obs_t)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def act(self, obs, deterministic: bool = True) -> int:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self.network.distribution(obs_t)
            if deterministic:
                return int(torch.argmax(dist.logits, dim=-1).item())
            return int(dist.sample().item())

    def _advantages_and_returns(self, rollout: PPORollout, last_value: float = 0.0):
        rewards = rollout.rewards
        dones = rollout.dones
        values = rollout.values + [float(last_value)]
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
        returns = advantages + np.asarray(rollout.values, dtype=np.float32)
        return advantages, returns

    def update(self, rollout: PPORollout, last_value: float = 0.0) -> dict[str, float]:
        if len(rollout) == 0:
            return {
                "policy_loss": np.nan,
                "value_loss": np.nan,
                "entropy": np.nan,
                "approx_kl": np.nan,
            }

        advantages, returns = self._advantages_and_returns(rollout, last_value)
        obs = torch.tensor(np.array(rollout.observations), dtype=torch.float32)
        actions = torch.tensor(rollout.actions, dtype=torch.long)
        old_log_probs = torch.tensor(rollout.log_probs, dtype=torch.float32)
        advantages_t = torch.tensor(advantages, dtype=torch.float32)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)

        n_samples = len(rollout)
        minibatch_size = min(self.minibatch_size, n_samples)
        indices = np.arange(n_samples)
        last_metrics = {}

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n_samples, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                mb_idx_t = torch.tensor(mb_idx, dtype=torch.long)

                dist, values = self.network.distribution(obs[mb_idx_t])
                new_log_probs = dist.log_prob(actions[mb_idx_t])
                entropy = dist.entropy().mean()
                log_ratio = new_log_probs - old_log_probs[mb_idx_t]
                ratio = torch.exp(log_ratio)

                unclipped = ratio * advantages_t[mb_idx_t]
                clipped = torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef) * advantages_t[mb_idx_t]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns_t[mb_idx_t])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                last_metrics = {
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                    "approx_kl": float(approx_kl.item()),
                }

        return last_metrics
