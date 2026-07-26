from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
from torch import nn


class OfflineTransitionDataset:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, int(action), float(reward), next_obs, bool(done)))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DiscreteQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0.0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


class DiscreteIQLAgent(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        expectile: float = 0.7,
        temperature: float = 3.0,
        max_advantage_weight: float = 100.0,
        target_tau: float = 0.005,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.max_advantage_weight = float(max_advantage_weight)
        self.target_tau = float(target_tau)
        self.q_net = DiscreteQNetwork(obs_dim, action_dim, hidden_dim)
        self.target_q_net = DiscreteQNetwork(obs_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(obs_dim, hidden_dim)
        self.policy_net = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def act(self, obs, deterministic: bool = True) -> int:
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.policy_net(obs_t)[0]
            if deterministic:
                return int(torch.argmax(logits).item())
            probs = torch.softmax(logits, dim=0)
            return int(torch.multinomial(probs, 1).item())

    def soft_update_target(self):
        with torch.no_grad():
            for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
                target_param.data.mul_(1.0 - self.target_tau)
                target_param.data.add_(self.target_tau * param.data)

    def update(self, batch, optimizers: dict[str, torch.optim.Optimizer], gamma: float) -> dict[str, float]:
        # ADDED: discrete IQL/IDQL-style offline learner for PyADM1.
        # Reason: the reference code used DDPM-IQL from jaxrl5, but the PyADM1
        # supervisory action is already discrete.
        # Role: fit value, Q, and behavior-improved policy from an offline
        # transition dataset collected with the same Gym environment.
        # Reference: user-supplied B_OfflineRL_IDQL_LowTMAH.py.
        obs, actions, rewards, next_obs, dones = batch

        with torch.no_grad():
            q_all_target = self.target_q_net(obs)
            q_action_target = q_all_target.gather(1, actions.unsqueeze(1)).squeeze(1)
        value = self.value_net(obs)
        value_loss = expectile_loss(q_action_target - value, self.expectile)
        optimizers["value"].zero_grad()
        value_loss.backward()
        optimizers["value"].step()

        with torch.no_grad():
            next_value = self.value_net(next_obs)
            q_target = rewards + gamma * (1.0 - dones) * next_value
        q_action = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_loss = nn.functional.mse_loss(q_action, q_target)
        optimizers["q"].zero_grad()
        q_loss.backward()
        optimizers["q"].step()

        with torch.no_grad():
            advantage = q_action_target - value
            weights = torch.exp(self.temperature * advantage).clamp(max=self.max_advantage_weight)
        logits = self.policy_net(obs)
        policy_loss = (nn.functional.cross_entropy(logits, actions, reduction="none") * weights).mean()
        optimizers["policy"].zero_grad()
        policy_loss.backward()
        optimizers["policy"].step()

        self.soft_update_target()
        return {
            "value_loss": float(value_loss.item()),
            "q_loss": float(q_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "advantage_mean": float(advantage.mean().item()),
            "advantage_weight_mean": float(weights.mean().item()),
        }
