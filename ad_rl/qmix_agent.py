from __future__ import annotations

from collections import deque
import random

import numpy as np
import torch
from torch import nn


class QMixReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done):
        self.buffer.append((obs, np.asarray(actions, dtype=np.int64), reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


class AgentQNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, n_agents: int = 1, hidden_dim: int = 64):
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_agents * self.action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        q = self.net(obs)
        return q.view(obs.shape[0], self.n_agents, self.action_dim)


class QMixer(nn.Module):
    def __init__(self, state_dim: int, n_agents: int = 1, mixing_embed_dim: int = 32):
        super().__init__()
        self.n_agents = int(n_agents)
        self.embed_dim = int(mixing_embed_dim)
        self.hyper_w_1 = nn.Linear(state_dim, self.n_agents * self.embed_dim)
        self.hyper_b_1 = nn.Linear(state_dim, self.embed_dim)
        self.hyper_w_final = nn.Linear(state_dim, self.embed_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        # ADDED: QMIX monotonic mixing network for the PyADM1 setpoint task.
        # Reason: the reference SBR code used QMIX for three control agents.
        # Role: combine per-agent Q values into one total Q while preserving
        # monotonicity. With n_agents=1 this is a comparable single-control
        # variant, and the same class can later accept multiple agents.
        # Reference: user-supplied A_ExpertDemo_QMIX_test_all_HighC.py.
        batch_size = states.shape[0]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states)).view(batch_size, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(batch_size, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(agent_qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states)).view(batch_size, self.embed_dim, 1)
        v = self.value(states).view(batch_size, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        return y.view(batch_size)


class QMixController(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int = 1,
        agent_hidden_dim: int = 64,
        mixing_embed_dim: int = 32,
    ):
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.agent_net = AgentQNetwork(obs_dim, action_dim, n_agents, agent_hidden_dim)
        self.mixer = QMixer(obs_dim, n_agents, mixing_embed_dim)
        self.target_agent_net = AgentQNetwork(obs_dim, action_dim, n_agents, agent_hidden_dim)
        self.target_mixer = QMixer(obs_dim, n_agents, mixing_embed_dim)
        self.update_targets()

    def online_parameters(self):
        return list(self.agent_net.parameters()) + list(self.mixer.parameters())

    def update_targets(self):
        self.target_agent_net.load_state_dict(self.agent_net.state_dict())
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def choose_actions(self, obs, epsilon: float) -> np.ndarray:
        if random.random() < epsilon:
            return np.array(
                [random.randrange(self.action_dim) for _ in range(self.n_agents)],
                dtype=np.int64,
            )
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.agent_net(obs_t)[0]
            return torch.argmax(q_values, dim=1).cpu().numpy().astype(np.int64)

    def greedy_action(self, obs) -> int:
        return int(self.choose_actions(obs, epsilon=0.0)[0])

    def loss(self, batch, gamma: float) -> torch.Tensor:
        obs, actions, rewards, next_obs, dones = batch
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)

        q_values = self.agent_net(obs)
        chosen_qs = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        q_total = self.mixer(chosen_qs, obs)

        with torch.no_grad():
            next_q_values = self.target_agent_net(next_obs)
            next_actions = torch.argmax(next_q_values, dim=2, keepdim=True)
            next_chosen_qs = next_q_values.gather(2, next_actions).squeeze(-1)
            next_q_total = self.target_mixer(next_chosen_qs, next_obs)
            target = rewards + gamma * (1.0 - dones) * next_q_total

        return nn.functional.mse_loss(q_total, target)
