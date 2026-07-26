from __future__ import annotations

from collections import deque
from dataclasses import asdict
import json
import os
import random
from pathlib import Path
from typing import Type

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import gym
import torch
from torch import nn

from config import RLConfig, ensure_run_dir
from env import PyADM1PISetpointEnv


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done = zip(*batch)
        return (
            torch.tensor(np.array(obs), dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(reward, dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(done, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def epsilon_by_step(step: int, config: RLConfig) -> float:
    frac = min(1.0, step / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def train_dqn(
    episodes: int = 5,
    episode_days: float | None = None,
    config: RLConfig | None = None,
    output_dir: Path | None = None,
    env_class: Type[gym.Env] = PyADM1PISetpointEnv,
):
    # ADDED: minimal DQN training loop.
    # Reason: provide an executable agent for the new T_setpoint optimization
    # task, while keeping full seasonal training optional.
    # Role: train/evaluate DQN against the PyADM1PISetpointEnv interface and
    # save logs/model artifacts for inspection.
    # Reference: user-requested DQN agent.
    config = config or RLConfig()
    output_dir = output_dir or ensure_run_dir() / "dqn_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = env_class(config=config, episode_days=episode_days or config.smoke_episode_days)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim)
    target_net = QNetwork(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)

    rows = []
    global_step = 0

    for ep in range(episodes):
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        ep_reward = 0.0
        ep_net_methane = 0.0
        ep_penalty = 0.0
        ep_steps = 0

        while not done:
            eps = epsilon_by_step(global_step, config)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            replay.push(obs, action, reward, next_obs, done)

            if len(replay) >= max(config.batch_size, config.warmup_steps):
                b_obs, b_action, b_reward, b_next_obs, b_done = replay.sample(config.batch_size)
                q_value = q_net(b_obs).gather(1, b_action.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(b_next_obs).max(dim=1).values
                    target = b_reward + config.gamma * (1.0 - b_done) * next_q
                loss = nn.functional.mse_loss(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if global_step % config.target_update_steps == 0:
                target_net.load_state_dict(q_net.state_dict())

            rows.append(
                {
                    "episode": ep,
                    "step": ep_steps,
                    "global_step": global_step,
                    "epsilon": eps,
                    "action": action,
                    **info,
                }
            )
            ep_reward += reward
            ep_net_methane += info["net_methane_m3"]
            ep_penalty += info["penalty"]
            obs = next_obs
            ep_steps += 1
            global_step += 1

        rows.append(
            {
                "episode": ep,
                "step": "summary",
                "global_step": global_step,
                "epsilon": epsilon_by_step(global_step, config),
                "episode_reward": ep_reward,
                "episode_net_methane_m3": ep_net_methane,
                "episode_penalty": ep_penalty,
                "episode_steps": ep_steps,
            }
        )

    log = pd.DataFrame(rows)
    log.to_csv(output_dir / "dqn_training_log.csv", index=False)
    torch.save(q_net.state_dict(), output_dir / "dqn_q_network.pt")
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        serializable = asdict(config)
        serializable = {k: str(v) if isinstance(v, Path) else v for k, v in serializable.items()}
        serializable["action_deltas_C"] = list(map(float, config.action_deltas_C))
        json.dump(serializable, f, indent=2)
    return log, output_dir


if __name__ == "__main__":
    train_dqn()
