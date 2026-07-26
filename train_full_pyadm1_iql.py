from __future__ import annotations

import argparse
import json
import os
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from ad_rl.config import RUN_DIR
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv
from ad_rl.iql_agent import DiscreteIQLAgent, OfflineTransitionDataset
from ad_rl.rl_utils import (
    deterministic_rollout_policy,
    json_ready_config,
    plot_rollout_basic,
    save_csv,
)
from train_full_pyadm1_temp_penalty_consumption import config_for_run


def collect_offline_dataset(config, output_dir, dataset_episodes: int, episode_days: float, capacity: int):
    env = FullPyADM1PISetpointEnv(config=config, episode_days=episode_days)
    dataset = OfflineTransitionDataset(capacity=capacity)
    rows: list[dict] = []
    for ep in range(int(dataset_episodes)):
        obs = env.reset(seed=config.random_seed + 50_000 + ep)
        done = False
        step = 0
        while not done:
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            dataset.push(obs, action, reward, next_obs, done)
            row = {
                "dataset_episode": ep,
                "dataset_step": step,
                "action": action,
                "reward": reward,
                "done": done,
            }
            row.update(info)
            rows.append(row)
            obs = next_obs
            step += 1
    pd.DataFrame(rows).to_csv(output_dir / "offline_dataset_steps.csv", index=False)
    return dataset


def evaluate_policy(config, agent: DiscreteIQLAgent, episode_days: float) -> dict:
    env = FullPyADM1PISetpointEnv(config=config, episode_days=episode_days)
    obs = env.reset(seed=config.random_seed + 90_000)
    done = False
    total_reward = 0.0
    total_physical = 0.0
    total_produced = 0.0
    total_consumed = 0.0
    total_penalty = 0.0
    steps = 0
    while not done:
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)
        total_physical += float(info["physical_reward"])
        total_produced += float(info["methane_produced_m3"])
        total_consumed += float(info["methane_consumed_m3"])
        total_penalty += float(info["penalty"])
        steps += 1
    return {
        "eval_reward": total_reward,
        "eval_physical_reward": total_physical,
        "eval_methane_produced_m3": total_produced,
        "eval_methane_consumed_m3": total_consumed,
        "eval_net_methane_m3": total_produced - total_consumed,
        "eval_penalty": total_penalty,
        "eval_steps": steps,
        "eval_final_T_reactor_C": env.plant.state.T_reactor_C,
        "eval_final_T_setpoint_C": env.plant.state.T_setpoint_C,
    }


def train(
    updates: int,
    dataset_episodes: int,
    episode_days: float,
    run_name: str,
    production_weight: float,
    consumption_weight: float,
    temp_penalty_per_event: float,
    reward_scale: float,
    learning_rate: float,
    replay_capacity: int,
    t_setpoint_min_C: float | None,
    t_setpoint_max_C: float | None,
    eval_interval: int,
):
    config = config_for_run(
        episode_days=episode_days,
        production_weight=production_weight,
        consumption_weight=consumption_weight,
        temp_penalty_per_event=temp_penalty_per_event,
        reward_scale=reward_scale,
        learning_rate=learning_rate,
        target_update_steps=500,
        replay_capacity=replay_capacity,
        t_setpoint_min_C=t_setpoint_min_C,
        t_setpoint_max_C=t_setpoint_max_C,
    )
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        meta = json_ready_config(config)
        meta["algorithm"] = "discrete_iql_offline"
        meta["dataset_episodes"] = int(dataset_episodes)
        meta["updates"] = int(updates)
        json.dump(meta, f, indent=2, ensure_ascii=False)

    dataset = collect_offline_dataset(
        config=config,
        output_dir=output_dir,
        dataset_episodes=dataset_episodes,
        episode_days=episode_days,
        capacity=replay_capacity,
    )
    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    agent = DiscreteIQLAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )
    optimizers = {
        "q": torch.optim.Adam(agent.q_net.parameters(), lr=learning_rate),
        "value": torch.optim.Adam(agent.value_net.parameters(), lr=learning_rate),
        "policy": torch.optim.Adam(agent.policy_net.parameters(), lr=learning_rate),
    }

    training_rows: list[dict] = []
    eval_rows: list[dict] = []
    run_t0 = time.time()
    batch_size = min(config.batch_size, len(dataset))
    if batch_size <= 0:
        raise RuntimeError("Offline dataset is empty.")

    for update_idx in range(int(updates)):
        metrics = agent.update(dataset.sample(batch_size), optimizers, gamma=config.gamma)
        row = {"update": update_idx, "elapsed_run_time_s": time.time() - run_t0}
        row.update(metrics)
        training_rows.append(row)

        if update_idx % max(1, eval_interval) == 0 or update_idx == updates - 1:
            eval_row = {"update": update_idx}
            eval_row.update(evaluate_policy(config, agent, episode_days))
            eval_rows.append(eval_row)
            save_csv(eval_rows, output_dir / "episode_summary.csv")
            torch.save(agent.state_dict(), output_dir / "checkpoint_latest.pt")
            print(
                f"iql update {update_idx:05d}/{updates} "
                f"eval_reward={eval_row['eval_reward']:.3f} "
                f"net={eval_row['eval_net_methane_m3']:.3f}",
                flush=True,
            )

        if update_idx % 100 == 0:
            save_csv(training_rows, output_dir / "training_steps.csv")

    save_csv(training_rows, output_dir / "training_steps.csv")
    torch.save(agent.state_dict(), output_dir / "iql_agent.pt")
    _, internal_df = deterministic_rollout_policy(
        config,
        output_dir,
        select_action=lambda obs: agent.act(obs, deterministic=True),
        episode_days=config.season_episode_days,
    )
    plot_rollout_basic(output_dir, internal_df, "Discrete IQL deterministic rollout")
    return output_dir


def nullable_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--dataset-episodes", type=int, default=2)
    parser.add_argument("--episode-days", type=float, default=1.0)
    parser.add_argument("--run-name", default="iql_smoke")
    parser.add_argument("--production-weight", type=float, default=1.0)
    parser.add_argument("--consumption-weight", type=float, default=1.0)
    parser.add_argument("--temp-penalty-per-event", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--t-setpoint-min-C", type=nullable_float, default=25.0)
    parser.add_argument("--t-setpoint-max-C", type=nullable_float, default=65.0)
    parser.add_argument("--eval-interval", type=int, default=20)
    args = parser.parse_args()

    output_dir = train(
        updates=args.updates,
        dataset_episodes=args.dataset_episodes,
        episode_days=args.episode_days,
        run_name=args.run_name,
        production_weight=args.production_weight,
        consumption_weight=args.consumption_weight,
        temp_penalty_per_event=args.temp_penalty_per_event,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        t_setpoint_min_C=args.t_setpoint_min_C,
        t_setpoint_max_C=args.t_setpoint_max_C,
        eval_interval=args.eval_interval,
    )
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
