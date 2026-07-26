from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
from torch import nn

from ad_rl.config import RLConfig, RUN_DIR
from ad_rl.dqn_agent import QNetwork, ReplayBuffer, epsilon_by_step
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv
from scripts.train_full_pyadm1_season_200_fixed import (
    deterministic_rollout,
    plot_rollout,
    plot_training,
)


def config_for_run(
    episode_days: float,
    production_weight: float = 1.0,
    consumption_weight: float = 1.0,
    temp_penalty_per_event: float = 100.0,
    reward_scale: float = 100.0,
    learning_rate: float = 2e-4,
    target_update_steps: int = 500,
    replay_capacity: int = 100_000,
    t_setpoint_min_C: float | None = 25.0,
    t_setpoint_max_C: float | None = 65.0,
) -> RLConfig:
    base_config = RLConfig()
    baseline = pd.read_csv(base_config.baseline_thermal_inputs_path)
    heater_accounted = (
        baseline["q_ch4_heater"].mean()
        * base_config.heater_methane_accounting_factor
    )
    baseline_reward_m3_d = float(
        production_weight * baseline["q_ch4"].mean()
        - consumption_weight * heater_accounted
    )
    return replace(
        base_config,
        decision_interval_h=6.0,
        simulation_dt_h=0.25,
        season_episode_days=float(episode_days),
        smoke_episode_days=float(episode_days),
        include_adm_state_observation=False,
        # ADDED: NPV-style methane production/consumption reward.
        # Reason: methane production should remain the main benefit while
        # heater methane use is treated as an operating cost.
        # Role: reward = price*(w_prod*produced - w_cons*consumed) - penalty.
        # Reference: user-requested NPV reward with higher production weighting.
        reward_include_heater_consumption=True,
        reward_methane_production_weight=float(production_weight),
        reward_heater_consumption_weight=float(consumption_weight),
        reward_scale=float(reward_scale),
        reward_subtract_baseline=True,
        reward_baseline_methane_m3_d=baseline_reward_m3_d,
        normalize_observation=True,
        # ADDED: actual-reactor-temperature shock penalty.
        # Reason: microbial shock relates to reactor temperature movement, not
        # only to setpoint command movement.
        # Role: one event penalty is applied when |T(t)-T(t-24h)| exceeds 1 C.
        # Reference: user-requested 24 h T_reactor-change count penalty.
        penalty_mode="reactor_24h_event",
        reactor_temp_change_limit_C=1.0,
        reactor_temp_penalty_per_event=float(temp_penalty_per_event),
        t_setpoint_min_C=t_setpoint_min_C,
        t_setpoint_max_C=t_setpoint_max_C,
        replay_capacity=int(replay_capacity),
        warmup_steps=1_000,
        batch_size=64,
        learning_rate=float(learning_rate),
        target_update_steps=int(target_update_steps),
        epsilon_decay_steps=50_000,
    )


def json_ready_config(config: RLConfig) -> dict:
    serializable = asdict(config)
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in serializable.items()
    }
    serializable["action_deltas_C"] = list(map(float, config.action_deltas_C))
    return serializable


def save_csv(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def checkpoint_episode(path: Path) -> int:
    return int(path.stem.replace("checkpoint_ep", ""))


def deterministic_checkpoint_metrics(config: RLConfig, checkpoint_path: Path) -> dict:
    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = QNetwork(obs_dim, action_dim)
    q_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    q_net.eval()

    obs = env.reset(seed=config.random_seed + 999_999)
    done = False
    step_count = 0
    learning_reward_total = 0.0
    physical_reward_total = 0.0
    methane_produced = 0.0
    methane_consumed = 0.0
    penalty_total = 0.0
    temp_events = 0.0
    actions: list[int] = []
    T_reactor: list[float] = []
    T_setpoint: list[float] = []

    while not done:
        with torch.no_grad():
            q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step_count += 1
        actions.append(action)
        learning_reward_total += float(reward)
        physical_reward_total += float(info["physical_reward"])
        methane_produced += float(info["methane_produced_m3"])
        methane_consumed += float(info["methane_consumed_m3"])
        penalty_total += float(info["penalty"])
        temp_events += float(info["reactor_temp_penalty_event"])
        T_reactor.append(float(info["T_reactor_C"]))
        T_setpoint.append(float(info["T_setpoint_C"]))

    action_counts = {f"action_{idx}_count": actions.count(idx) for idx in range(action_dim)}
    action_fracs = {
        f"action_{idx}_fraction": actions.count(idx) / max(1, len(actions))
        for idx in range(action_dim)
    }
    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_episode": checkpoint_episode(checkpoint_path),
        "steps": step_count,
        "learning_reward_total": learning_reward_total,
        "physical_reward_total": physical_reward_total,
        "methane_produced_m3": methane_produced,
        "methane_consumed_m3": methane_consumed,
        "net_methane_m3": methane_produced - methane_consumed,
        "penalty_total": penalty_total,
        "reactor_temp_penalty_events": temp_events,
        "final_T_reactor_C": float(env.plant.state.T_reactor_C),
        "final_T_setpoint_C": float(env.plant.state.T_setpoint_C),
        "mean_T_reactor_C": float(np.mean(T_reactor)) if T_reactor else np.nan,
        "mean_T_setpoint_C": float(np.mean(T_setpoint)) if T_setpoint else np.nan,
        "min_T_setpoint_C": float(np.min(T_setpoint)) if T_setpoint else np.nan,
        "max_T_setpoint_C": float(np.max(T_setpoint)) if T_setpoint else np.nan,
        **action_counts,
        **action_fracs,
    }


def evaluate_checkpoints(config: RLConfig, output_dir: Path) -> pd.DataFrame:
    # ADDED: deterministic best-checkpoint evaluation for long DQN runs.
    # Reason: the final DQN checkpoint can drift even when an earlier policy is
    # better under the requested net-methane reward.
    # Role: evaluate every saved checkpoint_ep*.pt and preserve the best one.
    # Reference: user-requested best checkpoint evaluation for the 1000 ep run.
    checkpoint_paths = sorted(
        output_dir.glob("checkpoint_ep*.pt"),
        key=checkpoint_episode,
    )
    rows: list[dict] = []
    eval_path = output_dir / "checkpoint_evaluation.csv"
    for checkpoint_path in checkpoint_paths:
        row = deterministic_checkpoint_metrics(config, checkpoint_path)
        rows.append(row)
        pd.DataFrame(rows).to_csv(eval_path, index=False)
        print(
            f"eval {row['checkpoint']} "
            f"learn_reward={row['learning_reward_total']:.3f} "
            f"net={row['net_methane_m3']:.3f}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    best = summary.loc[summary["learning_reward_total"].idxmax()]
    best_path = output_dir / str(best["checkpoint"])
    torch.save(torch.load(best_path, map_location="cpu"), output_dir / "checkpoint_best.pt")
    with open(output_dir / "checkpoint_best.json", "w", encoding="utf-8") as f:
        json.dump(best.to_dict(), f, indent=2, ensure_ascii=False)

    best_dir = output_dir / "best_checkpoint_rollout"
    best_dir.mkdir(parents=True, exist_ok=True)
    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    best_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    best_net.load_state_dict(torch.load(best_path, map_location="cpu"))
    _, internal_df = deterministic_rollout(config, best_dir, best_net)
    plot_rollout(best_dir, internal_df)
    print(
        f"best checkpoint: {best['checkpoint']} "
        f"learn_reward={best['learning_reward_total']:.3f} "
        f"net={best['net_methane_m3']:.3f}",
        flush=True,
    )
    return summary


def train(
    episodes: int,
    episode_days: float,
    run_name: str,
    production_weight: float = 1.0,
    consumption_weight: float = 1.0,
    temp_penalty_per_event: float = 100.0,
    reward_scale: float = 100.0,
    learning_rate: float = 2e-4,
    target_update_steps: int = 500,
    replay_capacity: int = 100_000,
    t_setpoint_min_C: float | None = 25.0,
    t_setpoint_max_C: float | None = 65.0,
    resume: bool = False,
) -> tuple[RLConfig, Path, QNetwork]:
    config = config_for_run(
        episode_days,
        production_weight=production_weight,
        consumption_weight=consumption_weight,
        temp_penalty_per_event=temp_penalty_per_event,
        reward_scale=reward_scale,
        learning_rate=learning_rate,
        target_update_steps=target_update_steps,
        replay_capacity=replay_capacity,
        t_setpoint_min_C=t_setpoint_min_C,
        t_setpoint_max_C=t_setpoint_max_C,
    )
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(obs_dim, action_dim)
    target_net = QNetwork(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)

    training_rows: list[dict] = []
    summary_rows: list[dict] = []
    global_step = 0
    start_episode = 0
    run_t0 = time.time()

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(json_ready_config(config), f, indent=2, ensure_ascii=False)

    training_path = output_dir / "training_steps.csv"
    summary_path = output_dir / "episode_summary.csv"
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    if resume:
        # ADDED: resume existing long RL runs without overwriting previous CSVs.
        # Reason: 90 d Full PyADM1 episodes are slow, so interrupted training
        # should continue from the latest saved episode/checkpoint.
        # Role: restore learned Q-network and append new episode/step rows.
        # Reference: user request to resume the previously stopped simulation.
        if summary_path.exists():
            summary_rows = pd.read_csv(summary_path).to_dict("records")
            if summary_rows:
                start_episode = int(max(row["episode"] for row in summary_rows)) + 1
        if training_path.exists():
            training_rows = pd.read_csv(training_path).to_dict("records")
            if training_rows:
                global_step = int(max(row["global_step"] for row in training_rows)) + 1
        if checkpoint_path.exists():
            q_net.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            target_net.load_state_dict(q_net.state_dict())
        print(
            f"resume enabled: start_episode={start_episode}, global_step={global_step}, "
            f"loaded_checkpoint={checkpoint_path.exists()}",
            flush=True,
        )

    for ep in range(start_episode, int(episodes)):
        ep_t0 = time.time()
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        ep_reward = 0.0
        ep_physical_reward = 0.0
        ep_reward_baseline = 0.0
        ep_produced = 0.0
        ep_consumed = 0.0
        ep_penalty = 0.0
        ep_temp_penalty_events = 0.0
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

            loss_value = np.nan
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
                loss_value = float(loss.item())

            if global_step % config.target_update_steps == 0:
                target_net.load_state_dict(q_net.state_dict())

            row = {
                "episode": ep,
                "step": ep_steps,
                "global_step": global_step,
                "epsilon": eps,
                "action": action,
                "loss": loss_value,
            }
            row.update(info)
            training_rows.append(row)

            ep_reward += float(reward)
            ep_physical_reward += float(info["physical_reward"])
            ep_reward_baseline += float(info["reward_baseline"])
            ep_produced += float(info["methane_produced_m3"])
            ep_consumed += float(info["methane_consumed_m3"])
            ep_penalty += float(info["penalty"])
            ep_temp_penalty_events += float(info["reactor_temp_penalty_event"])
            obs = next_obs
            ep_steps += 1
            global_step += 1

        ep_time_s = time.time() - ep_t0
        summary_rows.append(
            {
                "episode": ep,
                "steps": ep_steps,
                "global_step_end": global_step,
                "episode_reward": ep_reward,
                "episode_physical_reward": ep_physical_reward,
                "episode_reward_baseline": ep_reward_baseline,
                "episode_methane_produced_m3": ep_produced,
                "episode_methane_consumed_m3": ep_consumed,
                "episode_net_methane_m3": ep_produced - ep_consumed,
                "episode_penalty": ep_penalty,
                "episode_reactor_temp_penalty_events": ep_temp_penalty_events,
                "final_T_reactor_C": env.plant.state.T_reactor_C,
                "final_T_setpoint_C": env.plant.state.T_setpoint_C,
                "episode_time_s": ep_time_s,
                "elapsed_run_time_s": time.time() - run_t0,
            }
        )

        save_csv(training_rows, output_dir / "training_steps.csv")
        save_csv(summary_rows, output_dir / "episode_summary.csv")
        torch.save(q_net.state_dict(), output_dir / "checkpoint_latest.pt")
        if (ep + 1) % 10 == 0:
            torch.save(q_net.state_dict(), output_dir / f"checkpoint_ep{ep + 1:03d}.pt")

        print(
            f"episode {ep + 1:03d}/{episodes} "
            f"learn_reward={ep_reward:.3f} physical_reward={ep_physical_reward:.3f} "
            f"produced={ep_produced:.3f} consumed={ep_consumed:.3f} "
            f"penalty={ep_penalty:.3f} temp_events={ep_temp_penalty_events:.0f} "
            f"time={ep_time_s:.1f}s",
            flush=True,
        )

    torch.save(q_net.state_dict(), output_dir / "dqn_q_network.pt")
    return config, output_dir, q_net


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-days", type=float, default=90.0)
    parser.add_argument("--production-weight", type=float, default=1.0)
    parser.add_argument("--run-name", default="온도패널티_소비반영")
    parser.add_argument("--consumption-weight", type=float, default=1.0)
    parser.add_argument("--temp-penalty-per-event", type=float, default=100.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--t-setpoint-min-C", type=float, default=25.0)
    parser.add_argument("--t-setpoint-max-C", type=float, default=65.0)
    parser.add_argument("--eval-checkpoints", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config, output_dir, q_net = train(
        episodes=args.episodes,
        episode_days=args.episode_days,
        run_name=args.run_name,
        production_weight=args.production_weight,
        consumption_weight=args.consumption_weight,
        temp_penalty_per_event=args.temp_penalty_per_event,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        target_update_steps=args.target_update_steps,
        replay_capacity=args.replay_capacity,
        t_setpoint_min_C=args.t_setpoint_min_C,
        t_setpoint_max_C=args.t_setpoint_max_C,
        resume=args.resume,
    )
    _, internal_df = deterministic_rollout(config, output_dir, q_net)
    plot_training(output_dir)
    plot_rollout(output_dir, internal_df)
    if args.eval_checkpoints:
        evaluate_checkpoints(config, output_dir)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

