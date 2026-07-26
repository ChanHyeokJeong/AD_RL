from __future__ import annotations

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from config import RLConfig, RUN_DIR
from dqn_agent import QNetwork, ReplayBuffer, epsilon_by_step
from full_pyadm1_env import FullPyADM1PISetpointEnv


EPISODES = 200
EPISODE_DAYS = 90.0
DECISION_INTERVAL_H = 6.0
SIMULATION_DT_H = 0.25
RUN_NAME = "수정학습_90일200회"


def _config_for_run() -> RLConfig:
    base_config = RLConfig()
    baseline_q_ch4_m3_d = float(
        pd.read_csv(base_config.baseline_thermal_inputs_path)["q_ch4"].mean()
    )
    return replace(
        base_config,
        decision_interval_h=DECISION_INTERVAL_H,
        simulation_dt_h=SIMULATION_DT_H,
        season_episode_days=EPISODE_DAYS,
        smoke_episode_days=EPISODE_DAYS,
        include_adm_state_observation=False,
        reward_include_heater_consumption=False,
        reward_scale=100.0,
        reward_subtract_baseline=True,
        reward_baseline_methane_m3_d=baseline_q_ch4_m3_d,
        normalize_observation=True,
        replay_capacity=50_000,
        warmup_steps=1_000,
        batch_size=64,
        learning_rate=5e-4,
        target_update_steps=200,
        epsilon_decay_steps=50_000,
    )


def _json_ready_config(config: RLConfig) -> dict:
    serializable = asdict(config)
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in serializable.items()
    }
    serializable["action_deltas_C"] = list(map(float, config.action_deltas_C))
    return serializable


def _save_csv(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def train_full_pyadm1() -> tuple[RLConfig, Path, QNetwork]:
    config = _config_for_run()
    output_dir = RUN_DIR / RUN_NAME
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
    run_t0 = time.time()

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready_config(config), f, indent=2, ensure_ascii=False)

    for ep in range(EPISODES):
        ep_t0 = time.time()
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        ep_reward = 0.0
        ep_physical_reward = 0.0
        ep_reward_baseline = 0.0
        ep_produced = 0.0
        ep_consumed = 0.0
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
            obs = next_obs
            ep_steps += 1
            global_step += 1

        ep_time_s = time.time() - ep_t0
        summary = {
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
            "final_T_reactor_C": env.plant.state.T_reactor_C,
            "final_T_setpoint_C": env.plant.state.T_setpoint_C,
            "episode_time_s": ep_time_s,
            "elapsed_run_time_s": time.time() - run_t0,
        }
        summary_rows.append(summary)

        _save_csv(training_rows, output_dir / "training_steps.csv")
        _save_csv(summary_rows, output_dir / "episode_summary.csv")
        torch.save(q_net.state_dict(), output_dir / "checkpoint_latest.pt")
        if (ep + 1) % 10 == 0:
            torch.save(q_net.state_dict(), output_dir / f"checkpoint_ep{ep + 1:03d}.pt")

        print(
            f"episode {ep + 1:03d}/{EPISODES} "
            f"learn_reward={ep_reward:.3f} physical_reward={ep_physical_reward:.3f} "
            f"produced={ep_produced:.3f} "
            f"penalty={ep_penalty:.3f} time={ep_time_s:.1f}s",
            flush=True,
        )

    torch.save(q_net.state_dict(), output_dir / "dqn_q_network.pt")
    return config, output_dir, q_net


def deterministic_rollout(config: RLConfig, output_dir: Path, q_net: QNetwork) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    obs = env.reset(seed=config.random_seed + 10_000)
    done = False
    decision_rows: list[dict] = []
    internal_rows: list[dict] = []

    while not done:
        with torch.no_grad():
            q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())

        start_elapsed_d = env.plant.state.time_d - env.plant.state.episode_start_d
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        decision_row = {
            "action_start_elapsed_d": start_elapsed_d,
            "action": action,
            "reward": reward,
        }
        decision_row.update(info)
        decision_rows.append(decision_row)

        for row in env.plant.last_step_log:
            out = dict(row)
            out["episode_elapsed_d"] = row["time_d"] - env.plant.state.episode_start_d
            out["action"] = action
            out["action_delta_C"] = info["action_delta_C"]
            internal_rows.append(out)

        obs = next_obs

    decision_df = pd.DataFrame(decision_rows)
    internal_df = pd.DataFrame(internal_rows)
    decision_df.to_csv(output_dir / "deterministic_policy_decision_steps.csv", index=False)
    internal_df.to_csv(output_dir / "deterministic_policy_internal_timeseries.csv", index=False)
    return decision_df, internal_df


def plot_training(output_dir: Path) -> None:
    summary_path = output_dir / "episode_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary["episode"], summary["episode_reward"], lw=1.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title("DQN training reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_reward.png", dpi=200)
    plt.close(fig)


def plot_rollout(output_dir: Path, internal_df: pd.DataFrame) -> None:
    x = internal_df["episode_elapsed_d"]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x, internal_df["T_reactor_C"], label="T_reactor", color="tab:red", lw=1.3)
    ax1.plot(x, internal_df["T_setpoint_C"], label="T_setpoint", color="tab:orange", lw=1.0)
    ax1.set_xlabel("episode time (d)")
    ax1.set_ylabel("temperature (C)")
    ax2 = ax1.twinx()
    ax2.plot(x, internal_df["q_ch4_prod_m3_d"], label="CH4 production", color="tab:blue", lw=1.0)
    ax2.set_ylabel("CH4 production (m3/d)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_sp_Treactor_CH4production.png", dpi=200)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(x, internal_df["T_setpoint_C"], label="T_setpoint", color="tab:orange", lw=1.2)
    ax1.set_xlabel("episode time (d)")
    ax1.set_ylabel("T_setpoint (C)")
    ax2 = ax1.twinx()
    ax2.plot(x, internal_df["q_ch4_heater_m3_d"], label="heater methane use", color="tab:green", lw=1.0)
    ax2.set_ylabel("heater methane use (m3/d)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig2_sp_heater_methane_use.png", dpi=200)
    plt.close(fig)

    soluble_names = [
        "S_su",
        "S_aa",
        "S_fa",
        "S_va",
        "S_bu",
        "S_pro",
        "S_ac",
        "S_h2",
        "S_ch4",
        "S_IC",
        "S_IN",
        "S_I",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in soluble_names:
        col = f"ADM1_{name}"
        if col in internal_df:
            ax.plot(x, internal_df[col], label=name, lw=1.0)
    ax.set_xlabel("episode time (d)")
    ax.set_ylabel("reactor soluble state")
    ax.set_title("Reactor soluble ADM1 states")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig3_reactor_S_components.png", dpi=200)
    plt.close(fig)

    biomass_names = [
        "X_xc",
        "X_ch",
        "X_pr",
        "X_li",
        "X_su",
        "X_aa",
        "X_fa",
        "X_c4",
        "X_pro",
        "X_ac",
        "X_h2",
        "X_I",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in biomass_names:
        col = f"ADM1_{name}"
        if col in internal_df:
            ax.plot(x, internal_df[col], label=name, lw=1.0)
    ax.set_xlabel("episode time (d)")
    ax.set_ylabel("reactor biomass/particulate state")
    ax.set_title("Reactor biomass ADM1 states")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig4_reactor_biomass_components.png", dpi=200)
    plt.close(fig)


def main() -> None:
    config, output_dir, q_net = train_full_pyadm1()
    _, internal_df = deterministic_rollout(config, output_dir, q_net)
    plot_training(output_dir)
    plot_rollout(output_dir, internal_df)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
