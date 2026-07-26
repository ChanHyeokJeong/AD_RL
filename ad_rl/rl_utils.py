from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ad_rl.config import RLConfig
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv


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


def deterministic_rollout_policy(
    config: RLConfig,
    output_dir: Path,
    select_action,
    episode_days: float | None = None,
    prefix: str = "deterministic_policy",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    env = FullPyADM1PISetpointEnv(
        config=config,
        episode_days=config.season_episode_days if episode_days is None else float(episode_days),
    )
    obs = env.reset(seed=config.random_seed + 10_000)
    done = False
    decision_rows: list[dict] = []
    internal_rows: list[dict] = []

    while not done:
        action = int(select_action(obs))
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
    decision_df.to_csv(output_dir / f"{prefix}_decision_steps.csv", index=False)
    internal_df.to_csv(output_dir / f"{prefix}_internal_timeseries.csv", index=False)
    return decision_df, internal_df


def plot_training_reward(output_dir: Path, title: str) -> None:
    summary_path = output_dir / "episode_summary.csv"
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    if "episode_reward" not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(summary["episode"], summary["episode_reward"], lw=1.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "training_reward.png", dpi=200)
    plt.close(fig)


def plot_rollout_basic(output_dir: Path, internal_df: pd.DataFrame, title: str) -> None:
    if internal_df.empty:
        return
    x = internal_df["episode_elapsed_d"]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(x, internal_df["T_reactor_C"], label="T_reactor")
    axes[0].plot(x, internal_df["T_setpoint_C"], label="T_setpoint", alpha=0.8)
    if "T_in_C" in internal_df.columns:
        axes[0].plot(x, internal_df["T_in_C"], label="T_in", alpha=0.6)
    axes[0].set_ylabel("temperature (C)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    if "q_ch4_prod_m3_d" in internal_df.columns:
        axes[1].plot(x, internal_df["q_ch4_prod_m3_d"], label="CH4 production")
    if "q_ch4_heater_m3_d" in internal_df.columns:
        axes[1].plot(x, internal_df["q_ch4_heater_m3_d"], label="heater CH4 thermal")
    axes[1].set_ylabel("flow (m3/d)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    if "action_delta_C" in internal_df.columns:
        axes[2].step(x, internal_df["action_delta_C"], where="post", label="action delta")
    axes[2].set_xlabel("episode time (d)")
    axes[2].set_ylabel("action (C)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / "deterministic_rollout.png", dpi=200)
    plt.close(fig)
