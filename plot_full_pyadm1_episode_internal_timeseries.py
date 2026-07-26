from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RLConfig, ensure_run_dir
from full_pyadm1_env import FullPyADM1PISetpointEnv


SUBSTRATE_STATES = [
    "ADM1_S_su",
    "ADM1_S_aa",
    "ADM1_S_fa",
    "ADM1_S_va",
    "ADM1_S_bu",
    "ADM1_S_pro",
    "ADM1_S_ac",
]

BIOMASS_STATES = [
    "ADM1_X_su",
    "ADM1_X_aa",
    "ADM1_X_fa",
    "ADM1_X_c4",
    "ADM1_X_pro",
    "ADM1_X_ac",
    "ADM1_X_h2",
]

GAS_STATES = [
    "ADM1_S_gas_h2",
    "ADM1_S_gas_ch4",
    "ADM1_S_gas_co2",
]


def scripted_action(episode: int, decision_step: int) -> int:
    # ADDED: deterministic changing action sequence for internal diagnostics.
    # Reason: this script is not training a policy; it creates episode plots
    # where action changes and full ADM1 internal trajectories can be read
    # together.
    # Role: cycle through hold/up/down decisions from the same Gym interface.
    # Reference: user-requested per-episode time-series view of action effects.
    sequence = [1, 2, 2, 1, 0, 0, 1, 2]
    return int(sequence[(decision_step + episode) % len(sequence)])


def run_episode_timeseries(
    episode_days: float = 2.0,
    episodes: int = 3,
    output_name: str = "full_pyadm1_episode_timeseries",
) -> Path:
    config = replace(
        RLConfig(),
        include_adm_state_observation=False,
        decision_interval_h=6.0,
        control_interval_h=1.0,
        simulation_dt_h=0.25,
        smoke_episode_days=float(episode_days),
    )
    output_dir = ensure_run_dir() / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    for episode in range(episodes):
        env = FullPyADM1PISetpointEnv(config=config, episode_days=config.smoke_episode_days)
        env.reset(seed=config.random_seed + episode)
        done = False
        decision_step = 0
        episode_rows: list[dict] = []
        decision_rows: list[dict] = []

        while not done:
            action = scripted_action(episode, decision_step)
            action_start_time_d = float(env.plant.state.time_d)
            action_start_elapsed_d = float(
                action_start_time_d - env.plant.state.episode_start_d
            )
            _obs, reward, terminated, truncated, info = env.step(action)
            interval_log = pd.DataFrame(env.plant.last_step_log)
            if not interval_log.empty:
                interval_log.insert(0, "episode", episode)
                interval_log.insert(1, "decision_step", decision_step)
                interval_log.insert(2, "action_index", action)
                interval_log.insert(3, "action_delta_C", info["action_delta_C"])
                interval_log["action_start_time_d"] = action_start_time_d
                interval_log["action_start_elapsed_d"] = action_start_elapsed_d
                interval_log["old_T_setpoint_C"] = info["old_T_setpoint_C"]
                interval_log["new_T_setpoint_C"] = info["new_T_setpoint_C"]
                interval_log["episode_elapsed_d"] = (
                    interval_log["time_d"] - env.plant.state.episode_start_d
                )
                episode_rows.extend(interval_log.to_dict("records"))

            decision_rows.append(
                {
                    "episode": episode,
                    "decision_step": decision_step,
                    "action_index": action,
                    "action_delta_C": info["action_delta_C"],
                    "reward": reward,
                    "action_start_time_d": action_start_time_d,
                    "action_start_elapsed_d": action_start_elapsed_d,
                    "time_d": info["time_d"],
                    "episode_elapsed_d": info["episode_elapsed_d"],
                    "old_T_setpoint_C": info["old_T_setpoint_C"],
                    "new_T_setpoint_C": info["new_T_setpoint_C"],
                    "T_reactor_C": info["T_reactor_C"],
                    "q_ch4_prod_m3_d": info["q_ch4_prod_m3_d"],
                    "q_ch4_heater_m3_d": info["q_ch4_heater_m3_d"],
                    "q_ch4_heater_thermal_m3_d": info["q_ch4_heater_thermal_m3_d"],
                }
            )
            done = bool(terminated or truncated)
            decision_step += 1

        episode_df = pd.DataFrame(episode_rows)
        decision_df = pd.DataFrame(decision_rows)
        episode_csv = output_dir / f"episode_{episode:03d}_internal_timeseries.csv"
        decision_csv = output_dir / f"episode_{episode:03d}_decision_log.csv"
        episode_df.to_csv(episode_csv, index=False)
        decision_df.to_csv(decision_csv, index=False)
        plot_path = output_dir / f"episode_{episode:03d}_action_internal_timeseries.png"
        _plot_episode(episode_df, decision_df, plot_path)

        summary_rows.append(
            {
                "episode": episode,
                "steps": int(len(decision_df)),
                "internal_rows": int(len(episode_df)),
                "start_T_reactor_C": float(episode_df["T_reactor_C"].iloc[0]),
                "end_T_reactor_C": float(episode_df["T_reactor_C"].iloc[-1]),
                "total_CH4_production_m3": float(
                    np.trapz(
                        episode_df["q_ch4_prod_m3_d"],
                        episode_df["episode_elapsed_d"],
                    )
                ),
                "episode_csv": str(episode_csv),
                "plot": str(plot_path),
            }
        )

    pd.DataFrame(summary_rows).to_csv(output_dir / "episode_timeseries_summary.csv", index=False)
    return output_dir


def _plot_episode(episode_df: pd.DataFrame, decision_df: pd.DataFrame, output_path: Path) -> None:
    if episode_df.empty:
        return

    t = episode_df["episode_elapsed_d"]
    fig, axes = plt.subplots(5, 1, figsize=(12.0, 11.5), sharex=True)
    end_elapsed_d = float(max(episode_df["episode_elapsed_d"].max(), decision_df["episode_elapsed_d"].max()))
    sp_x, sp_y = _step_values(decision_df, "new_T_setpoint_C", end_elapsed_d)
    action_x, action_y = _step_values(decision_df, "action_delta_C", end_elapsed_d)

    axes[0].plot(t, episode_df["T_reactor_C"], label="T_reactor", linewidth=1.5)
    axes[0].step(sp_x, sp_y, where="post", label="T_setpoint", linewidth=1.1)
    ax_action = axes[0].twinx()
    ax_action.step(
        action_x,
        action_y,
        where="post",
        color="0.25",
        alpha=0.65,
        label="action delta",
    )
    axes[0].set_ylabel("temperature (C)")
    ax_action.set_ylabel("action (C)")
    axes[0].grid(True, alpha=0.25)
    lines, labels = axes[0].get_legend_handles_labels()
    lines2, labels2 = ax_action.get_legend_handles_labels()
    axes[0].legend(lines + lines2, labels + labels2, frameon=False, ncol=3, loc="upper left")

    axes[1].plot(t, episode_df["q_ch4_prod_m3_d"], label="CH4 production", linewidth=1.4)
    axes[1].plot(
        t,
        episode_df["q_ch4_heater_m3_d"],
        label="heater methane use (x0.01 applied)",
        linewidth=1.2,
    )
    axes[1].set_ylabel("CH4 flow (m3/d)")
    axes[1].legend(frameon=False, ncol=2)
    axes[1].grid(True, alpha=0.25)

    _plot_normalized_group(axes[2], episode_df, SUBSTRATE_STATES, "substrates/acids")
    _plot_normalized_group(axes[3], episode_df, BIOMASS_STATES, "biomass states")
    _plot_normalized_group(axes[4], episode_df, GAS_STATES, "gas states")
    axes[4].set_xlabel("episode time (d)")

    fig.suptitle("Full PyADM1 episode trajectory: action and internal states")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _step_values(decision_df: pd.DataFrame, value_column: str, end_elapsed_d: float) -> tuple[list[float], list[float]]:
    time_column = "action_start_elapsed_d"
    if time_column not in decision_df:
        time_column = "episode_elapsed_d"
    x = decision_df[time_column].astype(float).tolist()
    y = decision_df[value_column].astype(float).tolist()
    if x:
        x.append(float(end_elapsed_d))
        y.append(float(y[-1]))
    return x, y


def _plot_normalized_group(ax, df: pd.DataFrame, columns: list[str], ylabel: str) -> None:
    for column in columns:
        if column not in df:
            continue
        values = df[column].to_numpy(dtype=float)
        baseline = values[0]
        if abs(baseline) > 1e-30:
            series = values / baseline
            label = column.replace("ADM1_", "")
        else:
            series = values
            label = f"{column.replace('ADM1_', '')} raw"
        ax.plot(df["episode_elapsed_d"], series, linewidth=1.0, label=label)
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=0.8)
    ax.set_ylabel(f"{ylabel}\ninitial=1")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=8)


if __name__ == "__main__":
    print(run_episode_timeseries())
