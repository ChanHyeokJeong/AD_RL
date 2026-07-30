from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import replace
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

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2] if (THIS_DIR.parents[2] / "ad_rl").exists() else None
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ad_rl.dqn_agent import QNetwork, ReplayBuffer
from ph2stage_rl_env import (
    PHControlRLConfig,
    TwoStagePHDirectDosingEnv,
    config_to_json_dict,
    rollout_fixed_pH_policy,
    rollout_policy,
)


def epsilon_by_step(step: int, config: PHControlRLConfig) -> float:
    frac = min(1.0, step / max(1, config.epsilon_decay_steps))
    return float(config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start))


def output_base_dir() -> Path:
    return THIS_DIR.parents[0] / "runs"


def json_dump(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows: list[dict], path: Path) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def deterministic_action_from_network(q_net: QNetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(q_values, dim=1).item())


def train_dqn(
    config: PHControlRLConfig,
    output_dir: Path,
    episodes: int,
) -> tuple[QNetwork, pd.DataFrame, pd.DataFrame]:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = TwoStagePHDirectDosingEnv(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    q_net = QNetwork(obs_dim, action_dim)
    target_net = QNetwork(obs_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=config.learning_rate)
    replay = ReplayBuffer(config.replay_capacity)

    output_dir.mkdir(parents=True, exist_ok=True)
    env.action_table.to_csv(output_dir / "action_table.csv", index=False)
    json_dump(output_dir / "config.json", config_to_json_dict(config))

    training_rows: list[dict] = []
    summary_rows: list[dict] = []
    global_step = 0
    run_t0 = time.time()

    for ep in range(int(episodes)):
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        ep_rows_start = len(training_rows)
        ep_t0 = time.time()
        ep_step = 0

        while not done:
            eps = epsilon_by_step(global_step, config)
            if random.random() < eps:
                action = int(env.action_space.sample())
            else:
                action = deterministic_action_from_network(q_net, obs)

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

            if global_step % max(1, config.target_update_steps) == 0:
                target_net.load_state_dict(q_net.state_dict())

            row = {
                "episode": ep,
                "episode_step": ep_step,
                "global_step": global_step,
                "epsilon": eps,
                "loss": loss_value,
                "action": int(action),
            }
            row.update(info)
            training_rows.append(row)

            obs = next_obs
            global_step += 1
            ep_step += 1

        ep_df = pd.DataFrame(training_rows[ep_rows_start:])
        summary = {
            "episode": ep,
            "steps": int(len(ep_df)),
            "global_step_end": int(global_step),
            "episode_time_s": time.time() - ep_t0,
            "elapsed_run_time_s": time.time() - run_t0,
        }
        for col in [
            "reward",
            "reward_raw",
            "reward_benefit",
            "reward_chemical_cost",
            "reward_ph_cost",
            "stage1_vfa_produced_kgCOD",
            "stage2_vfa_removed_kgCOD",
            "stage2_ch4_m3",
            "chemical_kmol",
            "ph_violation_pH_d",
        ]:
            if col in ep_df.columns:
                summary[f"total_{col}"] = float(ep_df[col].sum())
        if not ep_df.empty:
            summary["final_stage1_pH"] = float(ep_df["stage1_pH"].iloc[-1])
            summary["final_stage2_pH"] = float(ep_df["stage2_pH"].iloc[-1])
            summary["mean_stage1_pH"] = float(ep_df["stage1_pH"].mean())
            summary["mean_stage2_pH"] = float(ep_df["stage2_pH"].mean())
        summary_rows.append(summary)

        save_csv(training_rows, output_dir / "training_steps.csv")
        save_csv(summary_rows, output_dir / "episode_summary.csv")
        torch.save(q_net.state_dict(), output_dir / "checkpoint_latest.pt")
        print(
            f"episode {ep + 1:03d}/{episodes} "
            f"reward={summary.get('total_reward', np.nan):.3f} "
            f"raw={summary.get('total_reward_raw', np.nan):.3f} "
            f"chem={summary.get('total_chemical_kmol', np.nan):.3f} "
            f"time={summary['episode_time_s']:.1f}s",
            flush=True,
        )

    torch.save(q_net.state_dict(), output_dir / "dqn_q_network.pt")
    return q_net, pd.DataFrame(training_rows), pd.DataFrame(summary_rows)


def evaluate_dqn_policy(config: PHControlRLConfig, q_net: QNetwork, output_dir: Path) -> dict:
    decision_df, internal_df, summary = rollout_policy(
        config,
        lambda obs, _env: deterministic_action_from_network(q_net, obs),
    )
    decision_df.to_csv(output_dir / "dqn_policy_decision_steps.csv", index=False)
    internal_df.to_csv(output_dir / "dqn_policy_internal_timeseries.csv", index=False)
    summary = dict(policy="dqn_deterministic", **summary)
    json_dump(output_dir / "dqn_policy_summary.json", summary)
    plot_rollout(output_dir, internal_df, "DQN deterministic policy", "dqn_policy_rollout.png")
    return summary


def evaluate_fixed_open_loop_baselines(
    config: PHControlRLConfig,
    output_dir: Path,
    max_actions: int | None = None,
) -> pd.DataFrame:
    env = TwoStagePHDirectDosingEnv(config)
    action_table = env.action_table
    if max_actions is not None:
        action_table = action_table.iloc[: int(max_actions)].copy()

    rows: list[dict] = []
    for _, action_row in action_table.iterrows():
        action = int(action_row["action"])
        decision_df, internal_df, summary = rollout_policy(config, lambda _obs, _env, a=action: a)
        row = {"policy": "fixed_open_loop_action", "fixed_action": action}
        row.update(action_row.to_dict())
        row.update(summary)
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_dir / "baseline_fixed_action_summary.csv", index=False)

        if len(rows) == 1:
            decision_df.to_csv(output_dir / "baseline_first_action_decision_steps.csv", index=False)
            internal_df.to_csv(output_dir / "baseline_first_action_internal_timeseries.csv", index=False)

        print(
            f"baseline action {action:02d} "
            f"reward={summary.get('total_reward', np.nan):.3f} "
            f"chem={summary.get('total_chemical_kmol', np.nan):.3f}",
            flush=True,
        )

    baseline_df = pd.DataFrame(rows)
    if not baseline_df.empty:
        best = baseline_df.loc[baseline_df["total_reward"].idxmax()]
        best_action = int(best["fixed_action"])
        decision_df, internal_df, summary = rollout_policy(config, lambda _obs, _env, a=best_action: a)
        decision_df.to_csv(output_dir / "baseline_best_action_decision_steps.csv", index=False)
        internal_df.to_csv(output_dir / "baseline_best_action_internal_timeseries.csv", index=False)
        json_dump(output_dir / "baseline_best_action_summary.json", dict(best.to_dict()))
        plot_rollout(
            output_dir,
            internal_df,
            f"Best fixed open-loop action {best_action}",
            "baseline_best_action_rollout.png",
        )
    return baseline_df


def evaluate_fixed_pH7_baseline(config: PHControlRLConfig, output_dir: Path) -> dict:
    decision_df, internal_df, summary = rollout_fixed_pH_policy(
        config,
        stage1_pH_sp=7.0,
        stage2_pH_sp=7.0,
    )
    decision_df.to_csv(output_dir / "baseline_fixed_pH7_PI_decision_steps.csv", index=False)
    internal_df.to_csv(output_dir / "baseline_fixed_pH7_PI_internal_timeseries.csv", index=False)
    summary = dict(policy="fixed_pH7_PI_both_stages", **summary)
    json_dump(output_dir / "baseline_fixed_pH7_PI_summary.json", summary)
    plot_rollout(
        output_dir,
        internal_df,
        "Fixed pH 7 PI baseline, both stages",
        "baseline_fixed_pH7_PI_rollout.png",
    )
    print(
        "fixed pH7 PI "
        f"reward={summary.get('total_reward', np.nan):.3f} "
        f"chem={summary.get('total_chemical_kmol', np.nan):.3f}",
        flush=True,
    )
    return summary


def build_comparison(
    output_dir: Path,
    dqn_summary: dict,
    baseline_df: pd.DataFrame,
    fixed_pH7_summary: dict | None = None,
) -> pd.DataFrame:
    rows = [dqn_summary]
    if not baseline_df.empty:
        best = baseline_df.loc[baseline_df["total_reward"].idxmax()].to_dict()
        best["policy"] = "best_fixed_open_loop_action"
        rows.append(best)
        hold_rows = baseline_df[
            (baseline_df["stage1_signed_m3_d"] == 0.0)
            & (baseline_df["stage2_signed_m3_d"] == 0.0)
        ]
        if not hold_rows.empty:
            hold = hold_rows.iloc[0].to_dict()
            hold["policy"] = "zero_dosing_open_loop"
            rows.append(hold)
    if fixed_pH7_summary is not None:
        rows.append(fixed_pH7_summary)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    return comparison


def plot_rollout(output_dir: Path, internal_df: pd.DataFrame, title: str, filename: str) -> None:
    if internal_df.empty:
        return
    x = internal_df["time_d"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].plot(x, internal_df["stage1_pH"], label="Stage 1 pH")
    axes[0].plot(x, internal_df["stage2_pH"], label="Stage 2 pH")
    axes[0].axhspan(4.8, 6.4, color="tab:blue", alpha=0.08)
    axes[0].axhspan(6.7, 7.8, color="tab:green", alpha=0.08)
    axes[0].set_ylabel("pH")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, internal_df["stage1_vfa_kgCOD_m3"], label="Stage 1 VFA")
    axes[1].plot(x, internal_df["stage2_vfa_kgCOD_m3"], label="Stage 2 VFA")
    axes[1].set_ylabel("VFA (kgCOD/m3)")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(x, internal_df["stage1_q_ch4_m3_d"], label="Stage 1 CH4")
    axes[2].plot(x, internal_df["stage2_q_ch4_m3_d"], label="Stage 2 CH4")
    axes[2].set_ylabel("CH4 (m3/d)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    axes[3].step(x, internal_df["stage1_signed_m3_d"], where="post", label="Stage 1 signed dose")
    axes[3].step(x, internal_df["stage2_signed_m3_d"], where="post", label="Stage 2 signed dose")
    axes[3].set_xlabel("time (d)")
    axes[3].set_ylabel("signed m3/d")
    axes[3].legend(loc="best")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-days", type=float, default=2.0)
    parser.add_argument("--run-name", default="ph2stage_dqn_4actuator_smoke")
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--reward-mode",
        choices=["methane_total", "stage2_methane", "staged_vfa_ch4"],
        default="methane_total",
    )
    parser.add_argument("--methane-reward-weight", type=float, default=1.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--chemical-kmol-weight", type=float, default=0.2)
    parser.add_argument("--ph-violation-weight", type=float, default=500.0)
    parser.add_argument("--influent-csv", default="digester_influent_mean_full.csv")
    parser.add_argument("--use-dynamic-flow", action="store_true")
    parser.add_argument(
        "--episode-start-mode",
        choices=["fixed", "random"],
        default="fixed",
    )
    parser.add_argument("--episode-start-day", type=float, default=0.0)
    parser.add_argument("--episode-start-day-min", type=float, default=0.0)
    parser.add_argument("--episode-start-day-max", type=float, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ph7-baseline", action="store_true")
    parser.add_argument("--baseline-max-actions", type=int, default=None)
    args = parser.parse_args()

    config = replace(
        PHControlRLConfig(),
        episode_days=float(args.episode_days),
        learning_rate=float(args.learning_rate),
        reward_mode=str(args.reward_mode),
        methane_reward_weight=float(args.methane_reward_weight),
        reward_scale=float(args.reward_scale),
        chemical_kmol_weight=float(args.chemical_kmol_weight),
        ph_violation_weight=float(args.ph_violation_weight),
        influent_csv=str(args.influent_csv),
        use_dynamic_flow=bool(args.use_dynamic_flow),
        episode_start_mode=str(args.episode_start_mode),
        episode_start_day=float(args.episode_start_day),
        episode_start_day_min=float(args.episode_start_day_min),
        episode_start_day_max=args.episode_start_day_max,
    )
    output_dir = output_base_dir() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    q_net, _training, _summary = train_dqn(config, output_dir, episodes=args.episodes)
    dqn_summary = evaluate_dqn_policy(config, q_net, output_dir)

    if args.skip_baselines:
        comparison = pd.DataFrame([dqn_summary])
        comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    else:
        baseline_df = evaluate_fixed_open_loop_baselines(
            config,
            output_dir,
            max_actions=args.baseline_max_actions,
        )
        fixed_pH7_summary = None
        if not args.skip_ph7_baseline:
            fixed_pH7_summary = evaluate_fixed_pH7_baseline(config, output_dir)
        comparison = build_comparison(output_dir, dqn_summary, baseline_df, fixed_pH7_summary)
        print(
            comparison[
                ["policy", "total_reward", "total_reward_raw", "total_chemical_kmol"]
            ].to_string(index=False)
        )

    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
