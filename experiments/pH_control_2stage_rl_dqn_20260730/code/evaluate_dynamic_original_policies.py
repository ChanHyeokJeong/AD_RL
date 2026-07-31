from __future__ import annotations

import argparse
import os
import sys
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

THIS_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = THIS_DIR.parent
REPO_ROOT = THIS_DIR.parents[2] if (THIS_DIR.parents[2] / "ad_rl").exists() else None
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ad_rl.dqn_agent import QNetwork
from ad_rl.ppo_agent import PPOAgent
from ph2stage_rl_env import (
    PHControlRLConfig,
    TwoStagePHDirectDosingEnv,
    config_to_json_dict,
    rollout_fixed_pH_policy,
    rollout_policy,
)
from train_dqn_4actuator import json_dump, output_base_dir, plot_rollout


def deterministic_action_from_network(q_net: QNetwork, obs: np.ndarray) -> int:
    with torch.no_grad():
        q_values = q_net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
    return int(torch.argmax(q_values, dim=1).item())


def load_dqn_policy(config: PHControlRLConfig, model_path: Path) -> QNetwork:
    env = TwoStagePHDirectDosingEnv(config)
    q_net = QNetwork(env.observation_space.shape[0], env.action_space.n)
    state = torch.load(model_path, map_location="cpu")
    q_net.load_state_dict(state)
    q_net.eval()
    return q_net


def load_ppo_policy(
    config: PHControlRLConfig,
    model_path: Path,
    hidden_dim: int = 128,
) -> PPOAgent:
    env = TwoStagePHDirectDosingEnv(config)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=int(hidden_dim),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
    )
    state = torch.load(model_path, map_location="cpu")
    agent.network.load_state_dict(state)
    agent.network.eval()
    return agent


def parse_start_days(text: str | None, config: PHControlRLConfig, window_count: int) -> list[float]:
    if text:
        return [float(part.strip()) for part in text.split(",") if part.strip()]

    csv_path = Path(config.influent_csv)
    if not csv_path.is_absolute():
        csv_path = THIS_DIR / csv_path
    influent = pd.read_csv(csv_path, usecols=["time"])
    max_start = max(0.0, float(influent["time"].iloc[-1]) - float(config.episode_days))
    if int(window_count) <= 1:
        return [0.0]
    return [float(v) for v in np.linspace(0.0, max_start, int(window_count))]


def action_label(action_table: pd.DataFrame, action: int) -> str:
    row = action_table.iloc[int(action)]
    return (
        f"action_{int(action):02d}"
        f"_s1_{float(row['stage1_signed_m3_d']):g}"
        f"_s2_{float(row['stage2_signed_m3_d']):g}"
    ).replace("-", "m").replace(".", "p")


def run_policy_window(
    config: PHControlRLConfig,
    policy: str,
    select_action,
    window_id: int,
    start_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    window_config = replace(
        config,
        episode_start_mode="fixed",
        episode_start_day=float(start_day),
    )
    decision_df, internal_df, summary = rollout_policy(window_config, select_action)
    metadata = {
        "policy": policy,
        "window_id": int(window_id),
        "window_start_d": float(start_day),
    }
    for df in (decision_df, internal_df):
        for key, value in metadata.items():
            df[key] = value
    summary = dict(metadata, **summary)
    return decision_df, internal_df, summary


def run_fixed_ph_window(
    config: PHControlRLConfig,
    window_id: int,
    start_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    window_config = replace(
        config,
        episode_start_mode="fixed",
        episode_start_day=float(start_day),
    )
    decision_df, internal_df, summary = rollout_fixed_pH_policy(
        window_config,
        stage1_pH_sp=7.0,
        stage2_pH_sp=7.0,
    )
    metadata = {
        "policy": "fixed_pH7_PI_both_stages",
        "window_id": int(window_id),
        "window_start_d": float(start_day),
    }
    for df in (decision_df, internal_df):
        for key, value in metadata.items():
            df[key] = value
    summary = dict(metadata, **summary)
    return decision_df, internal_df, summary


def summarize_by_policy(window_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "total_reward",
        "total_reward_raw",
        "total_reward_benefit",
        "total_reward_chemical_cost",
        "total_reward_ph_cost",
        "total_total_ch4_m3",
        "total_chemical_kmol",
        "total_ph_violation_pH_d",
        "mean_stage1_pH",
        "mean_stage2_pH",
        "mean_stage2_vfa_conversion",
        "mean_stage2_ch4_per_vfa_in",
    ]
    rows = []
    for policy, group in window_summary.groupby("policy", sort=False):
        row = {"policy": policy, "windows": int(len(group))}
        for metric in metrics:
            if metric in group.columns:
                values = pd.to_numeric(group[metric], errors="coerce")
                row[f"mean_{metric}"] = float(values.mean())
                row[f"std_{metric}"] = float(values.std(ddof=0))
                row[f"sum_{metric}"] = float(values.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_policy_summary(policy_summary: pd.DataFrame, output_dir: Path) -> None:
    if policy_summary.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    labels = policy_summary["policy"].astype(str).tolist()
    x = np.arange(len(labels))

    axes[0].bar(x, policy_summary["mean_total_reward"], color="tab:blue")
    axes[0].set_ylabel("Mean reward")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, policy_summary["mean_total_total_ch4_m3"], color="tab:green")
    axes[1].set_ylabel("Mean CH4 (m3)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, policy_summary["mean_total_chemical_kmol"], color="tab:orange")
    axes[2].set_ylabel("Mean chemical (kmol)")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")

    fig.suptitle("Original dynamic influent window evaluation")
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_original_policy_summary.png", dpi=200)
    plt.close(fig)


def plot_window_metrics(window_summary: pd.DataFrame, output_dir: Path) -> None:
    if window_summary.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for policy, group in window_summary.groupby("policy", sort=False):
        group = group.sort_values("window_start_d")
        axes[0].plot(group["window_start_d"], group["total_reward"], marker="o", label=policy)
        axes[1].plot(group["window_start_d"], group["total_chemical_kmol"], marker="o", label=policy)
    axes[0].set_ylabel("Reward")
    axes[1].set_ylabel("Chemical (kmol)")
    axes[1].set_xlabel("Window start day")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    fig.suptitle("Original dynamic influent window metrics")
    fig.tight_layout()
    fig.savefig(output_dir / "dynamic_original_window_metrics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="dynamic_original_dqn_ppo_7d_8windows")
    parser.add_argument("--episode-days", type=float, default=7.0)
    parser.add_argument("--decision-interval-h", type=float, default=1.0)
    parser.add_argument("--window-count", type=int, default=8)
    parser.add_argument("--start-days", default=None)
    parser.add_argument("--influent-csv", default="digester_influent_original_dynamic.csv")
    parser.add_argument("--use-dynamic-flow", action="store_true", default=True)
    parser.add_argument("--temperature-parameter-csv", default="adm1_temperature_parameters_long.csv")
    parser.add_argument("--use-temperature-kinetics", action="store_true")
    parser.add_argument(
        "--dqn-model",
        default=str(
            EXPERIMENT_ROOT
            / "runs"
            / "ph2stage_dqn_7d_1000ep_methane_reward_chem0p2_compare_full"
            / "dqn_q_network.pt"
        ),
    )
    parser.add_argument(
        "--ppo-model",
        default=str(
            EXPERIMENT_ROOT
            / "runs"
            / "ph2stage_ppo_7d_1000ep_methane_reward_chem0p2_compare_full"
            / "ppo_actor_critic.pt"
        ),
    )
    parser.add_argument("--ppo-hidden-dim", type=int, default=128)
    parser.add_argument("--chemical-kmol-weight", type=float, default=0.2)
    parser.add_argument("--ph-violation-weight", type=float, default=500.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--fixed-actions", default="22")
    parser.add_argument("--skip-zero", action="store_true")
    parser.add_argument("--skip-ph7", action="store_true")
    parser.add_argument("--save-internal-timeseries", action="store_true")
    args = parser.parse_args()

    config = replace(
        PHControlRLConfig(),
        episode_days=float(args.episode_days),
        decision_interval_h=float(args.decision_interval_h),
        influent_csv=str(args.influent_csv),
        use_dynamic_flow=bool(args.use_dynamic_flow),
        temperature_parameter_csv=str(args.temperature_parameter_csv),
        use_temperature_kinetics=bool(args.use_temperature_kinetics),
        chemical_kmol_weight=float(args.chemical_kmol_weight),
        ph_violation_weight=float(args.ph_violation_weight),
        reward_scale=float(args.reward_scale),
    )
    output_dir = output_base_dir() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    json_dump(output_dir / "config.json", config_to_json_dict(config))

    start_days = parse_start_days(args.start_days, config, args.window_count)
    pd.DataFrame(
        {"window_id": list(range(len(start_days))), "window_start_d": start_days}
    ).to_csv(output_dir / "window_start_days.csv", index=False)

    env_for_actions = TwoStagePHDirectDosingEnv(config)
    action_table = env_for_actions.action_table
    action_table.to_csv(output_dir / "action_table.csv", index=False)
    zero_action = env_for_actions.plant.hold_action_index()

    dqn_model = Path(args.dqn_model)
    ppo_model = Path(args.ppo_model)
    dqn_net = load_dqn_policy(config, dqn_model)
    ppo_agent = load_ppo_policy(config, ppo_model, hidden_dim=args.ppo_hidden_dim)

    fixed_actions = [int(part.strip()) for part in str(args.fixed_actions).split(",") if part.strip()]
    policies = [
        (
            "dqn1000_deterministic",
            lambda obs, _env: deterministic_action_from_network(dqn_net, obs),
        ),
        (
            "ppo1000_deterministic",
            lambda obs, _env: ppo_agent.act(obs, deterministic=True),
        ),
    ]
    for action in fixed_actions:
        policies.append(
            (
                f"fixed_{action_label(action_table, action)}",
                lambda _obs, _env, a=action: int(a),
            )
        )
    if not args.skip_zero:
        policies.append(("zero_dosing_open_loop", lambda _obs, _env, a=zero_action: int(a)))

    all_decisions: list[pd.DataFrame] = []
    all_internal: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for window_id, start_day in enumerate(start_days):
        for policy_name, selector in policies:
            decision_df, internal_df, summary = run_policy_window(
                config,
                policy_name,
                selector,
                window_id,
                start_day,
            )
            all_decisions.append(decision_df)
            if args.save_internal_timeseries:
                all_internal.append(internal_df)
            summaries.append(summary)
            print(
                f"window {window_id:02d} start={start_day:.3f} {policy_name} "
                f"reward={summary.get('total_reward', np.nan):.3f} "
                f"ch4={summary.get('total_total_ch4_m3', np.nan):.1f} "
                f"chem={summary.get('total_chemical_kmol', np.nan):.3f}",
                flush=True,
            )
            if window_id == 0 and policy_name in {
                "dqn1000_deterministic",
                "ppo1000_deterministic",
            }:
                plot_rollout(
                    output_dir,
                    internal_df,
                    f"{policy_name}, original dynamic window {window_id}",
                    f"{policy_name}_window{window_id:02d}_rollout.png",
                )

        if not args.skip_ph7:
            decision_df, internal_df, summary = run_fixed_ph_window(config, window_id, start_day)
            all_decisions.append(decision_df)
            if args.save_internal_timeseries:
                all_internal.append(internal_df)
            summaries.append(summary)
            print(
                f"window {window_id:02d} start={start_day:.3f} fixed_pH7_PI "
                f"reward={summary.get('total_reward', np.nan):.3f} "
                f"ch4={summary.get('total_total_ch4_m3', np.nan):.1f} "
                f"chem={summary.get('total_chemical_kmol', np.nan):.3f}",
                flush=True,
            )

    window_summary = pd.DataFrame(summaries)
    policy_summary = summarize_by_policy(window_summary)
    decision_summary = pd.concat(all_decisions, ignore_index=True) if all_decisions else pd.DataFrame()

    window_summary.to_csv(output_dir / "dynamic_window_summary.csv", index=False)
    policy_summary.to_csv(output_dir / "dynamic_policy_summary.csv", index=False)
    decision_summary.to_csv(output_dir / "dynamic_decision_steps.csv", index=False)
    if all_internal:
        pd.concat(all_internal, ignore_index=True).to_csv(
            output_dir / "dynamic_internal_timeseries.csv",
            index=False,
        )

    plot_policy_summary(policy_summary, output_dir)
    plot_window_metrics(window_summary, output_dir)

    print(
        policy_summary[
            [
                "policy",
                "mean_total_reward",
                "mean_total_total_ch4_m3",
                "mean_total_chemical_kmol",
                "mean_total_ph_violation_pH_d",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
