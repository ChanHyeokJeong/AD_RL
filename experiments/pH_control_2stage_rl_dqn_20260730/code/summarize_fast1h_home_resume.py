from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
SNAPSHOT = ROOT / "resume_snapshots" / "fast1h_multiscale_preoutage_20260801"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dominant_actions(path: Path, n: int = 3) -> str:
    actions = pd.read_csv(path)["action"].value_counts()
    total = actions.sum()
    return "; ".join(f"{int(action)} ({count / total:.1%})" for action, count in actions.head(n).items())


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    dqn_run = RUNS / "fast1h_home_dqn_resume_from_preoutage"
    ppo_run = RUNS / "fast1h_home_ppo_resume_from_preoutage"
    ep500_run = RUNS / "fast1h_home_dqn_ep500_eval"
    base_run = RUNS / "fast1h_home_static_comparison"

    baseline = pd.read_csv(base_run / "comparison_summary.csv")
    baseline = baseline[baseline["policy"] != "dqn_deterministic"].copy()
    policies = [
        ("DQN resumed latest", read_json(dqn_run / "dqn_policy_summary.json"), dominant_actions(dqn_run / "dqn_policy_decision_steps.csv")),
        ("PPO resumed latest", read_json(ppo_run / "ppo_policy_summary.json"), dominant_actions(ppo_run / "ppo_policy_decision_steps.csv")),
        ("DQN ep500 candidate", read_json(ep500_run / "dqn_policy_summary.json"), dominant_actions(ep500_run / "dqn_policy_decision_steps.csv")),
    ]
    rows = baseline.to_dict("records")
    for label, summary, actions in policies:
        rows.append(dict(summary, policy=label, dominant_actions=actions))
    final = pd.DataFrame(rows)
    keep = [
        "policy", "total_reward", "total_reward_raw", "total_total_ch4_m3",
        "total_chemical_kmol", "total_ph_violation_pH_d", "dominant_actions",
    ]
    final = final.reindex(columns=keep).sort_values("total_reward", ascending=False)
    final.to_csv(RESULTS / "fast1h_home_final_policy_comparison.csv", index=False)

    # Join pre-outage and home-resume episode summaries on a continuous episode axis.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, pre_name, run_dir, color in [
        ("DQN", "dqn_episode_summary.csv", dqn_run, "tab:blue"),
        ("PPO", "ppo_episode_summary.csv", ppo_run, "tab:orange"),
    ]:
        pre = pd.read_csv(SNAPSHOT / pre_name)
        home = pd.read_csv(run_dir / "episode_summary.csv")
        reward = pd.concat([pre["total_reward"], home["total_reward"]], ignore_index=True)
        smooth = reward.rolling(25, min_periods=1).mean()
        ax.plot(range(1, len(reward) + 1), smooth, label=f"{name} (25-episode mean)", color=color)
        ax.axvline(len(pre), color=color, linestyle="--", alpha=0.45)
    ax.set(xlabel="Episode", ylabel="Total reward", title="Fast 1 h multi-scale training reward")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "fast1h_home_reward_learning_curve.png", dpi=180)
    plt.close(fig)

    series = [
        ("Best fixed", base_run / "baseline_best_action_internal_timeseries.csv"),
        ("pH7 PI", base_run / "baseline_fixed_pH7_PI_internal_timeseries.csv"),
        ("DQN ep500", ep500_run / "dqn_policy_internal_timeseries.csv"),
        ("PPO latest", ppo_run / "ppo_policy_internal_timeseries.csv"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for label, path in series:
        df = pd.read_csv(path)
        axes[0].plot(df["time_d"], df["stage1_pH"], label=f"{label} stage 1")
        axes[1].plot(df["time_d"], df["stage2_pH"], label=f"{label} stage 2")
    for ax, stage in zip(axes, ("Stage 1", "Stage 2")):
        ax.axhline(7.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"{stage} pH")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[1].set_xlabel("Time, d")
    fig.suptitle("Deterministic policy rollout comparison")
    fig.tight_layout()
    fig.savefig(FIGURES / "fast1h_home_policy_rollout_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, row in final.iterrows():
        ax.scatter(row["total_chemical_kmol"], row["total_total_ch4_m3"], s=65)
        ax.annotate(row["policy"], (row["total_chemical_kmol"], row["total_total_ch4_m3"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Chemical use, kmol", ylabel="Total CH4 production, m3", title="Methane production vs chemical use")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fast1h_home_methane_vs_chemical.png", dpi=180)
    plt.close(fig)

    pi = final.loc[final["policy"] == "fixed_pH7_PI_both_stages"].iloc[0]
    delta = final.copy()
    for col in ["total_reward", "total_reward_raw", "total_total_ch4_m3", "total_chemical_kmol", "total_ph_violation_pH_d"]:
        delta[f"delta_vs_pH7_PI_{col}"] = delta[col] - pi[col]
    delta.to_csv(RESULTS / "fast1h_home_final_policy_comparison_vs_pH7_PI.csv", index=False)
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
