from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "biomass_reward_dqn_1d_30ep_test"
STAGES = [
    (30, "biomass_reward_dqn_1d_30ep_test", 0),
    (100, "biomass_reward_dqn_1d_resume30_to100", 30),
    (200, "biomass_reward_dqn_1d_resume100_to200", 0),
    (300, "biomass_reward_dqn_1d_resume200_to300", 0),
    (400, "biomass_reward_dqn_1d_resume300_to400", 0),
    (500, "biomass_reward_dqn_1d_resume400_to500", 0),
    (600, "biomass_reward_dqn_1d_resume500_to600", 0),
]
NO_PH_STAGES = [
    (500, "biomass_reward_no_ph_penalty_ep500_eval"),
    (700, "biomass_reward_no_ph_penalty_resume600_to700"),
    (800, "biomass_reward_no_ph_penalty_resume700_to800"),
    (900, "biomass_reward_no_ph_penalty_resume800_to900"),
]


def main() -> None:
    results = ROOT / "results"
    figures = ROOT / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    comparison = pd.read_csv(RUN / "comparison_summary.csv")
    columns = [
        "policy",
        "total_reward",
        "total_reward_benefit",
        "total_reward_chemical_cost",
        "total_reward_ph_cost",
        "total_stage1_active_biomass_relative_growth",
        "total_stage2_active_biomass_relative_growth",
        "final_stage1_active_biomass_kgCOD_m3",
        "final_stage2_active_biomass_kgCOD_m3",
        "total_total_ch4_m3",
        "total_chemical_kmol",
        "total_ph_violation_pH_d",
    ]
    comparison[columns].to_csv(
        results / "biomass_reward_dqn_1d_30ep_test_comparison.csv", index=False
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    episodes = pd.read_csv(RUN / "episode_summary.csv")
    axes[0].plot(episodes["episode"] + 1, episodes["total_reward"], alpha=0.45)
    axes[0].plot(
        episodes["episode"] + 1,
        episodes["total_reward"].rolling(5, min_periods=1).mean(),
        linewidth=2,
        label="5-episode mean",
    )
    axes[0].set(xlabel="Episode", ylabel="Reward", title="Biomass-reward DQN smoke training")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for label, filename in [
        ("DQN 30 ep", "dqn_policy_internal_timeseries.csv"),
        ("Best fixed", "baseline_best_action_internal_timeseries.csv"),
        ("pH7 PI", "baseline_fixed_pH7_PI_internal_timeseries.csv"),
    ]:
        df = pd.read_csv(RUN / filename)
        axes[1].plot(df["time_d"], df["stage1_active_biomass_kgCOD_m3"], label=f"{label} S1")
        axes[1].plot(
            df["time_d"],
            df["stage2_active_biomass_kgCOD_m3"],
            linestyle="--",
            label=f"{label} S2",
        )
    axes[1].set(
        xlabel="Time, d",
        ylabel="Active biomass, kgCOD/m3",
        title="Active microbial concentration (methane is not rewarded)",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "biomass_reward_dqn_1d_30ep_test.png", dpi=180)
    plt.close(fig)

    stage_rows = []
    training_parts = []
    for episodes_total, run_name, episode_offset in STAGES:
        run_dir = ROOT / "runs" / run_name
        summary = json.loads((run_dir / "dqn_policy_summary.json").read_text(encoding="utf-8"))
        stage_rows.append(
            {
                "policy": f"DQN deterministic ep{episodes_total}",
                "episodes": episodes_total,
                **{column: summary.get(column) for column in columns if column != "policy"},
            }
        )
        training = pd.read_csv(run_dir / "episode_summary.csv")
        if episode_offset:
            training["episode"] += episode_offset
        training_parts.append(training[["episode", "total_reward"]])

    for row in comparison[columns].to_dict("records"):
        if row["policy"] != "dqn_deterministic":
            stage_rows.append({"episodes": pd.NA, **row})
    staged = pd.DataFrame(stage_rows)
    staged.to_csv(results / "biomass_reward_dqn_1d_staged_comparison.csv", index=False)

    training = pd.concat(training_parts, ignore_index=True).sort_values("episode")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    axes[0].plot(training["episode"] + 1, training["total_reward"], alpha=0.2)
    axes[0].plot(
        training["episode"] + 1,
        training["total_reward"].rolling(25, min_periods=1).mean(),
        linewidth=2,
        label="25-episode mean",
    )
    dqn = staged[staged["episodes"].notna()].copy()
    axes[0].scatter(dqn["episodes"], dqn["total_reward"], color="black", label="Deterministic")
    for policy, style in [
        ("best_fixed_open_loop_action", "--"),
        ("fixed_pH7_PI_both_stages", ":"),
        ("zero_dosing_open_loop", "-."),
    ]:
        value = staged.loc[staged["policy"] == policy, "total_reward"].iloc[0]
        axes[0].axhline(value, linestyle=style, label=policy)
    axes[0].set(xlabel="Cumulative episode", ylabel="Reward", title="Staged biomass-reward DQN progress")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(
        dqn["episodes"],
        100 * dqn["total_stage1_active_biomass_relative_growth"],
        marker="o",
        label="Stage 1",
    )
    axes[1].plot(
        dqn["episodes"],
        100 * dqn["total_stage2_active_biomass_relative_growth"],
        marker="o",
        label="Stage 2",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set(
        xlabel="Cumulative episode",
        ylabel="Final biomass change, %",
        title="Deterministic active-biomass retention",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "biomass_reward_dqn_1d_staged_progress.png", dpi=180)
    plt.close(fig)

    no_ph_base = ROOT / "runs" / "biomass_reward_no_ph_penalty_ep500_eval"
    no_ph_comparison = pd.read_csv(no_ph_base / "comparison_summary.csv")
    no_ph_rows = []
    no_ph_training = []
    for episodes_total, run_name in NO_PH_STAGES:
        run_dir = ROOT / "runs" / run_name
        summary = json.loads((run_dir / "dqn_policy_summary.json").read_text(encoding="utf-8"))
        no_ph_rows.append(
            {
                "policy": f"DQN deterministic ep{episodes_total}",
                "episodes": episodes_total,
                **{column: summary.get(column) for column in columns if column != "policy"},
            }
        )
        episode_path = run_dir / "episode_summary.csv"
        if episode_path.exists():
            episode_df = pd.read_csv(episode_path)
            if not episode_df.empty:
                no_ph_training.append(episode_df[["episode", "total_reward"]])
    for row in no_ph_comparison[columns].to_dict("records"):
        if row["policy"] != "dqn_deterministic":
            no_ph_rows.append({"episodes": pd.NA, **row})
    no_ph = pd.DataFrame(no_ph_rows)
    no_ph.to_csv(results / "biomass_reward_no_ph_penalty_staged_comparison.csv", index=False)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    no_ph_dqn = no_ph[no_ph["episodes"].notna()].copy()
    if no_ph_training:
        training = pd.concat(no_ph_training, ignore_index=True).sort_values("episode")
        axes[0].plot(training["episode"] + 1, training["total_reward"], alpha=0.2)
        axes[0].plot(
            training["episode"] + 1,
            training["total_reward"].rolling(25, min_periods=1).mean(),
            linewidth=2,
            label="25-episode mean",
        )
    axes[0].scatter(
        no_ph_dqn["episodes"], no_ph_dqn["total_reward"], color="black", label="Deterministic"
    )
    for policy, style in [
        ("best_fixed_open_loop_action", "--"),
        ("fixed_pH7_PI_both_stages", ":"),
        ("zero_dosing_open_loop", "-."),
    ]:
        value = no_ph.loc[no_ph["policy"] == policy, "total_reward"].iloc[0]
        axes[0].axhline(value, linestyle=style, label=policy)
    axes[0].set(
        xlabel="Cumulative episode",
        ylabel="Reward",
        title="Active-biomass DQN after removing pH violation cost",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].plot(
        no_ph_dqn["episodes"],
        no_ph_dqn["total_ph_violation_pH_d"],
        marker="o",
        color="tab:red",
    )
    axes[1].set(
        xlabel="Cumulative episode",
        ylabel="Logged pH violation, pH*d",
        title="pH violation remains diagnostic only (zero reward weight)",
    )
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "biomass_reward_no_ph_penalty_staged_progress.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
