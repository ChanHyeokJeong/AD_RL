from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "biomass_reward_dqn_1d_30ep_test"


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


if __name__ == "__main__":
    main()
