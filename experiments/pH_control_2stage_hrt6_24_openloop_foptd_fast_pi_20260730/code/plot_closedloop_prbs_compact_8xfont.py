from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXP_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"
LOG = RESULTS_DIR / "closedloop_prbs_stage_specific_logs.csv"
SUMMARY = RESULTS_DIR / "closedloop_prbs_stage_specific_summary.csv"

FONT = 80
TITLE_FONT = 88
TICK_FONT = 64
LEGEND_FONT = 58
LINE_W = 7.0


def plot_stage(df: pd.DataFrame, summary: pd.Series, target: str, out_path: Path) -> None:
    stage_label = "1st Reactor" if target == "stage1" else "2nd Reactor"
    ph_col = f"{target}_pH"
    naoh_col = f"{target}_q_NaOH_m3_d"
    hcl_col = f"{target}_q_HCl_m3_d"

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(56, 32),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.35], "hspace": 0.16},
    )

    ax = axes[0]
    ax.plot(df["time"], df[ph_col], color="#0072B2", lw=LINE_W, label="pH")
    ax.set_ylabel("pH", fontsize=FONT)
    ax.set_title(stage_label, fontsize=TITLE_FONT, pad=26)
    ax.grid(True, alpha=0.24, linewidth=2.0)
    ax.set_ylim(6.78, 8.52)
    ax.tick_params(axis="both", labelsize=TICK_FONT, width=3.0, length=14)
    ax.legend(loc="upper center", ncol=1, fontsize=LEGEND_FONT, frameon=True)
    ax.text(
        0.015,
        0.055,
        f"median={summary['median_settle_time_0p05_d']:.3f} d, max={summary['max_settle_time_0p05_d']:.3f} d",
        transform=ax.transAxes,
        fontsize=LEGEND_FONT,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.65", "alpha": 0.85},
    )

    ax_naoh = axes[1]
    ax_hcl = ax_naoh.twinx()
    naoh_line = ax_naoh.plot(df["time"], df[naoh_col], color="#009E73", lw=LINE_W, label="NaOH")
    hcl_line = ax_hcl.plot(df["time"], df[hcl_col], color="#CC79A7", lw=LINE_W, label="HCl")

    ax_naoh.set_ylabel("NaOH (m3/d)", fontsize=FONT, color="#007A5A")
    ax_hcl.set_ylabel("HCl (m3/d)", fontsize=FONT, color="#A83D78")
    ax_naoh.set_xlabel("Time (d)", fontsize=FONT)
    ax_naoh.grid(True, alpha=0.24, linewidth=2.0)
    ax_naoh.set_ylim(0, max(5.0, df[naoh_col].max() * 1.20))
    ax_hcl.set_ylim(0, max(5.0, df[hcl_col].max() * 1.20))
    ax_naoh.tick_params(axis="both", labelsize=TICK_FONT, width=3.0, length=14)
    ax_hcl.tick_params(axis="y", labelsize=TICK_FONT, width=3.0, length=14)
    ax_naoh.tick_params(axis="y", colors="#007A5A")
    ax_hcl.tick_params(axis="y", colors="#A83D78")
    ax_naoh.yaxis.label.set_color("#007A5A")
    ax_hcl.yaxis.label.set_color("#A83D78")

    lines = naoh_line + hcl_line
    labels = [line.get_label() for line in lines]
    ax_naoh.legend(lines, labels, loc="upper center", ncol=2, fontsize=LEGEND_FONT, frameon=True)

    for axis in [ax, ax_naoh, ax_hcl]:
        for spine in axis.spines.values():
            spine.set_linewidth(3.0)

    fig.subplots_adjust(left=0.09, right=0.90, top=0.92, bottom=0.12)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOG)
    summary = pd.read_csv(SUMMARY).set_index("target")
    for target in ("stage1", "stage2"):
        part = df[df["target"] == target].copy()
        out = FIGURES_DIR / f"closedloop_prbs_test_{target}_compact_8xfont.png"
        plot_stage(part, summary.loc[target], target, out)
        print(out)


if __name__ == "__main__":
    main()
