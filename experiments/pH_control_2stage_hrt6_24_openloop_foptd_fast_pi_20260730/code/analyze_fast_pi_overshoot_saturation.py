from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE = Path(
    "Z:/backup/Chanhyeok Jeong/Project/KICHE_AD_control/"
    "\uc18c\ud654\uc870 \uc81c\uc5b4/PyADM1-master/pH_control_2stage"
)
OUT = Path("C:/Users/JCH/Documents/AD control/ph_control_diagnostics/hrt_6_24_fast_pi_prbs")

LOG = BASE / "acid_base_control_log_2stage_serial_PRBS.csv"
SCHEDULE = BASE / "pH_PRBS_schedule_2stage_serial.csv"

Q_NAOH_MAX = 100.0
Q_HCL_MAX = 100.0
SAT_TOL = 1e-6
SETTLE_BAND_PH = 0.05


def segment_metrics(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prev_sp = None
    for _, seg in schedule.iterrows():
        start = float(seg["start_d"])
        end = float(seg["end_d"])
        sp = float(seg["pH_sp"])
        idx = int(seg["segment"])
        mask = (df["time"] >= start) & (df["time"] < end if idx < int(schedule["segment"].iloc[-1]) else df["time"] <= end)
        part = df.loc[mask].copy()
        if part.empty:
            continue
        pH = part["stage2_pH"]
        err = sp - pH
        abs_err = err.abs()

        if prev_sp is None:
            direction = "initial"
            overshoot = np.nan
            settle_time = np.nan
        elif sp > prev_sp:
            direction = "up"
            overshoot = max(0.0, float((pH - sp).max()))
            settle_time = first_settled_time(part["time"].to_numpy(), abs_err.to_numpy(), start)
        elif sp < prev_sp:
            direction = "down"
            overshoot = max(0.0, float((sp - pH).max()))
            settle_time = first_settled_time(part["time"].to_numpy(), abs_err.to_numpy(), start)
        else:
            direction = "hold"
            overshoot = np.nan
            settle_time = first_settled_time(part["time"].to_numpy(), abs_err.to_numpy(), start)

        rows.append(
            {
                "segment": idx,
                "start_d": start,
                "end_d": end,
                "duration_d": end - start,
                "pH_sp": sp,
                "direction": direction,
                "pH_start": float(pH.iloc[0]),
                "pH_end": float(pH.iloc[-1]),
                "pH_min": float(pH.min()),
                "pH_max": float(pH.max()),
                "overshoot_pH": overshoot,
                "MAE_pH": float(abs_err.mean()),
                "RMSE_pH": float(np.sqrt(np.mean(err**2))),
                "final_abs_error_pH": float(abs_err.iloc[-1]),
                "settle_time_to_0p05_d": settle_time,
                "q_NaOH_max_m3_d": float(part["q_NaOH_m3_d"].max()),
                "q_HCl_max_m3_d": float(part["q_HCl_m3_d"].max()),
                "NaOH_sat_fraction": float((part["q_NaOH_m3_d"] >= Q_NAOH_MAX - SAT_TOL).mean()),
                "HCl_sat_fraction": float((part["q_HCl_m3_d"] >= Q_HCL_MAX - SAT_TOL).mean()),
            }
        )
        prev_sp = sp
    return pd.DataFrame(rows)


def first_settled_time(time_values: np.ndarray, abs_error: np.ndarray, start: float) -> float:
    ok = abs_error <= SETTLE_BAND_PH
    if not ok.any():
        return np.nan
    for i in np.where(ok)[0]:
        if ok[i:].all():
            return float(time_values[i] - start)
    return np.nan


def summary_metrics(df: pd.DataFrame, seg_df: pd.DataFrame) -> pd.DataFrame:
    err = df["pH_sp"] - df["stage2_pH"]
    abs_err = err.abs()
    high = df["pH_sp"] > df["pH_sp"].median()
    low = ~high

    up = seg_df[seg_df["direction"] == "up"]
    down = seg_df[seg_df["direction"] == "down"]

    rows = [
        ("rows", len(df)),
        ("time_start_d", df["time"].min()),
        ("time_end_d", df["time"].max()),
        ("stage1_pH_min", df["stage1_pH"].min()),
        ("stage1_pH_max", df["stage1_pH"].max()),
        ("stage2_pH_min", df["stage2_pH"].min()),
        ("stage2_pH_max", df["stage2_pH"].max()),
        ("overall_MAE_pH", abs_err.mean()),
        ("overall_RMSE_pH", np.sqrt(np.mean(err**2))),
        ("high_sp_MAE_pH", abs_err[high].mean()),
        ("low_sp_MAE_pH", abs_err[low].mean()),
        ("up_segments", len(up)),
        ("down_segments", len(down)),
        ("up_max_overshoot_pH", up["overshoot_pH"].max()),
        ("up_mean_overshoot_pH", up["overshoot_pH"].mean()),
        ("down_max_undershoot_pH", down["overshoot_pH"].max()),
        ("down_mean_undershoot_pH", down["overshoot_pH"].mean()),
        ("segments_settled_0p05_fraction", seg_df["settle_time_to_0p05_d"].notna().mean()),
        ("median_settle_time_0p05_d", seg_df["settle_time_to_0p05_d"].median()),
        ("max_settle_time_0p05_d", seg_df["settle_time_to_0p05_d"].max()),
        ("q_NaOH_max_m3_d", df["q_NaOH_m3_d"].max()),
        ("q_NaOH_mean_m3_d", df["q_NaOH_m3_d"].mean()),
        ("q_HCl_max_m3_d", df["q_HCl_m3_d"].max()),
        ("q_HCl_mean_m3_d", df["q_HCl_m3_d"].mean()),
        ("NaOH_sat_fraction", (df["q_NaOH_m3_d"] >= Q_NAOH_MAX - SAT_TOL).mean()),
        ("HCl_sat_fraction", (df["q_HCl_m3_d"] >= Q_HCL_MAX - SAT_TOL).mean()),
        ("NaOH_active_fraction", (df["q_NaOH_m3_d"] > 1e-9).mean()),
        ("HCl_active_fraction", (df["q_HCl_m3_d"] > 1e-9).mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def plot_tracking(df: pd.DataFrame, path: Path, start: float, end: float, title: str) -> None:
    win = df[(df["time"] >= start) & (df["time"] <= end)].copy()
    err = win["pH_sp"] - win["stage2_pH"]

    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0, 1.1, 1.1]})
    axes[0].step(win["time"], win["pH_sp"], where="post", color="black", lw=1.4, label="SP")
    axes[0].plot(win["time"], win["stage2_pH"], color="#0072B2", lw=1.2, label="Stage 2 pH")
    axes[0].plot(win["time"], win["stage1_pH"], color="#D55E00", lw=0.9, alpha=0.7, label="Stage 1 pH")
    axes[0].set_ylabel("pH")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncol=3, fontsize=9)

    axes[1].plot(win["time"], err, color="#555555", lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].axhline(0.05, color="#999999", lw=0.7, ls="--")
    axes[1].axhline(-0.05, color="#999999", lw=0.7, ls="--")
    axes[1].set_ylabel("SP-pH")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(win["time"], win["q_NaOH_m3_d"], color="#009E73", lw=1.0, label="NaOH")
    axes[2].axhline(Q_NAOH_MAX, color="red", lw=0.8, ls="--", label="max")
    axes[2].set_ylabel("NaOH\nm3/d")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=9)

    axes[3].plot(win["time"], win["q_HCl_m3_d"], color="#CC79A7", lw=1.0, label="HCl")
    axes[3].axhline(Q_HCL_MAX, color="red", lw=0.8, ls="--", label="max")
    axes[3].set_ylabel("HCl\nm3/d")
    axes[3].set_xlabel("Time (d)")
    axes[3].grid(True, alpha=0.25)
    axes[3].legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(LOG)
    schedule = pd.read_csv(SCHEDULE)
    seg = segment_metrics(df, schedule)
    summary = summary_metrics(df, seg)

    summary.to_csv(OUT / "fast_pi_prbs_overshoot_saturation_summary.csv", index=False)
    seg.to_csv(OUT / "fast_pi_prbs_segment_overshoot_saturation.csv", index=False)
    plot_tracking(df, OUT / "fast_pi_prbs_tracking_0_40d.png", 0.0, 40.0, "Aggressive PI PRBS tracking, 0-40 d")
    plot_tracking(df, OUT / "fast_pi_prbs_tracking_280d.png", float(df["time"].min()), float(df["time"].max()), "Aggressive PI PRBS tracking, full run")

    print(summary.to_string(index=False))
    print(OUT / "fast_pi_prbs_overshoot_saturation_summary.csv")
    print(OUT / "fast_pi_prbs_segment_overshoot_saturation.csv")
    print(OUT / "fast_pi_prbs_tracking_0_40d.png")
    print(OUT / "fast_pi_prbs_tracking_280d.png")


if __name__ == "__main__":
    main()
