from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import PyADM1_pH_2stage_PRBS as model


BASE_DIR = Path(__file__).resolve().parent
EXP_DIR = BASE_DIR.parent
RESULTS_DIR = EXP_DIR / "results"
FIGURES_DIR = EXP_DIR / "figures"

RUN_DAYS = 40.0
HOLD_PH = 7.0
SETTLE_BAND_PH = 0.05
SAT_TOL = 1e-6
CONTROL_DEADBAND_PH = {
    "stage1": 0.0,
    "stage2": 0.0,
}


def setup_reactors() -> tuple[dict, dict]:
    stage1 = model.fresh_reactor("stage1_55C", model.STAGE1_TEMP_K, model.STAGE1_VOLUME_FRACTION)
    stage2 = model.fresh_reactor("stage2_35C", model.STAGE2_TEMP_K, model.STAGE2_VOLUME_FRACTION)
    for reactor in (stage1, stage2):
        reactor["u_NaOH"] = 0.0
        reactor["q_NaOH"] = 0.0
        reactor["u_HCl"] = 0.0
        reactor["q_HCl"] = 0.0
        reactor["err_int"] = 0.0
        reactor["err_int_NaOH"] = 0.0
        reactor["err_int_HCl"] = 0.0
    return stage1, stage2


def prbs_sp(target: str, schedule: pd.DataFrame, idx: int) -> tuple[float, float]:
    level = float(schedule.iloc[idx]["pH_sp"])
    if target == "stage1":
        return level, HOLD_PH
    if target == "stage2":
        return HOLD_PH, level
    raise ValueError(f"Unknown target: {target}")


def control_row(target: str, time_d: float, segment: int, stage1_sp: float, stage2_sp: float, stage1: dict, stage2: dict) -> dict:
    return {
        "target": target,
        "time": time_d,
        "prbs_segment": segment,
        "stage1_pH_sp": stage1_sp,
        "stage2_pH_sp": stage2_sp,
        "stage1_pH": float(stage1["pH"]),
        "stage2_pH": float(stage2["pH"]),
        "stage1_q_NaOH_m3_d": float(stage1["q_NaOH"]),
        "stage1_u_NaOH_kmol_d": float(stage1["u_NaOH"]),
        "stage1_q_HCl_m3_d": float(stage1["q_HCl"]),
        "stage1_u_HCl_kmol_d": float(stage1["u_HCl"]),
        "stage2_q_NaOH_m3_d": float(stage2["q_NaOH"]),
        "stage2_u_NaOH_kmol_d": float(stage2["u_NaOH"]),
        "stage2_q_HCl_m3_d": float(stage2["q_HCl"]),
        "stage2_u_HCl_kmol_d": float(stage2["u_HCl"]),
    }


def apply_pi_with_deadband(reactor: dict, reactor_key: str, pH_sp: float, dt: float) -> None:
    err = float(pH_sp - reactor["pH"])
    if abs(err) <= CONTROL_DEADBAND_PH.get(reactor_key, 0.0):
        reactor["q_NaOH"] = 0.0
        reactor["u_NaOH"] = 0.0
        reactor["q_HCl"] = 0.0
        reactor["u_HCl"] = 0.0
        reactor["err_int"] = 0.0
        reactor["err_int_NaOH"] = 0.0
        reactor["err_int_HCl"] = 0.0
        return
    reactor["PI_pH_controller"](pH_sp, reactor["pH"], dt)


def run_target(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1, stage2 = setup_reactors()
    time_values = np.asarray(stage1["t"], dtype=float)
    time_values = time_values[time_values <= RUN_DAYS + 1e-12]
    schedule = model.build_prbs_schedule(float(time_values[-1]))

    prbs_idx = 0
    stage1_sp, stage2_sp = prbs_sp(target, schedule, prbs_idx)
    t0 = 0.0
    stage1_timer = 0.0
    stage2_timer = 0.0
    rows = [control_row(target, 0.0, prbs_idx, stage1_sp, stage2_sp, stage1, stage2)]

    for n, u in enumerate(time_values[1:], start=1):
        u = float(u)
        tstep = [t0, u]

        model.set_input_from_influent(stage1, n)
        model.run_reactor_step(stage1, tstep)

        model.set_input_from_reactor(stage2, stage1)
        model.run_reactor_step(stage2, tstep)

        dt = u - t0
        prbs_idx = model.prbs_segment_at(schedule, u, prbs_idx)
        stage1_sp, stage2_sp = prbs_sp(target, schedule, prbs_idx)

        stage1_timer += dt
        stage2_timer += dt
        if stage1_timer >= model.CONTROL_INTERVAL_DAYS - 1e-12:
            apply_pi_with_deadband(stage1, "stage1", stage1_sp, stage1_timer)
            stage1_timer = 0.0
        if stage2_timer >= model.CONTROL_INTERVAL_DAYS - 1e-12:
            apply_pi_with_deadband(stage2, "stage2", stage2_sp, stage2_timer)
            stage2_timer = 0.0

        rows.append(control_row(target, u, prbs_idx, stage1_sp, stage2_sp, stage1, stage2))
        t0 = u

    return pd.DataFrame(rows), schedule.assign(target=target)


def first_settled_time(time_values: np.ndarray, abs_error: np.ndarray, start: float) -> float:
    ok = abs_error <= SETTLE_BAND_PH
    if not ok.any():
        return np.nan
    for i in np.where(ok)[0]:
        if ok[i:].all():
            return float(time_values[i] - start)
    return np.nan


def segment_metrics(df: pd.DataFrame, schedule: pd.DataFrame, target: str) -> pd.DataFrame:
    pH_col = f"{target}_pH"
    sp_col = f"{target}_pH_sp"
    q_naoh_col = f"{target}_q_NaOH_m3_d"
    q_hcl_col = f"{target}_q_HCl_m3_d"
    rows = []
    prev_sp = None

    for _, seg in schedule.iterrows():
        start = float(seg["start_d"])
        end = float(seg["end_d"])
        idx = int(seg["segment"])
        mask = (df["time"] >= start) & (df["time"] < end if idx < int(schedule["segment"].iloc[-1]) else df["time"] <= end)
        part = df.loc[mask].copy()
        if part.empty:
            continue

        sp = float(part[sp_col].iloc[0])
        pH = part[pH_col]
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
                "target": target,
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
                "q_NaOH_max_m3_d": float(part[q_naoh_col].max()),
                "q_HCl_max_m3_d": float(part[q_hcl_col].max()),
                "NaOH_sat_fraction": float((part[q_naoh_col] >= model.Q_NAOH_MAX - SAT_TOL).mean()),
                "HCl_sat_fraction": float((part[q_hcl_col] >= model.Q_HCL_MAX - SAT_TOL).mean()),
            }
        )
        prev_sp = sp

    return pd.DataFrame(rows)


def summary_metrics(df: pd.DataFrame, seg: pd.DataFrame, target: str) -> dict:
    pH_col = f"{target}_pH"
    sp_col = f"{target}_pH_sp"
    q_naoh_col = f"{target}_q_NaOH_m3_d"
    q_hcl_col = f"{target}_q_HCl_m3_d"

    err = df[sp_col] - df[pH_col]
    abs_err = err.abs()
    settled = seg["settle_time_to_0p05_d"].dropna()
    up = seg[seg["direction"] == "up"]
    down = seg[seg["direction"] == "down"]

    return {
        "target": target,
        "run_days": float(df["time"].max() - df["time"].min()),
        "segments": int(len(seg)),
        "overall_MAE_pH": float(abs_err.mean()),
        "overall_RMSE_pH": float(np.sqrt(np.mean(err**2))),
        "max_abs_error_pH": float(abs_err.max()),
        "settled_fraction_0p05": float(seg["settle_time_to_0p05_d"].notna().mean()),
        "median_settle_time_0p05_d": float(settled.median()) if not settled.empty else np.nan,
        "max_settle_time_0p05_d": float(settled.max()) if not settled.empty else np.nan,
        "up_max_overshoot_pH": float(up["overshoot_pH"].max()) if not up.empty else np.nan,
        "down_max_undershoot_pH": float(down["overshoot_pH"].max()) if not down.empty else np.nan,
        "q_NaOH_max_m3_d": float(df[q_naoh_col].max()),
        "q_HCl_max_m3_d": float(df[q_hcl_col].max()),
        "NaOH_sat_fraction": float((df[q_naoh_col] >= model.Q_NAOH_MAX - SAT_TOL).mean()),
        "HCl_sat_fraction": float((df[q_hcl_col] >= model.Q_HCL_MAX - SAT_TOL).mean()),
    }


def plot_target(df: pd.DataFrame, summary: dict, target: str, path: Path) -> None:
    pH_col = f"{target}_pH"
    sp_col = f"{target}_pH_sp"
    q_naoh_col = f"{target}_q_NaOH_m3_d"
    q_hcl_col = f"{target}_q_HCl_m3_d"
    other = "stage2" if target == "stage1" else "stage1"

    label = "1st reactor" if target == "stage1" else "2nd reactor"
    title = "1st Reactor Closed-loop PRBS Tracking" if target == "stage1" else "2nd Reactor Closed-loop PRBS Tracking"

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(14, 9.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0, 1.1, 1.1]},
    )

    axes[0].step(df["time"], df[sp_col], where="post", color="black", lw=1.6, label="SP")
    axes[0].fill_between(df["time"], df[sp_col] - SETTLE_BAND_PH, df[sp_col] + SETTLE_BAND_PH, step="post", color="0.7", alpha=0.25, label="+/-0.05 pH")
    axes[0].plot(df["time"], df[pH_col], color="#0072B2", lw=1.3, label=f"{label} pH")
    axes[0].plot(df["time"], df[f"{other}_pH"], color="#999999", lw=0.8, alpha=0.65, label=f"{other} pH")
    axes[0].set_ylabel("pH")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", ncol=4, fontsize=9)
    axes[0].text(
        0.01,
        0.04,
        f"median settle={summary['median_settle_time_0p05_d']:.3f} d, max={summary['max_settle_time_0p05_d']:.3f} d",
        transform=axes[0].transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    err = df[sp_col] - df[pH_col]
    axes[1].plot(df["time"], err, color="#555555", lw=1.0)
    axes[1].axhline(0.0, color="black", lw=0.7)
    axes[1].axhline(SETTLE_BAND_PH, color="#999999", lw=0.8, ls="--")
    axes[1].axhline(-SETTLE_BAND_PH, color="#999999", lw=0.8, ls="--")
    axes[1].set_ylabel("SP-pH")
    axes[1].grid(True, alpha=0.25)

    axes[2].plot(df["time"], df[q_naoh_col], color="#009E73", lw=1.1)
    axes[2].axhline(model.Q_NAOH_MAX, color="red", lw=0.8, ls="--")
    axes[2].set_ylabel("NaOH\nm3/d")
    axes[2].grid(True, alpha=0.25)

    axes[3].plot(df["time"], df[q_hcl_col], color="#CC79A7", lw=1.1)
    axes[3].axhline(model.Q_HCL_MAX, color="red", lw=0.8, ls="--")
    axes[3].set_ylabel("HCl\nm3/d")
    axes[3].set_xlabel("Time (d)")
    axes[3].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logs = []
    schedules = []
    segments = []
    summaries = []

    for target in ("stage1", "stage2"):
        df, schedule = run_target(target)
        seg = segment_metrics(df, schedule, target)
        summary = summary_metrics(df, seg, target)

        logs.append(df)
        schedules.append(schedule)
        segments.append(seg)
        summaries.append(summary)

        suffix = "stage1" if target == "stage1" else "stage2"
        df.to_csv(RESULTS_DIR / f"closedloop_prbs_{suffix}_specific_log.csv", index=False)
        schedule.to_csv(RESULTS_DIR / f"closedloop_prbs_{suffix}_specific_schedule.csv", index=False)
        seg.to_csv(RESULTS_DIR / f"closedloop_prbs_{suffix}_specific_segment_metrics.csv", index=False)
        plot_target(df, summary, target, FIGURES_DIR / f"closedloop_prbs_test_{suffix}_tracking.png")

    pd.concat(logs, ignore_index=True).to_csv(RESULTS_DIR / "closedloop_prbs_stage_specific_logs.csv", index=False)
    pd.concat(schedules, ignore_index=True).to_csv(RESULTS_DIR / "closedloop_prbs_stage_specific_schedules.csv", index=False)
    pd.concat(segments, ignore_index=True).to_csv(RESULTS_DIR / "closedloop_prbs_stage_specific_segment_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(RESULTS_DIR / "closedloop_prbs_stage_specific_summary.csv", index=False)

    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
