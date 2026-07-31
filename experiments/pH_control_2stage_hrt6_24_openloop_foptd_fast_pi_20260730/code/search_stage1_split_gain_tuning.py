from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage_specific_closed_loop_prbs as runner


BASE_TUNING = copy.deepcopy(runner.model.CONTROL_TUNINGS["stage1_55C"])
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RUN_DAYS = 20.0


def set_stage1_gains(naoh_multiplier: float, hcl_multiplier: float) -> None:
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_NaOH"] = BASE_TUNING["Kp_NaOH"] * naoh_multiplier
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"] = BASE_TUNING["Ki_NaOH"] * naoh_multiplier
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_HCl"] = BASE_TUNING["Kp_HCl"] * hcl_multiplier
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"] = BASE_TUNING["Ki_HCl"] * hcl_multiplier


def stage1_row(time_d: float, segment: int, pH_sp: float, stage1: dict) -> dict:
    return {
        "target": "stage1",
        "time": time_d,
        "prbs_segment": segment,
        "stage1_pH_sp": pH_sp,
        "stage1_pH": float(stage1["pH"]),
        "stage1_q_NaOH_m3_d": float(stage1["q_NaOH"]),
        "stage1_u_NaOH_kmol_d": float(stage1["u_NaOH"]),
        "stage1_q_HCl_m3_d": float(stage1["q_HCl"]),
        "stage1_u_HCl_kmol_d": float(stage1["u_HCl"]),
    }


def run_stage1_only() -> tuple[pd.DataFrame, pd.DataFrame]:
    stage1 = runner.model.fresh_reactor("stage1_55C", runner.model.STAGE1_TEMP_K, runner.model.STAGE1_VOLUME_FRACTION)
    for key in ("u_NaOH", "q_NaOH", "u_HCl", "q_HCl", "err_int", "err_int_NaOH", "err_int_HCl"):
        stage1[key] = 0.0

    time_values = np.asarray(stage1["t"], dtype=float)
    time_values = time_values[time_values <= RUN_DAYS + 1e-12]
    schedule = runner.model.build_prbs_schedule(float(time_values[-1]))
    prbs_idx = 0
    pH_sp = float(schedule.iloc[prbs_idx]["pH_sp"])
    rows = [stage1_row(0.0, prbs_idx, pH_sp, stage1)]
    t0 = 0.0
    control_timer = 0.0

    for n, u in enumerate(time_values[1:], start=1):
        u = float(u)
        tstep = [t0, u]
        runner.model.set_input_from_influent(stage1, n)
        runner.model.run_reactor_step(stage1, tstep)

        dt = u - t0
        prbs_idx = runner.model.prbs_segment_at(schedule, u, prbs_idx)
        pH_sp = float(schedule.iloc[prbs_idx]["pH_sp"])
        control_timer += dt
        if control_timer >= runner.model.CONTROL_INTERVAL_DAYS - 1e-12:
            runner.apply_pi_with_deadband(stage1, "stage1", pH_sp, control_timer)
            control_timer = 0.0

        rows.append(stage1_row(u, prbs_idx, pH_sp, stage1))
        t0 = u

    return pd.DataFrame(rows), schedule.assign(target="stage1")


def objective(seg: pd.DataFrame, summary: dict) -> dict:
    eligible = seg[(seg["direction"].isin(["up", "down"])) & (seg["duration_d"] >= 1.0)].copy()
    settle = eligible["settle_time_to_0p05_d"]
    settled_1d = settle.le(1.0).fillna(False)
    max_overshoot = float(eligible["overshoot_pH"].max()) if len(eligible) else np.nan
    max_settle = float(settle.max()) if np.isfinite(settle.max()) else np.nan
    fraction = float(settled_1d.mean()) if len(eligible) else 0.0
    score = (
        float(summary["overall_MAE_pH"])
        + 3.0 * (1.0 - fraction)
        + 0.4 * max(0.0, max_overshoot - 0.15 if np.isfinite(max_overshoot) else 5.0)
        + 2.0 * float(summary["NaOH_sat_fraction"] + summary["HCl_sat_fraction"])
    )
    return {
        "eligible_segments_ge_1d": int(len(eligible)),
        "settled_under_1d_fraction_ge_1d": fraction,
        "max_settle_time_ge_1d": max_settle,
        "max_overshoot_ge_1d": max_overshoot,
        "objective": score,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    naoh_multipliers = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40]
    hcl_multipliers = [0.08, 0.12, 0.16, 0.22, 0.30, 0.40]

    for naoh_multiplier in naoh_multipliers:
        for hcl_multiplier in hcl_multipliers:
            set_stage1_gains(naoh_multiplier, hcl_multiplier)
            df, schedule = run_stage1_only()
            seg = runner.segment_metrics(df, schedule, "stage1")
            summary = runner.summary_metrics(df, seg, "stage1")
            scored = objective(seg, summary)
            row = {
                "naoh_multiplier": naoh_multiplier,
                "hcl_multiplier": hcl_multiplier,
                "Kp_NaOH": runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_NaOH"],
                "Ki_NaOH": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"],
                "Kp_HCl": runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_HCl"],
                "Ki_HCl": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"],
                **summary,
                **scored,
            }
            rows.append(row)
            print(row)

    out = pd.DataFrame(rows).sort_values("objective")
    out.to_csv(RESULTS_DIR / "stage1_split_gain_tuning_search.csv", index=False)
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        runner.model.CONTROL_TUNINGS["stage1_55C"].update(BASE_TUNING)
