from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage_specific_closed_loop_prbs as runner
import search_stage1_split_gain_tuning as stage1_only


BASE_TUNING = copy.deepcopy(runner.model.CONTROL_TUNINGS["stage1_55C"])
BASE_INTERVAL = runner.model.CONTROL_INTERVAL_DAYS
BASE_DEADBAND = copy.deepcopy(runner.CONTROL_DEADBAND_PH)
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def set_stage1_pi(kp_naoh: float, kp_hcl: float, ti_d: float) -> None:
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_NaOH"] = kp_naoh
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"] = kp_naoh / ti_d
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_HCl"] = kp_hcl
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"] = kp_hcl / ti_d


def score(seg: pd.DataFrame, summary: dict) -> dict:
    eligible = seg[(seg["direction"].isin(["up", "down"])) & (seg["duration_d"] >= 1.0)].copy()
    up = eligible[eligible["direction"] == "up"]
    down = eligible[eligible["direction"] == "down"]
    settle = eligible["settle_time_to_0p05_d"]
    settled_1d = settle.le(1.0).fillna(False)
    up_overshoot = float(up["overshoot_pH"].max()) if not up.empty else np.nan
    down_undershoot = float(down["overshoot_pH"].max()) if not down.empty else np.nan
    settle_fraction = float(settled_1d.mean()) if len(eligible) else 0.0
    overshoot_error = abs(up_overshoot - 0.1) if np.isfinite(up_overshoot) else 5.0
    objective = (
        3.0 * (1.0 - settle_fraction)
        + 2.0 * overshoot_error
        + 0.4 * float(summary["overall_MAE_pH"])
        + 0.5 * max(0.0, down_undershoot - 0.15 if np.isfinite(down_undershoot) else 0.0)
    )
    return {
        "eligible_segments_ge_1d": int(len(eligible)),
        "settled_under_1d_fraction_ge_1d": settle_fraction,
        "max_settle_time_ge_1d": float(settle.max()) if np.isfinite(settle.max()) else np.nan,
        "up_max_overshoot_ge_1d": up_overshoot,
        "down_max_undershoot_ge_1d": down_undershoot,
        "objective": objective,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runner.model.CONTROL_INTERVAL_DAYS = 1.0 / 24.0
    runner.CONTROL_DEADBAND_PH["stage1"] = 0.0
    stage1_only.RUN_DAYS = 20.0

    rows = []
    kp_naoh_values = [31.0, 32.0, 33.0]
    kp_hcl_values = [12.0]
    ti_values_d = [4.0, 5.0, 6.0, 8.0, 10.0]

    for kp_naoh in kp_naoh_values:
        for kp_hcl in kp_hcl_values:
            for ti_d in ti_values_d:
                set_stage1_pi(kp_naoh, kp_hcl, ti_d)
                df, schedule = stage1_only.run_stage1_only()
                seg = runner.segment_metrics(df, schedule, "stage1")
                summary = runner.summary_metrics(df, seg, "stage1")
                row = {
                    "control_interval_h": 1.0,
                    "stage1_deadband_pH": runner.CONTROL_DEADBAND_PH["stage1"],
                    "Kp_NaOH": kp_naoh,
                    "Ki_NaOH": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"],
                    "Kp_HCl": kp_hcl,
                    "Ki_HCl": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"],
                    "Ti_d": ti_d,
                    **summary,
                    **score(seg, summary),
                }
                rows.append(row)
                print(row)

    out = pd.DataFrame(rows).sort_values("objective")
    out.to_csv(RESULTS_DIR / "stage1_1h_cation_overshoot_search.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        runner.model.CONTROL_TUNINGS["stage1_55C"].update(BASE_TUNING)
        runner.model.CONTROL_INTERVAL_DAYS = BASE_INTERVAL
        runner.CONTROL_DEADBAND_PH.clear()
        runner.CONTROL_DEADBAND_PH.update(BASE_DEADBAND)
