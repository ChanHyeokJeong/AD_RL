from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage_specific_closed_loop_prbs as runner
from search_stage1_split_gain_tuning import run_stage1_only


BASE_TUNING = copy.deepcopy(runner.model.CONTROL_TUNINGS["stage1_55C"])
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def set_gains(kp_naoh: float, kp_hcl: float, ti_naoh: float | None, ti_hcl: float | None) -> None:
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_NaOH"] = kp_naoh
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Kp_HCl"] = kp_hcl
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"] = 0.0 if ti_naoh is None else kp_naoh / ti_naoh
    runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"] = 0.0 if ti_hcl is None else kp_hcl / ti_hcl


def score(seg: pd.DataFrame, summary: dict) -> dict:
    eligible = seg[(seg["direction"].isin(["up", "down"])) & (seg["duration_d"] >= 1.0)].copy()
    settle = eligible["settle_time_to_0p05_d"]
    settled_1d = settle.le(1.0).fillna(False)
    max_overshoot = float(eligible["overshoot_pH"].max()) if len(eligible) else np.nan
    max_settle = float(settle.max()) if np.isfinite(settle.max()) else np.nan
    fraction = float(settled_1d.mean()) if len(eligible) else 0.0
    objective = (
        float(summary["overall_MAE_pH"])
        + 4.0 * (1.0 - fraction)
        + 0.5 * max(0.0, max_overshoot - 0.15 if np.isfinite(max_overshoot) else 5.0)
        + 2.0 * float(summary["NaOH_sat_fraction"] + summary["HCl_sat_fraction"])
    )
    return {
        "eligible_segments_ge_1d": int(len(eligible)),
        "settled_under_1d_fraction_ge_1d": fraction,
        "max_settle_time_ge_1d": max_settle,
        "max_overshoot_ge_1d": max_overshoot,
        "objective": objective,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    kp_naoh_values = [8.0, 12.0, 16.0]
    kp_hcl_values = [4.0, 8.0, 12.0]
    ti_options = [None, 12.0]

    for kp_naoh in kp_naoh_values:
        for kp_hcl in kp_hcl_values:
            for ti in ti_options:
                set_gains(kp_naoh, kp_hcl, ti, ti)
                df, schedule = run_stage1_only()
                seg = runner.segment_metrics(df, schedule, "stage1")
                summary = runner.summary_metrics(df, seg, "stage1")
                row = {
                    "Kp_NaOH": kp_naoh,
                    "Ki_NaOH": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_NaOH"],
                    "Kp_HCl": kp_hcl,
                    "Ki_HCl": runner.model.CONTROL_TUNINGS["stage1_55C"]["Ki_HCl"],
                    "Ti_d": np.nan if ti is None else ti,
                    **summary,
                    **score(seg, summary),
                }
                rows.append(row)
                print(row)

    out = pd.DataFrame(rows).sort_values("objective")
    out.to_csv(RESULTS_DIR / "stage1_low_integral_tuning_search.csv", index=False)
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        runner.model.CONTROL_TUNINGS["stage1_55C"].update(BASE_TUNING)
