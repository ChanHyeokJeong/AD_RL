from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

import run_stage_specific_closed_loop_prbs as runner


BASE_TUNING = copy.deepcopy(runner.model.CONTROL_TUNINGS["stage1_55C"])
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def apply_stage1_multiplier(multiplier: float) -> None:
    for key, value in BASE_TUNING.items():
        runner.model.CONTROL_TUNINGS["stage1_55C"][key] = value * multiplier


def score_candidate(seg: pd.DataFrame, summary: dict) -> dict:
    eligible = seg[(seg["direction"].isin(["up", "down"])) & (seg["duration_d"] >= 1.0)].copy()
    settled_under_1d = eligible["settle_time_to_0p05_d"].le(1.0)
    max_settle = eligible["settle_time_to_0p05_d"].max()
    max_overshoot = eligible["overshoot_pH"].max()
    unsettled_penalty = float((~settled_under_1d.fillna(False)).mean()) if len(eligible) else 1.0
    overshoot_penalty = max(0.0, float(max_overshoot) - 0.15) if np.isfinite(max_overshoot) else 10.0
    saturation_penalty = float(summary["NaOH_sat_fraction"] + summary["HCl_sat_fraction"])
    objective = (
        float(summary["overall_MAE_pH"])
        + 2.0 * unsettled_penalty
        + 0.5 * overshoot_penalty
        + 2.0 * saturation_penalty
    )
    return {
        "eligible_segments_ge_1d": int(len(eligible)),
        "settled_under_1d_fraction_ge_1d": float(settled_under_1d.mean()) if len(eligible) else np.nan,
        "max_settle_time_ge_1d": float(max_settle) if np.isfinite(max_settle) else np.nan,
        "max_overshoot_ge_1d": float(max_overshoot) if np.isfinite(max_overshoot) else np.nan,
        "objective": objective,
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    runner.RUN_DAYS = 20.0
    rows = []

    for multiplier in [0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.32, 0.40, 0.50, 0.65, 0.80, 1.00]:
        apply_stage1_multiplier(multiplier)
        df, schedule = runner.run_target("stage1")
        seg = runner.segment_metrics(df, schedule, "stage1")
        summary = runner.summary_metrics(df, seg, "stage1")
        scored = score_candidate(seg, summary)
        rows.append({"multiplier": multiplier, **summary, **scored})
        print(rows[-1])

    result = pd.DataFrame(rows).sort_values("objective")
    result.to_csv(RESULTS_DIR / "stage1_closed_loop_tuning_search.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    finally:
        runner.model.CONTROL_TUNINGS["stage1_55C"].update(BASE_TUNING)
