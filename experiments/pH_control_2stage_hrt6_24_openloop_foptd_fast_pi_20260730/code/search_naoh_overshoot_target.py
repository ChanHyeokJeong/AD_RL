from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


SOURCE_DIR = Path(
    "\\\\?\\Z:\\backup\\Chanhyeok Jeong\\Project\\KICHE_AD_control\\"
    "\uc18c\ud654\uc870 \uc81c\uc5b4\\PyADM1-master\\pH_control_2stage"
)
OUT_DIR = Path("C:/Users/JCH/Documents/AD control/ph_control_diagnostics/naoh_overshoot_target_search")

sys.path.insert(0, str(SOURCE_DIR))
import PyADM1_pH_2stage_PRBS as model  # noqa: E402


BASE_NAOH = {
    "stage1_55C": {"Kp_NaOH": 16.241639, "Ki_NaOH": 2.706940},
    "stage2_35C": {"Kp_NaOH": 21.274926, "Ki_NaOH": 7.091642},
}

HCL_CURRENT = {
    "stage1_55C": {"Kp_HCl": 37.858077, "Ki_HCl": 6.309680},
    "stage2_35C": {"Kp_HCl": 123.104132, "Ki_HCl": 7.694008},
}


def apply_naoh_multiplier(multiplier: float) -> None:
    for label in ("stage1_55C", "stage2_35C"):
        model.CONTROL_TUNINGS[label]["Kp_NaOH"] = BASE_NAOH[label]["Kp_NaOH"] * multiplier
        model.CONTROL_TUNINGS[label]["Ki_NaOH"] = BASE_NAOH[label]["Ki_NaOH"] * multiplier
        model.CONTROL_TUNINGS[label]["Kp_HCl"] = HCL_CURRENT[label]["Kp_HCl"]
        model.CONTROL_TUNINGS[label]["Ki_HCl"] = HCL_CURRENT[label]["Ki_HCl"]


def reset_dosing(ctx: dict) -> None:
    ctx["u_NaOH"] = 0.0
    ctx["q_NaOH"] = 0.0
    ctx["u_HCl"] = 0.0
    ctx["q_HCl"] = 0.0
    ctx["err_int"] = 0.0
    ctx["err_int_NaOH"] = 0.0
    ctx["err_int_HCl"] = 0.0


def segment_metrics(df: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prev_sp = None
    for _, seg in schedule.iterrows():
        start = float(seg["start_d"])
        end = float(seg["end_d"])
        sp = float(seg["pH_sp"])
        idx = int(seg["segment"])
        if idx < int(schedule["segment"].iloc[-1]):
            part = df[(df["time"] >= start) & (df["time"] < end)].copy()
        else:
            part = df[(df["time"] >= start) & (df["time"] <= end)].copy()
        if part.empty:
            continue
        pH = part["stage2_pH"]
        err = sp - pH
        abs_err = err.abs()
        if prev_sp is None:
            direction = "initial"
            overshoot = np.nan
        elif sp > prev_sp:
            direction = "up"
            overshoot = max(0.0, float((pH - sp).max()))
        elif sp < prev_sp:
            direction = "down"
            overshoot = max(0.0, float((sp - pH).max()))
        else:
            direction = "hold"
            overshoot = np.nan
        rows.append(
            {
                "segment": idx,
                "start_d": start,
                "end_d": end,
                "duration_d": end - start,
                "pH_sp": sp,
                "direction": direction,
                "overshoot_pH": overshoot,
                "MAE_pH": float(abs_err.mean()),
                "q_NaOH_max_m3_d": float(part["q_NaOH_m3_d"].max()),
                "q_HCl_max_m3_d": float(part["q_HCl_m3_d"].max()),
            }
        )
        prev_sp = sp
    return pd.DataFrame(rows)


def run_prbs(multiplier: float, t_end_d: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    apply_naoh_multiplier(multiplier)
    stage1 = model.fresh_reactor("stage1_55C", model.STAGE1_TEMP_K, model.STAGE1_VOLUME_FRACTION)
    stage2 = model.fresh_reactor("stage2_35C", model.STAGE2_TEMP_K, model.STAGE2_VOLUME_FRACTION)
    reset_dosing(stage1)
    reset_dosing(stage2)

    time_values = np.asarray(stage1["t"], dtype=float)
    time_values = time_values[time_values <= t_end_d + 1e-12]
    schedule = model.build_prbs_schedule(float(time_values[-1]))
    prbs_idx = 0
    stage2["pH_sp"] = float(schedule.iloc[prbs_idx]["pH_sp"])
    control_timer = 0.0
    t0 = 0.0
    rows = [
        {
            "time": 0.0,
            "pH_sp": float(stage2["pH_sp"]),
            "stage1_pH": float(stage1["pH"]),
            "stage2_pH": float(stage2["pH"]),
            "q_NaOH_m3_d": float(stage2["q_NaOH"]),
            "q_HCl_m3_d": float(stage2["q_HCl"]),
        }
    ]

    for n, u in enumerate(time_values[1:], start=1):
        u = float(u)
        tstep = [t0, u]

        model.set_input_from_influent(stage1, n)
        stage1["u_NaOH"] = 0.0
        stage1["q_NaOH"] = 0.0
        stage1["u_HCl"] = 0.0
        stage1["q_HCl"] = 0.0
        model.run_reactor_step(stage1, tstep)

        model.set_input_from_reactor(stage2, stage1)
        model.run_reactor_step(stage2, tstep)

        dt = u - t0
        prbs_idx = model.prbs_segment_at(schedule, u, prbs_idx)
        stage2["pH_sp"] = float(schedule.iloc[prbs_idx]["pH_sp"])
        control_timer += dt
        if control_timer >= model.CONTROL_INTERVAL_DAYS - 1e-12:
            stage2["PI_pH_controller"](stage2["pH_sp"], stage2["pH"], control_timer)
            control_timer = 0.0

        rows.append(
            {
                "time": u,
                "pH_sp": float(stage2["pH_sp"]),
                "stage1_pH": float(stage1["pH"]),
                "stage2_pH": float(stage2["pH"]),
                "q_NaOH_m3_d": float(stage2["q_NaOH"]),
                "q_HCl_m3_d": float(stage2["q_HCl"]),
            }
        )
        t0 = u

    return pd.DataFrame(rows), schedule


def summarize_candidate(multiplier: float, t_end_d: float) -> dict:
    df, schedule = run_prbs(multiplier, t_end_d)
    seg = segment_metrics(df, schedule)
    up = seg[seg["direction"] == "up"]
    down = seg[seg["direction"] == "down"]
    err = df["pH_sp"] - df["stage2_pH"]
    row = {
        "NaOH_multiplier": multiplier,
        "t_end_d": t_end_d,
        "overall_MAE_pH": float(err.abs().mean()),
        "high_sp_MAE_pH": float((df.loc[df["pH_sp"] > df["pH_sp"].median(), "pH_sp"] - df.loc[df["pH_sp"] > df["pH_sp"].median(), "stage2_pH"]).abs().mean()),
        "up_max_overshoot_pH": float(up["overshoot_pH"].max()),
        "up_mean_overshoot_pH": float(up["overshoot_pH"].mean()),
        "down_max_undershoot_pH": float(down["overshoot_pH"].max()),
        "q_NaOH_max_m3_d": float(df["q_NaOH_m3_d"].max()),
        "NaOH_sat_fraction": float((df["q_NaOH_m3_d"] >= 100.0 - 1e-6).mean()),
        "q_HCl_max_m3_d": float(df["q_HCl_m3_d"].max()),
        "HCl_sat_fraction": float((df["q_HCl_m3_d"] >= 100.0 - 1e-6).mean()),
    }
    case_dir = OUT_DIR / f"naoh_x{multiplier:.2f}".replace(".", "p")
    case_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(case_dir / "prbs_log.csv", index=False)
    seg.to_csv(case_dir / "segment_metrics.csv", index=False)
    return row


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [1.35, 1.45, 1.55, 1.65, 1.75]
    rows = []
    for multiplier in candidates:
        print(f"RUN multiplier={multiplier}")
        rows.append(summarize_candidate(multiplier, 80.0))
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "naoh_multiplier_search_80d.csv", index=False)
    print(result.to_string(index=False))
    print(OUT_DIR / "naoh_multiplier_search_80d.csv")


if __name__ == "__main__":
    main()
