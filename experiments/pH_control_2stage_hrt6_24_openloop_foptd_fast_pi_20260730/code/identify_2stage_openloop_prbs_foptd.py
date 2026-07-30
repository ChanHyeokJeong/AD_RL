from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SOURCE_DIR = Path(
    "\\\\?\\Z:\\backup\\Chanhyeok Jeong\\Project\\KICHE_AD_control\\"
    "\uc18c\ud654\uc870 \uc81c\uc5b4\\PyADM1-master\\pH_control_2stage"
)
OUT_DIR = Path(
    "C:/Users/JCH/Documents/AD control/ph_control_diagnostics/"
    "hrt_6_24_openloop_prbs_foptd"
)

sys.path.insert(0, str(SOURCE_DIR))
import PyADM1_pH_2stage_PRBS as model  # noqa: E402


WARMUP_DAYS = 60.0
PRBS_DAYS = 150.0


@dataclass(frozen=True)
class Case:
    name: str
    target_stage: str
    chemical: str
    high_q_m3_d: float
    min_dwell_d: float
    max_dwell_d: float
    seed: int

    @property
    def concentration_kmol_m3(self) -> float:
        if self.chemical == "NaOH":
            return model.C_NAOH
        if self.chemical == "HCl":
            return model.C_HCL
        raise ValueError(self.chemical)

    @property
    def expected_gain_sign(self) -> int:
        return 1 if self.chemical == "NaOH" else -1


CASES = {
    "stage1_NaOH": Case("stage1_NaOH", "stage1", "NaOH", 0.15, 1.0, 6.0, 260711),
    "stage1_HCl": Case("stage1_HCl", "stage1", "HCl", 0.35, 1.0, 6.0, 260712),
    "stage2_NaOH": Case("stage2_NaOH", "stage2", "NaOH", 0.60, 4.0, 16.0, 260721),
    "stage2_HCl": Case("stage2_HCl", "stage2", "HCl", 1.35, 4.0, 16.0, 260722),
}


def reset_dosing(ctx: dict) -> None:
    ctx["q_NaOH"] = 0.0
    ctx["u_NaOH"] = 0.0
    ctx["q_HCl"] = 0.0
    ctx["u_HCl"] = 0.0


def set_stage_dose(target: dict, chemical: str, concentration_kmol_m3: float, q_m3_d: float) -> None:
    reset_dosing(target)
    u_kmol_d = q_m3_d * concentration_kmol_m3
    if chemical == "NaOH":
        target["q_NaOH"] = q_m3_d
        target["u_NaOH"] = u_kmol_d
    else:
        target["q_HCl"] = q_m3_d
        target["u_HCl"] = u_kmol_d


def apply_case_dose(stage1: dict, stage2: dict, case: Case, q_m3_d: float) -> None:
    target = stage1 if case.target_stage == "stage1" else stage2
    u_kmol_d = q_m3_d * case.concentration_kmol_m3
    if case.chemical == "NaOH":
        target["q_NaOH"] = q_m3_d
        target["u_NaOH"] = u_kmol_d
    else:
        target["q_HCl"] = q_m3_d
        target["u_HCl"] = u_kmol_d


def build_dose_schedule(case: Case, t_end_d: float) -> pd.DataFrame:
    rng = np.random.default_rng(case.seed)
    rows = []
    t = 0.0
    q = 0.0
    segment = 0
    while t < t_end_d - 1e-12:
        dwell = float(rng.uniform(case.min_dwell_d, case.max_dwell_d))
        end = min(t + dwell, t_end_d)
        rows.append(
            {
                "segment": segment,
                "start_d": t,
                "end_d": end,
                "duration_d": end - t,
                "q_m3_d": q,
                "u_kmol_d": q * case.concentration_kmol_m3,
            }
        )
        t = end
        q = case.high_q_m3_d if q <= 0.0 else 0.0
        segment += 1
    return pd.DataFrame(rows)


def segment_at(schedule: pd.DataFrame, t_day: float, current_idx: int) -> int:
    idx = current_idx
    while idx < len(schedule) - 1 and t_day >= float(schedule.iloc[idx]["end_d"]) - 1e-12:
        idx += 1
    return idx


def fresh_serial_reactors() -> tuple[dict, dict]:
    stage1 = model.fresh_reactor("stage1_55C", model.STAGE1_TEMP_K, model.STAGE1_VOLUME_FRACTION)
    stage2 = model.fresh_reactor("stage2_35C", model.STAGE2_TEMP_K, model.STAGE2_VOLUME_FRACTION)
    reset_dosing(stage1)
    reset_dosing(stage2)
    return stage1, stage2


def simulate_case(case: Case) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    case_dir = OUT_DIR / case.name
    case_dir.mkdir(parents=True, exist_ok=True)

    stage1, stage2 = fresh_serial_reactors()
    schedule = build_dose_schedule(case, PRBS_DAYS)

    source_time = np.asarray(stage1["t"], dtype=float)
    t_end = WARMUP_DAYS + PRBS_DAYS
    time_values = source_time[source_time <= t_end + 1e-12]

    t0 = 0.0
    prbs_idx = 0
    rows = []
    for n, u in enumerate(time_values[1:], start=1):
        u = float(u)
        tstep = [t0, u]

        if t0 < WARMUP_DAYS:
            q_m3_d = 0.0
            segment = -1
        else:
            t_prbs = t0 - WARMUP_DAYS
            prbs_idx = segment_at(schedule, t_prbs, prbs_idx)
            row = schedule.iloc[prbs_idx]
            q_m3_d = float(row["q_m3_d"])
            segment = int(row["segment"])

        model.set_input_from_influent(stage1, n)
        reset_dosing(stage1)
        reset_dosing(stage2)
        if case.target_stage == "stage1":
            set_stage_dose(stage1, case.chemical, case.concentration_kmol_m3, q_m3_d)
        model.run_reactor_step(stage1, tstep)

        model.set_input_from_reactor(stage2, stage1)
        if case.target_stage == "stage2":
            set_stage_dose(stage2, case.chemical, case.concentration_kmol_m3, q_m3_d)
        model.run_reactor_step(stage2, tstep)

        if u >= WARMUP_DAYS - 1e-12:
            dose_target = stage1 if case.target_stage == "stage1" else stage2
            rows.append(
                {
                    "case": case.name,
                    "time_d": u,
                    "time_prbs_d": max(0.0, u - WARMUP_DAYS),
                    "segment": segment,
                    "target_stage": case.target_stage,
                    "chemical": case.chemical,
                    "dose_q_m3_d": q_m3_d,
                    "dose_u_kmol_d": q_m3_d * case.concentration_kmol_m3,
                    "stage1_pH": float(stage1["pH"]),
                    "stage2_pH": float(stage2["pH"]),
                    "target_pH": float(dose_target["pH"]),
                    "stage1_q_NaOH_m3_d": float(stage1["q_NaOH"]),
                    "stage1_u_NaOH_kmol_d": float(stage1["u_NaOH"]),
                    "stage1_q_HCl_m3_d": float(stage1["q_HCl"]),
                    "stage1_u_HCl_kmol_d": float(stage1["u_HCl"]),
                    "stage2_q_NaOH_m3_d": float(stage2["q_NaOH"]),
                    "stage2_u_NaOH_kmol_d": float(stage2["u_NaOH"]),
                    "stage2_q_HCl_m3_d": float(stage2["q_HCl"]),
                    "stage2_u_HCl_kmol_d": float(stage2["u_HCl"]),
                    "stage1_V_liq_m3": float(stage1["V_liq"]),
                    "stage2_V_liq_m3": float(stage2["V_liq"]),
                    "stage1_HRT_d": float(stage1["V_liq"] / stage1["q_ad"]),
                    "stage2_HRT_d": float(stage2["V_liq"] / stage2["q_ad"]),
                }
            )
        t0 = u

    pd.DataFrame(rows).to_csv(case_dir / f"{case.name}_openloop_prbs_log.csv", index=False)
    schedule.to_csv(case_dir / f"{case.name}_dose_schedule.csv", index=False)
    print(f"{case.name}: rows={len(rows)} log={case_dir / f'{case.name}_openloop_prbs_log.csv'}")


def delayed_signal(t: np.ndarray, u: np.ndarray, theta: float) -> np.ndarray:
    return np.interp(t - theta, t, u, left=u[0], right=u[-1])


def first_order_basis(t: np.ndarray, u: np.ndarray, tau: float, theta: float) -> np.ndarray:
    u_del = delayed_signal(t, u, theta)
    x = np.empty_like(t)
    x[0] = u_del[0]
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = np.exp(-dt / tau)
        x[i] = alpha * x[i - 1] + (1.0 - alpha) * u_del[i - 1]
    return x


def fit_single_foptd(t: np.ndarray, y: np.ndarray, u: np.ndarray, expected_sign: int) -> dict:
    dt = float(np.median(np.diff(t)))
    tau_grid = np.unique(
        np.array(
            [
                dt,
                2 * dt,
                4 * dt,
                8 * dt,
                0.25,
                0.5,
                1.0,
                2.0,
                3.0,
                4.0,
                6.0,
                8.0,
                12.0,
                16.0,
                24.0,
                36.0,
                48.0,
                72.0,
            ]
        )
    )
    theta_grid = np.unique(np.array([0.0, dt, 2 * dt, 4 * dt, 8 * dt, 0.25, 0.5, 1.0, 2.0, 4.0]))

    best = None
    for tau in tau_grid:
        for theta in theta_grid:
            x = first_order_basis(t, u, float(tau), float(theta))
            a = np.column_stack([np.ones_like(t), x])
            coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
            y_hat = a @ coeffs
            y0, gain = coeffs
            err = y_hat - y
            rmse = float(np.sqrt(np.mean(err**2)))
            mae = float(np.mean(np.abs(err)))
            penalty = 0.0
            if expected_sign > 0 and gain < 0:
                penalty += 1000.0 * abs(gain)
            if expected_sign < 0 and gain > 0:
                penalty += 1000.0 * abs(gain)
            score = rmse + penalty
            if best is None or score < best["score"]:
                ss_res = float(np.sum(err**2))
                ss_tot = float(np.sum((y - np.mean(y)) ** 2))
                best = {
                    "score": score,
                    "y0": float(y0),
                    "gain_pH_per_kmol_d": float(gain),
                    "tau_d": float(tau),
                    "theta_d": float(theta),
                    "MAE_pH": mae,
                    "RMSE_pH": rmse,
                    "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
                    "y_hat": y_hat,
                }
    return best


def load_case_log(case: Case) -> pd.DataFrame:
    path = OUT_DIR / case.name / f"{case.name}_openloop_prbs_log.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fit_case(case: Case) -> tuple[dict, pd.DataFrame]:
    df = load_case_log(case)
    t = df["time_prbs_d"].to_numpy(dtype=float)
    u = df["dose_u_kmol_d"].to_numpy(dtype=float)
    y = df["target_pH"].to_numpy(dtype=float)
    result = fit_single_foptd(t, y, u, case.expected_gain_sign)
    y_hat = result.pop("y_hat")

    result.update(
        {
            "case": case.name,
            "target_stage": case.target_stage,
            "chemical": case.chemical,
            "high_q_m3_d": case.high_q_m3_d,
            "high_u_kmol_d": case.high_q_m3_d * case.concentration_kmol_m3,
            "gain_pH_per_m3_d": result["gain_pH_per_kmol_d"] * case.concentration_kmol_m3,
            "tau_h": result["tau_d"] * 24.0,
            "theta_h": result["theta_d"] * 24.0,
            "theta_min": result["theta_d"] * 24.0 * 60.0,
            "pH_min": float(y.min()),
            "pH_max": float(y.max()),
            "pH_delta": float(y.max() - y.min()),
        }
    )

    pred = df[["case", "time_prbs_d", "target_stage", "chemical", "dose_q_m3_d", "dose_u_kmol_d", "target_pH"]].copy()
    pred["target_pH_foptd"] = y_hat
    return result, pred


def fit_all_cases() -> None:
    rows = []
    pred_frames = []
    for case in CASES.values():
        result, pred = fit_case(case)
        rows.append(result)
        pred_frames.append(pred)

    params = pd.DataFrame(rows)
    params = params[
        [
            "case",
            "target_stage",
            "chemical",
            "high_q_m3_d",
            "high_u_kmol_d",
            "gain_pH_per_kmol_d",
            "gain_pH_per_m3_d",
            "tau_d",
            "tau_h",
            "theta_d",
            "theta_h",
            "theta_min",
            "MAE_pH",
            "RMSE_pH",
            "R2",
            "pH_min",
            "pH_max",
            "pH_delta",
            "y0",
        ]
    ]
    params.to_csv(OUT_DIR / "openloop_prbs_foptd_params_by_reactor_chemical.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(OUT_DIR / "openloop_prbs_foptd_predictions.csv", index=False)
    plot_fit_grid(params)
    plot_step_responses(params)
    print(params.to_string(index=False))
    print(OUT_DIR / "openloop_prbs_foptd_params_by_reactor_chemical.csv")


def plot_fit_grid(params: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=False)
    for ax, (_, row) in zip(axes, params.iterrows()):
        case = CASES[row["case"]]
        pred = pd.read_csv(OUT_DIR / case.name / f"{case.name}_openloop_prbs_log.csv")
        fit = pd.read_csv(OUT_DIR / "openloop_prbs_foptd_predictions.csv") if (OUT_DIR / "openloop_prbs_foptd_predictions.csv").exists() else None
        if fit is None:
            _, fit_case_pred = fit_case(case)
            fit = fit_case_pred
        else:
            fit = fit[fit["case"] == case.name]
        ax2 = ax.twinx()
        ax.plot(pred["time_prbs_d"], pred["target_pH"], color="#0072B2", lw=1.2, label="PyADM1 pH")
        ax.plot(fit["time_prbs_d"], fit["target_pH_foptd"], color="#D55E00", lw=1.0, label="FOPTD")
        ax2.step(pred["time_prbs_d"], pred["dose_q_m3_d"], where="post", color="#666666", alpha=0.35, lw=0.9, label="dose")
        ax.set_title(
            f"{case.name}: K={row['gain_pH_per_kmol_d']:.4g} pH/(kmol/d), "
            f"tau={row['tau_d']:.3g} d, theta={row['theta_h']:.2g} h"
        )
        ax.set_ylabel("pH")
        ax2.set_ylabel("m3/d")
        ax.grid(True, alpha=0.25)
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=8)
    axes[-1].set_xlabel("PRBS time (d)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "openloop_prbs_foptd_fit_by_case.png", dpi=180)
    plt.close(fig)


def plot_step_responses(params: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {
        "stage1_NaOH": "#009E73",
        "stage1_HCl": "#CC79A7",
        "stage2_NaOH": "#56B4E9",
        "stage2_HCl": "#D55E00",
    }
    t_h = np.linspace(0.0, 7.0 * 24.0, 700)
    for _, row in params.iterrows():
        t_d = t_h / 24.0
        active = t_d >= row["theta_d"]
        y = np.zeros_like(t_d)
        magnitude = row["high_u_kmol_d"]
        y[active] = (
            row["gain_pH_per_kmol_d"]
            * magnitude
            * (1.0 - np.exp(-(t_d[active] - row["theta_d"]) / row["tau_d"]))
        )
        ax.plot(t_h, y, lw=1.6, color=colors[row["case"]], label=f"{row['case']} high-step")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Time after step (h)")
    ax.set_ylabel("Delta pH")
    ax.set_title("Open-loop PRBS identified FOPTD step responses")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "openloop_prbs_foptd_step_responses.png", dpi=180)
    plt.close(fig)


def selected_cases(names: list[str]) -> list[Case]:
    if not names or names == ["all"]:
        return list(CASES.values())
    unknown = [name for name in names if name not in CASES]
    if unknown:
        raise SystemExit(f"unknown cases: {unknown}; available={list(CASES)}")
    return [CASES[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", nargs="*", default=["all"])
    parser.add_argument("--fit-only", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.fit_only:
        for case in selected_cases(args.case):
            simulate_case(case)
    if args.fit_only or args.case == ["all"]:
        fit_all_cases()


if __name__ == "__main__":
    main()
