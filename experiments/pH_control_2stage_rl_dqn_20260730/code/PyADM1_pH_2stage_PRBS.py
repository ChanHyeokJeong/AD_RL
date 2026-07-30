from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_SOURCE = BASE_DIR / "PyADM1_single_stage_reference.py"

STAGE1_TEMP_K = 328.15  # 55 C
STAGE2_TEMP_K = 308.15  # 35 C
STAGE1_VOLUME_FRACTION = 0.2
STAGE2_VOLUME_FRACTION = 0.8
STAGE1_HRT_DAYS = 6.0
STAGE2_HRT_DAYS = 24.0
Q_AD = 178.4674  # m3/d
TOTAL_LIQ_VOLUME = Q_AD * (STAGE1_HRT_DAYS + STAGE2_HRT_DAYS)
GAS_TO_LIQUID_VOLUME_RATIO = 300.0 / 3400.0

C_NAOH = 25.0  # kmol/m3
C_HCL = 11.3  # kmol/m3, approx. 35 wt% HCl

# Aggressive IMC PI tuning from 6 d / 24 d open-loop PRBS FOPTD fits.
# Target: NaOH tuned for about +0.1 pH initial up-step overshoot, HCl about 1 d settling.
# Units: Kp in (m3/d)/pH, Ki in (m3/d)/(pH*d).
CONTROL_TUNINGS = {
    "stage1_55C": {
        "Kp_NaOH": 25.986622,
        "Ki_NaOH": 4.331104,
        "Kp_HCl": 37.858077,
        "Ki_HCl": 6.309680,
    },
    "stage2_35C": {
        "Kp_NaOH": 34.039882,
        "Ki_NaOH": 11.346627,
        "Kp_HCl": 123.104132,
        "Ki_HCl": 7.694008,
    },
}
CONTROL_INTERVAL_DAYS = 3.0 / 24.0
Q_NAOH_MIN = 0.0
Q_NAOH_MAX = 100.0
Q_HCL_MIN = 0.0
Q_HCL_MAX = 100.0

PRBS_SEED = 260714
PRBS_LOW_PH = 7.0
PRBS_HIGH_PH = 8.4
PRBS_MIN_DWELL_D = 0.5
PRBS_MAX_DWELL_D = 3.0

SOLVER = "DOP853"

FEED_STATE_NAMES = [
    "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2",
    "S_ch4", "S_IC", "S_IN", "S_I", "X_xc", "X_ch", "X_pr", "X_li",
    "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
    "S_cation", "S_anion",
]

DEFAULT_INFLUENT_CSV = BASE_DIR / "digester_influent_mean_full.csv"
ACTIVE_INFLUENT_PATH = DEFAULT_INFLUENT_CSV
ACTIVE_INFLUENT_STATE: pd.DataFrame | None = None


def resolve_influent_csv(path: str | Path | None = None) -> Path:
    if path is None or str(path) == "":
        return DEFAULT_INFLUENT_CSV
    csv_path = Path(path)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    return csv_path


def load_influent_csv(path: str | Path | None = None) -> pd.DataFrame:
    global ACTIVE_INFLUENT_PATH, ACTIVE_INFLUENT_STATE
    csv_path = resolve_influent_csv(path)
    if ACTIVE_INFLUENT_STATE is None or csv_path != ACTIVE_INFLUENT_PATH:
        ACTIVE_INFLUENT_PATH = csv_path
        ACTIVE_INFLUENT_STATE = pd.read_csv(csv_path)
    return ACTIVE_INFLUENT_STATE


def active_influent_state() -> pd.DataFrame:
    return load_influent_csv(ACTIVE_INFLUENT_PATH)


def model_prefix() -> str:
    text = MODEL_SOURCE.read_text(encoding="utf-8")
    text = text.replace('print("START")\n\n', "")
    return text.split("\nsimulate_results = ", 1)[0]


MODEL_PREFIX = model_prefix()


def fresh_reactor(label: str, temp_k: float, volume_fraction: float) -> dict:
    ctx: dict = {"__file__": str(MODEL_SOURCE)}
    exec(MODEL_PREFIX, ctx)
    ctx["influent_path"] = str(ACTIVE_INFLUENT_PATH)
    ctx["influent_state"] = active_influent_state()
    ctx["t"] = ctx["influent_state"]["time"]
    ctx["setInfluent"](0)
    ctx["label"] = label
    ctx["solvermethod"] = SOLVER
    ctx["C_NAOH"] = C_NAOH
    ctx["C_HCL"] = C_HCL
    tuning = CONTROL_TUNINGS[label]
    ctx["Kp_NaOH"] = tuning["Kp_NaOH"]
    ctx["Ki_NaOH"] = tuning["Ki_NaOH"]
    ctx["Kp_HCl"] = tuning["Kp_HCl"]
    ctx["Ki_HCl"] = tuning["Ki_HCl"]
    ctx["Kp"] = ctx["Kp_NaOH"]
    ctx["Ki"] = ctx["Ki_NaOH"]
    ctx["q_NaOH_min"] = Q_NAOH_MIN
    ctx["q_NaOH_max"] = Q_NAOH_MAX
    ctx["q_HCl_min"] = Q_HCL_MIN
    ctx["q_HCl_max"] = Q_HCL_MAX
    ctx["q_NaOH"] = 0.0
    ctx["u_NaOH"] = 0.0
    ctx["q_HCl"] = 0.0
    ctx["u_HCl"] = 0.0
    ctx["err_int"] = 0.0
    ctx["err_int_NaOH"] = 0.0
    ctx["err_int_HCl"] = 0.0

    configure_volume(ctx, volume_fraction)
    configure_temperature(ctx, temp_k)
    refresh_state_input(ctx)
    ctx["DAESolve"]()
    sync_equilibrium_derived(ctx)
    update_gas_flow(ctx)
    ctx["state_zero"] = state_vector(ctx)
    return ctx


def configure_volume(ctx: dict, volume_fraction: float) -> None:
    ctx["V_liq"] = TOTAL_LIQ_VOLUME * volume_fraction
    ctx["V_gas"] = ctx["V_liq"] * GAS_TO_LIQUID_VOLUME_RATIO
    ctx["V_ad"] = ctx["V_liq"] + ctx["V_gas"]
    ctx["q_ad"] = Q_AD


def configure_temperature(ctx: dict, temp_k: float) -> None:
    r = ctx["R"]
    t_base = ctx["T_base"]
    ctx["T_ad"] = float(temp_k)
    ctx["T_op"] = ctx["T_ad"]
    t_ad = ctx["T_ad"]
    ctx["K_w"] = 10.0 ** -14.0 * np.exp((55900.0 / (100.0 * r)) * (1.0 / t_base - 1.0 / t_ad))
    ctx["K_a_co2"] = 10.0 ** -6.35 * np.exp((7646.0 / (100.0 * r)) * (1.0 / t_base - 1.0 / t_ad))
    ctx["K_a_IN"] = 10.0 ** -9.25 * np.exp((51965.0 / (100.0 * r)) * (1.0 / t_base - 1.0 / t_ad))
    ctx["p_gas_h2o"] = 0.0313 * np.exp(5290.0 * (1.0 / t_base - 1.0 / t_ad))
    ctx["K_H_co2"] = 0.035 * np.exp((-19410.0 / (100.0 * r)) * (1.0 / t_base - 1.0 / t_ad))
    ctx["K_H_ch4"] = 0.0014 * np.exp((-14240.0 / (100.0 * r)) * (1.0 / t_base - 1.0 / t_ad))
    ctx["K_H_h2"] = 7.8e-4 * np.exp(-4180.0 / (100.0 * r) * (1.0 / t_base - 1.0 / t_ad))


def refresh_state_input(ctx: dict) -> None:
    ctx["state_input"] = [
        ctx["S_su_in"], ctx["S_aa_in"], ctx["S_fa_in"], ctx["S_va_in"],
        ctx["S_bu_in"], ctx["S_pro_in"], ctx["S_ac_in"], ctx["S_h2_in"],
        ctx["S_ch4_in"], ctx["S_IC_in"], ctx["S_IN_in"], ctx["S_I_in"],
        ctx["X_xc_in"], ctx["X_ch_in"], ctx["X_pr_in"], ctx["X_li_in"],
        ctx["X_su_in"], ctx["X_aa_in"], ctx["X_fa_in"], ctx["X_c4_in"],
        ctx["X_pro_in"], ctx["X_ac_in"], ctx["X_h2_in"], ctx["X_I_in"],
        ctx["S_cation_in"], ctx["S_anion_in"],
    ]


def set_input_from_influent(ctx: dict, index: int) -> None:
    ctx["setInfluent"](index)
    refresh_state_input(ctx)


def set_input_from_reactor(ctx: dict, upstream: dict) -> None:
    for name in FEED_STATE_NAMES:
        ctx[f"{name}_in"] = float(upstream[name])
    refresh_state_input(ctx)


def state_vector(ctx: dict) -> list[float]:
    return [float(ctx[name]) for name in ctx["state_columns"]]


def assign_state(ctx: dict, values: np.ndarray) -> None:
    for name, value in zip(ctx["state_columns"], values):
        ctx[name] = float(value)


def sync_equilibrium_derived(ctx: dict) -> None:
    ctx["S_nh4_ion"] = ctx["S_IN"] - ctx["S_nh3"]
    ctx["S_co2"] = ctx["S_IC"] - ctx["S_hco3_ion"]


def update_gas_flow(ctx: dict) -> None:
    ctx["p_gas_h2"] = ctx["S_gas_h2"] * ctx["R"] * ctx["T_ad"] / 16.0
    ctx["p_gas_ch4"] = ctx["S_gas_ch4"] * ctx["R"] * ctx["T_ad"] / 64.0
    ctx["p_gas_co2"] = ctx["S_gas_co2"] * ctx["R"] * ctx["T_ad"]
    ctx["p_gas"] = ctx["p_gas_h2"] + ctx["p_gas_ch4"] + ctx["p_gas_co2"] + ctx["p_gas_h2o"]
    ctx["q_gas"] = ctx["k_p"] * (ctx["p_gas"] - ctx["p_atm"])
    if ctx["q_gas"] < 0.0:
        ctx["q_gas"] = 0.0
    ctx["q_ch4"] = ctx["q_gas"] * (ctx["p_gas_ch4"] / ctx["p_gas"]) if ctx["p_gas"] > 0.0 else 0.0
    if ctx["q_ch4"] < 0.0:
        ctx["q_ch4"] = 0.0


def run_reactor_step(ctx: dict, tstep: list[float]) -> None:
    ctx["state_zero"] = state_vector(ctx)
    sim = ctx["simulate"](tstep, ctx["solvermethod"])
    assign_state(ctx, np.asarray(sim[:, -1], dtype=float))
    ctx["DAESolve"]()
    sync_equilibrium_derived(ctx)
    update_gas_flow(ctx)
    ctx["state_zero"] = state_vector(ctx)


def state_row(ctx: dict) -> dict:
    return {name: float(ctx[name]) for name in ctx["state_columns"]}


def build_prbs_schedule(t_end_d: float) -> pd.DataFrame:
    rng = np.random.default_rng(PRBS_SEED)
    rows = []
    t = 0.0
    level = PRBS_LOW_PH
    segment = 0
    while t < t_end_d - 1e-12:
        dwell = float(rng.uniform(PRBS_MIN_DWELL_D, PRBS_MAX_DWELL_D))
        end = min(t + dwell, t_end_d)
        rows.append(
            {
                "segment": segment,
                "start_d": t,
                "end_d": end,
                "duration_d": end - t,
                "pH_sp": level,
            }
        )
        t = end
        level = PRBS_HIGH_PH if level == PRBS_LOW_PH else PRBS_LOW_PH
        segment += 1
    return pd.DataFrame(rows)

def prbs_segment_at(schedule: pd.DataFrame, t_day: float, current_idx: int) -> int:
    idx = current_idx
    while idx < len(schedule) - 1 and t_day >= float(schedule.iloc[idx]["end_d"]) - 1e-12:
        idx += 1
    return idx

def main() -> None:
    print("START pH control serial 2-reactor PRBS")
    stage1 = fresh_reactor("stage1_55C", STAGE1_TEMP_K, STAGE1_VOLUME_FRACTION)
    stage2 = fresh_reactor("stage2_35C", STAGE2_TEMP_K, STAGE2_VOLUME_FRACTION)

    stage1["u_NaOH"] = 0.0
    stage1["q_NaOH"] = 0.0
    stage1["u_HCl"] = 0.0
    stage1["q_HCl"] = 0.0
    stage2["u_NaOH"] = 0.0
    stage2["q_NaOH"] = 0.0
    stage2["u_HCl"] = 0.0
    stage2["q_HCl"] = 0.0

    time_values = np.asarray(stage1["t"], dtype=float)
    prbs_schedule = build_prbs_schedule(float(time_values[-1]))
    prbs_idx = 0
    stage2["pH_sp"] = float(prbs_schedule.iloc[prbs_idx]["pH_sp"])
    t0 = 0.0
    control_timer = 0.0

    stage1_rows = [dict(time=0.0, **state_row(stage1))]
    stage2_rows = [dict(time=0.0, **state_row(stage2))]
    gas_rows = [
        {
            "time": 0.0,
            "stage1_q_gas": float(stage1["q_gas"]),
            "stage1_q_ch4": float(stage1["q_ch4"]),
            "stage2_q_gas": float(stage2["q_gas"]),
            "stage2_q_ch4": float(stage2["q_ch4"]),
            "total_q_gas": float(stage1["q_gas"] + stage2["q_gas"]),
            "total_q_ch4": float(stage1["q_ch4"] + stage2["q_ch4"]),
        }
    ]
    control_rows = [
        {
            "time": 0.0,
            "pH_sp": float(stage2["pH_sp"]),
            "stage1_pH": float(stage1["pH"]),
            "stage2_pH": float(stage2["pH"]),
            "q_NaOH_m3_d": float(stage2["q_NaOH"]),
            "u_NaOH_kmol_d": float(stage2["u_NaOH"]),
            "q_HCl_m3_d": float(stage2["q_HCl"]),
            "u_HCl_kmol_d": float(stage2["u_HCl"]),
            "stage1_T_K": STAGE1_TEMP_K,
            "stage2_T_K": STAGE2_TEMP_K,
            "stage1_V_liq_m3": float(stage1["V_liq"]),
            "stage2_V_liq_m3": float(stage2["V_liq"]),
            "stage1_HRT_d": float(stage1["V_liq"] / stage1["q_ad"]),
            "stage2_HRT_d": float(stage2["V_liq"] / stage2["q_ad"]),
            "prbs_segment": int(prbs_idx),
        }
    ]

    for n, u in enumerate(time_values[1:], start=1):
        u = float(u)
        tstep = [t0, u]

        set_input_from_influent(stage1, n)
        stage1["u_NaOH"] = 0.0
        stage1["q_NaOH"] = 0.0
        stage1["u_HCl"] = 0.0
        stage1["q_HCl"] = 0.0
        run_reactor_step(stage1, tstep)

        set_input_from_reactor(stage2, stage1)
        run_reactor_step(stage2, tstep)

        dt = u - t0
        prbs_idx = prbs_segment_at(prbs_schedule, u, prbs_idx)
        stage2["pH_sp"] = float(prbs_schedule.iloc[prbs_idx]["pH_sp"])
        control_timer += dt
        if control_timer >= CONTROL_INTERVAL_DAYS - 1e-12:
            stage2["PI_pH_controller"](stage2["pH_sp"], stage2["pH"], control_timer)
            control_timer = 0.0

        stage1_rows.append(dict(time=u, **state_row(stage1)))
        stage2_rows.append(dict(time=u, **state_row(stage2)))
        gas_rows.append(
            {
                "time": u,
                "stage1_q_gas": float(stage1["q_gas"]),
                "stage1_q_ch4": float(stage1["q_ch4"]),
                "stage2_q_gas": float(stage2["q_gas"]),
                "stage2_q_ch4": float(stage2["q_ch4"]),
                "total_q_gas": float(stage1["q_gas"] + stage2["q_gas"]),
                "total_q_ch4": float(stage1["q_ch4"] + stage2["q_ch4"]),
            }
        )
        control_rows.append(
            {
                "time": u,
                "pH_sp": float(stage2["pH_sp"]),
                "stage1_pH": float(stage1["pH"]),
                "stage2_pH": float(stage2["pH"]),
                "q_NaOH_m3_d": float(stage2["q_NaOH"]),
                "u_NaOH_kmol_d": float(stage2["u_NaOH"]),
            "q_HCl_m3_d": float(stage2["q_HCl"]),
            "u_HCl_kmol_d": float(stage2["u_HCl"]),
                "stage1_T_K": STAGE1_TEMP_K,
                "stage2_T_K": STAGE2_TEMP_K,
                "stage1_V_liq_m3": float(stage1["V_liq"]),
                "stage2_V_liq_m3": float(stage2["V_liq"]),
                "stage1_HRT_d": float(stage1["V_liq"] / stage1["q_ad"]),
                "stage2_HRT_d": float(stage2["V_liq"] / stage2["q_ad"]),
            }
        )
        t0 = u

    out_stage1 = BASE_DIR / "dynamic_out_stage1_55C_PRBS.csv"
    out_stage2 = BASE_DIR / "dynamic_out_stage2_35C_PRBS.csv"
    out_gas = BASE_DIR / "gasflow_2stage_serial_PRBS.csv"
    out_ctl = BASE_DIR / "acid_base_control_log_2stage_serial_PRBS.csv"
    out_schedule = BASE_DIR / "pH_PRBS_schedule_2stage_serial.csv"

    pd.DataFrame(stage1_rows).to_csv(out_stage1, index=False)
    pd.DataFrame(stage2_rows).to_csv(out_stage2, index=False)
    pd.DataFrame(gas_rows).to_csv(out_gas, index=False)
    pd.DataFrame(control_rows).to_csv(out_ctl, index=False)
    prbs_schedule.to_csv(out_schedule, index=False)

    print("END")
    print(out_stage1)
    print(out_stage2)
    print(out_gas)
    print(out_ctl)
    print(out_schedule)


if __name__ == "__main__":
    main()
