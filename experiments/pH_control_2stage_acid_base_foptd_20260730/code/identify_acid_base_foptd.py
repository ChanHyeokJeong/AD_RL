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
LOG = BASE / "acid_base_control_log_2stage_serial_PRBS.csv"
OUT = Path("C:/Users/JCH/Documents/AD control/ph_control_diagnostics")

C_NAOH = 25.0
C_HCL = 11.3


def delayed_signal(t, u, theta):
    return np.interp(t - theta, t, u, left=u[0], right=u[-1])


def first_order_state(t, u, k, tau, theta, x0=0.0):
    u_del = delayed_signal(t, u, theta)
    x = np.empty_like(t)
    x[0] = x0
    for i in range(1, len(t)):
        dt = t[i] - t[i - 1]
        alpha = np.exp(-dt / tau)
        x[i] = alpha * x[i - 1] + (1.0 - alpha) * k * u_del[i - 1]
    return x


def predict(params, t, u_base, u_acid):
    y0, k_base, tau_base, theta_base, x0_base, k_acid, tau_acid, theta_acid, x0_acid = params
    x_base = first_order_state(t, u_base, k_base, tau_base, theta_base, x0_base)
    x_acid = first_order_state(t, u_acid, k_acid, tau_acid, theta_acid, x0_acid)
    return y0 + x_base + x_acid


def solve_linear_coefficients(t, y, xb, xa, tau_base, tau_acid):
    decay_base = np.exp(-t / tau_base)
    decay_acid = np.exp(-t / tau_acid)
    a = np.column_stack((np.ones_like(y), xb, xa, decay_base, decay_acid))
    coeffs, *_ = np.linalg.lstsq(a, y, rcond=None)
    y_hat = a @ coeffs
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    return coeffs, rmse


def make_basis_grid(t, u, tau_grid, theta_grid):
    basis = {}
    for tau in tau_grid:
        for theta in theta_grid:
            basis[(float(tau), float(theta))] = first_order_state(t, u, 1.0, float(tau), float(theta))
    return basis


def candidate_score(y, xb, xa):
    raise RuntimeError("candidate_score requires time constants")


def candidate_score_with_tau(t, y, xb, xa, tau_base, tau_acid):
    coeffs, rmse = solve_linear_coefficients(t, y, xb, xa, tau_base, tau_acid)
    y0, k_base, k_acid, x0_base, x0_acid = coeffs
    penalty = 0.0
    if not (6.0 <= y0 <= 8.8):
        penalty += 10.0 + abs(y0 - 7.2)
    if k_base < 0.0:
        penalty += 10.0 + abs(k_base) * 100.0
    if k_acid > 0.0:
        penalty += 10.0 + abs(k_acid) * 100.0
    if abs(x0_base) > 5.0:
        penalty += abs(x0_base)
    if abs(x0_acid) > 5.0:
        penalty += abs(x0_acid)
    return rmse + penalty, rmse, coeffs


def fit_model(t, y, u_base, u_acid):
    sample = slice(None, None, 4)
    tg = t[sample]
    yg = y[sample]
    ubg = u_base[sample]
    uag = u_acid[sample]

    tau_grid = np.array(
        [
            1.0 / 96.0,
            2.0 / 96.0,
            4.0 / 96.0,
            6.0 / 96.0,
            8.0 / 96.0,
            12.0 / 96.0,
            18.0 / 96.0,
            24.0 / 96.0,
            36.0 / 96.0,
            48.0 / 96.0,
            72.0 / 96.0,
            96.0 / 96.0,
            144.0 / 96.0,
            192.0 / 96.0,
        ]
    )
    theta_grid = np.array([0.0, 1.0 / 96.0, 2.0 / 96.0, 4.0 / 96.0, 8.0 / 96.0, 12.0 / 96.0, 24.0 / 96.0, 48.0 / 96.0])

    base_basis = make_basis_grid(tg, ubg, tau_grid, theta_grid)
    acid_basis = make_basis_grid(tg, uag, tau_grid, theta_grid)

    best = None
    for (tau_b, theta_b), xb in base_basis.items():
        for (tau_a, theta_a), xa in acid_basis.items():
            score, rmse, coeffs = candidate_score_with_tau(tg, yg, xb, xa, tau_b, tau_a)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "grid_rmse": rmse,
                    "coeffs": coeffs,
                    "tau_base": tau_b,
                    "theta_base": theta_b,
                    "tau_acid": tau_a,
                    "theta_acid": theta_a,
                }

    dt = float(np.median(np.diff(t)))
    tau_base_candidates = np.unique(
        np.clip(best["tau_base"] * np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.4, 2.0]), dt, 10.0)
    )
    tau_acid_candidates = np.unique(
        np.clip(best["tau_acid"] * np.array([0.5, 0.7, 0.85, 1.0, 1.15, 1.4, 2.0]), dt, 10.0)
    )
    theta_base_candidates = np.unique(np.clip(best["theta_base"] + dt * np.array([-2, -1, 0, 1, 2]), 0.0, 1.0))
    theta_acid_candidates = np.unique(np.clip(best["theta_acid"] + dt * np.array([-2, -1, 0, 1, 2]), 0.0, 1.0))

    base_basis_full = make_basis_grid(t, u_base, tau_base_candidates, theta_base_candidates)
    acid_basis_full = make_basis_grid(t, u_acid, tau_acid_candidates, theta_acid_candidates)

    refined = None
    for (tau_b, theta_b), xb in base_basis_full.items():
        for (tau_a, theta_a), xa in acid_basis_full.items():
            score, rmse, coeffs = candidate_score_with_tau(t, y, xb, xa, tau_b, tau_a)
            if refined is None or score < refined["score"]:
                refined = {
                    "score": score,
                    "rmse": rmse,
                    "coeffs": coeffs,
                    "tau_base": tau_b,
                    "theta_base": theta_b,
                    "tau_acid": tau_a,
                    "theta_acid": theta_a,
                }

    y0, k_base, k_acid, x0_base, x0_acid = refined["coeffs"]
    return np.array(
        [
            y0,
            k_base,
            refined["tau_base"],
            refined["theta_base"],
            x0_base,
            k_acid,
            refined["tau_acid"],
            refined["theta_acid"],
            x0_acid,
        ]
    )


def step_response(k, tau, theta, magnitude, t_end=3.0, dt=1.0 / 96.0):
    t = np.arange(0.0, t_end + dt / 2.0, dt)
    y = np.zeros_like(t)
    active = t >= theta
    y[active] = k * magnitude * (1.0 - np.exp(-(t[active] - theta) / tau))
    return t, y


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(LOG)
    df = df[df["time"] >= 2.0].copy()
    df["time"] = df["time"] - df["time"].iloc[0]
    df = df.reset_index(drop=True)

    t = df["time"].to_numpy(dtype=float)
    y = df["stage2_pH"].to_numpy(dtype=float)
    u_base = df["u_NaOH_kmol_d"].to_numpy(dtype=float)
    u_acid = df["u_HCl_kmol_d"].to_numpy(dtype=float)

    params = fit_model(t, y, u_base, u_acid)
    y_hat = predict(params, t, u_base, u_acid)
    residual = y_hat - y

    y0, k_base, tau_base, theta_base, x0_base, k_acid, tau_acid, theta_acid, x0_acid = params
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    r2 = float(1.0 - np.sum(residual**2) / np.sum((y - np.mean(y)) ** 2))

    rows = [
        {
            "input": "NaOH",
            "gain_pH_per_kmol_d": k_base,
            "gain_pH_per_m3_d": k_base * C_NAOH,
            "tau_d": tau_base,
            "tau_h": tau_base * 24.0,
            "theta_d": theta_base,
            "theta_h": theta_base * 24.0,
            "theta_min": theta_base * 24.0 * 60.0,
            "fit_initial_state_pH": x0_base,
        },
        {
            "input": "HCl",
            "gain_pH_per_kmol_d": k_acid,
            "gain_pH_per_m3_d": k_acid * C_HCL,
            "tau_d": tau_acid,
            "tau_h": tau_acid * 24.0,
            "theta_d": theta_acid,
            "theta_h": theta_acid * 24.0,
            "theta_min": theta_acid * 24.0 * 60.0,
            "fit_initial_state_pH": x0_acid,
        },
    ]
    param_df = pd.DataFrame(rows)
    fit_df = pd.DataFrame(
        {
            "metric": ["fit_start_original_d", "fit_end_original_d", "n_rows", "MAE_pH", "RMSE_pH", "R2"],
            "value": [2.0, df["time"].iloc[-1] + 2.0, len(df), mae, rmse, r2],
        }
    )

    param_path = OUT / "serial_2stage_PRBS_acid_base_foptd_params.csv"
    fit_path = OUT / "serial_2stage_PRBS_acid_base_foptd_fit_metrics.csv"
    pred_path = OUT / "serial_2stage_PRBS_acid_base_foptd_prediction.csv"
    plot_fit = OUT / "serial_2stage_PRBS_acid_base_foptd_fit_0_40d.png"
    plot_step = OUT / "serial_2stage_PRBS_acid_base_foptd_step_response.png"

    param_df.to_csv(param_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    pd.DataFrame(
        {
            "time_d_since_fit_start": t,
            "time_original_d": t + 2.0,
            "stage2_pH": y,
            "stage2_pH_foptd": y_hat,
            "u_NaOH_kmol_d": u_base,
            "u_HCl_kmol_d": u_acid,
        }
    ).to_csv(pred_path, index=False)

    win = t <= 38.0
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [2.0, 1, 1]})
    axes[0].plot(t[win] + 2.0, y[win], color="#0072B2", linewidth=1.4, label="PyADM1 stage2 pH")
    axes[0].plot(t[win] + 2.0, y_hat[win], color="#D55E00", linewidth=1.1, label="2-input FOPTD fit")
    axes[0].set_ylabel("pH")
    axes[0].set_title("Acid/base FOPTD fit, PRBS 2-40 d")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(t[win] + 2.0, u_base[win], color="#009E73", linewidth=1.1)
    axes[1].set_ylabel("NaOH\nkmol/d")
    axes[1].grid(True, alpha=0.25)
    axes[2].plot(t[win] + 2.0, u_acid[win], color="#CC79A7", linewidth=1.1)
    axes[2].set_ylabel("HCl\nkmol/d")
    axes[2].set_xlabel("Time (d)")
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_fit, dpi=180)
    plt.close(fig)

    t_step, y_base = step_response(k_base, tau_base, theta_base, 25.0)
    _, y_acid = step_response(k_acid, tau_acid, theta_acid, 11.3)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(t_step * 24.0, y_base, color="#009E73", linewidth=1.6, label="1 m3/d NaOH step (25 M)")
    ax.plot(t_step * 24.0, y_acid, color="#CC79A7", linewidth=1.6, label="1 m3/d HCl step (35 wt%)")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Time after step (h)")
    ax.set_ylabel("Delta pH")
    ax.set_title("Identified FOPTD step responses")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(plot_step, dpi=180)
    plt.close(fig)

    print("PARAMETERS")
    print(param_df.to_string(index=False))
    print("FIT")
    print(fit_df.to_string(index=False))
    print(f"param_csv={param_path}")
    print(f"fit_metrics_csv={fit_path}")
    print(f"prediction_csv={pred_path}")
    print(f"plot_fit={plot_fit}")
    print(f"plot_step={plot_step}")


if __name__ == "__main__":
    main()
