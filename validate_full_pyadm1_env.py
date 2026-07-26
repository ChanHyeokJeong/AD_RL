from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ad_rl.config import RLConfig, ensure_run_dir
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv


def main() -> Path:
    config = replace(
        RLConfig(),
        decision_interval_h=6.0,
        control_interval_h=1.0,
        simulation_dt_h=1.0,
        smoke_episode_days=0.5,
    )
    output_dir = ensure_run_dir() / "full_pyadm1_engine_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.smoke_episode_days)
    obs0 = env.reset(seed=config.random_seed)
    rows = []
    adm0 = env.plant.engine.adm_state_values().copy()

    done = False
    step = 0
    actions = [1, 2]
    while not done and step < len(actions):
        obs, reward, terminated, truncated, info = env.step(actions[step])
        rows.append({"step": step, "reward": reward, **info})
        done = bool(terminated or truncated)
        step += 1

    log = pd.DataFrame(rows)
    log.to_csv(output_dir / "full_pyadm1_env_step_log.csv", index=False)
    pd.DataFrame(
        {
            "observation_index": np.arange(len(obs0)),
            "observation_name": env.observation_names,
            "reset_value": obs0,
            "final_value": obs,
        }
    ).to_csv(output_dir / "full_pyadm1_observation_check.csv", index=False)
    pd.DataFrame(env.plant.last_step_log).to_csv(
        output_dir / "full_pyadm1_internal_1h_log_last_interval.csv",
        index=False,
    )

    adm_change = float(np.nanmax(np.abs(env.plant.engine.adm_state_values() - adm0)))
    pd.DataFrame(
        [
            {
                "source_path": str(config.full_pyadm1_source_path),
                "observation_dim": int(env.observation_space.shape[0]),
                "adm_observation_count": int(max(0, env.observation_space.shape[0] - 11)),
                "adm_engine_state_count": int(len(env.plant.engine.adm_state_names)),
                "max_abs_ADM1_engine_state_change": adm_change,
                "decision_interval_h": config.decision_interval_h,
                "simulation_dt_h": config.simulation_dt_h,
            }
        ]
    ).to_csv(output_dir / "full_pyadm1_engine_smoke_summary.csv", index=False)

    if not log.empty:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 7.2), sharex=True)
        t = log["episode_elapsed_d"]
        axes[0].plot(t, log["T_reactor_C"], marker="o", label="T_reactor")
        axes[0].step(t, log["T_setpoint_C"], where="post", label="T_setpoint")
        axes[0].set_ylabel("temperature (C)")
        axes[0].legend(frameon=False)
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(t, log["q_ch4_prod_m3_d"], marker="o")
        axes[1].set_ylabel("CH4 prod. (m3/d)")
        axes[1].grid(True, alpha=0.25)

        axes[2].step(t, log["action_delta_C"], where="post")
        axes[2].set_ylabel("action (C)")
        axes[2].set_xlabel("episode time (d)")
        axes[2].grid(True, alpha=0.25)

        fig.suptitle("Full PyADM1 engine smoke test")
        fig.tight_layout()
        fig.savefig(output_dir / "full_pyadm1_engine_smoke.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    return output_dir


if __name__ == "__main__":
    print(main())
