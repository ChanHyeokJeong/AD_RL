from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Callable

import gym
import numpy as np
import pandas as pd
from gym import spaces

import PyADM1_pH_2stage_PRBS as base_model


VFA_NAMES = ["S_va", "S_bu", "S_pro", "S_ac"]
FERMENTABLE_NAMES = ["S_su", "S_aa", "S_fa", "X_ch", "X_pr", "X_li"]


@dataclass
class PHControlRLConfig:
    episode_days: float = 2.0
    decision_interval_h: float = 3.0
    reward_scale: float = 100.0
    random_seed: int = 20260730

    # Signed flow convention: negative = HCl, positive = NaOH, unit = m3/d.
    # The table forbids simultaneous acid/base dosing within the same reactor.
    stage1_signed_flow_levels_m3_d: tuple[float, ...] = (-0.60, -0.20, 0.0, 0.10, 0.30)
    stage2_signed_flow_levels_m3_d: tuple[float, ...] = (-20.0, -5.0, 0.0, 5.0, 20.0)

    stage1_vfa_production_weight: float = 1.0
    stage2_vfa_removal_weight: float = 1.0
    stage2_methane_weight: float = 0.1
    chemical_kmol_weight: float = 0.03
    ph_violation_weight: float = 500.0

    stage1_pH_min: float = 4.8
    stage1_pH_max: float = 6.4
    stage2_pH_min: float = 6.7
    stage2_pH_max: float = 7.8

    gamma: float = 0.99
    learning_rate: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 50_000
    warmup_steps: int = 32
    target_update_steps: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2_000


def config_to_json_dict(config: PHControlRLConfig) -> dict:
    return asdict(config)


def split_signed_flow(flow_m3_d: float) -> tuple[float, float]:
    flow = float(flow_m3_d)
    if flow > 0.0:
        return flow, 0.0
    if flow < 0.0:
        return 0.0, abs(flow)
    return 0.0, 0.0


def build_action_table(config: PHControlRLConfig) -> pd.DataFrame:
    rows = []
    for action, (stage1_signed, stage2_signed) in enumerate(
        product(config.stage1_signed_flow_levels_m3_d, config.stage2_signed_flow_levels_m3_d)
    ):
        s1_naoh, s1_hcl = split_signed_flow(stage1_signed)
        s2_naoh, s2_hcl = split_signed_flow(stage2_signed)
        rows.append(
            {
                "action": action,
                "stage1_signed_m3_d": stage1_signed,
                "stage2_signed_m3_d": stage2_signed,
                "stage1_q_NaOH_m3_d": s1_naoh,
                "stage1_q_HCl_m3_d": s1_hcl,
                "stage2_q_NaOH_m3_d": s2_naoh,
                "stage2_q_HCl_m3_d": s2_hcl,
            }
        )
    return pd.DataFrame(rows)


def vfa_total(ctx: dict) -> float:
    return float(sum(float(ctx[name]) for name in VFA_NAMES))


def feed_total(ctx: dict, names: list[str]) -> float:
    return float(sum(float(ctx[f"{name}_in"]) for name in names))


def concentration_total(ctx: dict, names: list[str]) -> float:
    return float(sum(float(ctx[name]) for name in names))


def ph_violation(pH: float, low: float, high: float) -> float:
    if pH < low:
        return low - pH
    if pH > high:
        return pH - high
    return 0.0


class TwoStagePHDirectDosingPlant:
    def __init__(self, config: PHControlRLConfig | None = None):
        self.config = config or PHControlRLConfig()
        self.action_table = build_action_table(self.config)
        self.stage1: dict | None = None
        self.stage2: dict | None = None
        self.time_values = np.array([], dtype=float)
        self.current_index = 0
        self.current_time_d = 0.0
        self.episode_start_d = 0.0
        self.episode_end_d = float(self.config.episode_days)
        self.current_action = 0
        self.last_interval_totals: dict[str, float] = {}
        self.last_step_log: list[dict] = []

    def reset(self) -> np.ndarray:
        self.stage1 = base_model.fresh_reactor(
            "stage1_55C",
            base_model.STAGE1_TEMP_K,
            base_model.STAGE1_VOLUME_FRACTION,
        )
        self.stage2 = base_model.fresh_reactor(
            "stage2_35C",
            base_model.STAGE2_TEMP_K,
            base_model.STAGE2_VOLUME_FRACTION,
        )
        self.time_values = np.asarray(self.stage1["t"], dtype=float)
        self.current_index = 0
        self.current_time_d = 0.0
        self.episode_start_d = 0.0
        self.episode_end_d = float(self.config.episode_days)
        self.current_action = self.hold_action_index()
        self._apply_action_to_reactors(self.current_action)
        self.last_interval_totals = self._empty_totals()
        self.last_step_log = []
        return self.observation()

    def hold_action_index(self) -> int:
        hold_rows = self.action_table[
            (self.action_table["stage1_signed_m3_d"] == 0.0)
            & (self.action_table["stage2_signed_m3_d"] == 0.0)
        ]
        if hold_rows.empty:
            return 0
        return int(hold_rows.iloc[0]["action"])

    def _empty_totals(self) -> dict[str, float]:
        return {
            "stage1_vfa_in_kgCOD": 0.0,
            "stage1_vfa_out_kgCOD": 0.0,
            "stage1_vfa_produced_kgCOD": 0.0,
            "stage1_fermentable_in_kgCOD": 0.0,
            "stage2_vfa_in_kgCOD": 0.0,
            "stage2_vfa_out_kgCOD": 0.0,
            "stage2_vfa_removed_kgCOD": 0.0,
            "stage1_ch4_m3": 0.0,
            "stage2_ch4_m3": 0.0,
            "total_ch4_m3": 0.0,
            "chemical_m3": 0.0,
            "chemical_kmol": 0.0,
            "stage1_ph_violation_pH_d": 0.0,
            "stage2_ph_violation_pH_d": 0.0,
            "ph_violation_pH_d": 0.0,
        }

    def _set_stage_dosing(self, ctx: dict, q_naoh_m3_d: float, q_hcl_m3_d: float) -> None:
        ctx["q_NaOH"] = max(0.0, min(float(q_naoh_m3_d), float(ctx["q_NaOH_max"])))
        ctx["q_HCl"] = max(0.0, min(float(q_hcl_m3_d), float(ctx["q_HCl_max"])))
        ctx["u_NaOH"] = float(ctx["C_NAOH"]) * ctx["q_NaOH"]
        ctx["u_HCl"] = float(ctx["C_HCL"]) * ctx["q_HCl"]

    def _apply_action_to_reactors(self, action: int) -> dict:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Plant must be reset before applying an action.")
        row = self.action_table.iloc[int(action)].to_dict()
        self._set_stage_dosing(
            self.stage1,
            row["stage1_q_NaOH_m3_d"],
            row["stage1_q_HCl_m3_d"],
        )
        self._set_stage_dosing(
            self.stage2,
            row["stage2_q_NaOH_m3_d"],
            row["stage2_q_HCl_m3_d"],
        )
        self.current_action = int(action)
        return row

    def simulate_decision_interval(self, action: int) -> dict[str, float]:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Plant must be reset before simulation.")

        action_row = self._apply_action_to_reactors(action)
        return self._simulate_interval(int(action), action_row)

    def simulate_fixed_pH_interval(
        self,
        stage1_pH_sp: float = 7.0,
        stage2_pH_sp: float = 7.0,
    ) -> dict[str, float]:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Plant must be reset before simulation.")

        dt_d = min(
            float(self.config.decision_interval_h) / 24.0,
            max(0.0, self.episode_end_d - self.current_time_d),
        )
        self.stage1["PI_pH_controller"](float(stage1_pH_sp), float(self.stage1["pH"]), dt_d)
        self.stage2["PI_pH_controller"](float(stage2_pH_sp), float(self.stage2["pH"]), dt_d)
        action_row = {
            "action": -1,
            "stage1_signed_m3_d": float(self.stage1["q_NaOH"]) - float(self.stage1["q_HCl"]),
            "stage2_signed_m3_d": float(self.stage2["q_NaOH"]) - float(self.stage2["q_HCl"]),
            "stage1_q_NaOH_m3_d": float(self.stage1["q_NaOH"]),
            "stage1_q_HCl_m3_d": float(self.stage1["q_HCl"]),
            "stage2_q_NaOH_m3_d": float(self.stage2["q_NaOH"]),
            "stage2_q_HCl_m3_d": float(self.stage2["q_HCl"]),
            "stage1_pH_sp": float(stage1_pH_sp),
            "stage2_pH_sp": float(stage2_pH_sp),
        }
        self.current_action = -1
        return self._simulate_interval(-1, action_row)

    def _simulate_interval(self, action: int, action_row: dict) -> dict[str, float]:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Plant must be reset before simulation.")

        totals = self._empty_totals()
        self.last_step_log = []

        end_time = min(
            self.current_time_d + float(self.config.decision_interval_h) / 24.0,
            self.episode_end_d,
            float(self.time_values[-1]),
        )

        while self.current_time_d < end_time - 1e-12:
            if self.current_index + 1 >= len(self.time_values):
                break
            next_time = min(float(self.time_values[self.current_index + 1]), end_time)
            dt = next_time - self.current_time_d
            if dt <= 0:
                self.current_index += 1
                continue

            feed_index = min(self.current_index + 1, len(self.time_values) - 1)
            base_model.set_input_from_influent(self.stage1, feed_index)
            stage1_vfa_in = feed_total(self.stage1, VFA_NAMES)
            stage1_fermentable_in = feed_total(self.stage1, FERMENTABLE_NAMES)

            base_model.run_reactor_step(self.stage1, [self.current_time_d, next_time])
            stage1_vfa_out = vfa_total(self.stage1)

            base_model.set_input_from_reactor(self.stage2, self.stage1)
            stage2_vfa_in = stage1_vfa_out
            base_model.run_reactor_step(self.stage2, [self.current_time_d, next_time])
            stage2_vfa_out = vfa_total(self.stage2)

            q_ad = float(self.stage1["q_ad"])
            stage1_vfa_in_load = stage1_vfa_in * q_ad * dt
            stage1_vfa_out_load = stage1_vfa_out * q_ad * dt
            stage1_fermentable_load = stage1_fermentable_in * q_ad * dt
            stage1_vfa_produced = max(0.0, stage1_vfa_out_load - stage1_vfa_in_load)

            stage2_vfa_in_load = stage2_vfa_in * q_ad * dt
            stage2_vfa_out_load = stage2_vfa_out * q_ad * dt
            stage2_vfa_removed = max(0.0, stage2_vfa_in_load - stage2_vfa_out_load)

            stage1_ch4 = max(0.0, float(self.stage1["q_ch4"])) * dt
            stage2_ch4 = max(0.0, float(self.stage2["q_ch4"])) * dt
            chemical_m3 = (
                float(self.stage1["q_NaOH"])
                + float(self.stage1["q_HCl"])
                + float(self.stage2["q_NaOH"])
                + float(self.stage2["q_HCl"])
            ) * dt
            chemical_kmol = (
                float(self.stage1["u_NaOH"])
                + float(self.stage1["u_HCl"])
                + float(self.stage2["u_NaOH"])
                + float(self.stage2["u_HCl"])
            ) * dt

            stage1_pH = float(self.stage1["pH"])
            stage2_pH = float(self.stage2["pH"])
            stage1_ph_violation = ph_violation(
                stage1_pH,
                self.config.stage1_pH_min,
                self.config.stage1_pH_max,
            ) * dt
            stage2_ph_violation = ph_violation(
                stage2_pH,
                self.config.stage2_pH_min,
                self.config.stage2_pH_max,
            ) * dt

            increments = {
                "stage1_vfa_in_kgCOD": stage1_vfa_in_load,
                "stage1_vfa_out_kgCOD": stage1_vfa_out_load,
                "stage1_vfa_produced_kgCOD": stage1_vfa_produced,
                "stage1_fermentable_in_kgCOD": stage1_fermentable_load,
                "stage2_vfa_in_kgCOD": stage2_vfa_in_load,
                "stage2_vfa_out_kgCOD": stage2_vfa_out_load,
                "stage2_vfa_removed_kgCOD": stage2_vfa_removed,
                "stage1_ch4_m3": stage1_ch4,
                "stage2_ch4_m3": stage2_ch4,
                "total_ch4_m3": stage1_ch4 + stage2_ch4,
                "chemical_m3": chemical_m3,
                "chemical_kmol": chemical_kmol,
                "stage1_ph_violation_pH_d": stage1_ph_violation,
                "stage2_ph_violation_pH_d": stage2_ph_violation,
                "ph_violation_pH_d": stage1_ph_violation + stage2_ph_violation,
            }
            for key, value in increments.items():
                totals[key] += float(value)

            row = {
                "time_d": next_time,
                "action": int(action),
                "stage1_pH": stage1_pH,
                "stage2_pH": stage2_pH,
                "stage1_vfa_kgCOD_m3": stage1_vfa_out,
                "stage2_vfa_kgCOD_m3": stage2_vfa_out,
                "stage1_q_ch4_m3_d": float(self.stage1["q_ch4"]),
                "stage2_q_ch4_m3_d": float(self.stage2["q_ch4"]),
                "stage1_q_NaOH_m3_d": float(self.stage1["q_NaOH"]),
                "stage1_q_HCl_m3_d": float(self.stage1["q_HCl"]),
                "stage2_q_NaOH_m3_d": float(self.stage2["q_NaOH"]),
                "stage2_q_HCl_m3_d": float(self.stage2["q_HCl"]),
            }
            row.update(action_row)
            row.update(increments)
            self.last_step_log.append(row)

            self.current_time_d = float(next_time)
            if next_time >= float(self.time_values[self.current_index + 1]) - 1e-12:
                self.current_index += 1

        reward_terms = self.reward_terms(totals)
        totals.update(reward_terms)
        totals.update(
            {
                "time_d": self.current_time_d,
                "episode_elapsed_d": self.current_time_d - self.episode_start_d,
                "action": int(action),
                "stage1_signed_m3_d": float(
                    action_row.get(
                        "stage1_signed_m3_d",
                        float(self.stage1["q_NaOH"]) - float(self.stage1["q_HCl"]),
                    )
                ),
                "stage2_signed_m3_d": float(
                    action_row.get(
                        "stage2_signed_m3_d",
                        float(self.stage2["q_NaOH"]) - float(self.stage2["q_HCl"]),
                    )
                ),
                "stage1_pH_sp": float(action_row.get("stage1_pH_sp", np.nan)),
                "stage2_pH_sp": float(action_row.get("stage2_pH_sp", np.nan)),
                "stage1_pH": float(self.stage1["pH"]),
                "stage2_pH": float(self.stage2["pH"]),
                "stage1_vfa_kgCOD_m3": vfa_total(self.stage1),
                "stage2_vfa_kgCOD_m3": vfa_total(self.stage2),
                "stage1_q_NaOH_m3_d": float(self.stage1["q_NaOH"]),
                "stage1_q_HCl_m3_d": float(self.stage1["q_HCl"]),
                "stage2_q_NaOH_m3_d": float(self.stage2["q_NaOH"]),
                "stage2_q_HCl_m3_d": float(self.stage2["q_HCl"]),
                "stage1_q_ch4_m3_d": float(self.stage1["q_ch4"]),
                "stage2_q_ch4_m3_d": float(self.stage2["q_ch4"]),
                "stage1_acid_yield": totals["stage1_vfa_produced_kgCOD"]
                / max(1e-12, totals["stage1_fermentable_in_kgCOD"]),
                "stage2_vfa_conversion": totals["stage2_vfa_removed_kgCOD"]
                / max(1e-12, totals["stage2_vfa_in_kgCOD"]),
                "stage2_ch4_per_vfa_in": totals["stage2_ch4_m3"]
                / max(1e-12, totals["stage2_vfa_in_kgCOD"]),
            }
        )
        self.last_interval_totals = totals
        return totals

    def reward_terms(self, totals: dict[str, float]) -> dict[str, float]:
        benefit = (
            self.config.stage1_vfa_production_weight * totals["stage1_vfa_produced_kgCOD"]
            + self.config.stage2_vfa_removal_weight * totals["stage2_vfa_removed_kgCOD"]
            + self.config.stage2_methane_weight * totals["stage2_ch4_m3"]
        )
        chemical_cost = self.config.chemical_kmol_weight * totals["chemical_kmol"]
        ph_cost = self.config.ph_violation_weight * totals["ph_violation_pH_d"]
        raw_reward = benefit - chemical_cost - ph_cost
        return {
            "reward_benefit": benefit,
            "reward_chemical_cost": chemical_cost,
            "reward_ph_cost": ph_cost,
            "reward_raw": raw_reward,
            "reward": raw_reward / max(1e-12, self.config.reward_scale),
        }

    def observation(self) -> np.ndarray:
        if self.stage1 is None or self.stage2 is None:
            raise RuntimeError("Plant must be reset before observation.")
        elapsed_frac = (
            (self.current_time_d - self.episode_start_d)
            / max(1e-12, self.episode_end_d - self.episode_start_d)
        )
        totals = self.last_interval_totals or self._empty_totals()
        obs = np.asarray(
            [
                (float(self.stage1["pH"]) - 6.0) / 2.0,
                (float(self.stage2["pH"]) - 7.2) / 2.0,
                np.log1p(vfa_total(self.stage1)),
                np.log1p(vfa_total(self.stage2)),
                float(self.stage1["q_ch4"]) / 1000.0,
                float(self.stage2["q_ch4"]) / 1000.0,
                float(self.stage1["q_NaOH"]) / max(1e-12, base_model.Q_NAOH_MAX),
                float(self.stage1["q_HCl"]) / max(1e-12, base_model.Q_HCL_MAX),
                float(self.stage2["q_NaOH"]) / max(1e-12, base_model.Q_NAOH_MAX),
                float(self.stage2["q_HCl"]) / max(1e-12, base_model.Q_HCL_MAX),
                totals.get("stage1_acid_yield", 0.0),
                totals.get("stage2_vfa_conversion", 0.0),
                elapsed_frac,
            ],
            dtype=np.float32,
        )
        return obs

    @property
    def observation_names(self) -> list[str]:
        return [
            "stage1_pH_scaled",
            "stage2_pH_scaled",
            "stage1_log1p_vfa",
            "stage2_log1p_vfa",
            "stage1_q_ch4_scaled",
            "stage2_q_ch4_scaled",
            "stage1_q_NaOH_scaled",
            "stage1_q_HCl_scaled",
            "stage2_q_NaOH_scaled",
            "stage2_q_HCl_scaled",
            "last_stage1_acid_yield",
            "last_stage2_vfa_conversion",
            "episode_elapsed_fraction",
        ]


class TwoStagePHDirectDosingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: PHControlRLConfig | None = None):
        super().__init__()
        self.config = config or PHControlRLConfig()
        self.plant = TwoStagePHDirectDosingPlant(self.config)
        self.action_table = self.plant.action_table
        self.action_space = spaces.Discrete(len(self.action_table))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.plant.observation_names),),
            dtype=np.float32,
        )
        self.last_info: dict = {}

    @property
    def observation_names(self) -> list[str]:
        return self.plant.observation_names

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.last_info = {}
        return self.plant.reset()

    def step(self, action: int):
        totals = self.plant.simulate_decision_interval(int(action))
        obs = self.plant.observation()
        terminated = self.plant.current_time_d >= self.plant.episode_end_d - 1e-12
        truncated = False
        self.last_info = totals
        return obs, float(totals["reward"]), bool(terminated), truncated, totals

    def render(self):
        return self.last_info


def rollout_policy(
    config: PHControlRLConfig,
    select_action: Callable[[np.ndarray, TwoStagePHDirectDosingEnv], int],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    env = TwoStagePHDirectDosingEnv(config)
    obs = env.reset(seed=config.random_seed)
    done = False
    decision_rows: list[dict] = []
    internal_rows: list[dict] = []
    step = 0

    while not done:
        action = int(select_action(obs, env))
        next_obs, reward, terminated, truncated, info = env.step(action)
        row = {"decision_step": step, "reward": reward}
        row.update(info)
        decision_rows.append(row)
        for internal in env.plant.last_step_log:
            internal_rows.append(dict(internal, decision_step=step))
        obs = next_obs
        done = bool(terminated or truncated)
        step += 1

    decision_df = pd.DataFrame(decision_rows)
    internal_df = pd.DataFrame(internal_rows)
    summary = summarize_decisions(decision_df)
    return decision_df, internal_df, summary


def rollout_fixed_pH_policy(
    config: PHControlRLConfig,
    stage1_pH_sp: float = 7.0,
    stage2_pH_sp: float = 7.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    env = TwoStagePHDirectDosingEnv(config)
    env.reset(seed=config.random_seed)
    done = False
    decision_rows: list[dict] = []
    internal_rows: list[dict] = []
    step = 0

    while not done:
        info = env.plant.simulate_fixed_pH_interval(
            stage1_pH_sp=stage1_pH_sp,
            stage2_pH_sp=stage2_pH_sp,
        )
        reward = float(info["reward"])
        row = {"decision_step": step, "reward": reward}
        row.update(info)
        decision_rows.append(row)
        for internal in env.plant.last_step_log:
            internal_rows.append(dict(internal, decision_step=step))
        done = env.plant.current_time_d >= env.plant.episode_end_d - 1e-12
        step += 1

    decision_df = pd.DataFrame(decision_rows)
    internal_df = pd.DataFrame(internal_rows)
    summary = summarize_decisions(decision_df)
    summary["stage1_pH_sp"] = float(stage1_pH_sp)
    summary["stage2_pH_sp"] = float(stage2_pH_sp)
    return decision_df, internal_df, summary


def summarize_decisions(decision_df: pd.DataFrame) -> dict[str, float]:
    if decision_df.empty:
        return {}
    sum_cols = [
        "reward",
        "reward_raw",
        "reward_benefit",
        "reward_chemical_cost",
        "reward_ph_cost",
        "stage1_vfa_produced_kgCOD",
        "stage2_vfa_removed_kgCOD",
        "stage1_ch4_m3",
        "stage2_ch4_m3",
        "total_ch4_m3",
        "chemical_m3",
        "chemical_kmol",
        "ph_violation_pH_d",
    ]
    out = {f"total_{col}": float(decision_df[col].sum()) for col in sum_cols if col in decision_df.columns}
    last = decision_df.iloc[-1]
    out.update(
        {
            "steps": int(len(decision_df)),
            "final_time_d": float(last["time_d"]),
            "final_stage1_pH": float(last["stage1_pH"]),
            "final_stage2_pH": float(last["stage2_pH"]),
            "mean_stage1_pH": float(decision_df["stage1_pH"].mean()),
            "mean_stage2_pH": float(decision_df["stage2_pH"].mean()),
            "mean_stage1_acid_yield": float(decision_df["stage1_acid_yield"].mean()),
            "mean_stage2_vfa_conversion": float(decision_df["stage2_vfa_conversion"].mean()),
            "mean_stage2_ch4_per_vfa_in": float(decision_df["stage2_ch4_per_vfa_in"].mean()),
        }
    )
    return out
