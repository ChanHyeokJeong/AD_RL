from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

from config import RLConfig
from plant import PlantState, StepTotals
from pyadm1_thermal_engine import PyADM1ThermalEngine


class FullPyADM1PIPlant:
    """PI-controlled plant that calls the full PyADM1 thermal engine."""

    def __init__(self, config: RLConfig):
        self.config = config
        self.params = pd.read_csv(config.thermal_parameters_path).iloc[0].to_dict()
        self.engine = PyADM1ThermalEngine(config)
        self.state: PlantState | None = None
        self.last_step_log: list[dict] = []
        self.last_interval_adm_state_before: dict[str, float] = {}
        self.last_interval_adm_state_after: dict[str, float] = {}
        self.last_interval_adm_state_delta: dict[str, float] = {}
        self.last_daily_setpoint_change_C = 0.0
        self.last_setpoint_penalty_excess_C = 0.0
        self.last_reactor_24h_change_C = np.nan
        self.last_reactor_temp_penalty_event = 0.0
        self.last_reactor_temp_penalty_excess_C = 0.0

    def reset(self, start_day: float = 0.0, setpoint_C: float | None = None) -> PlantState:
        setpoint = self.config.initial_t_setpoint_C if setpoint_C is None else float(setpoint_C)
        result = self.engine.reset(start_day=start_day)
        T0 = float(result.diagnostics["T_reactor_C"])
        q0 = self.engine.open_loop_ch4_flow(setpoint, start_day)
        error0 = setpoint - T0
        self.state = PlantState(
            time_d=float(start_day),
            episode_start_d=float(start_day),
            T_reactor_C=T0,
            T_setpoint_C=float(setpoint),
            q_ch4_heater_m3_d=q0,
            q_ch4_prod_m3_d=float(result.diagnostics["q_ch4_prod_m3_d"]),
            pi_error_C=error0,
            pi_error_integral_C_d=0.0,
            pi_last_error_C=error0,
            next_control_time_d=float(start_day),
            daily_setpoint_changes=deque(),
            reactor_temp_history=deque([(float(start_day), T0)]),
        )
        self.last_step_log = []
        self.last_interval_adm_state_before = self._adm_state_dict()
        self.last_interval_adm_state_after = self._adm_state_dict()
        self.last_interval_adm_state_delta = {
            name: 0.0 for name in self.engine.adm_state_names
        }
        self.last_daily_setpoint_change_C = 0.0
        self.last_setpoint_penalty_excess_C = 0.0
        self.last_reactor_24h_change_C = np.nan
        self.last_reactor_temp_penalty_event = 0.0
        self.last_reactor_temp_penalty_excess_C = 0.0
        return self.state

    def influent_at(self, time_d: float) -> tuple[float, float]:
        return self.engine.influent_at(time_d)

    def apply_setpoint_delta(self, delta_C: float) -> float:
        if self.state is None:
            raise RuntimeError("Plant must be reset before applying an action.")
        old_sp = self.state.T_setpoint_C
        requested_sp = old_sp + float(delta_C)
        new_sp = requested_sp
        if self.config.t_setpoint_min_C is not None:
            new_sp = max(float(self.config.t_setpoint_min_C), new_sp)
        if self.config.t_setpoint_max_C is not None:
            new_sp = min(float(self.config.t_setpoint_max_C), new_sp)
        applied_delta = new_sp - old_sp
        self.state.T_setpoint_C = new_sp
        return self._setpoint_change_penalty(applied_delta)

    def simulate_interval(self, duration_h: float) -> StepTotals:
        if self.state is None:
            raise RuntimeError("Plant must be reset before simulation.")

        totals = StepTotals()
        end_time = self.state.time_d + float(duration_h) / 24.0
        sim_dt_d = self.config.simulation_dt_h / 24.0
        self.last_step_log = []
        self.last_interval_adm_state_before = self._adm_state_dict()

        while self.state.time_d < end_time - 1e-12:
            if self.state.time_d >= self.state.next_control_time_d - 1e-12:
                self._update_pi_controller()

            t0 = self.state.time_d
            dt_d = min(sim_dt_d, end_time - t0)
            result = self.engine.step(
                t0,
                t0 + dt_d,
                self.state.q_ch4_heater_m3_d,
            )
            diag = result.diagnostics
            q_prod = float(diag["q_ch4_prod_m3_d"])
            q_heat_thermal = float(self.state.q_ch4_heater_m3_d)
            q_heat_accounted = self.accounted_heater_use_rate(q_heat_thermal)

            self.state.time_d = float(result.time_d)
            self.state.T_reactor_C = float(diag["T_reactor_C"])
            self.state.q_ch4_prod_m3_d = q_prod
            self._record_reactor_temperature()

            totals.methane_produced_m3 += q_prod * dt_d
            totals.methane_consumed_m3 += q_heat_accounted * dt_d

            row = {
                "time_d": self.state.time_d,
                "T_reactor_C": self.state.T_reactor_C,
                "T_setpoint_C": self.state.T_setpoint_C,
                "T_in_C": float(diag["T_in_C"]),
                "Q_m3_d": float(diag["Q_m3_d"]),
                "q_ch4_prod_m3_d": q_prod,
                "q_ch4_heater_m3_d": q_heat_accounted,
                "q_ch4_heater_thermal_m3_d": q_heat_thermal,
                "pi_error_C": self.state.pi_error_C,
                "pi_error_integral_C_d": self.state.pi_error_integral_C_d,
                "pH": float(diag["pH"]),
                "T_adapt_C": float(diag.get("T_adapt_C", np.nan)),
                "methanogenesis_shock_factor": float(
                    diag.get("methanogenesis_shock_factor", 1.0)
                ),
                "methanogenesis_temp_mismatch_K": float(
                    diag.get("methanogenesis_temp_mismatch_K", 0.0)
                ),
                "feed_heat_MJ_d": float(diag["feed_heat_MJ_d"]),
                "wall_heat_loss_MJ_d": float(diag["wall_heat_loss_MJ_d"]),
                "heater_heat_MJ_d": float(diag["heater_heat_MJ_d"]),
            }
            row.update(
                {
                    f"ADM1_{name}": float(value)
                    for name, value in zip(
                        self.engine.adm_state_names,
                        self.engine.adm_state_values(),
                    )
                }
            )
            self.last_step_log.append(row)

        self.last_interval_adm_state_after = self._adm_state_dict()
        self.last_interval_adm_state_delta = {
            name: self.last_interval_adm_state_after[name]
            - self.last_interval_adm_state_before[name]
            for name in self.engine.adm_state_names
        }
        return totals

    def reward_from_totals(self, totals: StepTotals, penalty: float) -> StepTotals:
        totals.penalty = penalty
        # ADDED: NPV-style methane term with independent production benefit.
        # Reason: methane production should dominate the objective more than
        # heater-use cost, without weakening the physical consumption log.
        # Role: reward = price*(w_prod*CH4_prod - w_cons*CH4_used) - penalty.
        # Reference: user-requested NPV reward with higher methane-production weight.
        methane_term = (
            self.config.reward_methane_production_weight
            * totals.methane_produced_m3
        )
        if self.config.reward_include_heater_consumption:
            methane_term -= (
                self.config.reward_heater_consumption_weight
                * totals.methane_consumed_m3
            )
        totals.reward = self.config.methane_price * methane_term - penalty
        return totals

    def observation(self) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("Plant must be reset before observation.")
        Tin_C, Q_m3_d = self.influent_at(self.state.time_d)
        day_of_year = self._wrapped_time(self.state.time_d) % 365.0
        base_values = [
            self.state.T_reactor_C,
            self.state.T_setpoint_C,
            self.last_daily_setpoint_change_C,
            max(
                0.0,
                self.config.daily_setpoint_change_limit_C
                - self.last_daily_setpoint_change_C,
            ),
            Tin_C,
            Q_m3_d,
            self.state.q_ch4_prod_m3_d,
            self.accounted_heater_use_rate(self.state.q_ch4_heater_m3_d),
            self.state.pi_error_C,
            self.state.pi_error_integral_C_d,
            self.state.time_d - self.state.episode_start_d,
            np.sin(2.0 * np.pi * day_of_year / 365.0),
            np.cos(2.0 * np.pi * day_of_year / 365.0),
        ]
        if self.config.include_shock_state_observation:
            # ADDED: expose the slow methanogen adaptation state to DQN.
            # Reason: once F_K affects methane production, T_reactor alone is
            # not a Markov state for methanogenesis response.
            # Role: let the agent distinguish adapted operation from a recent
            # temperature shock at the same T_reactor.
            # Reference: Yuki thesis Eq. 4.54-4.55.
            base_values.extend(
                [
                    self.engine.adapted_temperature_C(),
                    self.engine.methanogenesis_shock_factor(),
                    self.engine.methanogenesis_temp_mismatch_K(),
                ]
            )
        base_obs = np.asarray(base_values, dtype=np.float32)
        if self.config.normalize_observation:
            base_obs = self._normalize_base_observation(base_obs)
        if not self.config.include_adm_state_observation:
            return base_obs
        return np.concatenate(
            [base_obs, self.engine.adm_state_values().astype(np.float32)],
            dtype=np.float32,
        )

    @property
    def observation_names(self) -> list[str]:
        base_names = [
            "T_reactor_C",
            "T_setpoint_C",
            "daily_setpoint_change_C",
            "remaining_daily_setpoint_change_C",
            "T_in_C",
            "Q_m3_d",
            "q_ch4_prod_m3_d",
            "q_ch4_heater_m3_d",
            "pi_error_C",
            "pi_error_integral_C_d",
            "episode_elapsed_d",
            "season_sin",
            "season_cos",
        ]
        if self.config.include_shock_state_observation:
            base_names += [
                "T_adapt_C",
                "methanogenesis_shock_factor",
                "methanogenesis_temp_mismatch_K",
            ]
        if not self.config.include_adm_state_observation:
            return base_names
        return base_names + [f"ADM1_{name}" for name in self.engine.adm_state_names]

    def _normalize_base_observation(self, obs: np.ndarray) -> np.ndarray:
        # ADDED: normalize only the compact control observation for DQN.
        # Reason: raw methane production and flow terms are O(10^2-10^3),
        # while temperature/action-relevant terms are O(1); this pushed the
        # Q-network toward a nearly constant hold policy.
        # Role: keep logged variables physical, but give the neural network a
        # centered/scaled observation vector.
        # Reference: post-run diagnosis of hold-only deterministic policy.
        center_values = [
            35.0,  # T_reactor_C
            35.0,  # T_setpoint_C
            0.0,  # daily_setpoint_change_C
            self.config.daily_setpoint_change_limit_C,  # remaining limit
            35.0,  # T_in_C
            175.0,  # Q_m3_d
            1650.0,  # q_ch4_prod_m3_d
            15.0,  # q_ch4_heater_m3_d
            0.0,  # pi_error_C
            0.0,  # pi_error_integral_C_d
            45.0,  # episode_elapsed_d
            0.0,  # season_sin
            0.0,  # season_cos
        ]
        scale_values = [
            10.0,
            10.0,
            max(1.0, self.config.daily_setpoint_change_limit_C),
            max(1.0, self.config.daily_setpoint_change_limit_C),
            10.0,
            50.0,
            200.0,
            50.0,
            5.0,
            10.0,
            45.0,
            1.0,
            1.0,
        ]
        if self.config.include_shock_state_observation:
            center_values.extend([35.0, 1.0, 0.0])
            scale_values.extend([10.0, 1.0, 5.0])
        center = np.asarray(center_values, dtype=np.float32)
        scale = np.asarray(scale_values, dtype=np.float32)
        return (obs - center) / scale

    def accounted_heater_use_rate(self, thermal_q_ch4_m3_d: float) -> float:
        return float(thermal_q_ch4_m3_d) * self.config.heater_methane_accounting_factor

    def internal_delta_summary(self, top_n: int = 8) -> dict[str, float]:
        if not self.last_interval_adm_state_delta:
            return {
                "adm_delta_max_abs": 0.0,
                "adm_delta_top_state_count": 0,
            }
        sorted_items = sorted(
            self.last_interval_adm_state_delta.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
        summary = {
            "adm_delta_max_abs": float(abs(sorted_items[0][1])),
            "adm_delta_top_state_count": int(min(top_n, len(sorted_items))),
        }
        for rank, (name, delta) in enumerate(sorted_items[:top_n], start=1):
            summary[f"adm_delta_rank{rank}_name"] = name
            summary[f"adm_delta_rank{rank}_value"] = float(delta)
            summary[f"adm_delta_rank{rank}_before"] = float(
                self.last_interval_adm_state_before[name]
            )
            summary[f"adm_delta_rank{rank}_after"] = float(
                self.last_interval_adm_state_after[name]
            )
        return summary

    def _adm_state_dict(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(
                self.engine.adm_state_names,
                self.engine.adm_state_values(),
            )
        }

    def _update_pi_controller(self) -> None:
        assert self.state is not None
        error = self.state.T_setpoint_C - self.state.T_reactor_C
        dt_control = self.config.control_interval_h / 24.0
        integral_candidate = self.state.pi_error_integral_C_d + error * dt_control
        q_bias = self.engine.open_loop_ch4_flow(self.state.T_setpoint_C, self.state.time_d)
        q_unclamped = q_bias + self._p("controller_Kc_m3_d_C") * (
            error + integral_candidate / self._p("controller_Ti_d")
        )

        self.state.pi_error_C = error
        self.state.pi_error_integral_C_d = integral_candidate
        self.state.pi_last_error_C = error
        self.state.q_ch4_heater_m3_d = float(max(0.0, q_unclamped))
        self.state.next_control_time_d += self.config.control_interval_h / 24.0

    def _setpoint_change_penalty(self, applied_delta_C: float) -> float:
        assert self.state is not None
        now = self.state.time_d
        self.state.daily_setpoint_changes.append((now, abs(float(applied_delta_C))))
        while self.state.daily_setpoint_changes and self.state.daily_setpoint_changes[0][0] < now - 1.0:
            self.state.daily_setpoint_changes.popleft()
        daily_change = sum(delta for _, delta in self.state.daily_setpoint_changes)
        excess = max(0.0, daily_change - self.config.daily_setpoint_change_limit_C)
        self.last_daily_setpoint_change_C = daily_change
        self.last_setpoint_penalty_excess_C = excess
        if self.config.penalty_mode != "setpoint_change_24h":
            return 0.0
        return self.config.shock_penalty_per_C * excess

    def reactor_temperature_change_penalty(self) -> float:
        if self.config.penalty_mode not in {"reactor_24h_event", "reactor_24h_excess"}:
            self.last_reactor_24h_change_C = np.nan
            self.last_reactor_temp_penalty_event = 0.0
            self.last_reactor_temp_penalty_excess_C = 0.0
            return 0.0
        assert self.state is not None
        window_d = self.config.reactor_temp_change_window_h / 24.0
        if self.state.time_d - self.state.episode_start_d < window_d - 1e-12:
            self.last_reactor_24h_change_C = np.nan
            self.last_reactor_temp_penalty_event = 0.0
            self.last_reactor_temp_penalty_excess_C = 0.0
            return 0.0

        target_time = self.state.time_d - window_d
        reference_temp = self._reactor_temperature_at(target_time)
        if reference_temp is None:
            self.last_reactor_24h_change_C = np.nan
            self.last_reactor_temp_penalty_event = 0.0
            self.last_reactor_temp_penalty_excess_C = 0.0
            return 0.0

        change_C = abs(self.state.T_reactor_C - reference_temp)
        excess_C = max(0.0, change_C - self.config.reactor_temp_change_limit_C)
        event = 1.0 if excess_C > 0.0 else 0.0
        self.last_reactor_24h_change_C = change_C
        self.last_reactor_temp_penalty_event = event
        self.last_reactor_temp_penalty_excess_C = excess_C
        if self.config.penalty_mode == "reactor_24h_event":
            return self.config.reactor_temp_penalty_per_event * event
        return self.config.reactor_temp_penalty_per_C * excess_C

    def _record_reactor_temperature(self) -> None:
        assert self.state is not None
        history = self.state.reactor_temp_history
        history.append((self.state.time_d, self.state.T_reactor_C))
        keep_from = self.state.time_d - self.config.reactor_temp_change_window_h / 24.0 - 1.0
        while len(history) > 2 and history[0][0] < keep_from:
            history.popleft()

    def _reactor_temperature_at(self, target_time_d: float) -> float | None:
        assert self.state is not None
        history = list(self.state.reactor_temp_history)
        if not history or target_time_d < history[0][0] or target_time_d > history[-1][0]:
            return None
        times = np.asarray([item[0] for item in history], dtype=float)
        temps = np.asarray([item[1] for item in history], dtype=float)
        return float(np.interp(target_time_d, times, temps))

    def _wrapped_time(self, time_d: float) -> float:
        if self.engine.time_grid.size == 0:
            return float(time_d)
        max_time = float(self.engine.time_grid[-1])
        if time_d <= max_time:
            return float(time_d)
        return float(time_d % max_time)

    def _p(self, name: str, default: float | None = None) -> float:
        if name in self.params and pd.notna(self.params[name]):
            return float(self.params[name])
        if default is not None:
            return float(default)
        raise KeyError(f"Missing thermal parameter: {name}")

