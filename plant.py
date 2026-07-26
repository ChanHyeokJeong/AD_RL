from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import RLConfig


@dataclass
class StepTotals:
    methane_produced_m3: float = 0.0
    methane_consumed_m3: float = 0.0
    penalty: float = 0.0
    reward: float = 0.0


@dataclass
class PlantState:
    time_d: float
    episode_start_d: float
    T_reactor_C: float
    T_setpoint_C: float
    q_ch4_heater_m3_d: float
    q_ch4_prod_m3_d: float
    pi_error_C: float
    pi_error_integral_C_d: float
    pi_last_error_C: float
    next_control_time_d: float
    daily_setpoint_changes: deque = field(default_factory=deque)
    reactor_temp_history: deque = field(default_factory=deque)


class ThermalPIPlant:
    """Fast PyADM1+PI-compatible plant wrapper for RL smoke tests.

    This class intentionally keeps the same heat-balance and PI-controller
    constants used in the validated PyADM1 thermal experiments, but it uses a
    fast interval simulator so DQN code can be tested before the full PyADM1
    interval integrator is refactored out of the monolithic script.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.influent = pd.read_csv(config.influent_path)
        self.initial_state = pd.read_csv(config.initial_state_path)
        self.params = pd.read_csv(config.thermal_parameters_path).iloc[0].to_dict()
        self.baseline = pd.read_csv(config.baseline_thermal_inputs_path)

        self.time_grid = self.influent["time"].to_numpy(dtype=float)
        self.Tin_grid = self.influent["T_in"].to_numpy(dtype=float)
        self.Q_grid = self.influent["Q"].to_numpy(dtype=float)

        self.baseline_time = self.baseline["time"].to_numpy(dtype=float)
        self.baseline_q_ch4 = self.baseline["q_ch4"].to_numpy(dtype=float)
        self.baseline_q_ch4_mean = float(np.nanmean(self.baseline_q_ch4))
        # ADDED: ADM1 composition states exposed to the RL observation.
        # Reason: diagnose whether the DQN keeps increasing T_setpoint because
        # it only sees thermal variables and cannot observe microbial/substrate
        # conditions. In this fast surrogate these ADM1 values are initialized
        # from the validated equilibrium CSV and remain constant until the full
        # PyADM1 integrator is wrapped.
        # Role: make all ADM1 soluble, particulate, ion, and gas states visible
        # to the agent without changing the current thermal/reward dynamics.
        # Reference: user-requested state expansion to ADM1 detailed components.
        self.adm_state_names = [
            column for column in self.initial_state.columns if column != "T_reactor"
        ]
        self.adm_state_values = (
            self.initial_state[self.adm_state_names].iloc[0].to_numpy(dtype=float)
        )

        self.state: PlantState | None = None
        self.last_step_log: list[dict] = []
        self.last_daily_setpoint_change_C = 0.0
        self.last_setpoint_penalty_excess_C = 0.0

    def reset(self, start_day: float = 0.0, setpoint_C: float | None = None) -> PlantState:
        # ADDED: deterministic reset to ADM1 equilibrium state.
        # Reason: RL episodes should start from a repeatable SI/equilibrium
        # state, while the dynamic influent time can be shifted by season.
        # Role: initialize reactor temperature, PI integral, and heater command.
        # Reference: user-requested reset = season first day + ADM1 default
        # equilibrium state.
        setpoint = self.config.initial_t_setpoint_C if setpoint_C is None else float(setpoint_C)
        setpoint = float(setpoint)
        T0 = float(self.initial_state["T_reactor"].iloc[0])
        error0 = setpoint - T0
        q0 = self.open_loop_ch4_flow(setpoint, start_day)
        self.state = PlantState(
            time_d=float(start_day),
            episode_start_d=float(start_day),
            T_reactor_C=T0,
            T_setpoint_C=setpoint,
            q_ch4_heater_m3_d=q0,
            q_ch4_prod_m3_d=self.methane_production_rate(start_day, T0),
            pi_error_C=error0,
            pi_error_integral_C_d=0.0,
            pi_last_error_C=error0,
            next_control_time_d=float(start_day),
        )
        self.last_step_log = []
        self.last_daily_setpoint_change_C = 0.0
        self.last_setpoint_penalty_excess_C = 0.0
        return self.state

    def influent_at(self, time_d: float) -> tuple[float, float]:
        t = self._wrapped_time(time_d)
        Tin = float(np.interp(t, self.time_grid, self.Tin_grid))
        Q = float(np.interp(t, self.time_grid, self.Q_grid))
        return Tin, Q

    def methane_production_rate(self, time_d: float, T_reactor_C: float) -> float:
        # ADDED: fast methane-production surrogate for the RL smoke env.
        # Reason: the monolithic PyADM1 ODE is too slow to call inside every DQN
        # step before refactoring; baseline q_ch4 comes from the validated PyADM1
        # run and is adjusted mildly by reactor temperature.
        # Role: provide the "total methane produced" term in the prototype NPV
        # reward while preserving the env/action/reward interface.
        # Reference: current PyADM1+BMS2+ITAE baseline output; Yuki kinetics
        # motivate temperature sensitivity, but this is a smoke-test surrogate.
        if self.config.use_static_methane_production_baseline:
            baseline_q = self.baseline_q_ch4_mean
        else:
            t = self._wrapped_time(time_d)
            baseline_q = float(np.interp(t, self.baseline_time, self.baseline_q_ch4))
        temp_factor = np.clip(1.0 + 0.01 * (T_reactor_C - 35.0), 0.85, 1.20)
        return max(0.0, baseline_q * temp_factor)

    def apply_setpoint_delta(self, delta_C: float) -> float:
        if self.state is None:
            raise RuntimeError("Plant must be reset before applying an action.")
        old_sp = self.state.T_setpoint_C
        new_sp = old_sp + delta_C
        applied_delta = new_sp - old_sp
        self.state.T_setpoint_C = new_sp
        return self._setpoint_change_penalty(applied_delta)

    def simulate_interval(self, duration_h: float) -> StepTotals:
        if self.state is None:
            raise RuntimeError("Plant must be reset before simulation.")

        totals = StepTotals()
        end_time = self.state.time_d + duration_h / 24.0
        sim_dt_d = self.config.simulation_dt_h / 24.0
        self.last_step_log = []

        while self.state.time_d < end_time - 1e-12:
            if self.state.time_d >= self.state.next_control_time_d - 1e-12:
                self._update_pi_controller()

            dt_d = min(sim_dt_d, end_time - self.state.time_d)
            Tin_C, Q_m3_d = self.influent_at(self.state.time_d)
            q_prod = self.methane_production_rate(self.state.time_d, self.state.T_reactor_C)
            q_heat_thermal = self.state.q_ch4_heater_m3_d
            q_heat_accounted = self.accounted_heater_use_rate(q_heat_thermal)

            dTdt = self.temperature_derivative_C_d(self.state.T_reactor_C, Tin_C, Q_m3_d, q_heat_thermal)
            self.state.T_reactor_C += dTdt * dt_d
            self.state.time_d += dt_d
            self.state.q_ch4_prod_m3_d = q_prod

            totals.methane_produced_m3 += q_prod * dt_d
            totals.methane_consumed_m3 += q_heat_accounted * dt_d

            self.last_step_log.append(
                {
                    "time_d": self.state.time_d,
                    "T_reactor_C": self.state.T_reactor_C,
                    "T_setpoint_C": self.state.T_setpoint_C,
                    "T_in_C": Tin_C,
                    "Q_m3_d": Q_m3_d,
                    "q_ch4_prod_m3_d": q_prod,
                    "q_ch4_heater_m3_d": q_heat_accounted,
                    "q_ch4_heater_thermal_m3_d": q_heat_thermal,
                    "pi_error_C": self.state.pi_error_C,
                    "pi_error_integral_C_d": self.state.pi_error_integral_C_d,
                }
            )

        return totals

    def reward_from_totals(self, totals: StepTotals, penalty: float) -> StepTotals:
        totals.penalty = penalty
        # ADDED: match the full PyADM1 plant's NPV-style reward.
        # Reason: the fast plant and full plant should score the same objective
        # whenever both are used for RL smoke tests or checklist validation.
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
        base_obs = np.array(
            [
                self.state.T_reactor_C,
                self.state.T_setpoint_C,
                Tin_C,
                Q_m3_d,
                self.state.q_ch4_prod_m3_d,
                self.accounted_heater_use_rate(self.state.q_ch4_heater_m3_d),
                self.state.pi_error_C,
                self.state.pi_error_integral_C_d,
                self.state.time_d - self.state.episode_start_d,
                np.sin(2.0 * np.pi * day_of_year / 365.0),
                np.cos(2.0 * np.pi * day_of_year / 365.0),
            ],
            dtype=np.float32,
        )
        if not self.config.include_adm_state_observation:
            return base_obs
        return np.concatenate(
            [base_obs, self.adm_state_values.astype(np.float32)],
            dtype=np.float32,
        )

    @property
    def observation_names(self) -> list[str]:
        base_names = [
            "T_reactor_C",
            "T_setpoint_C",
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
        if not self.config.include_adm_state_observation:
            return base_names
        return base_names + [f"ADM1_{name}" for name in self.adm_state_names]

    def temperature_derivative_C_d(
        self,
        T_reactor_C: float,
        Tin_C: float,
        Q_m3_d: float,
        q_ch4_heater_m3_d: float,
    ) -> float:
        T_K = T_reactor_C + 273.15
        Tin_K = Tin_C + 273.15
        feed_heat = (
            Q_m3_d
            * self._p("sludge_density_kg_m3")
            * self._p("sludge_specific_heat_MJ_kg_K")
            * (Tin_K - T_K)
            * self._p("feed_heat_time_factor")
        )
        loss = self.heat_loss_MJ_d(T_reactor_C)
        heater = (
            q_ch4_heater_m3_d
            * self._p("methane_LHV_MJ_m3")
            * self._p("heater_thermal_efficiency")
            * self._p("heater_heat_time_factor")
        )
        denominator = self._p("reactor_vol_heat_capacity_MJ_m3_K") * self._p("V_liq_m3", default=3400.0)
        return (feed_heat - loss + heater) / denominator

    def heat_loss_MJ_d(self, T_reactor_C: float) -> float:
        air_temp = self._p("air_temp_C")
        soil_temp = self._p("soil_temp_C")
        factor = self._p("heat_loss_time_factor")
        cover = self._p("digester_cover_area_m2") * self._p("cover_U_J_m2_h_K") * (T_reactor_C - air_temp) * factor / 1_000_000.0
        air_wall = self._p("digester_air_wall_area_m2") * self._p("air_wall_U_J_m2_h_K") * (T_reactor_C - air_temp) * factor / 1_000_000.0
        soil_wall = self._p("digester_soil_wall_area_m2") * self._p("soil_wall_U_J_m2_h_K") * (T_reactor_C - soil_temp) * factor / 1_000_000.0
        wet_soil = self._p("digester_soil_wall_area_m2") * self._p("wet_soil_wall_U_J_m2_h_K") * (T_reactor_C - soil_temp) * factor / 1_000_000.0
        return cover + air_wall + soil_wall + wet_soil

    def open_loop_ch4_flow(self, target_C: float, time_d: float) -> float:
        Tin_C, Q_m3_d = self.influent_at(time_d)
        T_K = target_C + 273.15
        Tin_K = Tin_C + 273.15
        feed_heat = (
            Q_m3_d
            * self._p("sludge_density_kg_m3")
            * self._p("sludge_specific_heat_MJ_kg_K")
            * (Tin_K - T_K)
            * self._p("feed_heat_time_factor")
        )
        required_heat = self.heat_loss_MJ_d(target_C) - feed_heat
        if required_heat <= 0.0:
            return 0.0
        return required_heat / (
            self._p("methane_LHV_MJ_m3")
            * self._p("heater_thermal_efficiency")
            * self._p("heater_heat_time_factor")
        )

    def accounted_heater_use_rate(self, thermal_q_ch4_m3_d: float) -> float:
        # ADDED: separate heat-balance MV from methane accounting.
        # Reason: the requested test keeps the same heater heat applied to the
        # reactor but assumes the methane amount required for that heat is 100x
        # smaller in reward/log accounting.
        # Role: report and accumulate methane use as thermal equivalent flow
        # multiplied by heater_methane_accounting_factor.
        # Reference: user-requested methane use factor under the unchanged
        # loss-factor-1440 thermal model.
        return float(thermal_q_ch4_m3_d) * self.config.heater_methane_accounting_factor

    def _update_pi_controller(self) -> None:
        assert self.state is not None
        error = self.state.T_setpoint_C - self.state.T_reactor_C
        dt_control = self.config.control_interval_h / 24.0
        integral_candidate = self.state.pi_error_integral_C_d + error * dt_control
        q_unclamped = self._p("controller_q_bias_m3_d") + self._p("controller_Kc_m3_d_C") * (
            error + integral_candidate / self._p("controller_Ti_d")
        )

        self.state.pi_error_C = error
        self.state.pi_error_integral_C_d = integral_candidate
        self.state.pi_last_error_C = error
        # ADDED: remove controller MV saturation for unconstrained RL testing.
        # Reason: user requested no lower/upper bounds from controller_q_min_m3_d
        # and controller_q_max_m3_d, so the PI output is applied directly.
        # Role: let T_setpoint actions explore reactor heating without the
        # previous heater-flow saturation around q_max.
        # Reference: user-requested removal of controller q min/max limits.
        self.state.q_ch4_heater_m3_d = float(q_unclamped)
        self.state.next_control_time_d += self.config.control_interval_h / 24.0

    def _setpoint_change_penalty(self, applied_delta_C: float) -> float:
        assert self.state is not None
        now = self.state.time_d
        self.state.daily_setpoint_changes.append((now, abs(applied_delta_C)))
        while self.state.daily_setpoint_changes and self.state.daily_setpoint_changes[0][0] < now - 1.0:
            self.state.daily_setpoint_changes.popleft()
        daily_change = sum(delta for _, delta in self.state.daily_setpoint_changes)
        excess = max(0.0, daily_change - self.config.daily_setpoint_change_limit_C)
        self.last_daily_setpoint_change_C = daily_change
        self.last_setpoint_penalty_excess_C = excess
        return self.config.shock_penalty_per_C * excess

    def _wrapped_time(self, time_d: float) -> float:
        max_time = float(self.time_grid[-1])
        if time_d <= max_time:
            return float(time_d)
        return float(time_d % max_time)

    def _p(self, name: str, default: float | None = None) -> float:
        if name in self.params and pd.notna(self.params[name]):
            return float(self.params[name])
        if default is not None:
            return float(default)
        raise KeyError(f"Missing thermal parameter: {name}")
