from __future__ import annotations

import gym
import numpy as np
from gym import spaces

from ad_rl.config import RLConfig
from ad_rl.full_pyadm1_plant import FullPyADM1PIPlant


class FullPyADM1PISetpointEnv(gym.Env):
    """Gym wrapper that calls full PyADM1 dynamics inside each step."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: RLConfig | None = None,
        episode_days: float | None = None,
        start_day: float = 0.0,
    ):
        super().__init__()
        self.config = config or RLConfig()
        self.episode_days = self.config.smoke_episode_days if episode_days is None else float(episode_days)
        self.start_day = float(start_day)
        self.max_episode_steps = int(np.ceil(self.episode_days * 24.0 / self.config.decision_interval_h))
        self.current_step = 0
        self.plant = FullPyADM1PIPlant(self.config)
        self.action_space = spaces.Discrete(len(self.config.action_deltas_C))
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
        options = options or {}
        start_day = float(options.get("start_day", self.start_day))
        setpoint = float(options.get("initial_t_setpoint_C", self.config.initial_t_setpoint_C))
        self.plant.reset(start_day=start_day, setpoint_C=setpoint)
        self.current_step = 0
        self.last_info = {"reset_start_day": start_day}
        return self.plant.observation()

    def step(self, action: int):
        action = int(action)
        if action < 0 or action >= len(self.config.action_deltas_C):
            raise ValueError(f"Invalid action index: {action}")

        delta_C = float(self.config.action_deltas_C[action])
        old_sp = self.plant.state.T_setpoint_C if self.plant.state is not None else np.nan
        penalty = self.plant.apply_setpoint_delta(delta_C)
        new_sp = self.plant.state.T_setpoint_C

        elapsed_before = self.plant.state.time_d - self.plant.state.episode_start_d
        remaining_h = max(0.0, (self.episode_days - elapsed_before) * 24.0)
        interval_h = min(self.config.decision_interval_h, remaining_h)
        totals = self.plant.simulate_interval(interval_h)
        penalty += self.plant.reactor_temperature_change_penalty()
        totals = self.plant.reward_from_totals(totals, penalty)
        physical_reward = float(totals.reward)
        reward_baseline = self._reward_baseline(interval_h)
        learning_reward = (
            physical_reward - reward_baseline
        ) / max(1e-12, self.config.reward_scale)

        reward_without_penalty = (
            self.config.reward_methane_production_weight
            * totals.methane_produced_m3
        )
        if self.config.reward_include_heater_consumption:
            reward_without_penalty -= (
                self.config.reward_heater_consumption_weight
                * totals.methane_consumed_m3
            )
        reward_without_penalty *= self.config.methane_price

        Tin_C, Q_m3_d = self.plant.influent_at(self.plant.state.time_d)
        tracking_error_C = self.plant.state.T_setpoint_C - self.plant.state.T_reactor_C
        heater_use_to_prod_pct = (
            self.plant.accounted_heater_use_rate(self.plant.state.q_ch4_heater_m3_d)
            / self.plant.state.q_ch4_prod_m3_d
            * 100.0
            if self.plant.state.q_ch4_prod_m3_d > 0.0
            else np.nan
        )

        elapsed = self.plant.state.time_d - self.plant.state.episode_start_d
        terminated = elapsed >= self.episode_days - 1e-12
        truncated = False
        obs = self.plant.observation()
        self.current_step += 1
        info = {
            "action_delta_C": delta_C,
            "old_T_setpoint_C": old_sp,
            "new_T_setpoint_C": new_sp,
            "methane_produced_m3": totals.methane_produced_m3,
            "methane_consumed_m3": totals.methane_consumed_m3,
            "net_methane_m3": totals.methane_produced_m3 - totals.methane_consumed_m3,
            "penalty": totals.penalty,
            "reward_without_penalty": reward_without_penalty,
            "physical_reward": physical_reward,
            "reward_baseline": reward_baseline,
            "reward": learning_reward,
            "time_d": self.plant.state.time_d,
            "episode_elapsed_d": elapsed,
            "episode_step": self.current_step,
            "max_episode_steps": self.max_episode_steps,
            "interval_h": interval_h,
            "T_reactor_C": self.plant.state.T_reactor_C,
            "T_setpoint_C": self.plant.state.T_setpoint_C,
            "T_in_C": Tin_C,
            "Q_m3_d": Q_m3_d,
            "pi_error_C": self.plant.state.pi_error_C,
            "pi_error_integral_C_d": self.plant.state.pi_error_integral_C_d,
            "tracking_error_C": tracking_error_C,
            "tracking_abs_error_C": abs(tracking_error_C),
            "tracking_squared_error_C2": tracking_error_C * tracking_error_C,
            "daily_setpoint_change_C": self.plant.last_daily_setpoint_change_C,
            "setpoint_penalty_excess_C": self.plant.last_setpoint_penalty_excess_C,
            "reactor_24h_change_C": self.plant.last_reactor_24h_change_C,
            "reactor_temp_penalty_event": self.plant.last_reactor_temp_penalty_event,
            "reactor_temp_penalty_excess_C": self.plant.last_reactor_temp_penalty_excess_C,
            "T_adapt_C": self.plant.engine.adapted_temperature_C(),
            "methanogenesis_shock_factor": self.plant.engine.methanogenesis_shock_factor(),
            "methanogenesis_temp_mismatch_K": self.plant.engine.methanogenesis_temp_mismatch_K(),
            "q_ch4_heater_m3_d": self.plant.accounted_heater_use_rate(
                self.plant.state.q_ch4_heater_m3_d
            ),
            "q_ch4_heater_thermal_m3_d": self.plant.state.q_ch4_heater_m3_d,
            "q_ch4_prod_m3_d": self.plant.state.q_ch4_prod_m3_d,
            "heater_use_to_prod_pct": heater_use_to_prod_pct,
        }
        info.update(self.plant.internal_delta_summary(top_n=8))
        terminated = terminated or self.current_step >= self.max_episode_steps
        self.last_info = info
        return obs, float(learning_reward), terminated, truncated, info

    def render(self):
        return self.last_info

    def _reward_baseline(self, interval_h: float) -> float:
        if not self.config.reward_subtract_baseline:
            return 0.0
        interval_d = float(interval_h) / 24.0
        return (
            self.config.methane_price
            * self.config.reward_baseline_methane_m3_d
            * interval_d
        )
