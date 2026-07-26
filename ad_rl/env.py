from __future__ import annotations

import gym
import numpy as np
from gym import spaces

from ad_rl.config import RLConfig
from ad_rl.plant import ThermalPIPlant


class PyADM1PISetpointEnv(gym.Env):
    """Gym wrapper for RL-based T_setpoint optimization."""

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
        self.plant = ThermalPIPlant(self.config)
        self.action_space = spaces.Discrete(len(self.config.action_deltas_C))

        obs_dim = len(self.plant.observation_names)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.last_info: dict = {}

    @property
    def observation_names(self) -> list[str]:
        return self.plant.observation_names

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # ADDED: deterministic Gym reset.
        # Reason: RL validation requires reset reproducibility and seasonal
        # episodes require configurable start days.
        # Role: reset the wrapped plant to ADM1 equilibrium plus current season
        # day and return an observation containing T_setpoint.
        # Reference: RL checklist R4 and R5.
        super().reset(seed=seed)
        options = options or {}
        start_day = float(options.get("start_day", self.start_day))
        setpoint = float(options.get("initial_t_setpoint_C", self.config.initial_t_setpoint_C))
        self.plant.reset(start_day=start_day, setpoint_C=setpoint)
        self.current_step = 0
        self.last_info = {"reset_start_day": start_day}
        return self.plant.observation()

    def step(self, action: int):
        # ADDED: RL action-to-setpoint bridge.
        # Reason: DQN action must change the PI controller's T_setpoint, then
        # the PI controller handles methane-heater MV inside the interval.
        # Role: map action index to delta T_setpoint, simulate N h, and return
        # NPV-style methane reward plus safety penalty.
        # Reference: user-requested action, reward, and penalty definitions.
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
        totals = self.plant.reward_from_totals(totals, penalty)
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
            "reward": totals.reward,
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
            "q_ch4_heater_m3_d": self.plant.accounted_heater_use_rate(
                self.plant.state.q_ch4_heater_m3_d
            ),
            "q_ch4_heater_thermal_m3_d": self.plant.state.q_ch4_heater_m3_d,
            "q_ch4_prod_m3_d": self.plant.state.q_ch4_prod_m3_d,
            "heater_use_to_prod_pct": heater_use_to_prod_pct,
        }
        terminated = terminated or self.current_step >= self.max_episode_steps
        self.last_info = info
        return obs, float(totals.reward), terminated, truncated, info

    def render(self):
        return self.last_info
