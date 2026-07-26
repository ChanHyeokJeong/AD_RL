from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RUN_DIR = PROJECT_DIR / "runs"


@dataclass
class RLConfig:
    # ADDED: RL experiment file locations.
    # Reason: keep the RL wrapper separate from the validated PI/PyADM1
    # experiment folders while reusing their BSM2 influent and PI parameters.
    # Role: provide deterministic data paths to the Gym environment and tests.
    # Reference: user-requested separate RL task folder.
    influent_path: Path = DATA_DIR / "digester_influent_static_mean.csv"
    initial_state_path: Path = DATA_DIR / "digester_initial_equilibrium.csv"
    thermal_parameters_path: Path = DATA_DIR / "thermal_parameters_bsm2Tin_itaePI.csv"
    baseline_thermal_inputs_path: Path = DATA_DIR / "thermal_inputs_bsm2Tin_itaePI_baseline.csv"

    # ADDED: full PyADM1 engine source for publication-grade RL validation.
    # Reason: the fast RL plant exposes ADM1 states but does not integrate
    # them; this path points to the monolithic PyADM1_thermal.py that will be
    # wrapped as an engine and called inside Gym step intervals.
    # Role: keep the full engine optional and configurable without replacing
    # the current smoke-test surrogate plant.
    # Reference: user-requested "wrap original PyADM1_thermal as an engine".
    full_pyadm1_source_path: Path = DATA_DIR / "full_pyadm1_source" / "PyADM1_thermal.py"
    full_pyadm1_engine_input_dir: Path = DATA_DIR / "full_pyadm1_engine_inputs"
    full_pyadm1_solver_method: str = "DOP853"
    full_pyadm1_rtol: float = 1e-6
    full_pyadm1_atol: float = 1e-9

    # ADDED: decision and episode horizons.
    # Reason: RL should change T_setpoint every N hours, while code smoke tests
    # should finish quickly before seasonal 90 d episodes are used.
    # Role: default 24 h aligns with the 1 d microbial-shock penalty and can be
    # changed after a dedicated literature review of AD supervisory intervals.
    # Reference: user-requested 1 season episode and 1 d code test.
    decision_interval_h: float = 24.0
    control_interval_h: float = 1.0
    simulation_dt_h: float = 0.25
    smoke_episode_days: float = 1.0
    season_episode_days: float = 90.0

    # ADDED: setpoint action definition.
    # Reason: DQN needs a discrete action set; start with the requested three
    # actions and leave a single place to extend to five or seven actions.
    # Role: action index maps to T_setpoint change in degC.
    # Reference: user-requested initial actions [-0.5, 0, +0.5].
    action_deltas_C: np.ndarray = field(
        default_factory=lambda: np.array([-0.5, 0.0, 0.5], dtype=float)
    )
    initial_t_setpoint_C: float = 35.0
    # ADDED: hard operating range for DQN supervisory setpoints.
    # Reason: unbounded T_setpoint drift made policies move to unrealistic
    # low/high temperatures during long net-methane training.
    # Role: clamp action-updated T_setpoint to a broad but physical range.
    # Reference: user-requested 25-65 C setpoint range for this run.
    t_setpoint_min_C: float | None = 25.0
    t_setpoint_max_C: float | None = 65.0
    # ADDED: keep full ADM1 states out of the default agent observation.
    # Reason: the publication question is not that the DQN must observe every
    # ADM1 component, but that each T_setpoint action can be diagnosed by its
    # effect on the reactor internals after the full PyADM1 step.
    # Role: use compact observations for control and write ADM1 pre/post/delta
    # diagnostics separately.
    # Reference: user-requested action-impact view instead of all-state obs.
    include_adm_state_observation: bool = False
    # ADDED: methanogenesis temperature-shock state/observation switch.
    # Reason: quick reactor-temperature changes can inhibit methanogens before
    # biomass adapts to the new temperature.
    # Role: append T_adapt to the full PyADM1 engine state and expose
    # T_adapt/F_K diagnostics to the Gym observation when enabled.
    # Reference: Yuki thesis Eq. 4.54-4.55, tau_a=30 d and s_hg=5 K.
    use_methanogenesis_temp_shock: bool = True
    include_shock_state_observation: bool = True

    # ADDED: reward and safety constants.
    # Reason: this static-input RL variant tests a reward that ignores heater
    # methane loss and scores only produced methane.
    # Role: keep the same thermal heater power in the heat balance, but account
    # only 1% of the methane-equivalent heater flow as methane use in logs.
    # Reference: user-requested production-only reward and 100x lower methane
    # use factor under the same 1440 heat-loss scaling.
    methane_price: float = 1.0
    reward_include_heater_consumption: bool = False
    # ADDED: independent production benefit multiplier for NPV-style reward.
    # Reason: methane production is the primary objective, while heater methane
    # use and shock penalties are secondary costs; production needs a separate
    # coefficient instead of sharing one price factor with consumption.
    # Role: reward = price*(w_prod*CH4_prod - w_cons*CH4_used) - penalty.
    # Reference: user-requested NPV reward with a higher methane-production weight.
    reward_methane_production_weight: float = 1.0
    reward_heater_consumption_weight: float = 1.0
    reward_scale: float = 1.0
    reward_subtract_baseline: bool = False
    reward_baseline_methane_m3_d: float = 0.0
    heater_methane_accounting_factor: float = 0.01
    use_static_methane_production_baseline: bool = True
    daily_setpoint_change_limit_C: float = 1.0
    shock_penalty_per_C: float = 10_000.0
    # ADDED: selectable thermal-shock penalty definition.
    # Reason: setpoint-change penalties can punish command movement even when
    # the reactor itself changes slowly; the publication model should also test
    # a penalty based on actual reactor temperature change over 24 h.
    # Role: "setpoint_change_24h" preserves the previous behavior, while
    # "reactor_24h_event" penalizes each decision step where the reactor
    # temperature differs from its 24 h previous value by more than the limit.
    # Reference: user-requested 24 h T_reactor-change count penalty.
    penalty_mode: str = "setpoint_change_24h"
    reactor_temp_change_window_h: float = 24.0
    reactor_temp_change_limit_C: float = 1.0
    reactor_temp_penalty_per_event: float = 100.0
    reactor_temp_penalty_per_C: float = 10_000.0
    normalize_observation: bool = False

    # ADDED: DQN defaults for smoke testing.
    # Reason: keep the implementation executable without a long training run.
    # Role: test that the agent loop, replay buffer, and env interface work.
    # Reference: user-requested DQN agent and 1 d code test.
    random_seed: int = 20260715
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 32
    replay_capacity: int = 10_000
    warmup_steps: int = 16
    target_update_steps: int = 50
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 500


def ensure_run_dir() -> Path:
    RUN_DIR.mkdir(exist_ok=True)
    return RUN_DIR

