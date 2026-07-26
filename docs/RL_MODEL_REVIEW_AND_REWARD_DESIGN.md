# RL model review and reward design notes

## Current diagnosis

The current RL environment is useful for smoke testing DQN logic, but it is not yet a paper-grade PyADM1 RL environment.

Current observation vector:

```text
T_reactor_C
T_setpoint_C
T_in_C
Q_m3_d
q_ch4_prod_m3_d
q_ch4_heater_m3_d
pi_error_C
pi_error_integral_C_d
episode_elapsed_d
season_sin
season_cos
```

The full ADM1 concentration and biomass states are not dynamically integrated in this fast wrapper. The plant reads `digester_initial_equilibrium.csv`, but those ADM variables are used only as the reset/equilibrium source, not as time-varying RL states.

## Why DQN keeps increasing T_setpoint after methane saturation

The fast methane model is:

```text
q_ch4_prod = baseline_q_ch4 * clip(1 + 0.01 * (T_reactor_C - 35), 0.85, 1.20)
```

Therefore methane production saturates once:

```text
1 + 0.01 * (T_reactor_C - 35) >= 1.20
T_reactor_C >= 55 C
```

The current reward is:

```text
reward = methane_produced - penalty
```

Heater methane use, high-temperature biological risk, and temperature feasibility are not subtracted. Once methane production has saturated, raising `T_setpoint` gives nearly flat immediate reward, but it is not punished. With DQN bootstrapping and limited training, this can leave a learned bias toward the positive action.

## State expansion needed for paper-grade RL

For a proper PyADM1-based RL model, add dynamic ADM states to the observation. At minimum:

```text
Biomass:
X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2

Key soluble states:
S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4, S_IC, S_IN

Gas states:
S_gas_h2, S_gas_ch4, S_gas_co2

Thermal and control:
T_reactor_C, T_setpoint_C, T_in_C, Q_m3_d, q_ch4_heater_m3_d, PI error/integral
```

These should come from the actual PyADM1 integration result, not from static initial-state columns.

## NPV reward design

A paper-grade reward should use a normalized NPV-like one-step profit:

```text
r_t =
  w_ch4 * norm(CH4_produced)
- w_heat * norm(CH4_heater_used)
- w_shock * norm(max(0, daily_delta_Tsp - 1 C))
- w_temp * norm(max(0, T_reactor - T_safe))
- w_violation * I(process_constraint_violation)
```

Prototype physical form:

```text
NPV_t =
  methane_price * (CH4_produced_m3 - CH4_heater_used_m3)
- operating_cost
- biological_risk_penalty
- setpoint_rate_penalty
```

Recommended fixed normalization:

```text
norm(CH4_produced) = CH4_produced_m3 / baseline_daily_CH4_m3
norm(CH4_heater_used) = CH4_heater_used_m3 / baseline_daily_CH4_m3
norm(delta_Tsp_excess) = max(0, daily_delta_Tsp - 1) / 1 C
norm(T_excess) = max(0, T_reactor - 55) / 10 C
```

This keeps reward magnitudes around order 1 and helps DQN training stability.

## Training checks

Before claiming convergence:

```text
1. Train longer than 200 episodes, then run greedy evaluation with epsilon=0.
2. Plot reward, action distribution, T_setpoint, T_reactor, q_ch4 production/use.
3. Confirm methane use and high-temperature penalties change action preference.
4. Compare standalone PI simulation vs Gym-wrapped simulation.
5. Verify reset reproducibility at the beginning of every episode.
6. Move from the fast surrogate to full PyADM1 state integration for final results.
```
