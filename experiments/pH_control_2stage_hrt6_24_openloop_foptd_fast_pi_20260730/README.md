# 2-stage AD pH control: HRT 6/24 acid-base FOPTD and PI tuning

Date: 2026-07-30

This directory preserves the current two-stage PyADM1 pH-control result after changing the process split to 2:8 and HRT to 6 d / 24 d.

## Process configuration

- Stage 1: 55 degC, 20% of liquid/gas volume, 6 d HRT.
- Stage 2: 35 degC, 80% of liquid/gas volume, 24 d HRT.
- Stage 1 effluent feeds Stage 2 influent.
- Influent flow: 178.4674 m3/d.
- Liquid volumes: Stage 1 = 1070.8044 m3, Stage 2 = 4283.2176 m3.
- Base dosing: 25 M NaOH, represented as `S_cation` addition.
- Acid dosing: 35 wt% HCl, approximated as 11.3 kmol/m3, represented as `S_anion` addition.
- PI output update interval: 3 h. Model output/log interval: 15 min.
- PRBS setpoint: pH 7.0 / 8.4 with irregular 0.5-3.0 d dwell length, seed 260714.

## Open-loop FOPTD/FORTD models

The models below are fitted from separate open-loop PRBS tests for each reactor and chemical. The transfer-function form is:

```text
G(s) = K exp(-theta s) / (tau s + 1)
```

For the flow-basis model, `K` is pH/(m3/d), `tau` and `theta` are in days, and `s` is in 1/d.

| reactor | chemical | model_flow_basis | gain_pH_per_kmol_d | tau_d | theta_h | RMSE_pH | R2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stage 1 (55 degC) | NaOH | +1.477683 exp(-0.000s)/(6.000s+1) | +0.059107 | 6.000 | 0.00 | 0.0005 | 0.9998 |
| Stage 1 (55 degC) | HCl | -0.633947 exp(-0.000s)/(6.000s+1) | -0.056101 | 6.000 | 0.00 | 0.0002 | 1.0000 |
| Stage 2 (35 degC) | NaOH | +0.564044 exp(-0.000s)/(3.000s+1) | +0.022562 | 3.000 | 0.00 | 0.0343 | 0.9169 |
| Stage 2 (35 degC) | HCl | -0.519885 exp(-0.000s)/(16.000s+1) | -0.046008 | 16.000 | 0.00 | 0.0503 | 0.8119 |

`theta = 0` means the delay was not identifiable above the 15 min simulation/logging resolution. It should not be interpreted as proof of zero physical transport or mixing delay.

## Controller tuning used in code

NaOH gains were moved more aggressively than the 1 d settling baseline to target about +0.1 pH initial up-step overshoot. HCl gains remain near the aggressive 1 d settling basis used before this NaOH overshoot search.

| reactor | chemical | Kp_m3_d_per_pH | Ki_m3_d_per_pH_d | active_in_current_serial_run |
| --- | --- | --- | --- | --- |
| Stage 1 | NaOH | 25.986622 | 4.331104 | False |
| Stage 1 | HCl | 37.858077 | 6.309680 | False |
| Stage 2 | NaOH | 34.039882 | 11.346627 | True |
| Stage 2 | HCl | 123.104132 | 7.694008 | True |

Current serial scripts control Stage 2 pH. Stage 1 acid/base gains are stored as code inputs for reactor-specific control experiments, but Stage 1 dosing is held at zero in the present serial PRBS run.

## ITAE and tracking result

ITAE is computed segment by segment as `integral(t_rel * abs(pH_sp - pH_stage2) dt)`, where `t_rel = t - segment_start`. Segment ITAE values are summed for the total score.

| metric | value | unit |
| --- | --- | --- |
| total_IAE | 32.134668 | pH*d |
| total_ITAE | 7.163673 | pH*d^2 |
| mean_abs_error_time_weighted | 0.114767 | pH |
| mean_segment_ITAE | 0.044773 | pH*d^2 |
| median_segment_ITAE | 0.042290 | pH*d^2 |
| settled_fraction_0p05 | 0.975000 | - |
| median_settle_time_0p05 | 0.296126 | d |
| max_settle_time_0p05 | 0.915160 | d |

Overshoot and pump saturation summary from the latest fast-PI PRBS check:

| metric | value | unit |
| --- | --- | --- |
| up_max_overshoot_pH | 0.114186 | pH |
| up_mean_overshoot_pH | 0.087812 | pH |
| down_max_undershoot_pH | 0.138345 | pH |
| q_NaOH_max_m3_d | 49.482616 | m3/d |
| q_HCl_max_m3_d | 100.000000 | m3/d |
| NaOH_sat_fraction | 0.000000 | - |
| HCl_sat_fraction | 0.035267 | - |

## Key figures

### Open-loop FOPTD fit by reactor and chemical

![Open-loop FOPTD fit](figures/openloop_prbs_foptd_fit_by_case.png)

### Open-loop step responses

![Open-loop FOPTD step responses](figures/openloop_prbs_foptd_step_responses.png)

### Fast PI PRBS tracking, 0-40 d

![Fast PI PRBS tracking, 0-40 d](figures/fast_pi_prbs_tracking_0_40d.png)

### Fast PI PRBS tracking, full 280 d

![Fast PI PRBS tracking, 280 d](figures/fast_pi_prbs_tracking_280d.png)

## Contents

- `code/`
  - Current pH-control PyADM1 scripts and local identification/analysis scripts.
- `results/openloop_prbs_foptd_params_by_reactor_chemical.csv`
  - Reactor-specific NaOH/HCl FOPTD fit parameters.
- `results/controller_tuning_parameters.csv`
  - PI gains currently entered in the code, with the linked FOPTD values.
- `results/controller_actual_input_values.csv`
  - Actual process, dosing, PRBS, and control constants used by the scripts.
- `results/fast_pi_prbs_itae_summary.csv`
  - Full-run ITAE, IAE, settling, and pump-use metrics.
- `results/fast_pi_prbs_segment_itae.csv`
  - Segment-level IAE/ITAE and dosing metrics.
- `results/acid_base_control_log_2stage_serial_PRBS.csv`
  - Latest 280 d PRBS tracking log used for the ITAE calculation.
