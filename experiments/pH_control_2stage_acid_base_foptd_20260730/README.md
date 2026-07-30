# 2-stage AD pH control: acid/base PRBS FOPTD result

Date: 2026-07-30

This directory preserves the pH-control result for a serial two-stage PyADM1 setup:

- Stage 1: 55 degC, 80% of liquid/gas volume
- Stage 2: 35 degC, 20% of liquid/gas volume
- Stage 1 effluent feeds Stage 2 influent
- pH controller observes and doses Stage 2
- Base dosing: 25 M NaOH through `S_cation`
- Acid dosing: 35 wt% HCl, approximated as 11.3 kmol/m3, through `S_anion`
- PRBS setpoint: pH 7.0 / 8.4 with random dwell length from 0.5 d to 3.0 d

## Identified equivalent FOPTD models

The fitted model uses the closed-loop PRBS acid/base dosing log after excluding the first 2 d transient.

Input basis: molar dosing rate, kmol/d.

```text
NaOH: G_base(s) =  0.1408 * exp(-0*s) / (2.30*s + 1)
HCl : G_acid(s) = -0.1325 * exp(-0*s) / (2.30*s + 1)
```

`s` is in 1/d.

Flow-rate equivalent gains:

| Input | Concentration | Gain | Tau | Theta |
| --- | ---: | ---: | ---: | ---: |
| NaOH | 25.0 kmol/m3 | +3.520 pH/(m3/d) | 2.30 d / 55.2 h | 0 h |
| HCl | 11.3 kmol/m3 | -1.497 pH/(m3/d) | 2.30 d / 55.2 h | 0 h |

Fit quality:

| Metric | Value |
| --- | ---: |
| MAE | 0.027 pH |
| RMSE | 0.037 pH |
| R2 | 0.997 |

`theta = 0` means delay was not identifiable above the 15 min simulation/logging resolution, not that the real process has no physical delay.

## Contents

- `code/`
  - `PyADM1_single_stage_reference.py`: shared ADM1 and pH-control routines
  - `PyADM1_pH_2stage.py`: serial 2-stage acid/base pH-control run
  - `PyADM1_pH_2stage_PRBS.py`: serial 2-stage PRBS tracking run
  - `identify_acid_base_foptd.py`: FOPTD identification script
- `results/`
  - `acid_base_control_log_2stage_serial_PRBS.csv`: pH setpoint, Stage 1/2 pH, NaOH/HCl dosing log
  - `pH_PRBS_schedule_2stage_serial.csv`: random dwell PRBS schedule
  - `serial_2stage_PRBS_acid_base_summary.csv`: PRBS tracking summary
  - `serial_2stage_PRBS_acid_base_segment_metrics.csv`: per-segment PRBS metrics
  - `serial_2stage_PRBS_acid_base_foptd_params.csv`: fitted FOPTD parameters
  - `serial_2stage_PRBS_acid_base_foptd_fit_metrics.csv`: FOPTD fit metrics
  - `serial_2stage_PRBS_acid_base_foptd_prediction.csv`: fitted pH trace
- `figures/`
  - PRBS tracking plots and FOPTD fit/step-response plots

## Interpretation note

These are closed-loop equivalent FOPTD parameters from PI-generated acid/base dosing, so they are useful as controller-tuning starting values. For pure process identification, run separate open-loop NaOH and HCl bump tests.
