# AD_RL

Reinforcement-learning experiments for anaerobic digester temperature setpoint
optimization using a PyADM1 thermal model, PI methane-heater control, and a DQN
supervisory controller.

## What Is Included

This repository contains the minimal runnable version of the latest experiment:

- full PyADM1 thermal engine wrapper
- PI controller for methane-based heating
- Gym environment for `T_setpoint` control
- DQN training code
- static-mean influent and equilibrium initial state data
- representative figures and 60-70 d diagnostics from the best checkpoint

Large run artifacts are intentionally excluded from git:

- full `training_steps.csv`
- full internal rollout time series
- all intermediate checkpoints
- `__pycache__` and local logs

## Main Files

Root-level scripts are the commands most users run directly:

- `train_full_pyadm1_temp_penalty_consumption.py`: main DQN training script
- `validate_full_pyadm1_env.py`: quick full-engine validation
- `plot_full_pyadm1_episode_internal_timeseries.py`: diagnostic rollout plotter

Supporting code is grouped under subfolders:

- `ad_rl/`: reusable environment, plant, PyADM1 engine, config, and DQN modules
- `scripts/`: helper or legacy training/plotting utilities imported by the main scripts
- `docs/`: model/reward design notes
- `data/`: minimal input files needed to run the model
- `examples/`: lightweight representative figures and summary CSVs

## Data

Required input data are under `data/`:

- `digester_influent_static_mean.csv`
- `digester_initial_equilibrium.csv`
- `thermal_parameters_bsm2Tin_itaePI.csv`
- `thermal_inputs_bsm2Tin_itaePI_baseline.csv`
- `full_pyadm1_source/PyADM1_thermal.py`
- `full_pyadm1_engine_inputs/`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Validation

```powershell
python .\validate_full_pyadm1_env.py
```

This validates that the Gym wrapper can reset the full PyADM1 state and advance
one or more RL decision steps.

## Example Training Command

Small smoke run:

```powershell
python .\train_full_pyadm1_temp_penalty_consumption.py --episodes 2 --episode-days 1 --run-name smoke_full_pyadm1
```

Long run used for the current saved result:

```powershell
python .\train_full_pyadm1_temp_penalty_consumption.py --episodes 1000 --episode-days 90 --run-name net_methane_no_penalty_1000ep --production-weight 1 --consumption-weight 1 --temp-penalty-per-event 0 --reward-scale 100 --learning-rate 2e-4 --target-update-steps 500 --replay-capacity 100000 --t-setpoint-min-C 25 --t-setpoint-max-C 65 --eval-checkpoints
```

## Current Reference Result

Best checkpoint from the 1000-episode run:

- checkpoint episode: 820
- 90 d methane produced: 153,031.61 m3
- 90 d accounted heater methane used: 1,938.70 m3
- 90 d net methane: 151,092.91 m3
- final reactor temperature: 47.80 C
- final setpoint: 48.00 C

Representative figures are stored under `examples/best_checkpoint_rollout/`.

## Additional Algorithm Scripts

The same PyADM1 + PI-control Gym environment can also be trained with the two
reference-inspired algorithms below. The action remains the same supervisory
`T_setpoint` change used by the DQN script.

QMIX-style online smoke run:

```powershell
python .\train_full_pyadm1_qmix.py --episodes 2 --episode-days 1 --run-name qmix_smoke
```



PPO-style online smoke run:

```powershell
python .\train_full_pyadm1_ppo.py --episodes 2 --episode-days 1 --run-name ppo_smoke
```

Discrete IQL/IDQL-style offline smoke run:

```powershell
python .\train_full_pyadm1_iql.py --dataset-episodes 2 --updates 100 --episode-days 1 --run-name iql_smoke
```

Notes:

- `train_full_pyadm1_qmix.py` keeps `n_agents=1` so that the control mechanism
  is identical to the DQN setpoint controller. The mixer remains in the code so
  the implementation can later be extended to multiple coordinated decisions.
- `train_full_pyadm1_iql.py` implements a discrete-action IQL variant in PyTorch.
  The supplied SBR reference uses JAXRL5/DDPM-IQL for offline RL, but PyADM1
  currently uses a discrete setpoint action, so a discrete IQL implementation is
  the minimal dependency version.
