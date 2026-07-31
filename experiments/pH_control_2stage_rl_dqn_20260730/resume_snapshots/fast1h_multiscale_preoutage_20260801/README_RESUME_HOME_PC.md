# Fast 1h multi-scale pH RL resume snapshot

This snapshot was prepared before the scheduled power outage on 2026-08-01.

It contains the lightweight files needed to resume or evaluate the current 2-stage pH-control RL work on another PC:

- `dqn_checkpoint_latest.pt`
- `ppo_checkpoint_latest.pt`
- `dqn_checkpoint_eval_ep500.pt`
- `dqn_episode_summary.csv`
- `ppo_episode_summary.csv`
- `action_table.csv`
- baseline and checkpoint evaluation summaries

The full `runs/` directories and large `training_steps.csv` files are intentionally not included.

## Snapshot status

The school-PC runs were stopped after the latest saved checkpoints were copied and verified.

```text
DQN latest checkpoint: episode 926 completed, best stochastic training reward = 113.128
PPO latest checkpoint: episode 906 completed, best stochastic training reward = 107.191
```

## Resume DQN from latest checkpoint

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_dqn_4actuator.py `
  --episodes 150 `
  --episode-days 7.0 `
  --decision-interval-h 1.0 `
  --run-name fast1h_home_dqn_resume_from_preoutage `
  --use-temperature-kinetics `
  --skip-baselines `
  --init-model .\experiments\pH_control_2stage_rl_dqn_20260730\resume_snapshots\fast1h_multiscale_preoutage_20260801\dqn_checkpoint_latest.pt
```

## Resume PPO from latest checkpoint

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_ppo_4actuator.py `
  --episodes 150 `
  --episode-days 7.0 `
  --decision-interval-h 1.0 `
  --run-name fast1h_home_ppo_resume_from_preoutage `
  --use-temperature-kinetics `
  --learning-rate 1e-4 `
  --update-every-episodes 8 `
  --update-epochs 6 `
  --minibatch-size 512 `
  --clip-coef 0.1 `
  --entropy-coef-start 0.002 `
  --entropy-coef-end 0.0002 `
  --entropy-decay-episodes 300 `
  --initial-dose-prior-strength 3.0 `
  --skip-baselines `
  --init-model .\experiments\pH_control_2stage_rl_dqn_20260730\resume_snapshots\fast1h_multiscale_preoutage_20260801\ppo_checkpoint_latest.pt
```

## Best known deterministic candidate before this snapshot

The best deterministic policy observed before preparing this snapshot was the DQN ep500 candidate:

```text
reward = 104.369
CH4 = 12746.2 m3
chemical = 1802.0 kmol
pH violation = 3.898 pH*d
```

It is saved as `dqn_checkpoint_eval_ep500.pt`.
