# Active-biomass DQN staged-training snapshot

Static 1 d horizon, 1 h decisions, temperature kinetics enabled. Training was
extended in stages through 600 cumulative episodes.

- Best deterministic checkpoint: `dqn_checkpoint_best_ep500.pt`
- Latest continuation checkpoint: `dqn_checkpoint_latest_ep600.pt`
- Episode 500 reward: 16.8848
- Episode 600 reward: 16.1783
- Best fixed open-loop reward: 16.3576
- Fixed pH7 PI reward: 15.5821

The checkpoint files include the Q network, target network, optimizer,
`global_step`, and completed episode count. Replay memory is not persisted.

Resume training from episode 600:

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_dqn_4actuator.py `
  --episodes 100 --episode-days 1.0 --decision-interval-h 1.0 `
  --run-name biomass_reward_dqn_1d_resume600_to700 `
  --reward-mode active_biomass --use-temperature-kinetics --skip-baselines `
  --init-model .\experiments\pH_control_2stage_rl_dqn_20260730\resume_snapshots\biomass_reward_staged_20260801\dqn_checkpoint_latest_ep600.pt
```

Use the ep500 checkpoint for evaluation or rollback, not as the continuation
head, unless intentionally branching the experiment from the best policy.
