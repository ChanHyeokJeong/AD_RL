# Active-biomass DQN without pH violation cost

The active-biomass reward now uses zero pH-violation weight by default. pH
violations remain logged diagnostics and are not subtracted from reward.

Static 1 d horizon, 1 h decisions, temperature kinetics enabled:

- Best fine-tuned checkpoint: `dqn_checkpoint_best_ep700.pt`
- Latest continuation checkpoint: `dqn_checkpoint_latest_ep900.pt`
- Ep700 deterministic reward: 18.7431
- Ep900 deterministic reward: 18.3318
- Best fixed open-loop reward: 19.1333
- Zero-dosing reward: 19.1292
- Fixed pH7 PI reward: 19.0120

The ep700 policy preserved active biomass well but accumulated 0.943 pH*d of
diagnostic pH violation. The ep900 policy increased this to 1.354 pH*d while
scoring lower, so it is not the best policy.

Resume from episode 900:

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_dqn_4actuator.py `
  --episodes 100 --episode-days 1.0 --decision-interval-h 1.0 `
  --run-name biomass_reward_no_ph_penalty_resume900_to1000 `
  --reward-mode active_biomass --use-temperature-kinetics --skip-baselines `
  --init-model .\experiments\pH_control_2stage_rl_dqn_20260730\resume_snapshots\biomass_reward_no_ph_penalty_20260801\dqn_checkpoint_latest_ep900.pt
```
