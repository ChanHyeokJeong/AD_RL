# Codex Home-PC Continuation Prompt

Copy and paste the prompt below into Codex on the home PC.

````text
You are continuing my AD_RL two-stage pH-control reinforcement-learning work on my home PC.

Main goal:
1. Work from GitHub branch `agent/ph-acid-base-foptd-results`.
2. Resume the fast 1h multi-scale DQN/PPO runs from the pre-outage snapshot.
3. Compare zero dosing, best fixed open-loop dosing, fixed pH7 PI, DQN, and PPO under the same static 7d temperature-kinetic condition.
4. Save the final CSV summaries and plots.
5. Commit and push the relevant small result files if useful.

First prepare the repository.

If the repository is not cloned yet:

```powershell
git clone https://github.com/ChanHyeokJeong/AD_RL.git
cd AD_RL
git checkout agent/ph-acid-base-foptd-results
pip install -r requirements.txt
```

If the repository already exists:

```powershell
cd AD_RL
git fetch origin
git checkout agent/ph-acid-base-foptd-results
git pull origin agent/ph-acid-base-foptd-results
pip install -r requirements.txt
```

Important paths:

```text
experiments/pH_control_2stage_rl_dqn_20260730/
experiments/pH_control_2stage_rl_dqn_20260730/resume_snapshots/fast1h_multiscale_preoutage_20260801/
```

Snapshot status:

```text
DQN latest checkpoint: episode 926 completed, 73 episodes remaining to reach 1000
PPO latest checkpoint: episode 906 completed, 93 episodes remaining to reach 1000
Best known deterministic candidate before outage: dqn_checkpoint_eval_ep500.pt
```

Resume DQN:

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_dqn_4actuator.py `
  --episodes 73 `
  --episode-days 7.0 `
  --decision-interval-h 1.0 `
  --run-name fast1h_home_dqn_resume_from_preoutage `
  --use-temperature-kinetics `
  --skip-baselines `
  --init-model .\experiments\pH_control_2stage_rl_dqn_20260730\resume_snapshots\fast1h_multiscale_preoutage_20260801\dqn_checkpoint_latest.pt
```

Resume PPO:

```powershell
python .\experiments\pH_control_2stage_rl_dqn_20260730\code\train_ppo_4actuator.py `
  --episodes 93 `
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

After both runs finish:

1. Read each run folder under:
   `experiments/pH_control_2stage_rl_dqn_20260730/runs/`
2. Compare:
   - zero dosing
   - best fixed open-loop dosing
   - fixed pH7 PI
   - resumed DQN deterministic policy
   - resumed PPO deterministic policy
   - saved DQN ep500 deterministic candidate
3. Summarize these metrics:
   - total reward
   - total raw reward
   - total CH4 production, m3
   - chemical use, kmol
   - pH violation, pH*d
   - dominant actions
4. Create final plots:
   - DQN/PPO reward learning curve
   - policy rollout comparison
   - methane production vs chemical use comparison
5. Save outputs under:
   - `experiments/pH_control_2stage_rl_dqn_20260730/results/`
   - `experiments/pH_control_2stage_rl_dqn_20260730/figures/`

Important notes:

- `runs/` and `*.pt` are ignored by `.gitignore`.
- Do not commit large `training_steps.csv` files.
- If a small checkpoint must be shared, use `git add -f`.
- The latest checkpoint is not always the best deterministic policy.
- Always choose the final DQN/PPO policy by deterministic rollout performance, not by the latest episode number alone.

Report back to me with:

```text
DQN final reward / CH4 / chemical / pH violation
PPO final reward / CH4 / chemical / pH violation
Difference vs fixed pH7 PI
Best deterministic checkpoint: latest or ep500 candidate
Saved result paths
Whether GitHub was pushed
```
````
