from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2] if (THIS_DIR.parents[2] / "ad_rl").exists() else None
if REPO_ROOT is not None and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ad_rl.ppo_agent import PPOAgent, PPORollout
from ph2stage_rl_env import (
    PHControlRLConfig,
    TwoStagePHDirectDosingEnv,
    config_to_json_dict,
    rollout_policy,
)
from train_dqn_4actuator import (
    build_comparison,
    evaluate_fixed_open_loop_baselines,
    evaluate_fixed_pH7_baseline,
    json_dump,
    output_base_dir,
    plot_rollout,
    save_csv,
)


def entropy_by_episode(
    episode: int,
    start: float,
    end: float,
    decay_episodes: int,
) -> float:
    frac = min(1.0, float(episode) / max(1.0, float(decay_episodes)))
    return float(start + frac * (end - start))


def summarize_episode(
    ep: int,
    global_step: int,
    ep_df: pd.DataFrame,
    episode_time_s: float,
    elapsed_run_time_s: float,
) -> dict:
    summary = {
        "episode": int(ep),
        "steps": int(len(ep_df)),
        "global_step_end": int(global_step),
        "episode_time_s": float(episode_time_s),
        "elapsed_run_time_s": float(elapsed_run_time_s),
    }
    for col in [
        "reward",
        "reward_raw",
        "reward_benefit",
        "reward_chemical_cost",
        "reward_ph_cost",
        "stage1_vfa_produced_kgCOD",
        "stage2_vfa_removed_kgCOD",
        "stage1_ch4_m3",
        "stage2_ch4_m3",
        "total_ch4_m3",
        "chemical_m3",
        "chemical_kmol",
        "ph_violation_pH_d",
        "stage1_active_biomass_ratio_d",
        "stage2_active_biomass_ratio_d",
        "stage1_active_biomass_relative_growth",
        "stage2_active_biomass_relative_growth",
    ]:
        if col in ep_df.columns:
            summary[f"total_{col}"] = float(ep_df[col].sum())
    if not ep_df.empty:
        summary["final_stage1_pH"] = float(ep_df["stage1_pH"].iloc[-1])
        summary["final_stage2_pH"] = float(ep_df["stage2_pH"].iloc[-1])
        summary["mean_stage1_pH"] = float(ep_df["stage1_pH"].mean())
        summary["mean_stage2_pH"] = float(ep_df["stage2_pH"].mean())
    return summary


def apply_initial_dose_prior(
    agent: PPOAgent,
    env: TwoStagePHDirectDosingEnv,
    strength: float,
) -> None:
    if strength <= 0.0:
        return
    table = env.action_table.copy()
    stage1_scale = max(1e-12, table["stage1_signed_m3_d"].abs().max())
    stage2_scale = max(1e-12, table["stage2_signed_m3_d"].abs().max())
    dose_distance = (
        table["stage1_signed_m3_d"].abs().to_numpy(dtype=float) / stage1_scale
        + table["stage2_signed_m3_d"].abs().to_numpy(dtype=float) / stage2_scale
    )
    logits = -float(strength) * dose_distance
    logits = logits - float(np.max(logits))
    with torch.no_grad():
        nn.init.zeros_(agent.network.policy_head.weight)
        agent.network.policy_head.bias.copy_(torch.tensor(logits, dtype=torch.float32))


def train_ppo(
    config: PHControlRLConfig,
    output_dir: Path,
    episodes: int,
    hidden_dim: int,
    update_every_episodes: int,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    gae_lambda: float,
    entropy_coef_start: float,
    entropy_coef_end: float,
    entropy_decay_episodes: int,
    value_coef: float,
    max_grad_norm: float,
    initial_dose_prior_strength: float,
    init_model: str | None = None,
) -> tuple[PPOAgent, pd.DataFrame, pd.DataFrame]:
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = TwoStagePHDirectDosingEnv(config)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(hidden_dim),
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=float(gae_lambda),
        clip_coef=float(clip_coef),
        value_coef=float(value_coef),
        entropy_coef=float(entropy_coef_start),
        max_grad_norm=float(max_grad_norm),
        update_epochs=int(update_epochs),
        minibatch_size=int(minibatch_size),
    )
    apply_initial_dose_prior(agent, env, float(initial_dose_prior_strength))
    if init_model:
        init_path = Path(init_model)
        state = torch.load(init_path, map_location="cpu")
        if isinstance(state, dict) and "network_state_dict" in state:
            agent.network.load_state_dict(state["network_state_dict"])
            if "optimizer_state_dict" in state:
                agent.optimizer.load_state_dict(state["optimizer_state_dict"])
        else:
            agent.network.load_state_dict(state)
        print(f"loaded PPO init model: {init_path}", flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    env.action_table.to_csv(output_dir / "action_table.csv", index=False)
    json_dump(output_dir / "config.json", config_to_json_dict(config))
    json_dump(
        output_dir / "ppo_hyperparameters.json",
        {
            "hidden_dim": int(hidden_dim),
            "update_every_episodes": int(update_every_episodes),
            "update_epochs": int(update_epochs),
            "minibatch_size": int(minibatch_size),
            "clip_coef": float(clip_coef),
            "gae_lambda": float(gae_lambda),
            "entropy_coef_start": float(entropy_coef_start),
            "entropy_coef_end": float(entropy_coef_end),
            "entropy_decay_episodes": int(entropy_decay_episodes),
            "value_coef": float(value_coef),
            "max_grad_norm": float(max_grad_norm),
            "initial_dose_prior_strength": float(initial_dose_prior_strength),
        },
    )

    training_rows: list[dict] = []
    summary_rows: list[dict] = []
    update_rows: list[dict] = []
    batch_rollout = PPORollout()
    batch_episode_count = 0
    global_step = 0
    update_index = 0
    run_t0 = time.time()

    for ep in range(int(episodes)):
        agent.entropy_coef = entropy_by_episode(
            ep,
            entropy_coef_start,
            entropy_coef_end,
            entropy_decay_episodes,
        )
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        episode_rollout = PPORollout()
        ep_rows_start = len(training_rows)
        ep_t0 = time.time()
        ep_step = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_rollout.push(obs, action, log_prob, reward, done, value)

            row = {
                "episode": ep,
                "episode_step": ep_step,
                "global_step": global_step,
                "action": int(action),
                "log_prob": float(log_prob),
                "value_estimate": float(value),
                "entropy_coef": float(agent.entropy_coef),
                "update_index": int(update_index),
            }
            row.update(info)
            training_rows.append(row)

            obs = next_obs
            global_step += 1
            ep_step += 1

        batch_rollout.extend(episode_rollout)
        batch_episode_count += 1

        ep_df = pd.DataFrame(training_rows[ep_rows_start:])
        summary = summarize_episode(
            ep,
            global_step,
            ep_df,
            time.time() - ep_t0,
            time.time() - run_t0,
        )

        should_update = (
            batch_episode_count >= max(1, int(update_every_episodes))
            or ep == int(episodes) - 1
        )
        metrics = {
            "policy_loss": np.nan,
            "value_loss": np.nan,
            "entropy": np.nan,
            "approx_kl": np.nan,
        }
        if should_update:
            metrics = agent.update(batch_rollout, last_value=0.0)
            update_row = {
                "update_index": int(update_index),
                "episode_end": int(ep),
                "batch_episodes": int(batch_episode_count),
                "batch_steps": int(len(batch_rollout)),
                "entropy_coef": float(agent.entropy_coef),
            }
            update_row.update(metrics)
            update_rows.append(update_row)
            update_index += 1
            batch_rollout = PPORollout()
            batch_episode_count = 0

        summary.update({f"update_{key}": float(value) for key, value in metrics.items()})
        summary_rows.append(summary)

        save_csv(training_rows, output_dir / "training_steps.csv")
        save_csv(summary_rows, output_dir / "episode_summary.csv")
        save_csv(update_rows, output_dir / "ppo_update_summary.csv")
        torch.save(
            {
                "network_state_dict": agent.network.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "episode": int(ep),
                "global_step": int(global_step),
            },
            output_dir / "checkpoint_latest.pt",
        )
        print(
            f"episode {ep + 1:03d}/{episodes} "
            f"reward={summary.get('total_reward', np.nan):.3f} "
            f"raw={summary.get('total_reward_raw', np.nan):.3f} "
            f"chem={summary.get('total_chemical_kmol', np.nan):.3f} "
            f"policy_loss={summary.get('update_policy_loss', np.nan):.3f} "
            f"time={summary['episode_time_s']:.1f}s",
            flush=True,
        )

    torch.save(agent.network.state_dict(), output_dir / "ppo_actor_critic.pt")
    return agent, pd.DataFrame(training_rows), pd.DataFrame(summary_rows)


def evaluate_ppo_policy(config: PHControlRLConfig, agent: PPOAgent, output_dir: Path) -> dict:
    decision_df, internal_df, summary = rollout_policy(
        config,
        lambda obs, _env: agent.act(obs, deterministic=True),
    )
    decision_df.to_csv(output_dir / "ppo_policy_decision_steps.csv", index=False)
    internal_df.to_csv(output_dir / "ppo_policy_internal_timeseries.csv", index=False)
    summary = dict(policy="ppo_deterministic", **summary)
    json_dump(output_dir / "ppo_policy_summary.json", summary)
    plot_rollout(output_dir, internal_df, "PPO deterministic policy", "ppo_policy_rollout.png")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--episode-days", type=float, default=2.0)
    parser.add_argument("--decision-interval-h", type=float, default=1.0)
    parser.add_argument("--run-name", default="ph2stage_ppo_4actuator_smoke")
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--update-every-episodes", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--entropy-coef-start", type=float, default=0.02)
    parser.add_argument("--entropy-coef-end", type=float, default=0.005)
    parser.add_argument("--entropy-decay-episodes", type=int, default=100)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--initial-dose-prior-strength", type=float, default=0.0)
    parser.add_argument(
        "--reward-mode",
        choices=["methane_total", "stage2_methane", "staged_vfa_ch4", "active_biomass"],
        default="methane_total",
    )
    parser.add_argument("--methane-reward-weight", type=float, default=1.0)
    parser.add_argument("--biomass-maintenance-weight", type=float, default=1000.0)
    parser.add_argument("--biomass-growth-weight", type=float, default=5000.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--chemical-kmol-weight", type=float, default=0.2)
    parser.add_argument("--ph-violation-weight", type=float, default=500.0)
    parser.add_argument("--influent-csv", default="digester_influent_mean_full.csv")
    parser.add_argument("--use-dynamic-flow", action="store_true")
    parser.add_argument("--temperature-parameter-csv", default="adm1_temperature_parameters_long.csv")
    parser.add_argument("--use-temperature-kinetics", action="store_true")
    parser.add_argument(
        "--episode-start-mode",
        choices=["fixed", "random"],
        default="fixed",
    )
    parser.add_argument("--episode-start-day", type=float, default=0.0)
    parser.add_argument("--episode-start-day-min", type=float, default=0.0)
    parser.add_argument("--episode-start-day-max", type=float, default=None)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ph7-baseline", action="store_true")
    parser.add_argument("--baseline-max-actions", type=int, default=None)
    args = parser.parse_args()

    config = replace(
        PHControlRLConfig(),
        episode_days=float(args.episode_days),
        decision_interval_h=float(args.decision_interval_h),
        learning_rate=float(args.learning_rate),
        reward_mode=str(args.reward_mode),
        methane_reward_weight=float(args.methane_reward_weight),
        biomass_maintenance_weight=float(args.biomass_maintenance_weight),
        biomass_growth_weight=float(args.biomass_growth_weight),
        reward_scale=float(args.reward_scale),
        chemical_kmol_weight=float(args.chemical_kmol_weight),
        ph_violation_weight=float(args.ph_violation_weight),
        influent_csv=str(args.influent_csv),
        use_dynamic_flow=bool(args.use_dynamic_flow),
        temperature_parameter_csv=str(args.temperature_parameter_csv),
        use_temperature_kinetics=bool(args.use_temperature_kinetics),
        episode_start_mode=str(args.episode_start_mode),
        episode_start_day=float(args.episode_start_day),
        episode_start_day_min=float(args.episode_start_day_min),
        episode_start_day_max=args.episode_start_day_max,
    )
    output_dir = output_base_dir() / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    agent, _training, _summary = train_ppo(
        config=config,
        output_dir=output_dir,
        episodes=args.episodes,
        hidden_dim=args.hidden_dim,
        update_every_episodes=args.update_every_episodes,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        clip_coef=args.clip_coef,
        gae_lambda=args.gae_lambda,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        entropy_decay_episodes=args.entropy_decay_episodes,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        initial_dose_prior_strength=args.initial_dose_prior_strength,
        init_model=args.init_model,
    )
    ppo_summary = evaluate_ppo_policy(config, agent, output_dir)

    if args.skip_baselines:
        comparison = pd.DataFrame([ppo_summary])
        comparison.to_csv(output_dir / "comparison_summary.csv", index=False)
    else:
        baseline_df = evaluate_fixed_open_loop_baselines(
            config,
            output_dir,
            max_actions=args.baseline_max_actions,
        )
        fixed_pH7_summary = None
        if not args.skip_ph7_baseline:
            fixed_pH7_summary = evaluate_fixed_pH7_baseline(config, output_dir)
        comparison = build_comparison(output_dir, ppo_summary, baseline_df, fixed_pH7_summary)
        print(
            comparison[
                ["policy", "total_reward", "total_reward_raw", "total_chemical_kmol"]
            ].to_string(index=False)
        )

    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
