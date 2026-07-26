from __future__ import annotations

import argparse
import json
import os
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from ad_rl.config import RUN_DIR
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv
from ad_rl.ppo_agent import PPOAgent, PPORollout
from ad_rl.rl_utils import (
    deterministic_rollout_policy,
    json_ready_config,
    plot_rollout_basic,
    plot_training_reward,
    save_csv,
)
from train_full_pyadm1_temp_penalty_consumption import config_for_run


def entropy_coef_at_episode(
    episode: int,
    entropy_coef_start: float,
    entropy_coef_end: float,
    entropy_decay_episodes: int,
) -> float:
    if entropy_decay_episodes <= 0:
        return float(entropy_coef_end)
    frac = min(1.0, max(0.0, episode / float(entropy_decay_episodes)))
    return float(entropy_coef_start + frac * (entropy_coef_end - entropy_coef_start))


def train(
    episodes: int,
    episode_days: float,
    run_name: str,
    production_weight: float,
    consumption_weight: float,
    temp_penalty_per_event: float,
    reward_scale: float,
    learning_rate: float,
    t_setpoint_min_C: float | None,
    t_setpoint_max_C: float | None,
    update_epochs: int,
    minibatch_size: int,
    gae_lambda: float,
    clip_coef: float,
    entropy_coef_start: float,
    entropy_coef_end: float,
    entropy_decay_episodes: int,
    value_coef: float,
    rollout_episodes_per_update: int,
):
    config = config_for_run(
        episode_days=episode_days,
        production_weight=production_weight,
        consumption_weight=consumption_weight,
        temp_penalty_per_event=temp_penalty_per_event,
        reward_scale=reward_scale,
        learning_rate=learning_rate,
        target_update_steps=500,
        replay_capacity=100_000,
        t_setpoint_min_C=t_setpoint_min_C,
        t_setpoint_max_C=t_setpoint_max_C,
    )
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        learning_rate=learning_rate,
        gamma=config.gamma,
        gae_lambda=gae_lambda,
        clip_coef=clip_coef,
        entropy_coef=entropy_coef_start,
        value_coef=value_coef,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
    )

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        meta = json_ready_config(config)
        meta["algorithm"] = "ppo_discrete_setpoint"
        meta["ppo_update_epochs"] = int(update_epochs)
        meta["ppo_minibatch_size"] = int(minibatch_size)
        meta["ppo_gae_lambda"] = float(gae_lambda)
        meta["ppo_clip_coef"] = float(clip_coef)
        meta["ppo_entropy_coef_start"] = float(entropy_coef_start)
        meta["ppo_entropy_coef_end"] = float(entropy_coef_end)
        meta["ppo_entropy_decay_episodes"] = int(entropy_decay_episodes)
        meta["ppo_value_coef"] = float(value_coef)
        meta["ppo_rollout_episodes_per_update"] = int(rollout_episodes_per_update)
        json.dump(meta, f, indent=2, ensure_ascii=False)

    training_rows: list[dict] = []
    summary_rows: list[dict] = []
    global_step = 0
    run_t0 = time.time()
    batch_rollout = PPORollout()
    batch_episode_count = 0
    rollout_episodes_per_update = max(1, int(rollout_episodes_per_update))

    for ep in range(int(episodes)):
        ep_t0 = time.time()
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        episode_rollout = PPORollout()
        ep_reward = 0.0
        ep_physical_reward = 0.0
        ep_reward_baseline = 0.0
        ep_produced = 0.0
        ep_consumed = 0.0
        ep_penalty = 0.0
        ep_temp_penalty_events = 0.0
        ep_steps = 0

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            episode_rollout.push(obs, action, log_prob, reward, done, value)

            row = {
                "episode": ep,
                "step": ep_steps,
                "global_step": global_step,
                "action": action,
                "log_prob": log_prob,
                "value": value,
            }
            row.update(info)
            training_rows.append(row)

            ep_reward += float(reward)
            ep_physical_reward += float(info["physical_reward"])
            ep_reward_baseline += float(info["reward_baseline"])
            ep_produced += float(info["methane_produced_m3"])
            ep_consumed += float(info["methane_consumed_m3"])
            ep_penalty += float(info["penalty"])
            ep_temp_penalty_events += float(info["reactor_temp_penalty_event"])
            obs = next_obs
            ep_steps += 1
            global_step += 1

        batch_rollout.extend(episode_rollout)
        batch_episode_count += 1
        metrics = {
            "policy_loss": np.nan,
            "value_loss": np.nan,
            "entropy": np.nan,
            "approx_kl": np.nan,
            "ppo_updated": 0,
            "ppo_batch_episodes": batch_episode_count,
            "ppo_entropy_coef": agent.entropy_coef,
        }
        should_update = (
            batch_episode_count >= rollout_episodes_per_update
            or ep == int(episodes) - 1
        )
        if should_update:
            current_entropy_coef = entropy_coef_at_episode(
                ep,
                entropy_coef_start,
                entropy_coef_end,
                entropy_decay_episodes,
            )
            agent.entropy_coef = current_entropy_coef
            metrics = agent.update(batch_rollout, last_value=0.0)
            metrics.update(
                {
                    "ppo_updated": 1,
                    "ppo_batch_episodes": batch_episode_count,
                    "ppo_entropy_coef": current_entropy_coef,
                }
            )
            batch_rollout = PPORollout()
            batch_episode_count = 0

        summary_rows.append(
            {
                "episode": ep,
                "steps": ep_steps,
                "global_step_end": global_step,
                "episode_reward": ep_reward,
                "episode_physical_reward": ep_physical_reward,
                "episode_reward_baseline": ep_reward_baseline,
                "episode_methane_produced_m3": ep_produced,
                "episode_methane_consumed_m3": ep_consumed,
                "episode_net_methane_m3": ep_produced - ep_consumed,
                "episode_penalty": ep_penalty,
                "episode_reactor_temp_penalty_events": ep_temp_penalty_events,
                "final_T_reactor_C": env.plant.state.T_reactor_C,
                "final_T_setpoint_C": env.plant.state.T_setpoint_C,
                "episode_time_s": time.time() - ep_t0,
                "elapsed_run_time_s": time.time() - run_t0,
                **metrics,
            }
        )

        save_csv(training_rows, output_dir / "training_steps.csv")
        save_csv(summary_rows, output_dir / "episode_summary.csv")
        torch.save(agent.network.state_dict(), output_dir / "checkpoint_latest.pt")
        if (ep + 1) % 10 == 0:
            torch.save(agent.network.state_dict(), output_dir / f"checkpoint_ep{ep + 1:03d}.pt")

        print(
            f"ppo episode {ep + 1:03d}/{episodes} "
            f"reward={ep_reward:.3f} produced={ep_produced:.3f} "
            f"consumed={ep_consumed:.3f} penalty={ep_penalty:.3f} "
            f"updated={metrics['ppo_updated']} entropy_coef={metrics['ppo_entropy_coef']:.5f}",
            flush=True,
        )

    _, deterministic_internal_df = deterministic_rollout_policy(
        config,
        output_dir,
        select_action=lambda obs: agent.act(obs, deterministic=True),
        episode_days=config.season_episode_days,
        prefix="deterministic_policy",
    )
    torch.manual_seed(config.random_seed + 20_000)
    _, stochastic_internal_df = deterministic_rollout_policy(
        config,
        output_dir,
        select_action=lambda obs: agent.act(obs, deterministic=False),
        episode_days=config.season_episode_days,
        prefix="stochastic_policy",
    )
    plot_training_reward(output_dir, "PPO training reward")
    plot_rollout_basic(
        output_dir,
        deterministic_internal_df,
        "PPO deterministic rollout",
        filename="deterministic_rollout.png",
    )
    plot_rollout_basic(
        output_dir,
        stochastic_internal_df,
        "PPO stochastic rollout",
        filename="stochastic_rollout.png",
    )
    torch.save(agent.network.state_dict(), output_dir / "ppo_actor_critic.pt")
    return output_dir


def nullable_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-days", type=float, default=1.0)
    parser.add_argument("--run-name", default="ppo_smoke")
    parser.add_argument("--production-weight", type=float, default=1.0)
    parser.add_argument("--consumption-weight", type=float, default=1.0)
    parser.add_argument("--temp-penalty-per-event", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--t-setpoint-min-C", type=nullable_float, default=25.0)
    parser.add_argument("--t-setpoint-max-C", type=nullable_float, default=65.0)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--entropy-coef-start", type=float, default=0.01)
    parser.add_argument("--entropy-coef-end", type=float, default=0.001)
    parser.add_argument("--entropy-decay-episodes", type=int, default=100)
    parser.add_argument("--rollout-episodes-per-update", type=int, default=10)
    parser.add_argument("--value-coef", type=float, default=0.5)
    args = parser.parse_args()

    output_dir = train(
        episodes=args.episodes,
        episode_days=args.episode_days,
        run_name=args.run_name,
        production_weight=args.production_weight,
        consumption_weight=args.consumption_weight,
        temp_penalty_per_event=args.temp_penalty_per_event,
        reward_scale=args.reward_scale,
        learning_rate=args.learning_rate,
        t_setpoint_min_C=args.t_setpoint_min_C,
        t_setpoint_max_C=args.t_setpoint_max_C,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef_start=args.entropy_coef_start,
        entropy_coef_end=args.entropy_coef_end,
        entropy_decay_episodes=args.entropy_decay_episodes,
        value_coef=args.value_coef,
        rollout_episodes_per_update=args.rollout_episodes_per_update,
    )
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
