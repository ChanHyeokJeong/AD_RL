from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

from ad_rl.config import RUN_DIR
from ad_rl.full_pyadm1_env import FullPyADM1PISetpointEnv
from ad_rl.qmix_agent import QMixController, QMixReplayBuffer
from ad_rl.rl_utils import (
    deterministic_rollout_policy,
    json_ready_config,
    plot_rollout_basic,
    plot_training_reward,
    save_csv,
)
from train_full_pyadm1_temp_penalty_consumption import config_for_run


def epsilon_by_step(step: int, epsilon_start: float, epsilon_end: float, decay_steps: int) -> float:
    frac = min(1.0, step / max(1, decay_steps))
    return epsilon_start + frac * (epsilon_end - epsilon_start)


def train(
    episodes: int,
    episode_days: float,
    run_name: str,
    production_weight: float,
    consumption_weight: float,
    temp_penalty_per_event: float,
    reward_scale: float,
    learning_rate: float,
    target_update_steps: int,
    replay_capacity: int,
    t_setpoint_min_C: float | None,
    t_setpoint_max_C: float | None,
):
    config = config_for_run(
        episode_days=episode_days,
        production_weight=production_weight,
        consumption_weight=consumption_weight,
        temp_penalty_per_event=temp_penalty_per_event,
        reward_scale=reward_scale,
        learning_rate=learning_rate,
        target_update_steps=target_update_steps,
        replay_capacity=replay_capacity,
        t_setpoint_min_C=t_setpoint_min_C,
        t_setpoint_max_C=t_setpoint_max_C,
    )
    output_dir = RUN_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    env = FullPyADM1PISetpointEnv(config=config, episode_days=config.season_episode_days)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    controller = QMixController(obs_dim=obs_dim, action_dim=action_dim, n_agents=1)
    optimizer = torch.optim.Adam(controller.online_parameters(), lr=config.learning_rate)
    replay = QMixReplayBuffer(config.replay_capacity)

    training_rows: list[dict] = []
    summary_rows: list[dict] = []
    global_step = 0
    run_t0 = time.time()

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        meta = json_ready_config(config)
        meta["algorithm"] = "qmix_single_setpoint_agent"
        meta["n_agents"] = 1
        json.dump(meta, f, indent=2, ensure_ascii=False)

    for ep in range(int(episodes)):
        ep_t0 = time.time()
        obs = env.reset(seed=config.random_seed + ep)
        done = False
        ep_reward = 0.0
        ep_physical_reward = 0.0
        ep_reward_baseline = 0.0
        ep_produced = 0.0
        ep_consumed = 0.0
        ep_penalty = 0.0
        ep_temp_penalty_events = 0.0
        ep_steps = 0

        while not done:
            eps = epsilon_by_step(
                global_step,
                config.epsilon_start,
                config.epsilon_end,
                config.epsilon_decay_steps,
            )
            joint_actions = controller.choose_actions(obs, eps)
            action = int(joint_actions[0])
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            replay.push(obs, joint_actions, reward, next_obs, done)

            loss_value = np.nan
            if len(replay) >= max(config.batch_size, config.warmup_steps):
                batch = replay.sample(config.batch_size)
                loss = controller.loss(batch, config.gamma)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = float(loss.item())

            if global_step % config.target_update_steps == 0:
                controller.update_targets()

            row = {
                "episode": ep,
                "step": ep_steps,
                "global_step": global_step,
                "epsilon": eps,
                "action": action,
                "loss": loss_value,
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
            }
        )

        save_csv(training_rows, output_dir / "training_steps.csv")
        save_csv(summary_rows, output_dir / "episode_summary.csv")
        torch.save(controller.state_dict(), output_dir / "checkpoint_latest.pt")
        if (ep + 1) % 10 == 0:
            torch.save(controller.state_dict(), output_dir / f"checkpoint_ep{ep + 1:03d}.pt")

        print(
            f"qmix episode {ep + 1:03d}/{episodes} "
            f"reward={ep_reward:.3f} produced={ep_produced:.3f} "
            f"consumed={ep_consumed:.3f} penalty={ep_penalty:.3f}",
            flush=True,
        )

    _, internal_df = deterministic_rollout_policy(
        config,
        output_dir,
        select_action=controller.greedy_action,
        episode_days=config.season_episode_days,
    )
    plot_training_reward(output_dir, "QMIX training reward")
    plot_rollout_basic(output_dir, internal_df, "QMIX deterministic rollout")
    torch.save(controller.state_dict(), output_dir / "qmix_controller.pt")
    return output_dir


def nullable_float(value: str) -> float | None:
    if value.lower() in {"none", "null"}:
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--episode-days", type=float, default=1.0)
    parser.add_argument("--run-name", default="qmix_smoke")
    parser.add_argument("--production-weight", type=float, default=1.0)
    parser.add_argument("--consumption-weight", type=float, default=1.0)
    parser.add_argument("--temp-penalty-per-event", type=float, default=0.0)
    parser.add_argument("--reward-scale", type=float, default=100.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--target-update-steps", type=int, default=500)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--t-setpoint-min-C", type=nullable_float, default=25.0)
    parser.add_argument("--t-setpoint-max-C", type=nullable_float, default=65.0)
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
        target_update_steps=args.target_update_steps,
        replay_capacity=args.replay_capacity,
        t_setpoint_min_C=args.t_setpoint_min_C,
        t_setpoint_max_C=args.t_setpoint_max_C,
    )
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
