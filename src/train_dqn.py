import os
import time
import csv
import math
import numpy as np

from src.env.racetrack_env import RaceTrackEnv
from src.rl.replay_buffer import ReplayBuffer
from src.rl.dqn_agent import DQNAgent

def linear_epsilon(step, eps_start, eps_end, decay_steps):
    if step >= decay_steps:
        return eps_end
    t = step / decay_steps
    return eps_start + t * (eps_end - eps_start)

def main():
    # --- Hyperparams (good starting point) ---
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    replay_size = 100_000
    warmup_steps = 2_000
    target_update = 1_000
    total_steps = 50_000

    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 50_000

    log_every = 1_000
    save_every = 5_000

    # --- Run folder ---
    run_id = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)

    # --- Env (NO pygame init) ---
    env = RaceTrackEnv()
    obs = env.reset()

    obs_dim = int(obs.shape[0])
    num_actions = 9

    buffer = ReplayBuffer(replay_size, obs_dim)
    agent = DQNAgent(obs_dim, num_actions, lr=lr, gamma=gamma)

    # --- Logging ---
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["global_step", "episode", "ep_reward", "ep_len", "crashed", "lap_complete", "epsilon", "loss", "q_mean"])

    episode = 0
    ep_reward = 0.0
    ep_len = 0
    crash_count = 0
    lap_count = 0

    last_log = time.time()

    # Training loop
    try:
        # Training loop
        for step in range(1, total_steps + 1):
            epsilon = linear_epsilon(step, eps_start, eps_end, eps_decay_steps)
            action = agent.act(obs, epsilon)

            next_obs, reward, done, info = env.step(action)

            buffer.push(obs, action, reward, next_obs, done)

            obs = next_obs
            ep_reward += reward
            ep_len += 1

            loss_val = math.nan
            q_mean = math.nan

            if len(buffer) >= warmup_steps:
                batch = buffer.sample(batch_size)
                loss_val, q_mean = agent.update(batch)

                if step % target_update == 0:
                    agent.sync_target()

            if done:
                episode += 1
                crashed = bool(info.get("crashed", False))
                lap_complete = bool(info.get("lap_complete", False))
                if crashed:
                    crash_count += 1
                if lap_complete:
                    lap_count += 1

                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([step, episode, ep_reward, ep_len, int(crashed), int(lap_complete), epsilon, loss_val, q_mean])

                obs = env.reset()
                ep_reward = 0.0
                ep_len = 0

            if step % log_every == 0:
                now = time.time()
                dt = now - last_log
                last_log = now
                print(
                    f"step={step}/{total_steps} eps={epsilon:.3f} "
                    f"buffer={len(buffer)} device={agent.device} "
                    f"laps={lap_count} crashes={crash_count} "
                    f"({log_every/dt:.0f} steps/s)"
                )

            if step % save_every == 0:
                ckpt_path = os.path.join(run_dir, "checkpoints", f"dqn_step_{step}.pt")
                agent.save(ckpt_path)
                print(f"saved: {ckpt_path}")

    finally:
        final_path = os.path.join(run_dir, "dqn_final.pt")
        agent.save(final_path)
        print(f"saved final (even on stop): {final_path}")
        print(f"metrics: {csv_path}")

    

if __name__ == "__main__":
    main()