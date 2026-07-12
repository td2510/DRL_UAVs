"""
Training script for the DRL-based UAV trajectory optimization.

Usage:
    python train.py                          # Default training (5000 episodes)
    python train.py --episodes 100           # Quick smoke test
    python train.py --episodes 10000 --save  # Full training with model saving
"""
import argparse
import os
import time
import numpy as np
import config as cfg
from uav_env import UAVEnvironment
from ddpg_agent import DDPGAgent


def train(args):
    print("=" * 60)
    print("DRL-based UAV Trajectory Optimization - Training")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {cfg.N_SLOTS}")
    print(f"Device: cuda" if __import__('torch').cuda.is_available() else "Device: cpu")
    print(f"S_DEMAND: {cfg.S_DEMAND} Mbits, T: {cfg.T}s, N: {cfg.N_SLOTS} slots")
    print("=" * 60)

    env = UAVEnvironment()
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        print(f"Resumed from checkpoint: {args.resume}")

    # Logging arrays
    episode_rewards = []
    episode_throughputs = []
    episode_feasible = []
    best_throughput = -np.inf
    best_feasible_throughput = -np.inf

    # Create output directory
    save_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()

    for ep in range(1, args.episodes + 1):
        state, _ = env.reset()
        agent.reset_noise()
        ep_reward = 0.0

        for step in range(cfg.N_SLOTS):
            # Select action
            action = agent.predict(state, add_noise=True)

            # Execute action
            next_state, reward, terminated, truncated, info = env.step(action)

            # Store transition
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)

            # Update networks
            agent.update()

            ep_reward += reward
            state = next_state

            if done:
                break

        # Decay noise
        agent.noise.sigma = max(0.01, agent.noise.sigma * getattr(cfg, 'NOISE_DECAY', 0.995))

        # Episode summary
        results = env.get_results()
        total_throughput = results["total_throughput"]
        is_feasible = (
            results["energy_feasible"]
            and results["relay_feasible"]
            and results["demand_met"]
        )

        episode_rewards.append(ep_reward)
        episode_throughputs.append(total_throughput)
        episode_feasible.append(is_feasible)

        # Track best
        if total_throughput > best_throughput:
            best_throughput = total_throughput
        if is_feasible and total_throughput > best_feasible_throughput:
            best_feasible_throughput = total_throughput
            if args.save:
                agent.save(os.path.join(save_dir, "best_feasible_model.pth"))

        # Logging
        if ep % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:])
            avg_throughput = np.mean(episode_throughputs[-args.log_interval:])
            feasible_rate = np.mean(episode_feasible[-args.log_interval:]) * 100
            elapsed = time.time() - start_time

            print(
                f"Ep {ep:5d}/{args.episodes} | "
                f"Avg Reward: {avg_reward:8.2f} | "
                f"Avg Throughput: {avg_throughput:8.3f} Mbits | "
                f"Best: {best_throughput:8.3f} | "
                f"Best Feasible: {best_feasible_throughput:8.3f} | "
                f"Feasible: {feasible_rate:5.1f}% | "
                f"Time: {elapsed:6.1f}s"
            )

        # Periodic save
        if args.save and ep % cfg.SAVE_INTERVAL == 0:
            agent.save(os.path.join(save_dir, f"model_ep{ep}.pth"))

    # Final save
    if args.save:
        agent.save(os.path.join(save_dir, "final_model.pth"))

    # Save training history
    np.savez(
        os.path.join(save_dir, "training_history.npz"),
        rewards=np.array(episode_rewards),
        throughputs=np.array(episode_throughputs),
        feasible=np.array(episode_feasible),
    )

    total_time = time.time() - start_time
    print("=" * 60)
    print("Training Complete!")
    print(f"Total time: {total_time:.1f}s")
    print(f"Best throughput: {best_throughput:.3f} Mbits")
    print(f"Best feasible throughput: {best_feasible_throughput:.3f} Mbits")
    print(f"Final feasible rate: {np.mean(episode_feasible[-100:]) * 100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DRL agent for UAV optimization")
    parser.add_argument("--episodes", type=int, default=cfg.MAX_EPISODES, help="Number of training episodes")
    parser.add_argument("--log-interval", type=int, default=cfg.LOG_INTERVAL, help="Logging interval")
    parser.add_argument("--save", action="store_true", default=True, help="Save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)
