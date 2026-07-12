"""
Testing/evaluation script for the trained DRL agent.

Usage:
    python test.py --model results/best_feasible_model.pth
    python test.py --model results/final_model.pth --num-episodes 10
"""
import argparse
import os
import numpy as np
import config as cfg
from uav_env import UAVEnvironment
from ddpg_agent import DDPGAgent
from plot_results import plot_trajectory_3d, plot_comparison


def evaluate(args):
    print("=" * 60)
    print("DRL-based UAV Trajectory Optimization - Evaluation")
    print("=" * 60)

    env = UAVEnvironment()
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return

    agent.load(args.model)
    print(f"Loaded model: {args.model}")

    all_throughputs = []
    all_feasible = []
    best_results = None
    best_throughput = -np.inf

    for ep in range(args.num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0

        for step in range(cfg.N_SLOTS):
            # No noise during evaluation
            action = agent.predict(state, add_noise=False)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state
            if terminated or truncated:
                break

        results = env.get_results()
        total_throughput = results["total_throughput"]
        is_feasible = (
            results["energy_feasible"]
            and results["relay_feasible"]
            and results["demand_met"]
        )

        all_throughputs.append(total_throughput)
        all_feasible.append(is_feasible)

        if total_throughput > best_throughput:
            best_throughput = total_throughput
            best_results = results

        print(f"\nEpisode {ep + 1}/{args.num_episodes}:")
        print(f"  Total Throughput: {total_throughput:.4f} Mbits")
        print(f"  Per-UAV Throughput: {results['throughput_per_uav']}")
        print(f"  Cum R_u: {results['cum_R_u']}")
        print(f"  Cum R_d: {results['cum_R_d']}")
        print(f"  Energy consumed: {[f'{e:.2f}' for e in results['energy_consumed']]}")
        print(f"  Energy harvested: {[f'{e:.2f}' for e in results['energy_harvested']]}")
        print(f"  Energy feasible: {results['energy_feasible']}")
        print(f"  Relay feasible: {results['relay_feasible']}")
        print(f"  Demand met: {results['demand_met']}")
        print(f"  Overall feasible: {is_feasible}")
        print(f"  Final positions: {results['final_positions']}")
        print(f"  Reward: {ep_reward:.4f}")

    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Average throughput: {np.mean(all_throughputs):.4f} Mbits")
    print(f"  Best throughput: {best_throughput:.4f} Mbits")
    print(f"  Feasibility rate: {np.mean(all_feasible) * 100:.1f}%")
    print(f"  S_DEMAND: {cfg.S_DEMAND} Mbits")
    print("=" * 60)

    # Plot best trajectory
    if best_results is not None:
        save_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(save_dir, exist_ok=True)

        import time
        timestamp = int(time.time())
        save_file = f"best_trajectory_{timestamp}.png"
        plot_trajectory_3d(
            best_results["trajectories"],
            save_path=os.path.join(save_dir, save_file),
        )
        print(f"\nTrajectory plot saved to results/{save_file}")
    else:
        print("\nNo trajectory found to plot.")

    return best_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained DRL agent")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results", "best_feasible_model.pth"),
        help="Path to trained model",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    args = parser.parse_args()
    evaluate(args)
