"""
Visualization utilities for UAV trajectory optimization results.

Provides:
- 3D trajectory plotting
- Training curves
- Comparison plots: DRL vs GA vs BCD
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import config as cfg
import os


def plot_trajectory_3d(trajectories, save_path=None, title="UAV Trajectories (DRL)"):
    """
    Plot 3D trajectories of multiple UAVs.

    Args:
        trajectories: List of np.array, each shape (N+1, 3)
        save_path: Optional path to save the figure
        title: Plot title
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    labels = [f"UAV_{i+1}" for i in range(len(trajectories))]

    for i, traj in enumerate(trajectories):
        ax.plot(
            traj[:, 0], traj[:, 1], traj[:, 2],
            marker="o", linewidth=2, markersize=4,
            color=colors[i % len(colors)],
            label=labels[i],
        )
        # Mark start and end
        ax.scatter(*traj[0], color="green", s=80, zorder=5)
        ax.scatter(*traj[-1], color="red", s=80, zorder=5)

    # Plot source and destination
    ax.scatter(*cfg.W_S, color="red", s=100, marker="^", zorder=5)
    ax.text(cfg.W_S[0], cfg.W_S[1], cfg.W_S[2] + 0.5, "Source", color="red", fontsize=10)

    ax.scatter(*cfg.W_D, color="blue", s=100, marker="s", zorder=5)
    ax.text(cfg.W_D[0], cfg.W_D[1], cfg.W_D[2] + 0.5, "Destination", color="blue", fontsize=10)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Trajectory plot saved: {save_path}")
    plt.close()


def plot_training_curves(history_path, save_dir=None):
    """
    Plot training curves from saved history.

    Args:
        history_path: Path to training_history.npz
        save_dir: Directory to save plots
    """
    data = np.load(history_path)
    rewards = data["rewards"]
    throughputs = data["throughputs"]
    feasible = data["feasible"]

    episodes = np.arange(1, len(rewards) + 1)
    window = min(50, len(rewards) // 5)

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Reward curve
    axes[0].plot(episodes, rewards, alpha=0.3, color="tab:blue")
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(
            episodes[window - 1:], smoothed,
            color="tab:blue", linewidth=2, label=f"Moving avg ({window})"
        )
    axes[0].set_ylabel("Episode Reward")
    axes[0].set_title("Training Curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Throughput curve
    axes[1].plot(episodes, throughputs, alpha=0.3, color="tab:orange")
    if len(throughputs) >= window:
        smoothed = np.convolve(throughputs, np.ones(window) / window, mode="valid")
        axes[1].plot(
            episodes[window - 1:], smoothed,
            color="tab:orange", linewidth=2, label=f"Moving avg ({window})"
        )
    axes[1].axhline(y=cfg.S_DEMAND, color="red", linestyle="--", label=f"S_DEMAND={cfg.S_DEMAND}")
    axes[1].set_ylabel("Total Throughput (Mbits)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Feasibility rate
    if len(feasible) >= window:
        feas_rate = np.convolve(feasible.astype(float), np.ones(window) / window, mode="valid") * 100
        axes[2].plot(episodes[window - 1:], feas_rate, color="tab:green", linewidth=2)
    axes[2].set_ylabel("Feasibility Rate (%)")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_dir:
        path = os.path.join(save_dir, "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved: {path}")
    plt.close()


def plot_comparison(drl_throughput, ga_throughput=None, bcd_throughput=None,
                    x_values=None, x_label="T (seconds)", save_path=None):
    """
    Plot throughput comparison between DRL, GA, and BCD methods.

    Args:
        drl_throughput: Array of DRL throughput values
        ga_throughput: Optional array of GA throughput values
        bcd_throughput: Optional array of BCD throughput values
        x_values: X-axis values (e.g., different T values)
        x_label: Label for X-axis
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    if x_values is None:
        x_values = np.arange(len(drl_throughput))

    ax.plot(x_values, drl_throughput, "o-", linewidth=2, markersize=8, label="DRL (DDPG)")

    if ga_throughput is not None:
        ax.plot(x_values, ga_throughput, "s--", linewidth=2, markersize=8, label="GA")

    if bcd_throughput is not None:
        ax.plot(x_values, bcd_throughput, "^-.", linewidth=2, markersize=8, label="BCD")

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Total Throughput (Mbits)", fontsize=12)
    ax.set_title("Throughput Comparison: DRL vs GA vs BCD", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison plot saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Quick test: plot training curves if history exists
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    history_path = os.path.join(results_dir, "training_history.npz")

    if os.path.exists(history_path):
        plot_training_curves(history_path, save_dir=results_dir)
    else:
        print("No training history found. Run train.py first.")
