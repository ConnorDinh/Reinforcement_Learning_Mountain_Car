
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIGURATIONS = [
    (10, 10),
    (20, 20),
    (40, 40),
    (20, 40),
    (40, 20),
]

SEEDS = [42, 123, 456]

MODELS_DIR = Path("results/models")
PLOTS_DIR = Path("results/plots")


def moving_average(values, window=100):
    """Compute a moving average without shortening the output."""
    values = np.asarray(values, dtype=float)

    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    counts = np.minimum(np.arange(1, len(values) + 1), window)

    sums = cumulative[1:] - cumulative[
        np.maximum(np.arange(1, len(values) + 1) - window, 0)
    ]

    return sums / counts


def plot_learning_curves():
    """Plot mean training rewards across seeds for each configuration."""
    plt.figure(figsize=(11, 6))

    for position_bins, velocity_bins in CONFIGURATIONS:
        histories = []

        for seed in SEEDS:
            path = MODELS_DIR / (
                f"reward_history_{position_bins}x{velocity_bins}_seed{seed}.npy"
            )

            rewards = np.load(path)
            histories.append(rewards)

        histories = np.array(histories)

        # Average rewards across seeds at each episode.
        mean_rewards = np.mean(histories, axis=0)

        # Smooth the mean rewards.
        smoothed_rewards = moving_average(mean_rewards, window=100)

        episodes = np.arange(1, len(smoothed_rewards) + 1)

        plt.plot(
            episodes,
            smoothed_rewards,
            label=f"{position_bins}x{velocity_bins}",
        )

    plt.title("MountainCar: Q-Learning Training Performance")
    plt.xlabel("Training Episode")
    plt.ylabel("Average Episode Reward (100-Episode Moving Average)")
    plt.legend(title="Discretization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "learning_curves.png"
    plt.savefig(output, dpi=300)
    plt.close()

    print(f"Saved: {output}")


def plot_evaluation_results():
    """Compare greedy evaluation performance across configurations."""

    # Results from your 100-episode evaluations.
    # Rows correspond to configurations; columns to seeds 42, 123, 456.
    success_rates = np.array([
        [1, 15, 100],
        [88, 100, 100],
        [74, 71, 83],
        [76, 96, 100],
        [100, 85, 85],
    ], dtype=float)

    average_rewards = np.array([
        [-199.83, -187.70, -154.83],
        [-171.13, -126.53, -134.79],
        [-162.93, -177.54, -150.04],
        [-161.79, -174.76, -161.23],
        [-148.98, -156.06, -157.69],
    ], dtype=float)

    labels = [f"{p}x{v}" for p, v in CONFIGURATIONS]
    x = np.arange(len(labels))

    # Success rate chart
    plt.figure(figsize=(9, 5))

    plt.bar(
        x,
        np.mean(success_rates, axis=1),
        yerr=np.std(success_rates, axis=1, ddof=1),
        capsize=5,
    )

    plt.xticks(x, labels)
    plt.ylim(0, 110)
    plt.xlabel("Discretization (Position x Velocity)")
    plt.ylabel("Success Rate (%)")
    plt.title("Greedy Evaluation Success Rate Across 3 Seeds")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "evaluation_success.png"
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")

    # Average reward chart
    plt.figure(figsize=(9, 5))

    plt.bar(
        x,
        np.mean(average_rewards, axis=1),
        yerr=np.std(average_rewards, axis=1, ddof=1),
        capsize=5,
    )

    plt.xticks(x, labels)
    plt.xlabel("Discretization (Position x Velocity)")
    plt.ylabel("Average Episode Reward")
    plt.title("Greedy Evaluation Reward Across 3 Seeds")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output = PLOTS_DIR / "evaluation_rewards.png"
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def plot_policy_heatmap(config=(20, 20), seed=123):
    """Visualize the greedy action at each discretized state."""

    position_bins, velocity_bins = config

    path = MODELS_DIR / (
        f"q_table_{position_bins}x{velocity_bins}_seed{seed}.npy"
    )

    q_table = np.load(path)

    # Actions: 0 = left, 1 = neutral, 2 = right
    policy = np.argmax(q_table, axis=2)

    plt.figure(figsize=(9, 6))

    plt.imshow(
        policy.T,
        origin="lower",
        aspect="auto",
        extent=[-1.2, 0.6, -0.07, 0.07],
        vmin=0,
        vmax=2,
        cmap="viridis",
    )

    colorbar = plt.colorbar(ticks=[0, 1, 2])
    colorbar.ax.set_yticklabels(["Left", "Neutral", "Right"])

    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(
        f"Learned Greedy Policy ({position_bins}x{velocity_bins}, Seed {seed})"
    )
    plt.tight_layout()

    output = PLOTS_DIR / f"policy_{position_bins}x{velocity_bins}_seed{seed}.png"
    plt.savefig(output, dpi=300)
    plt.close()
    print(f"Saved: {output}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_learning_curves()
    plot_evaluation_results()
    plot_policy_heatmap(config=(20, 20), seed=123)


if __name__ == "__main__":
    main()
