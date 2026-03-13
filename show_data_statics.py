from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SIZE = 7000


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_path = project_root / "data" / "raw" / f"grid({SIZE}).npy"
    output_path = project_root / "data" / "statics" / f"data_mean_distribution({SIZE}).png"

    grid = np.load(data_path)  # (T, H, W)
    mean_grid = grid.mean(axis=0, keepdims=True)  # (1, H, W)
    flattened_mean = mean_grid.reshape(-1)  # (N,)

    plt.figure(figsize=(10, 6))
    plt.hist(flattened_mean, bins=50, edgecolor="black", alpha=0.8)
    plt.title("Distribution of Mean Demand Across Grid Cells")
    plt.xlabel("Mean Value")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"loaded shape: {grid.shape}")
    print(f"mean shape: {mean_grid.shape}")
    print(f"flattened shape: {flattened_mean.shape}")
    print(f"saved figure to: {output_path}")
    
    print(f'num over 1: {np.sum(flattened_mean > 1)}')
if __name__ == "__main__":
    main()
