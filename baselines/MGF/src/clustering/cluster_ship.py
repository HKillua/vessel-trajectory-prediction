"""KMeans clustering on ship future trajectories for GMM base distribution.

Usage:
    python -m src.clustering.cluster_ship --data_root <path> --n_clusters 8 --pred_len 30
"""

import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.ship_loader import extract_futures_for_clustering


def rotate_trajectories(futures, base_dirs):
    """Rotate trajectories to align COG direction to x-axis.

    Args:
        futures:   (N, pred_len, 2) position trajectories
        base_dirs: (N, 2) (cos_cog, sin_cog)

    Returns:
        rotated: (N, pred_len, 2) aligned trajectories
    """
    cos_a = base_dirs[:, 0]   # cos_cog
    sin_a = base_dirs[:, 1]   # sin_cog
    # Inverse rotation: rotate by -angle to align to x-axis
    # R(-a) = [[cos, sin], [-sin, cos]]
    x = futures[:, :, 0]
    y = futures[:, :, 1]
    rot_x = cos_a[:, None] * x + sin_a[:, None] * y
    rot_y = -sin_a[:, None] * x + cos_a[:, None] * y
    return np.stack([rot_x, rot_y], axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to pred{N}/ directory with train/val/test")
    parser.add_argument("--n_clusters", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for pkl (default: src/clustering/models/)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dataset_name", type=str, default="ship")
    args = parser.parse_args()

    print(f"Extracting futures from {args.data_root} ...")
    futures, base_dirs, _ = extract_futures_for_clustering(
        args.data_root, max_samples=args.max_samples
    )
    print(f"  Got {futures.shape[0]} trajectories, pred_len={futures.shape[1]}")

    print("Rotating trajectories by COG direction ...")
    rotated = rotate_trajectories(futures, base_dirs)

    # Flatten to (N, pred_len*2) for KMeans
    flat = rotated.reshape(rotated.shape[0], -1)
    print(f"  Feature dim for KMeans: {flat.shape[1]}")

    print(f"Running KMeans with K={args.n_clusters} ...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(flat)
    print(f"  Inertia: {kmeans.inertia_:.4f}")

    # Print cluster sizes
    _, counts = np.unique(kmeans.labels_, return_counts=True)
    for i, c in enumerate(counts):
        print(f"  Cluster {i}: {c} samples ({c/len(flat)*100:.1f}%)")

    # Save
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(args.output_dir, exist_ok=True)

    pred_len = futures.shape[1]
    out_path = os.path.join(
        args.output_dir,
        f"{args.dataset_name}_train_kmeans_{args.n_clusters}_pred{pred_len}.pkl"
    )
    with open(out_path, "wb") as f:
        pickle.dump(kmeans, f)
    print(f"Saved cluster model to {out_path}")


if __name__ == "__main__":
    main()
