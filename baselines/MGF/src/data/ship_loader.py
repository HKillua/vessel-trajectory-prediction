"""Ship trajectory data adapter for MGF.

Wraps MultiVesselDataset to extract target-ship-only data
in the format expected by ShipMGF model.
"""

import os
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from data_provider.dataloader_multivessel import (
    MultiVesselDataset,
    multivessel_collate_fn,
)


def extract_target_ship(batch):
    """Extract target ship data from multi-vessel collated batch.

    Returns dict compatible with ShipMGF.update() / ShipMGF.predict():
        obs_st:   (B, obs_len, 7)  normalized observation
        gt_st:    (B, pred_len, 2) normalized future position (lat_nm, lon_nm)
        base_pos: (B, 2)          (cos_cog, sin_cog) from last obs for GMM rotation
    """
    B = batch["obs"].shape[0]
    idx = batch["target_ship_idx"]

    obs = batch["obs"][torch.arange(B), idx]   # (B, obs_len, 7)
    pred = batch["pred"][torch.arange(B), idx]  # (B, pred_len, 7)

    # base_pos: direction vector from last obs COG
    # features: [lat, lon, sog, sin_cog, cos_cog, sin_hdg, cos_hdg]
    cos_cog = obs[:, -1, 4]
    sin_cog = obs[:, -1, 3]
    base_pos = torch.stack([cos_cog, sin_cog], dim=-1)  # (B, 2)

    return {
        "obs_st": obs,
        "gt_st": pred[:, :, :2],
        "base_pos": base_pos,
    }


def create_ship_dataloaders(data_root, batch_size=64, num_workers=4):
    """Create train/val/test DataLoaders for ship trajectory prediction.

    Args:
        data_root: path to pred{10,20,30}/ directory containing train/val/test
        batch_size: batch size
        num_workers: DataLoader workers

    Returns:
        loaders: dict of DataLoaders {"train": ..., "val": ..., "test": ...}
        norm_params: dict with 'mean' (3,) and 'std' (3,) arrays
    """
    loaders = {}
    train_norm_params = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            print(f"  Warning: {split_dir} does not exist, skipping")
            continue

        dataset = MultiVesselDataset(
            split_dir,
            normalize=True,
            random_target=(split == "train"),
            norm_params=train_norm_params if split != "train" else None,
            split=split,
        )

        if split == "train":
            train_norm_params = dataset.norm_params

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=multivessel_collate_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        print(f"  {split}: {len(dataset)} samples, {len(loaders[split])} batches")

    return loaders, train_norm_params


def extract_futures_for_clustering(data_root, max_samples=None):
    """Extract target ship future trajectories for KMeans clustering.

    Returns:
        futures_nm: (N, pred_len, 2) normalized position trajectories
        base_dirs:  (N, 2) (cos_cog, sin_cog) direction vectors
    """
    train_dir = os.path.join(data_root, "train")
    dataset = MultiVesselDataset(train_dir, normalize=True, random_target=False)
    norm_params = dataset.norm_params

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        collate_fn=multivessel_collate_fn,
        pin_memory=False,
    )

    all_futures = []
    all_dirs = []
    count = 0

    for batch in loader:
        data = extract_target_ship(batch)
        all_futures.append(data["gt_st"])
        all_dirs.append(data["base_pos"])
        count += data["gt_st"].shape[0]
        if max_samples and count >= max_samples:
            break

    futures = torch.cat(all_futures, dim=0)
    dirs = torch.cat(all_dirs, dim=0)

    if max_samples and futures.shape[0] > max_samples:
        futures = futures[:max_samples]
        dirs = dirs[:max_samples]

    return futures.numpy(), dirs.numpy(), norm_params

