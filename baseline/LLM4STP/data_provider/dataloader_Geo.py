"""
TrajectoryDataset adapted for our ship trajectory npz data.
Loads multi-ship npz samples and extracts target ship's (lon, lat) trajectory.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Loads npz trajectory files and returns target ship trajectories.

    Expected npz keys:
        obs: (n_ships, T_obs, 5)  - [lat, lon, speed, head_cos, head_sin]
        pred: (n_ships, T_pred, 5)
        target_ship_idx: scalar
        n_ships: scalar
    """

    def __init__(self, data_dir, cache_dir=None, load_dataset=True,
                 obs_len=30, pred_len=30, skip=1, min_traj_len=10):
        super().__init__()
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len

        self.data_dir = data_dir
        self.samples = []

        # Collect all npz files
        npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        print(f"Loading dataset from {data_dir}: {len(npz_files)} files")

        # Compute global normalization stats from a subset
        all_lats = []
        all_lons = []
        sample_files = npz_files[:min(500, len(npz_files))]
        for f in sample_files:
            d = np.load(f)
            target_idx = int(d['target_ship_idx'])
            all_lats.append(d['obs'][target_idx, :, 0])
            all_lats.append(d['pred'][target_idx, :, 0])
            all_lons.append(d['obs'][target_idx, :, 1])
            all_lons.append(d['pred'][target_idx, :, 1])

        all_lats = np.concatenate(all_lats)
        all_lons = np.concatenate(all_lons)
        self.lat_min, self.lat_max = all_lats.min(), all_lats.max()
        self.lon_min, self.lon_max = all_lons.min(), all_lons.max()
        # Add small margin
        lat_range = self.lat_max - self.lat_min + 1e-6
        lon_range = self.lon_max - self.lon_max + 1e-6
        self.lat_min -= lat_range * 0.01
        self.lat_max += lat_range * 0.01
        self.lon_min -= lon_range * 0.01
        self.lon_max += lon_range * 0.01

        # Store file paths (skip by step)
        self.npz_files = npz_files[::skip]
        print(f"  Using {len(self.npz_files)} samples (skip={skip})")
        print(f"  Coord range: lat=[{self.lat_min:.4f}, {self.lat_max:.4f}], "
              f"lon=[{self.lon_min:.4f}, {self.lon_max:.4f}]")

    def _normalize(self, lat, lon):
        """Normalize to [0, 1] range for geohash encoding."""
        lat_norm = (lat - self.lat_min) / (self.lat_max - self.lat_min + 1e-8)
        lon_norm = (lon - self.lon_min) / (self.lon_max - self.lon_min + 1e-8)
        return lon_norm, lat_norm  # return (x, y) = (lon, lat)

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        try:
            filepath = self.npz_files[idx]
            with np.load(filepath) as d:
                target_idx_arr = d['target_ship_idx']
                target_idx = int(target_idx_arr.item() if target_idx_arr.ndim == 0 else target_idx_arr[0])
                obs = d['obs'][target_idx].copy()    # [T_obs, 5]
                pred = d['pred'][target_idx].copy()  # [T_pred, 5]
        except Exception as e:
            # Fallback: try another random sample
            import random
            alt_idx = random.randint(0, len(self) - 1)
            filepath = self.npz_files[alt_idx]
            with np.load(filepath) as d:
                target_idx_arr = d['target_ship_idx']
                target_idx = int(target_idx_arr.item() if target_idx_arr.ndim == 0 else target_idx_arr[0])
                obs = d['obs'][target_idx].copy()
                pred = d['pred'][target_idx].copy()

        # Extract (lon, lat) and normalize
        obs_lon, obs_lat = self._normalize(obs[:, 0], obs[:, 1])
        pred_lon, pred_lat = self._normalize(pred[:, 0], pred[:, 1])

        # Stack as [T, 2] with (x=lon, y=lat)
        past_traj = np.stack([obs_lon, obs_lat], axis=-1).astype(np.float32)
        future_traj = np.stack([pred_lon, pred_lat], axis=-1).astype(np.float32)

        # Pad/truncate to match expected lengths
        if past_traj.shape[0] < self.obs_len:
            pad = np.tile(past_traj[-1:], (self.obs_len - past_traj.shape[0], 1))
            past_traj = np.concatenate([past_traj, pad], axis=0)
        elif past_traj.shape[0] > self.obs_len:
            past_traj = past_traj[:self.obs_len]

        if future_traj.shape[0] < self.pred_len:
            pad = np.tile(future_traj[-1:], (self.pred_len - future_traj.shape[0], 1))
            future_traj = np.concatenate([future_traj, pad], axis=0)
        elif future_traj.shape[0] > self.pred_len:
            future_traj = future_traj[:self.pred_len]

        # Return in format expected by mainSTP_Geo.py:
        # past_traj: [2, T_obs] (will be permuted to [T_obs, 2])
        # future_traj: [2, T_pred]
        return {
            'past_traj': torch.from_numpy(past_traj.T),     # [2, T_obs]
            'future_traj': torch.from_numpy(future_traj.T),  # [2, T_pred]
        }

    def generate_gaussian_map(self, nodes_current, grid_size=64, sigma_x=5, sigma_y=5):
        """
        Generate Gaussian density maps from trajectory positions.

        Args:
            nodes_current: [B, T, 2] tensor of (x, y) positions
            grid_size: size of the output grid
            sigma_x, sigma_y: Gaussian spread parameters

        Returns:
            gaussian_maps: [B, grid_size, grid_size] tensor
        """
        if isinstance(nodes_current, torch.Tensor):
            device = nodes_current.device
            # Use last position for the gaussian map center
            x = nodes_current[:, -1, 0].cpu().numpy()  # [B]
            y = nodes_current[:, -1, 1].cpu().numpy()  # [B]
        else:
            x = nodes_current[:, -1, 0]
            y = nodes_current[:, -1, 1]

        B = len(x)
        # Create grid
        gx = np.linspace(0, 1, grid_size)
        gy = np.linspace(0, 1, grid_size)
        gx_grid, gy_grid = np.meshgrid(gx, gy)  # [G, G]

        # Convert sigma from grid units to normalized units
        sx = sigma_x / grid_size
        sy = sigma_y / grid_size

        # Compute Gaussian for each sample
        gaussian_maps = np.zeros((B, grid_size, grid_size), dtype=np.float32)
        for i in range(B):
            gaussian_maps[i] = np.exp(
                -((gx_grid - x[i]) ** 2 / (2 * sx ** 2 + 1e-8) +
                  (gy_grid - y[i]) ** 2 / (2 * sy ** 2 + 1e-8))
            )

        return torch.from_numpy(gaussian_maps).to(device) if isinstance(nodes_current, torch.Tensor) else gaussian_maps
