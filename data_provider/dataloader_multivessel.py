"""多船交互轨迹数据集

读取 08_generate_splits.py 输出的 NPZ 样本文件，
提供 obs/pred/adj/DCPA/TCPA/CRI/encounter_type 等字段。
支持不同场景的不同船数（通过 collate_fn padding 到 batch 内最大值）。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MultiVesselDataset(Dataset):
    """多船交互轨迹数据集

    每个样本 NPZ 包含:
    - obs: [N_ships, obs_steps, 5]  (lat, lon, sog, cog, heading)
    - pred: [N_ships, pred_steps, 5]
    - adj_matrix: [N_ships, N_ships]
    - dcpa_matrix: [N_ships, N_ships]
    - tcpa_matrix: [N_ships, N_ships]
    - cri_matrix: [N_ships, N_ships]
    - encounter_type: [N_ships, N_ships]
    - target_ship_idx: int
    - n_ships: int

    归一化策略:
    - lat/lon/sog: 线性归一化 (x - mean) / std
    - cog/heading: 转为 (sin, cos) 对，保持环形语义
    - 输出特征维度: 7 (lat, lon, sog, sin_cog, cos_cog, sin_hdg, cos_hdg)
    """

    def __init__(self, data_dir, normalize=True, random_target=True, norm_params=None, split=None):
        self.data_dir = data_dir
        self.normalize = normalize
        self.random_target = random_target

        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.startswith("sample_") and f.endswith(".npz")
        ])

        if not self.files:
            raise FileNotFoundError(f"No sample NPZ files in {data_dir}")

        sample0 = np.load(self.files[0])
        self.obs_steps = sample0["obs"].shape[1]
        self.pred_steps = sample0["pred"].shape[1]
        self.n_feat = sample0["obs"].shape[2]
        self.has_encounter = "dcpa_matrix" in sample0.files

        self.norm_params = None
        if normalize:
            if norm_params is not None:
                self.norm_params = norm_params
            elif split is not None and split != "train":
                raise ValueError(
                    f"norm_params must be provided for '{split}' split to avoid data leakage. "
                    f"Compute norm_params from the train split first."
                )
            else:
                self._compute_norm_params()

    def _compute_norm_params(self):
        """计算归一化参数

        局部坐标模式下，lat/lon 已经是相对位移，std 量级远小于绝对位置。
        只对 lat(0), lon(1), sog(2) 做线性归一化，sin/cos 天然在 [-1,1]。
        """
        n_sample = min(5000, len(self.files))
        indices = np.linspace(0, len(self.files) - 1, n_sample, dtype=int)

        all_vals = []
        rng = np.random.RandomState(42)
        for idx in indices:
            data = np.load(self.files[idx])
            obs = data["obs"]
            pred = data["pred"]
            n_ships = int(data["n_ships"])
            target_idx = rng.randint(0, n_ships)
            origin_lat = obs[target_idx, -1, 0]
            origin_lon = obs[target_idx, -1, 1]

            combined = np.concatenate([obs[:n_ships], pred[:n_ships]], axis=1)
            local = combined.copy()
            cos_lat = np.cos(np.radians(origin_lat))
            local[:, :, 0] = (local[:, :, 0] - origin_lat) * 60.0
            local[:, :, 1] = (local[:, :, 1] - origin_lon) * 60.0 * cos_lat
            all_vals.append(local.reshape(-1, combined.shape[-1]))

        combined = np.concatenate(all_vals, axis=0)
        self.norm_params = {
            "mean": np.array([0.0, 0.0, combined[:, 2].mean()], dtype=np.float32),
            "std": np.array([
                combined[:, 0].std() + 1e-8,
                combined[:, 1].std() + 1e-8,
                combined[:, 2].std() + 1e-8,
            ], dtype=np.float32),
        }

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _encode_angles(arr):
        """将 [N, T, 5] 中的 cog(3)/heading(4) 转为 sin/cos 对 → [N, T, 7]

        输入: [lat, lon, sog, cog, heading]
        输出: [lat, lon, sog, sin_cog, cos_cog, sin_hdg, cos_hdg]
        """
        lat_lon_sog = arr[..., :3]  # [N, T, 3]
        cog_rad = np.radians(arr[..., 3:4])   # [N, T, 1]
        hdg_rad = np.radians(arr[..., 4:5])   # [N, T, 1]
        sin_cog = np.sin(cog_rad)
        cos_cog = np.cos(cog_rad)
        sin_hdg = np.sin(hdg_rad)
        cos_hdg = np.cos(hdg_rad)
        return np.concatenate([lat_lon_sog, sin_cog, cos_cog, sin_hdg, cos_hdg], axis=-1)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])

        obs = data["obs"].astype(np.float32)
        pred = data["pred"].astype(np.float32)
        adj = data["adj_matrix"].astype(np.float32)
        target_idx = int(data["target_ship_idx"])
        n_ships = int(data["n_ships"])

        # 训练时随机选择 target ship，val/test 时确定性轮转覆盖不同视角
        if self.random_target and n_ships > 1:
            target_idx = np.random.randint(0, n_ships)
        elif not self.random_target and n_ships > 1:
            target_idx = idx % n_ships

        # 局部坐标变换：以 target 船最后一个 obs 点为原点，转换为海里
        origin_lat = obs[target_idx, -1, 0]
        origin_lon = obs[target_idx, -1, 1]
        cos_lat = np.cos(np.radians(origin_lat))
        obs[:, :, 0] = (obs[:, :, 0] - origin_lat) * 60.0
        obs[:, :, 1] = (obs[:, :, 1] - origin_lon) * 60.0 * cos_lat
        pred[:, :, 0] = (pred[:, :, 0] - origin_lat) * 60.0
        pred[:, :, 1] = (pred[:, :, 1] - origin_lon) * 60.0 * cos_lat

        # COG/Heading → sin/cos 编码 (5 feat → 7 feat)
        obs = self._encode_angles(obs)
        pred = self._encode_angles(pred)

        if self.normalize and self.norm_params is not None:
            mean = self.norm_params["mean"]  # [3] for lat/lon/sog
            std = self.norm_params["std"]    # [3]
            obs[..., :3] = (obs[..., :3] - mean) / std
            pred[..., :3] = (pred[..., :3] - mean) / std

        result = {
            "obs": torch.from_numpy(obs),
            "pred": torch.from_numpy(pred),
            "adj_matrix": torch.from_numpy(adj),
            "target_ship_idx": target_idx,
            "n_ships": n_ships,
            "origin_latlon": torch.tensor([origin_lat, origin_lon], dtype=torch.float32),
            "cos_lat": torch.tensor(cos_lat, dtype=torch.float32),
        }

        if self.has_encounter:
            result["dcpa_matrix"] = torch.from_numpy(data["dcpa_matrix"].astype(np.float32))
            result["tcpa_matrix"] = torch.from_numpy(data["tcpa_matrix"].astype(np.float32))
            result["cri_matrix"] = torch.from_numpy(data["cri_matrix"].astype(np.float32))
            result["encounter_type"] = torch.from_numpy(data["encounter_type"].astype(np.int64))

        return result


def multivessel_collate_fn(batch):
    """将不同船数的样本 padding 到 batch 内最大船数"""
    max_ships = max(item["n_ships"] for item in batch)
    obs_steps = batch[0]["obs"].shape[1]
    pred_steps = batch[0]["pred"].shape[1]
    n_feat = batch[0]["obs"].shape[2]  # 7 after sin/cos encoding
    batch_size = len(batch)
    has_encounter = "dcpa_matrix" in batch[0]

    obs_padded = torch.zeros(batch_size, max_ships, obs_steps, n_feat)
    pred_padded = torch.zeros(batch_size, max_ships, pred_steps, n_feat)
    adj_padded = torch.zeros(batch_size, max_ships, max_ships)
    mask = torch.zeros(batch_size, max_ships, dtype=torch.bool)
    target_idx = torch.zeros(batch_size, dtype=torch.long)
    n_ships = torch.zeros(batch_size, dtype=torch.long)
    origin_latlon = torch.zeros(batch_size, 2)
    cos_lat = torch.zeros(batch_size)

    if has_encounter:
        dcpa_padded = torch.full((batch_size, max_ships, max_ships), float("inf"))
        tcpa_padded = torch.zeros(batch_size, max_ships, max_ships)
        cri_padded = torch.zeros(batch_size, max_ships, max_ships)
        enc_padded = torch.zeros(batch_size, max_ships, max_ships, dtype=torch.long)

    for i, item in enumerate(batch):
        ns = item["n_ships"]
        obs_padded[i, :ns] = item["obs"][:ns]
        pred_padded[i, :ns] = item["pred"][:ns]
        adj_padded[i, :ns, :ns] = item["adj_matrix"][:ns, :ns]
        mask[i, :ns] = True
        target_idx[i] = item["target_ship_idx"]
        n_ships[i] = ns
        origin_latlon[i] = item["origin_latlon"]
        cos_lat[i] = item["cos_lat"]

        if has_encounter:
            dcpa_padded[i, :ns, :ns] = item["dcpa_matrix"][:ns, :ns]
            tcpa_padded[i, :ns, :ns] = item["tcpa_matrix"][:ns, :ns]
            cri_padded[i, :ns, :ns] = item["cri_matrix"][:ns, :ns]
            enc_padded[i, :ns, :ns] = item["encounter_type"][:ns, :ns]

    result = {
        "obs": obs_padded,
        "pred": pred_padded,
        "adj_matrix": adj_padded,
        "mask": mask,
        "target_ship_idx": target_idx,
        "n_ships": n_ships,
        "origin_latlon": origin_latlon,
        "cos_lat": cos_lat,
    }

    if has_encounter:
        result["dcpa_matrix"] = dcpa_padded
        result["tcpa_matrix"] = tcpa_padded
        result["cri_matrix"] = cri_padded
        result["encounter_type"] = enc_padded

    return result


def create_dataloaders(data_root, batch_size=32, num_workers=4, normalize=True):
    """创建 train/val/test DataLoader

    Args:
        data_root: obs10_pred10 目录路径
        batch_size: batch size
        num_workers: DataLoader workers
        normalize: 是否归一化

    Returns:
        dict of DataLoaders: {"train": ..., "val": ..., "test": ...}
    """
    loaders = {}
    train_norm_params = None

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            print(f"  警告: {split_dir} 不存在，跳过")
            continue

        sample_files = [f for f in os.listdir(split_dir) if f.startswith("sample_") and f.endswith(".npz")]
        if not sample_files:
            print(f"  警告: {split_dir} 无样本文件，跳过")
            continue

        dataset = MultiVesselDataset(
            split_dir, normalize=normalize,
            random_target=(split == "train"),
            norm_params=train_norm_params if split != "train" else None,
            split=split,
        )

        if split == "train" and dataset.norm_params is not None:
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

    return loaders


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    data_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "ship_trajectory_prediction", "data", "final", "obs10_pred10"
    )

    if not os.path.exists(data_root):
        print(f"数据目录不存在: {data_root}")
        print("请先运行数据处理管线 (02→04→07→11→08)")
        sys.exit(1)

    loaders = create_dataloaders(data_root, batch_size=4, num_workers=0, normalize=False)

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n{split} batch:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape} {v.dtype}")
            else:
                print(f"  {k}: {v}")