"""
统一 Baseline 数据适配器
复用 vessel-trajectory-prediction 的 MultiVesselDataset
提取目标船的 obs/pred 序列供 baseline 模型使用
"""
import sys
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# 让本脚本能 import vessel-trajectory-prediction 的数据加载器
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)

from data_provider.dataloader_multivessel import MultiVesselDataset


class BaselineDataset(Dataset):
    """适配 MultiVesselDataset -> baseline 模型所需格式

    MultiVesselDataset 返回 dict:
      obs: [N_ships, T_obs, 7], pred: [N_ships, T_pred, 7], target_ship_idx: int, ...

    本适配器提取目标船:
      obs:       [T_obs, 7]    目标船观测特征 (归一化后)
      pred_lat:  [T_pred, 2]   目标船真值 (lat, lon 归一化后)
    """

    def __init__(self, data_dir, normalize=True, norm_params=None, split=None):
        self.inner = MultiVesselDataset(
            data_dir, normalize=normalize,
            random_target=True, norm_params=norm_params, split=split,
        )
        self.obs_steps = self.inner.obs_steps
        self.pred_steps = self.inner.pred_steps
        self.output_dim = 7  # after angle encoding (lat,lon,sog,sin_cog,cos_cog,sin_hdg,cos_hdg)
        self.norm_params = self.inner.norm_params

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]  # dict
        target_idx = item['target_ship_idx']
        obs = item['obs'][target_idx].float()          # [T_obs, 7]
        pred = item['pred'][target_idx, :, :2].float() # [T_pred, 2] (lat, lon only)
        return obs, pred


class SocialBaselineDataset(Dataset):
    """带邻居信息的数据集，用于 Social-LSTM / Social-STGCNN 等交互模型

    返回:
      target_obs:    [T_obs, 7]
      target_pred:   [T_pred, 2]
      neighbor_obs:  [N_max, T_obs, 7]   (padding)
      mask:          [N_max]
    """

    def __init__(self, data_dir, normalize=True, norm_params=None, split=None, max_neighbors=15):
        self.inner = MultiVesselDataset(
            data_dir, normalize=normalize,
            random_target=True, norm_params=norm_params, split=split,
        )
        self.max_neighbors = max_neighbors
        self.obs_steps = self.inner.obs_steps
        self.pred_steps = self.inner.pred_steps
        self.norm_params = self.inner.norm_params

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]  # dict
        target_idx = item['target_ship_idx']
        n_ships = item['n_ships']

        target_obs = item['obs'][target_idx].float()          # [T_obs, 7]
        target_pred = item['pred'][target_idx, :, :2].float() # [T_pred, 2]

        # Neighbor ships (exclude target, pad to max_neighbors)
        T_obs = self.inner.obs_steps
        neighbor_obs = torch.zeros(self.max_neighbors, T_obs, 7)
        mask = torch.zeros(self.max_neighbors)

        neighbors = [i for i in range(n_ships) if i != target_idx][:self.max_neighbors]
        for j, ni in enumerate(neighbors):
            neighbor_obs[j] = item['obs'][ni].float()
            mask[j] = 1.0

        return target_obs, target_pred, neighbor_obs, mask


def build_dataloaders(data_root, batch_size=64, num_workers=4, model_type='simple'):
    """构建 train/val/test dataloaders

    Args:
        data_root: 数据根目录 (e.g. .../obs10_pred10)
        batch_size: batch size
        model_type: 'simple' 用 BaselineDataset, 'social' 用 SocialBaselineDataset
    """
    DatasetCls = BaselineDataset if model_type == 'simple' else SocialBaselineDataset

    train_ds = DatasetCls(os.path.join(data_root, 'train'), normalize=True, split='train')
    norm_params = train_ds.norm_params
    val_ds = DatasetCls(os.path.join(data_root, 'val'), normalize=True, norm_params=norm_params, split='val')
    test_ds = DatasetCls(os.path.join(data_root, 'test'), normalize=True, norm_params=norm_params, split='test')

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, norm_params
