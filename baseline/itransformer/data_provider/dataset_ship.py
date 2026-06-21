"""
NPZ ship trajectory dataset adapter for official iTransformer.

Reuses MultiVesselDataset to get properly encoded 7D features,
then wraps them into the (batch_x, batch_y, batch_x_mark, batch_y_mark)
format expected by the iTransformer framework.
"""
import importlib.util
import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset

_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_MVD_PATH = os.path.join(_REPO_DIR, 'data_provider', 'dataloader_multivessel.py')
_spec = importlib.util.spec_from_file_location('dataloader_multivessel', _MVD_PATH)
_mvd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mvd)
MultiVesselDataset = _mvd.MultiVesselDataset


class Dataset_Ship(Dataset):
    """Adapts NPZ multi-vessel data to iTransformer's expected format.

    Returns:
        seq_x:      [seq_len, 7]   observed 7D features
        seq_y:      [pred_len, 7]  prediction target (full 7D)
        seq_x_mark: [seq_len, 1]   dummy time marks (zeros)
        seq_y_mark: [pred_len, 1]  dummy time marks (zeros)
    """

    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path=None, target=None,
                 scale=True, timeenc=0, freq='t'):
        self.scale = scale
        self.flag = flag

        if size is not None:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        else:
            self.seq_len = 30
            self.label_len = 0
            self.pred_len = 10

        split_map = {'train': 'train', 'val': 'val', 'test': 'test'}
        split = split_map.get(flag, 'train')
        data_dir = os.path.join(root_path, split)

        is_train = (flag == 'train')

        if is_train:
            self.inner = MultiVesselDataset(
                data_dir, normalize=scale,
                random_target=True, split='train',
            )
            self.norm_params = self.inner.norm_params
        else:
            train_dir = os.path.join(root_path, 'train')
            train_ds = MultiVesselDataset(train_dir, normalize=scale, split='train')
            self.norm_params = train_ds.norm_params
            self.inner = MultiVesselDataset(
                data_dir, normalize=scale,
                random_target=False, norm_params=self.norm_params, split=split,
            )

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        target_idx = item['target_ship_idx']

        obs = item['obs'][target_idx].float()    # [T_obs, 7]
        pred = item['pred'][target_idx].float()  # [T_pred, 7]

        seq_x = obs                              # [seq_len, 7]
        seq_y = pred                             # [pred_len, 7]

        seq_x_mark = torch.zeros(seq_x.shape[0], 1)
        seq_y_mark = torch.zeros(seq_y.shape[0], 1)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def inverse_transform(self, data):
        if self.norm_params is None:
            return data
        mean = np.array(self.norm_params['mean'][:3])
        std = np.array(self.norm_params['std'][:3])
        result = data.copy()
        result[..., :3] = result[..., :3] * std + mean
        return result
