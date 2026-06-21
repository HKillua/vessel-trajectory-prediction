"""Ship trajectory data adapter for MoFlow.

Wraps MultiVesselDataset and outputs x_data dicts compatible with
MoFlow's FlowMatcher / ShipMotionTransformer.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from data_provider.dataloader_multivessel import MultiVesselDataset


def _ship_collate_fixed_agents(batch, max_agents, norm_params=None):
    """Collate ship samples, always padding to max_agents.

    This ensures consistent tensor shapes matching cfg.agents,
    which FlowMatcher.sample uses to create noise tensors.
    """
    obs_steps = batch[0]['obs'].shape[1]
    pred_steps = batch[0]['pred'].shape[1]
    n_feat = batch[0]['obs'].shape[2]
    batch_size = len(batch)
    has_encounter = 'dcpa_matrix' in batch[0]
    A = max_agents

    obs_padded = torch.zeros(batch_size, A, obs_steps, n_feat)
    pred_padded = torch.zeros(batch_size, A, pred_steps, n_feat)
    adj_padded = torch.zeros(batch_size, A, A)
    mask = torch.zeros(batch_size, A, dtype=torch.bool)
    target_idx = torch.zeros(batch_size, dtype=torch.long)
    n_ships = torch.zeros(batch_size, dtype=torch.long)
    origin_latlon = torch.zeros(batch_size, 2)
    cos_lat = torch.zeros(batch_size)

    if has_encounter:
        dcpa_padded = torch.zeros(batch_size, A, A)
        tcpa_padded = torch.zeros(batch_size, A, A)
        cri_padded = torch.zeros(batch_size, A, A)
        enc_padded = torch.zeros(batch_size, A, A, dtype=torch.long)

    for i, item in enumerate(batch):
        ns = min(item['n_ships'], max_agents)
        obs_padded[i, :ns] = item['obs'][:ns]
        pred_padded[i, :ns] = item['pred'][:ns]
        adj_padded[i, :ns, :ns] = item['adj_matrix'][:ns, :ns]
        mask[i, :ns] = True
        target_idx[i] = min(item['target_ship_idx'], ns - 1)
        n_ships[i] = ns
        origin_latlon[i] = item['origin_latlon']
        cos_lat[i] = item['cos_lat']

        if has_encounter:
            dcpa_padded[i, :ns, :ns] = item['dcpa_matrix'][:ns, :ns]
            tcpa_padded[i, :ns, :ns] = item['tcpa_matrix'][:ns, :ns]
            cri_padded[i, :ns, :ns] = item['cri_matrix'][:ns, :ns]
            enc_padded[i, :ns, :ns] = item['encounter_type'][:ns, :ns]

    fut_xy = pred_padded[..., :2]  # [B, A, F, 2]

    if norm_params is not None:
        mean_xy = torch.tensor(norm_params['mean'][:2], dtype=torch.float32)
        std_xy = torch.tensor(norm_params['std'][:2], dtype=torch.float32)
        fut_xy_orig = fut_xy * std_xy + mean_xy
    else:
        fut_xy_orig = fut_xy

    data = {
        'batch_size': batch_size,
        'past_traj': obs_padded,
        'past_traj_original_scale': obs_padded,
        'fut_traj': fut_xy,
        'fut_traj_original_scale': fut_xy_orig,
        'adj_matrix': adj_padded,
        'mask': mask,
        'target_ship_idx': target_idx,
        'n_ships': n_ships,
        'origin_latlon': origin_latlon,
        'cos_lat': cos_lat,
    }

    if has_encounter:
        data['dcpa_matrix'] = dcpa_padded
        data['tcpa_matrix'] = tcpa_padded
        data['cri_matrix'] = cri_padded
        data['encounter_type'] = enc_padded

    return data


def build_ship_dataloaders(data_root, cfg, batch_size_train=64, batch_size_test=256, num_workers=4):
    """Build train / val / test DataLoaders for ship data.

    Args:
        data_root: path to obs{X}_pred{Y} directory
        cfg: config object — will have norm_params attached
        batch_size_train: training batch size
        batch_size_test: test/val batch size
        num_workers: dataloader workers

    Returns:
        train_loader, val_loader, test_loader
    """
    from functools import partial

    max_agents = cfg.agents

    loaders = {}
    train_norm_params = None

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_root, split)
        if not os.path.exists(split_dir):
            continue

        dataset = MultiVesselDataset(
            split_dir,
            normalize=True,
            random_target=(split == 'train'),
            norm_params=train_norm_params if split != 'train' else None,
            split=split,
        )

        if split == 'train' and dataset.norm_params is not None:
            train_norm_params = dataset.norm_params

        split_collate = partial(_ship_collate_fixed_agents,
                                max_agents=max_agents,
                                norm_params=train_norm_params)

        bs = batch_size_train if split == 'train' else batch_size_test
        loaders[split] = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            collate_fn=split_collate,
            pin_memory=True,
            drop_last=(split == 'train'),
        )

    cfg.norm_params = train_norm_params
    cfg.data_norm = 'pre_normalized'

    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test', val_loader)

    return train_loader, val_loader, test_loader
