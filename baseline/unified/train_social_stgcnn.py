"""
Social-STGCNN 统一评估脚本
输入: [B, 7, T_obs, V], 邻接矩阵: [B, T_obs, V, V] (per-sample)
输出: [B, 2, T_pred, V] (lat, lon)
"""
import os, sys, glob, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse

sys.path.insert(0, '/home/wangguangjie/djs/vessel-trajectory-prediction')
sys.path.insert(0, '/home/wangguangjie/djs/baseline/social-stgcnn')
torch.backends.cudnn.enabled = False

from data_provider.dataloader_multivessel import MultiVesselDataset
from model import social_stgcnn


class SocialSTGCNNWrapper(nn.Module):
    """Wrapper: internal hidden_dim for capacity + 1x1 projection to pred_dim.

    The raw social_stgcnn uses output_feat as both the internal channel width
    and the output channel count. Setting output_feat=256 wastes 254 channels
    when only 2 are used. This wrapper keeps a moderate hidden_dim for the
    graph/temporal convolutions and projects to the exact pred_dim at the end.
    """

    def __init__(self, hidden_dim=64, pred_dim=2, **stgcnn_kwargs):
        super().__init__()
        self.stgcnn = social_stgcnn(output_feat=hidden_dim, **stgcnn_kwargs)
        self.output_proj = nn.Conv2d(hidden_dim, pred_dim, kernel_size=1)

    def forward(self, v, a):
        v, a = self.stgcnn(v, a)
        v = self.output_proj(v)
        return v, a

DEFAULT_DATA_ROOT = '/home/wangguangjie/djs/vessel-trajectory-prediction/ship_trajectory_prediction/data/final/obs10_pred10'
RESULTS_BASE_DIR = '/home/wangguangjie/djs/vessel-trajectory-prediction/baseline/results'
DEVICE = 'cuda:0'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 20
MAX_SHIPS = 12


class STGCNNDataset(Dataset):
    def __init__(self, data_dir, normalize=True, norm_params=None, split=None, max_ships=8):
        self.inner = MultiVesselDataset(data_dir, normalize=normalize,
            random_target=(split == 'train' or split is None),
            norm_params=norm_params, split=split)
        self.max_ships = max_ships
        self.obs_steps = self.inner.obs_steps
        self.pred_steps = self.inner.pred_steps
        self.norm_params = self.inner.norm_params

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, idx):
        item = self.inner[idx]
        n_ships = min(item['n_ships'], self.max_ships)
        target_idx = min(item['target_ship_idx'], self.max_ships - 1)
        if target_idx >= n_ships:
            target_idx = 0

        D = 7
        obs = np.zeros((D, self.obs_steps, self.max_ships), dtype=np.float32)
        pred = np.zeros((2, self.pred_steps, self.max_ships), dtype=np.float32)
        mask = np.zeros(self.max_ships, dtype=np.float32)

        for s in range(n_ships):
            obs_ship = item['obs'][s].numpy()
            obs[:, :, s] = obs_ship[:self.obs_steps, :D].T
            pred_ship = item['pred'][s].numpy()
            pred[:, :, s] = pred_ship[:self.pred_steps, :2].T
            mask[s] = 1.0

        adj = np.eye(self.max_ships, dtype=np.float32)
        adj_raw = item['adj_matrix'].numpy()
        n_adj = min(n_ships, adj_raw.shape[0])
        adj[:n_adj, :n_adj] = adj_raw[:n_adj, :n_adj]
        for s in range(n_adj):
            adj[s, s] = 1.0
        # Threshold weak edges
        adj[adj < 0.05] = 0.0
        for s in range(n_adj):
            adj[s, s] = 1.0
        # Row-normalize: D^{-1} A
        row_sum = adj.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj = adj / row_sum

        return {
            'obs': torch.from_numpy(obs),
            'pred': torch.from_numpy(pred),
            'adj': torch.from_numpy(adj),
            'target_idx': int(target_idx),
            'mask': torch.from_numpy(mask),
        }


def collate_stgcnn(batch):
    obs = torch.stack([b['obs'] for b in batch])
    pred = torch.stack([b['pred'] for b in batch])
    target_idx = torch.tensor([b['target_idx'] for b in batch])
    mask = torch.stack([b['mask'] for b in batch])
    adj = torch.stack([b['adj'] for b in batch])
    T = obs.shape[2]
    adj_expanded = adj.unsqueeze(1).expand(-1, T, -1, -1)
    return obs, pred, adj_expanded, target_idx, mask


def run_epoch(model, loader, device, norm_params, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    all_ades, all_fdes = [], []
    mean_t = torch.tensor(norm_params['mean'][:2], device=device)
    std_t = torch.tensor(norm_params['std'][:2], device=device)

    for obs, pred, adj, target_idx, mask in loader:
        obs, pred, adj, mask = obs.to(device), pred.to(device), adj.to(device), mask.to(device)
        target_idx = target_idx.to(device)

        if is_train:
            optimizer.zero_grad()

        out, _ = model(obs, adj)  # [B, 2, T_pred, V]

        # Primary loss: target ship
        ti = target_idx.view(-1, 1, 1, 1).expand(-1, 2, out.shape[2], 1)
        pred_target = out.gather(3, ti).squeeze(3)  # [B, 2, T_pred]
        gt_target = pred.gather(3, ti).squeeze(3)        # [B, 2, T_pred]
        loss_target = criterion(pred_target, gt_target)

        # Auxiliary loss: all valid ships
        mask_exp = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, V]
        loss_all = criterion(out * mask_exp, pred * mask_exp)
        loss = loss_target + 0.5 * loss_all

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            B = out.shape[0]
            for b in range(B):
                t_idx = target_idx[b]
                p = out[b, :, :, t_idx].permute(1, 0) * std_t + mean_t  # [T_pred, 2]
                t = pred[b, :, :, t_idx].permute(1, 0) * std_t + mean_t  # [T_pred, 2]
                dist = torch.norm(p - t, dim=-1)
                all_ades.append(dist.mean().item())
                all_fdes.append(dist[-1].item())

    return total_loss / len(loader), np.mean(all_ades), np.mean(all_fdes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    DATA_ROOT = args.data_root
    device_str = f'cuda:{args.gpu}'
    dataset_name = os.path.basename(DATA_ROOT)
    if 'NOAANY' in DATA_ROOT:
        RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, 'NOAANY', f'obs30_{dataset_name}', 'social_stgcnn')
    else:
        RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, 'DMA', dataset_name, 'social_stgcnn')

    print("=" * 60)
    print("  Social-STGCNN 训练")
    print(f"  Data: {DATA_ROOT}")
    print("=" * 60)

    train_dir = os.path.join(DATA_ROOT, 'train')
    train_set = STGCNNDataset(train_dir, split='train', max_ships=MAX_SHIPS)
    norm_params = train_set.norm_params
    val_set = STGCNNDataset(os.path.join(DATA_ROOT, 'val'),
        norm_params=norm_params, split='val', max_ships=MAX_SHIPS)

    obs_steps = train_set.obs_steps
    pred_steps = train_set.pred_steps
    print(f"Train: {len(train_set)}, Val: {len(val_set)}")
    print(f"Max ships: {MAX_SHIPS}, Obs: {obs_steps}, Pred: {pred_steps}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_stgcnn, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_stgcnn, num_workers=2)

    device = torch.device(device_str)
    model = SocialSTGCNNWrapper(
        hidden_dim=128, pred_dim=2,
        n_stgcnn=2, n_txpcnn=5,
        input_feat=7,
        seq_len=obs_steps, pred_seq_len=pred_steps,
        kernel_size=3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_ade = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_ade, tr_fde = run_epoch(
            model, train_loader, device, norm_params, criterion, optimizer)
        scheduler.step()
        vl_loss, vl_ade, vl_fde = run_epoch(
            model, val_loader, device, norm_params, criterion)
        dt = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={tr_loss:.6f} val={vl_loss:.6f} | "
              f"ADE={vl_ade:.4f}nm FDE={vl_fde:.4f}nm | lr={lr:.2e} t={dt:.1f}s")

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': vl_loss,
                        'ade_nm': vl_ade, 'fde_nm': vl_fde})

        if vl_ade < best_val_ade:
            best_val_ade = vl_ade
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test
    print(f"\n  Testing (best val ADE={best_val_ade:.4f}nm)")
    test_set = STGCNNDataset(os.path.join(DATA_ROOT, 'test'),
        norm_params=norm_params, split='test', max_ships=MAX_SHIPS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_stgcnn, num_workers=2)

    model.load_state_dict(best_state)
    model.to(device).eval()
    _, test_ade, test_fde = run_epoch(
        model, test_loader, device, norm_params, criterion)

    print(f"  TEST ADE = {test_ade:.4f} nm")
    print(f"  TEST FDE = {test_fde:.4f} nm")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {'model': 'social_stgcnn', 'test_ade_nm': float(test_ade),
              'test_fde_nm': float(test_fde), 'n_params': n_params, 'history': history}
    with open(os.path.join(RESULTS_DIR, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/results.json")


if __name__ == '__main__':
    main()