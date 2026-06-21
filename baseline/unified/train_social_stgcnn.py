"""
Social-STGCNN 统一评估脚本
输入: [B, 2, T_obs, V], 邻接矩阵: [T_obs, V, V] (全局固定)
输出: [B, T_pred, 5, V] → 取 target ship 前2维
"""
import os, sys, glob, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from functools import partial

import argparse

sys.path.insert(0, '/home/wangguangjie/djs/vessel-trajectory-prediction')
sys.path.insert(0, '/home/wangguangjie/djs/baseline/social-stgcnn')
torch.backends.cudnn.enabled = False

from data_provider.dataloader_multivessel import MultiVesselDataset
from model import social_stgcnn

DEFAULT_DATA_ROOT = '/home/wangguangjie/djs/vessel-trajectory-prediction/ship_trajectory_prediction/data/final/obs10_pred10'
RESULTS_BASE_DIR = '/home/wangguangjie/djs/vessel-trajectory-prediction/baseline/results'
DEVICE = 'cuda:0'
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
PATIENCE = 20
MAX_SHIPS = 18


class STGCNNDataset(Dataset):
    def __init__(self, data_dir, normalize=True, norm_params=None, split=None, max_ships=8):
        self.inner = MultiVesselDataset(data_dir, normalize=normalize,
            random_target=True, norm_params=norm_params, split=split)
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

        # 确保 target_idx < max_ships
        if target_idx >= self.max_ships:
            target_idx = 0  # 回退到第一艘船

        obs = np.zeros((2, self.obs_steps, self.max_ships), dtype=np.float32)
        pred = np.zeros((2, self.pred_steps, self.max_ships), dtype=np.float32)
        mask = np.zeros(self.max_ships, dtype=np.float32)

        for s in range(n_ships):
            obs_ship = item['obs'][s].numpy()
            obs[0, :, s] = obs_ship[:self.obs_steps, 0]
            obs[1, :, s] = obs_ship[:self.obs_steps, 1]
            pred_ship = item['pred'][s].numpy()
            pred[0, :, s] = pred_ship[:self.pred_steps, 0]
            pred[1, :, s] = pred_ship[:self.pred_steps, 1]
            mask[s] = 1.0

        return {
            'obs': torch.from_numpy(obs),
            'pred': torch.from_numpy(pred),
            'target_idx': int(target_idx),
            'mask': torch.from_numpy(mask),
        }


def build_global_adj(data_dir, seq_len, max_ships, n_samples=500):
    """构建全局邻接矩阵 [seq_len, V, V]"""
    npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))[:n_samples]
    adj_sum = np.zeros((max_ships, max_ships))
    count = 0
    for f in npz_files:
        d = np.load(f)
        n = min(int(d['n_ships']), max_ships)
        pos = np.zeros((2, n))
        for s in range(n):
            pos[0, s] = d['obs'][s, -1, 0]
            pos[1, s] = d['obs'][s, -1, 1]
        for i in range(n):
            for j in range(n):
                dist = np.sqrt((pos[0, i] - pos[0, j])**2 + (pos[1, i] - pos[1, j])**2)
                adj_sum[i, j] += np.exp(-dist / 5.0)
        count += 1

    adj_avg = adj_sum / max(count, 1)
    np.fill_diagonal(adj_avg, 0.0)

    # [seq_len, V, V]: 每个时间步用相同距离邻接 + 自环
    adj_stack = np.stack([adj_avg] * seq_len, axis=0).astype(np.float32)
    for t in range(seq_len):
        np.fill_diagonal(adj_stack[t], 1.0)
    return adj_stack


def collate_stgcnn(batch, global_adj_tensor):
    obs = torch.stack([b['obs'] for b in batch])
    pred = torch.stack([b['pred'] for b in batch])
    target_idx = torch.tensor([b['target_idx'] for b in batch])
    mask = torch.stack([b['mask'] for b in batch])
    return obs, pred, global_adj_tensor, target_idx, mask


def run_epoch(model, loader, adj_tensor, device, norm_params, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0
    all_ades, all_fdes = [], []
    mean_t = torch.tensor(norm_params['mean'][:2], device=device)
    std_t = torch.tensor(norm_params['std'][:2], device=device)

    for obs, pred, adj, target_idx, mask in loader:
        obs, pred, adj, mask = obs.to(device), pred.to(device), adj.to(device), mask.to(device)
        if is_train:
            optimizer.zero_grad()

        out, _ = model(obs, adj)  # [B, 5, T_pred, V]
        target = pred.permute(0, 2, 1, 3)  # [B, T_pred, 2, V]
        pred_xy = out[:, :2, :, :].permute(0, 2, 1, 3)  # [B, T_pred, 2, V]
        loss = criterion(pred_xy, target)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            B = out.shape[0]
            for b in range(B):
                ti = target_idx[b]
                p = out[b, :2, :, ti].permute(1, 0) * std_t + mean_t  # [T_pred, 2]
                t = target[b, :, :, ti] * std_t + mean_t
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
    RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, f'social_stgcnn_{dataset_name}') if dataset_name != 'obs10_pred10' else os.path.join(RESULTS_BASE_DIR, 'social_stgcnn')

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

    print("Building global adjacency matrix...")
    global_adj = build_global_adj(train_dir, seq_len=obs_steps, max_ships=MAX_SHIPS)
    global_adj_tensor = torch.from_numpy(global_adj)
    print(f"Adj shape: {global_adj.shape}")

    _collate = partial(collate_stgcnn, global_adj_tensor=global_adj_tensor)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=_collate, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=2)

    device = torch.device(device_str)
    model = social_stgcnn(
        n_stgcnn=1, n_txpcnn=1,
        input_feat=2, output_feat=5,
        seq_len=obs_steps, pred_seq_len=pred_steps,
        kernel_size=3
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_ade, tr_fde = run_epoch(
            model, train_loader, global_adj_tensor, device, norm_params, criterion, optimizer)
        scheduler.step()
        vl_loss, vl_ade, vl_fde = run_epoch(
            model, val_loader, global_adj_tensor, device, norm_params, criterion)
        dt = time.time() - t0

        lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:3d}/{EPOCHS} | train={tr_loss:.6f} val={vl_loss:.6f} | "
              f"ADE={vl_ade:.4f}nm FDE={vl_fde:.4f}nm | lr={lr:.2e} t={dt:.1f}s")

        history.append({'epoch': epoch, 'train_loss': tr_loss, 'val_loss': vl_loss,
                        'ade_nm': vl_ade, 'fde_nm': vl_fde})

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test
    best_ade = min(h['ade_nm'] for h in history)
    print(f"\n  Testing (best val_loss={best_val_loss:.6f}, ADE={best_ade:.4f}nm)")
    test_set = STGCNNDataset(os.path.join(DATA_ROOT, 'test'),
        norm_params=norm_params, split='test', max_ships=MAX_SHIPS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=_collate, num_workers=2)

    model.load_state_dict(best_state)
    model.to(device).eval()
    _, test_ade, test_fde = run_epoch(
        model, test_loader, global_adj_tensor, device, norm_params, criterion)

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
