"""
统一 Baseline 训练脚本
用法:
  python -m unified.train --model lstm  --gpu 1
  python -m unified.train --model bilstm --gpu 2
"""
import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

# 确保能找到模块
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

from data_adapter import build_dataloaders
from models import build_model, MODEL_REGISTRY

# 数据路径 (绝对路径)
DATA_ROOT = '/home/wangguangjie/djs/vessel-trajectory-prediction/ship_trajectory_prediction/data/final/obs10_pred10'


def compute_metrics(pred_nm, target_nm, norm_params):
    """计算 ADE/FDE (海里)

    坐标系: 局部坐标，已转换为海里 (60nm/deg lat)
    norm_params: mean=[0,0,sog_mean], std=[lat_std_nm, lon_std_nm, sog_std]

    pred_nm:   [B, T, 2]  归一化坐标
    target_nm: [B, T, 2]  归一化坐标
    """
    # 反归一化: 归一化坐标 -> 海里坐标
    mean = torch.tensor(norm_params['mean'][:2], device=pred_nm.device)
    std = torch.tensor(norm_params['std'][:2], device=pred_nm.device)

    pred_real = pred_nm * std + mean     # [B, T, 2] in nautical miles
    target_real = target_nm * std + mean

    # 欧氏距离 (已在海里坐标系)
    disp = pred_real - target_real        # [B, T, 2]
    dist = torch.norm(disp, dim=-1)       # [B, T]

    ade_nm = dist.mean().item()            # Average Displacement Error (nm)
    fde_nm = dist[:, -1].mean().item()     # Final Displacement Error (nm)
    return ade_nm, fde_nm


def _forward_model(model, model_name, batch, pred_steps, device):
    """统一前向传播接口"""
    if model_name == 'social_lstm':
        target_obs, target_pred, neighbor_obs, mask = [x.to(device) for x in batch]
        pred_out = model(target_obs, neighbor_obs=neighbor_obs)
        return pred_out, target_pred
    else:
        obs, pred_target = batch
        obs, pred_target = obs.to(device), pred_target.to(device)
        # Check if model needs pred_steps arg
        fwd_args = model.forward.__code__.co_varnames
        if 'pred_steps' in fwd_args:
            pred_out = model(obs, pred_steps=pred_steps)
        else:
            pred_out = model(obs)
        return pred_out, pred_target


def train_one_model(model_name, args):
    print(f"\n{'='*60}")
    print(f"  Training: {model_name.upper()}")
    print(f"  GPU: {args.gpu}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"{'='*60}\n")

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 避免 cuDNN 多 GPU 冲突
    torch.backends.cudnn.enabled = False

    # Data
    model_type = 'social' if model_name in ('social_lstm',) else 'simple'
    train_loader, val_loader, test_loader, norm_params = build_dataloaders(
        DATA_ROOT, batch_size=args.batch_size, num_workers=args.num_workers,
        model_type=model_type,
    )
    pred_steps = train_loader.dataset.pred_steps
    print(f"  Data: train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")
    print(f"  Pred steps: {pred_steps}")
    print(f"  Norm std (nm): lat={norm_params['std'][0]:.4f} lon={norm_params['std'][1]:.4f}")

    # Model
    model = build_model(model_name, pred_steps=pred_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params: {n_params:,}")

    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    # Results dir
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results', model_name)
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float('inf')
    best_ade = float('inf')
    best_fde = float('inf')
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        t0 = time.time()
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            pred_out, target = _forward_model(model, model_name, batch, pred_steps, device)
            loss = criterion(pred_out, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        all_pred = []
        all_target = []
        with torch.no_grad():
            for batch in val_loader:
                pred_out, target = _forward_model(model, model_name, batch, pred_steps, device)
                loss = criterion(pred_out, target)
                val_losses.append(loss.item())
                all_pred.append(pred_out.cpu())
                all_target.append(target.cpu())

        scheduler.step()

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        # Compute metrics
        all_pred_cat = torch.cat(all_pred, dim=0)
        all_target_cat = torch.cat(all_target, dim=0)
        ade_nm, fde_nm = compute_metrics(all_pred_cat, all_target_cat, norm_params)

        dt = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch+1:3d}/{args.epochs} | "
              f"train={train_loss:.6f} val={val_loss:.6f} | "
              f"ADE={ade_nm:.4f}nm FDE={fde_nm:.4f}nm | "
              f"lr={lr_now:.2e} t={dt:.1f}s")

        history.append({
            'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
            'ade_nm': ade_nm, 'fde_nm': fde_nm, 'lr': lr_now,
        })

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ade = ade_nm
            best_fde = fde_nm
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(results_dir, 'best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
                break

    # Final test
    print(f"\n{'='*60}")
    print(f"  Testing {model_name.upper()} (best val_loss={best_val_loss:.6f}, ADE={best_ade:.4f}nm)")
    print(f"{'='*60}")

    model.load_state_dict(torch.load(os.path.join(results_dir, 'best.pt')))
    model.eval()
    test_pred = []
    test_target = []
    with torch.no_grad():
        for batch in test_loader:
            pred_out, target = _forward_model(model, model_name, batch, pred_steps, device)
            test_pred.append(pred_out.cpu())
            test_target.append(target.cpu())

    test_pred_cat = torch.cat(test_pred, dim=0)
    test_target_cat = torch.cat(test_target, dim=0)
    test_ade, test_fde = compute_metrics(test_pred_cat, test_target_cat, norm_params)

    print(f"  TEST ADE = {test_ade:.4f} nm")
    print(f"  TEST FDE = {test_fde:.4f} nm")

    # Save results
    results = {
        'model': model_name,
        'test_ade_nm': test_ade,
        'test_fde_nm': test_fde,
        'best_val_ade_nm': best_ade,
        'best_val_fde_nm': best_fde,
        'best_val_loss': best_val_loss,
        'n_params': n_params,
        'config': vars(args),
        'history': history,
    }
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {results_dir}/results.json\n")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    train_one_model(args.model, args)


if __name__ == '__main__':
    main()
