#!/usr/bin/env python3
"""
Fair evaluation of official iTransformer on ship trajectory prediction.
Uses NPZ data via MultiVesselDataset.

Key design choices vs vanilla iTransformer:
  - use_norm=False by default (data is already normalized by dataset stats;
    instance norm on top removes the positional trend signal)
  - Loss computed on lat/lon only (first 2 variates), matching ADE/FDE eval
  - Early stopping on val ADE, not on 7-variate MSE
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from model.iTransformer import Model
from data_provider.dataset_ship import Dataset_Ship


def _model_config(seq_len, pred_len, d_model, n_heads, e_layers, d_ff,
                  dropout, use_norm):
    from types import SimpleNamespace
    return SimpleNamespace(
        seq_len=seq_len, pred_len=pred_len, output_attention=False,
        use_norm=use_norm, d_model=d_model, n_heads=n_heads, e_layers=e_layers,
        d_ff=d_ff, factor=1, dropout=dropout, embed='timeF', freq='t',
        activation='gelu', class_strategy='projection',
    )


def compute_ade_fde(pred, target, norm_params):
    """ADE/FDE on lat/lon in nautical miles."""
    mean = torch.tensor(norm_params['mean'][:2], device=pred.device)
    std = torch.tensor(norm_params['std'][:2], device=pred.device)
    pred_real = pred * std + mean
    target_real = target * std + mean
    dist = torch.norm(pred_real - target_real, dim=-1)
    return dist.mean().item(), dist[:, -1].mean().item()


def run_epoch(model, loader, criterion, device, norm_params, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    losses, ades, fdes = [], [], []

    ctx = torch.no_grad() if not is_train else torch.enable_grad()
    with ctx:
        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x, None, None, None)  # [B, pred_len, 7]
            loss = criterion(outputs[:, :, :2], batch_y[:, :, :2])

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())

            with torch.no_grad():
                ade, fde = compute_ade_fde(
                    outputs[:, :, :2].detach(), batch_y[:, :, :2].detach(),
                    norm_params)
                ades.append(ade)
                fdes.append(fde)

    return np.mean(losses), np.mean(ades), np.mean(fdes)


def main():
    parser = argparse.ArgumentParser(description='Official iTransformer – ship trajectories')
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(_REPO_DIR,
                                             'ship_trajectory_prediction/data/final/pred10'))
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_norm', action='store_true', default=False,
                        help='Enable instance normalization (default OFF for pre-normalized data)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (1e-3 matches unified baselines)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Data ────────────────────────────────────────────────────────────
    size = [args.seq_len, 0, args.pred_len]
    train_ds = Dataset_Ship(root_path=args.data_root, flag='train', size=size)
    val_ds = Dataset_Ship(root_path=args.data_root, flag='val', size=size)
    test_ds = Dataset_Ship(root_path=args.data_root, flag='test', size=size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=False)

    norm_params = train_ds.norm_params
    print(f'Data: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}')
    print(f'Seq: {args.seq_len} -> {args.pred_len}')
    print(f'Norm std (nm): lat={norm_params["std"][0]:.4f} lon={norm_params["std"][1]:.4f}')

    # ── Model ───────────────────────────────────────────────────────────
    cfg = _model_config(args.seq_len, args.pred_len, args.d_model,
                        args.n_heads, args.e_layers, args.d_ff,
                        args.dropout, args.use_norm)
    model = Model(cfg).float().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'iTransformer: d_model={args.d_model} n_heads={args.n_heads} '
          f'e_layers={args.e_layers} d_ff={args.d_ff} use_norm={args.use_norm} '
          f'| params={n_params:,}')

    # ── Optimizer ───────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    criterion = nn.MSELoss()

    results_dir = os.path.join(_SCRIPT_DIR, 'results_ship', f'pred{args.pred_len}')
    os.makedirs(results_dir, exist_ok=True)

    # ── Training (early stop on val ADE) ────────────────────────────────
    best_ade = float('inf')
    best_fde = float('inf')
    best_val_loss = float('inf')
    patience_ctr = 0
    history = []

    print(f'\nTraining: epochs={args.epochs} lr={args.lr} patience={args.patience}')
    print(f'Loss: MSE on lat/lon only | Early stop: val ADE')
    print(f'Results → {results_dir}\n')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, _, _ = run_epoch(model, train_loader, criterion, device,
                                     norm_params, optimizer)
        val_loss, val_ade, val_fde = run_epoch(model, val_loader, criterion,
                                                device, norm_params)

        scheduler.step()
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history.append({'epoch': epoch, 'train_loss': train_loss,
                        'val_loss': val_loss, 'ade_nm': val_ade,
                        'fde_nm': val_fde, 'lr': lr_now})

        improved = val_ade < best_ade
        if improved:
            best_ade, best_fde, best_val_loss = val_ade, val_fde, val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(results_dir, 'best.pt'))

        print(f'Epoch {epoch:3d}/{args.epochs} ({elapsed:.1f}s) | '
              f'train={train_loss:.6f} val={val_loss:.6f} | '
              f'ADE={val_ade:.4f}nm FDE={val_fde:.4f}nm | lr={lr_now:.2e}'
              f'{" *" if improved else f" (pat {patience_ctr}/{args.patience})"}')

        if not improved:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print('Early stopping.')
                break

    # ── Test ────────────────────────────────────────────────────────────
    print(f'\nTesting (best val ADE={best_ade:.4f}nm)')
    model.load_state_dict(torch.load(os.path.join(results_dir, 'best.pt'),
                                     map_location=device, weights_only=True))
    test_loss, test_ade, test_fde = run_epoch(model, test_loader, criterion,
                                              device, norm_params)

    print(f'  TEST ADE = {test_ade:.4f} nm')
    print(f'  TEST FDE = {test_fde:.4f} nm')

    results = {
        'model': 'iTransformer_official',
        'test_ade_nm': test_ade,
        'test_fde_nm': test_fde,
        'best_val_ade_nm': best_ade,
        'best_val_fde_nm': best_fde,
        'best_val_loss': best_val_loss,
        'n_params': n_params,
        'config': {
            'd_model': args.d_model, 'n_heads': args.n_heads,
            'e_layers': args.e_layers, 'd_ff': args.d_ff,
            'dropout': args.dropout, 'lr': args.lr,
            'batch_size': args.batch_size, 'seq_len': args.seq_len,
            'pred_len': args.pred_len, 'seed': args.seed,
            'use_norm': args.use_norm, 'activation': 'gelu',
        },
        'history': history,
    }
    with open(os.path.join(results_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'  Results saved to {results_dir}/results.json')


if __name__ == '__main__':
    main()
 