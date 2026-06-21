"""
Constant Velocity (CV) Baseline
用 obs 最后几步的速度线性外推预测轨迹
如果 ADE ≈ 0.46nm，则证实"直接解码模型 ≈ 线性外推"假说
"""
import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# 复用数据加载器
_REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO_DIR not in sys.path:
    sys.path.insert(0, _REPO_DIR)
from data_provider.dataloader_multivessel import MultiVesselDataset

def compute_cv_prediction(obs_seq, pred_steps):
    """Constant Velocity prediction: 用最后两步的速度外推
    
    obs_seq: [T_obs, D] 归一化后的观测序列
    Returns: [pred_steps, 2] 预测的 (lat, lon)
    """
    # 用最后两步计算速度
    last_pos = obs_seq[-1, :2]  # [2] lat, lon
    prev_pos = obs_seq[-2, :2]
    velocity = last_pos - prev_pos  # [2] delta_lat, delta_lon per step
    
    # 外推 pred_steps 步
    steps = torch.arange(1, pred_steps + 1, device=obs_seq.device).unsqueeze(1)  # [T_pred, 1]
    pred = last_pos.unsqueeze(0) + steps * velocity.unsqueeze(0)  # [T_pred, 2]
    return pred


def evaluate_cv(data_dir, batch_size=64, num_workers=4):
    """评估 CV baseline"""
    # 创建数据集
    train_ds = MultiVesselDataset(os.path.join(data_dir, 'train'), normalize=True, split='train')
    norm_params = train_ds.norm_params
    
    test_ds = MultiVesselDataset(
        os.path.join(data_dir, 'test'), 
        normalize=True, 
        norm_params=norm_params, 
        split='test'
    )
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=0, pin_memory=True)
    
    # 反归一化到海里坐标系 (已经是海里, 不需要再乘60)
    mean = torch.tensor(norm_params['mean'][:2])  # [lat, lon]
    std = torch.tensor(norm_params['std'][:2])
    
    all_ades = []
    all_fdes = []
    
    for batch in test_loader:
        obs = batch['obs']  # [B, N, T_obs, 7]
        pred = batch['pred']  # [B, N, T_pred, 7]
        target_idx = batch['target_ship_idx']  # [B]
        
        B = obs.shape[0]
        
        for b in range(B):
            idx = target_idx[b].item()
            obs_b = obs[b, idx]  # [T_obs, 7]
            pred_b = pred[b, idx]  # [T_pred, 7]
            
            # CV prediction (归一化空间)
            cv_pred_norm = compute_cv_prediction(obs_b, pred_b.shape[0])  # [T_pred, 2]
            
            # 反归一化
            cv_pred_real = cv_pred_norm * std + mean
            pred_real = pred_b[:, :2] * std + mean
            
            # 计算误差
            diff = (cv_pred_real - pred_real) ** 2
            ade = torch.sqrt(diff.sum(dim=-1)).mean().item()
            fde = torch.sqrt(diff[-1].sum()).item()
            
            all_ades.append(ade)
            all_fdes.append(fde)
    
    # 已经是海里坐标系, 不需要转换
    ade_nm = np.mean(all_ades)
    fde_nm = np.mean(all_fdes)
    
    return ade_nm, fde_nm


if __name__ == '__main__':
    data_dir = '/home/wangguangjie/djs/vessel-trajectory-prediction/ship_trajectory_prediction/data/final/obs10_pred10'
    
    print("=" * 60)
    print("  Constant Velocity (CV) Baseline Evaluation")
    print("=" * 60)
    print()
    
    ade, fde = evaluate_cv(data_dir)
    
    print(f"  TEST ADE = {ade:.4f} nm")
    print(f"  TEST FDE = {fde:.4f} nm")
    print()
    print("=" * 60)
    print()
    print("Comparison with direct-decoding models:")
    print(f"  LSTM:    ADE=0.4641, FDE=0.8993")
    print(f"  GRU:     ADE=0.4646, FDE=0.9042")
    print(f"  Bi-LSTM: ADE=0.4644, FDE=0.8844")
    print(f"  Mamba:   ADE=0.4667, FDE=0.8997")
    print(f"  CV:      ADE={ade:.4f}, FDE={fde:.4f}")
    print()
    
    if abs(ade - 0.46) < 0.1:
        print("✓ Confirmed: Direct-decoding models ≈ Constant Velocity extrapolation!")
    else:
        print(f"✗ CV baseline differs from direct-decoding models by {abs(ade - 0.46):.2f} nm")
    
    # Save results
    result_dir = '/home/wangguangjie/djs/vessel-trajectory-prediction/baseline/results/cv_baseline'
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump({
            'model': 'constant_velocity',
            'test_ade_nm': ade,
            'test_fde_nm': fde,
            'description': 'Linear extrapolation using last 2 steps velocity'
        }, f, indent=2)
    print(f"\nResults saved to {result_dir}/results.json")
