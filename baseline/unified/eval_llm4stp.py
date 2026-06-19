"""
统一评估: LLM4STP 在局部海里坐标系下的 ADE/FDE

思路:
1. 用 LLM4STP 原始 dataloader 加载数据 (MinMax 归一化 lon/lat)
2. 运行模型预测
3. 反归一化回真实 lat/lon
4. 转局部坐标 (以 target 最后 obs 点为原点, nm 单位)
5. 计算 ADE/FDE (nm), 与 baseline 公平对比
"""
import os, sys, glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import rearrange

# === 配置 ===
LLM4STP_DIR = '/home/wangguangjie/djs/baseline/LLM4STP'
CHECKPOINT = os.path.join(LLM4STP_DIR, 'checkpoints_STPGeo',
    'train_pl30_dm768_nh16_ps16_gl3_df512_stride8_itr0_ship_traj', 'checkpoint.pth')
DEVICE = 'cuda:0'
BATCH_SIZE = 32
GRID_SIZE = 64
SIGMA_X = 5
SIGMA_Y = 5
PRED_LEN = 30

sys.path.insert(0, LLM4STP_DIR)

# 确保 LLM4STP 的模块能正确加载
import importlib.util

def load_llm4stp_module(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(LLM4STP_DIR, 'models', f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def generate_gaussian_map(nodes_current, grid_size=64, sigma_x=5, sigma_y=5):
    """生成高斯密度图"""
    device = nodes_current.device
    x = nodes_current[:, -1, 0].cpu().numpy()
    y = nodes_current[:, -1, 1].cpu().numpy()
    B = len(x)
    gx = np.linspace(0, 1, grid_size)
    gy = np.linspace(0, 1, grid_size)
    gx_grid, gy_grid = np.meshgrid(gx, gy)
    sx = sigma_x / grid_size
    sy = sigma_y / grid_size
    gaussian_maps = np.zeros((B, grid_size, grid_size), dtype=np.float32)
    for i in range(B):
        gaussian_maps[i] = np.exp(
            -((gx_grid - x[i])**2 / (2 * sx**2 + 1e-8) +
              (gy_grid - y[i])**2 / (2 * sy**2 + 1e-8))
        )
    return torch.from_numpy(gaussian_maps).to(device)


def main():
    print("=" * 60)
    print("  统一评估: LLM4STP (局部海里坐标系)")
    print("=" * 60)

    # 1. 加载数据集
    from data_provider.dataloader_Geo import TrajectoryDataset

    def seq_collate(batch):
        past = torch.stack([item['past_traj'] for item in batch], dim=0)
        future = torch.stack([item['future_traj'] for item in batch], dim=0)
        return {'past_traj': past, 'future_traj': future}

    TEST_DIR = '/home/wangguangjie/djs/vessel-trajectory-prediction/ship_trajectory_prediction/data/final/obs10_pred10/test'
    test_set = TrajectoryDataset(TEST_DIR, obs_len=30, pred_len=30, skip=1)

    loader = DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=seq_collate, num_workers=0, pin_memory=False)

    lat_min, lat_max = test_set.lat_min, test_set.lat_max
    lon_min, lon_max = test_set.lon_min, test_set.lon_max
    print(f"Samples: {len(test_set)}")
    print(f"Norm: lat=[{lat_min:.4f}, {lat_max:.4f}], lon=[{lon_min:.4f}, {lon_max:.4f}]")

    # 2. 加载模型
    import yaml
    from utils.dict_as_object import DictAsObject
    with open(os.path.join(LLM4STP_DIR, 'config.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    args = DictAsObject(config)

    from models.GPT4STP_Geohash import GPT4STP
    from models.geohash import geohash_encoding

    device = torch.device(DEVICE)
    model = GPT4STP(args, device).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    print(f"Model: GPT4STP ({sum(p.numel() for p in model.parameters()):,} params)")

    # 3. 推理
    all_ades, all_fdes = [], []

    with torch.no_grad():
        for batch_data in tqdm(loader, desc='Evaluating'):
            # batch_data: dict with 'past_traj' [B,2,T], 'future_traj' [B,2,T]
            past = batch_data['past_traj']    # [B, 2, T_obs]
            future = batch_data['future_traj'] # [B, 2, T_pred]

            # 转 [B, T, 2]
            batch_x = past.permute(0, 2, 1).to(device)    # [B, T_obs, 2]
            batch_y = future.permute(0, 2, 1).to(device)   # [B, T_pred, 2]

            # Geohash + Gaussian
            gh = geohash_encoding(batch_x).to(device).float()
            gh = gh.reshape(gh.shape[0], gh.shape[1], -1)
            gmap = generate_gaussian_map(batch_x, GRID_SIZE, SIGMA_X, SIGMA_Y)

            # 推理
            outputs = model(batch_x, gmap, gh)
            pred_out = outputs[:, -PRED_LEN:, :]  # [B, T_pred, 2] (lon_n, lat_n)

            # === 反归一化 → 真实 lat/lon → 局部海里坐标 ===
            # 预测 (normalized)
            p_lon_n = pred_out[:, :, 0].cpu().numpy()
            p_lat_n = pred_out[:, :, 1].cpu().numpy()

            # 真值 (normalized)
            t_lon_n = batch_y[:, :, 0].cpu().numpy()
            t_lat_n = batch_y[:, :, 1].cpu().numpy()

            # 最后观测点 (normalized)
            o_lon_n = batch_x[:, -1, 0].cpu().numpy()
            o_lat_n = batch_x[:, -1, 1].cpu().numpy()

            # 反归一化: norm [0,1] → 真实 degrees
            lat_range = lat_max - lat_min + 1e-8
            lon_range = lon_max - lon_min + 1e-8

            pred_lat = p_lat_n * lat_range + lat_min
            pred_lon = p_lon_n * lon_range + lon_min
            tgt_lat  = t_lat_n * lat_range + lat_min
            tgt_lon  = t_lon_n * lon_range + lon_min
            obs_lat  = o_lat_n * lat_range + lat_min  # origin
            obs_lon  = o_lon_n * lon_range + lon_min

            B = pred_lat.shape[0]
            for b in range(B):
                cos_lat = np.cos(np.radians(obs_lat[b]))

                # 预测 → 局部 nm
                d_lat_p = (pred_lat[b] - obs_lat[b]) * 60.0
                d_lon_p = (pred_lon[b] - obs_lon[b]) * 60.0 * cos_lat

                # 真值 → 局部 nm
                d_lat_t = (tgt_lat[b] - obs_lat[b]) * 60.0
                d_lon_t = (tgt_lon[b] - obs_lon[b]) * 60.0 * cos_lat

                dist = np.sqrt((d_lat_p - d_lat_t)**2 + (d_lon_p - d_lon_t)**2)
                all_ades.append(dist.mean())
                all_fdes.append(dist[-1])

    ade = np.mean(all_ades)
    fde = np.mean(all_fdes)

    print(f"\n{'='*60}")
    print(f"  LLM4STP (局部海里坐标系) ADE = {ade:.4f} nm")
    print(f"  LLM4STP (局部海里坐标系) FDE = {fde:.4f} nm")
    print(f"{'='*60}\n")

    # 保存
    import json
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'llm4stp_unified')
    os.makedirs(out_dir, exist_ok=True)
    result = {'model': 'LLM4STP', 'test_ade_nm': float(ade), 'test_fde_nm': float(fde),
              'n_samples': len(all_ades), 'eval': 'local_nm_unified'}
    out_path = os.path.join(out_dir, 'results.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
