"""
05_generate_splits.py - Time-based train/val/test split + sliding window sample generation.
Output NPZ compatible with dataloader_multivessel.py.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, compute_mean_adjacency


def timestamp_to_date(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d')


def get_split(timestamps, splits_cfg):
    mid_ts = timestamps[len(timestamps) // 2]
    scene_date = datetime.fromtimestamp(mid_ts, tz=timezone.utc).strftime('%Y-%m-%d')

    for split_name in ['train', 'val', 'test']:
        s = splits_cfg[split_name]
        if s['start'] <= scene_date <= s['end']:
            return split_name
    return None


def generate_samples_from_scene(scene_data, config):
    splits_cfg = config['splits']
    obs_steps = splits_cfg['obs_steps']
    pred_variants = splits_cfg['pred_variants']
    slide_step = splits_cfg['slide_step']
    min_pred_sog = splits_cfg.get('min_pred_mean_sog', 0.5)
    min_tail_sog = splits_cfg.get('min_obs_tail_sog', 1.0)
    tail_steps = splits_cfg.get('obs_tail_steps', 5)

    trajs = scene_data['trajectories']
    timestamps = scene_data['timestamps']
    n_ships, n_steps, n_feat = trajs.shape

    split = get_split(timestamps, splits_cfg)
    if split is None:
        return []

    max_pred = max(pred_variants)
    min_pred = min(pred_variants)
    min_window = obs_steps + min_pred

    if n_steps < min_window:
        return []

    has_dcpa = 'dcpa_matrix' in scene_data
    has_enc = 'encounter_type_temporal' in scene_data

    samples = []
    for start in range(0, n_steps - min_window + 1, slide_step):
        obs_end = start + obs_steps

        obs = trajs[:, start:obs_end, :]

        obs_tail_sog = obs[:, -tail_steps:, 2].mean(axis=1)
        moving_ratio = np.mean(obs_tail_sog >= min_tail_sog)
        if moving_ratio < 0.5:
            continue

        obs_adj = compute_mean_adjacency(obs, threshold_nm=3.0)
        if obs_adj.max() < 1e-6:
            continue

        for pred_len in pred_variants:
            pred_end = obs_end + pred_len
            if pred_end > n_steps:
                continue

            pred = trajs[:, obs_end:pred_end, :]

            pred_mean_sog = pred[:, :, 2].mean(axis=1)
            pred_moving = np.mean(pred_mean_sog >= min_pred_sog)
            if pred_moving < 0.5:
                continue

            sample = {
                'obs': obs.astype(np.float32),
                'pred': pred.astype(np.float32),
                'adj_matrix': obs_adj,
                'n_ships': np.int32(n_ships),
                'scene_id': np.int32(scene_data.get('scene_id', 0)),
                'target_ship_idx': np.int32(0),
                'pred_steps': np.int32(pred_len),
                'split': split,
            }

            if has_dcpa:
                dcpa_3d = scene_data['dcpa_matrix'][:, :, start:obs_end]
                sample['dcpa_matrix'] = dcpa_3d.min(axis=2).astype(np.float32)
                tcpa_3d = scene_data['tcpa_matrix'][:, :, start:obs_end]
                min_dcpa_t = np.argmin(dcpa_3d, axis=2)
                tcpa_at_min = np.zeros((n_ships, n_ships), dtype=np.float32)
                for i in range(n_ships):
                    for j in range(n_ships):
                        tcpa_at_min[i, j] = tcpa_3d[i, j, min_dcpa_t[i, j]]
                sample['tcpa_matrix'] = tcpa_at_min
                cri_3d = scene_data['cri_matrix'][:, :, start:obs_end]
                sample['cri_matrix'] = cri_3d.max(axis=2).astype(np.float32)

            if has_enc:
                enc_temporal = scene_data['encounter_type_temporal'][:, :, start:obs_end]
                window_dominant = np.zeros((n_ships, n_ships), dtype=np.int8)
                for i in range(n_ships):
                    for j in range(i + 1, n_ships):
                        dcpa_window = dcpa_3d[i, j] if has_dcpa else np.zeros(obs_steps)
                        min_t = np.argmin(dcpa_window)
                        window_dominant[i, j] = enc_temporal[i, j, min_t]
                        window_dominant[j, i] = enc_temporal[j, i, min_t]
                sample['encounter_type'] = window_dominant

            samples.append(sample)

    return samples


def main():
    parser = argparse.ArgumentParser(description='Generate train/val/test splits')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    scenes_dir = base / config['scenes']['scenes_dir']
    output_dir = base / config['splits']['output_dir']

    for pv in config['splits']['pred_variants']:
        for split in ['train', 'val', 'test']:
            (output_dir / f"pred{pv}" / split).mkdir(parents=True, exist_ok=True)

    scene_files = sorted(scenes_dir.glob('scene_*.npz'))
    print(f"Generating samples from {len(scene_files)} scenes")

    counts = {f"{s}_pred{p}": 0 for s in ['train', 'val', 'test']
              for p in config['splits']['pred_variants']}
    sample_counter = {k: 0 for k in counts}

    for sf in tqdm(scene_files, desc="Generating samples"):
        scene_data = dict(np.load(sf, allow_pickle=True))
        samples = generate_samples_from_scene(scene_data, config)

        for sample in samples:
            split = sample.pop('split')
            pred_len = int(sample['pred_steps'])
            key = f"{split}_pred{pred_len}"

            idx = sample_counter[key]
            out_path = output_dir / f"pred{pred_len}" / split / f"sample_{idx:06d}.npz"
            np.savez_compressed(out_path, **sample)
            counts[key] += 1
            sample_counter[key] += 1

    print(f"\nSample generation complete:")
    for key, count in sorted(counts.items()):
        print(f"  {key}: {count}")
    print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
