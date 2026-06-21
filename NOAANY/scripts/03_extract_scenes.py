"""
03_extract_scenes.py - Encounter-centric multi-ship scene extraction.
Finds real <2nm ship pairs first, then builds scenes around them.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, haversine_nm, align_trajectories_to_grid,
    post_quality_check, compute_mean_adjacency
)


def find_encounter_seeds(day_df, config):
    """
    Find ship pairs within encounter_distance_nm at any overlapping timestep.
    Uses spatial grid pre-filtering to avoid full O(N²) comparison.
    """
    scenes_cfg = config['scenes']
    enc_dist = scenes_cfg.get('encounter_distance_nm', 2.0)
    interval = config['preprocess']['sampling_interval']
    time_window = scenes_cfg.get('time_window_min', 60) * 60

    ships = day_df.groupby('mmsi')
    ship_data = {}
    for mmsi, grp in ships:
        segs = grp.groupby('segment_id')
        for seg_id, seg in segs:
            if len(seg) < 10:
                continue
            key = (mmsi, seg_id)
            ship_data[key] = seg[['timestamp', 'lat', 'lon', 'sog', 'cog', 'heading']].values

    keys = list(ship_data.keys())
    if len(keys) < 2:
        return []

    grid_size_deg = enc_dist / 60.0 * 2
    grid_cells = defaultdict(list)
    for idx, key in enumerate(keys):
        data = ship_data[key]
        lat_min, lat_max = data[:, 1].min(), data[:, 1].max()
        lon_min, lon_max = data[:, 2].min(), data[:, 2].max()
        gi_min = int(np.floor(lat_min / grid_size_deg))
        gi_max = int(np.floor(lat_max / grid_size_deg))
        gj_min = int(np.floor(lon_min / grid_size_deg))
        gj_max = int(np.floor(lon_max / grid_size_deg))
        for gi in range(gi_min, gi_max + 1):
            for gj in range(gj_min, gj_max + 1):
                grid_cells[(gi, gj)].append(idx)

    candidate_pairs = set()
    for cell_indices in grid_cells.values():
        for a in range(len(cell_indices)):
            for b in range(a + 1, len(cell_indices)):
                i, j = cell_indices[a], cell_indices[b]
                if i > j:
                    i, j = j, i
                if keys[i][0] != keys[j][0]:
                    candidate_pairs.add((i, j))

    seeds = []
    for i, j in candidate_pairs:
        data_i = ship_data[keys[i]]
        data_j = ship_data[keys[j]]

        t_start = max(data_i[0, 0], data_j[0, 0])
        t_end = min(data_i[-1, 0], data_j[-1, 0])
        overlap = t_end - t_start
        if overlap < time_window:
            continue

        ts_i = data_i[:, 0]
        ts_j = data_j[:, 0]
        common_mask_i = (ts_i >= t_start) & (ts_i <= t_end)
        common_mask_j = (ts_j >= t_start) & (ts_j <= t_end)

        pts_i = data_i[common_mask_i]
        pts_j = data_j[common_mask_j]

        if len(pts_i) < 5 or len(pts_j) < 5:
            continue

        sample_step = max(1, len(pts_i) // 30)
        min_dist = float('inf')
        best_t = t_start
        best_lat = pts_i[0, 1]
        best_lon = pts_i[0, 2]

        for k in range(0, len(pts_i), sample_step):
            t_k = pts_i[k, 0]
            idx_j = np.argmin(np.abs(pts_j[:, 0] - t_k))
            if abs(pts_j[idx_j, 0] - t_k) > interval * 2:
                continue
            d = haversine_nm(pts_i[k, 1], pts_i[k, 2],
                             pts_j[idx_j, 1], pts_j[idx_j, 2])
            if d < min_dist:
                min_dist = d
                best_t = t_k
                best_lat = (pts_i[k, 1] + pts_j[idx_j, 1]) / 2
                best_lon = (pts_i[k, 2] + pts_j[idx_j, 2]) / 2

        if min_dist <= enc_dist:
            seeds.append({
                'mmsi_i': keys[i][0], 'seg_i': keys[i][1],
                'mmsi_j': keys[j][0], 'seg_j': keys[j][1],
                't_closest': best_t, 'dist_min': min_dist,
                'center_lat': best_lat, 'center_lon': best_lon
            })

    return seeds


def build_scene_from_seed(seed, day_df, config):
    """
    Build a multi-ship scene around an encounter seed.
    Collect all ships within radius_nm of the encounter center.
    """
    scenes_cfg = config['scenes']
    radius = scenes_cfg.get('radius_nm', 4.0)
    time_window = scenes_cfg.get('time_window_min', 60) * 60
    max_ships = scenes_cfg.get('max_ships', 20)
    min_coexist = scenes_cfg.get('min_coexist_min', 30) * 60
    interval = config['preprocess']['sampling_interval']

    t_center = seed['t_closest']
    t_start = t_center - time_window
    t_end = t_center + time_window

    window_df = day_df[
        (day_df['timestamp'] >= t_start) & (day_df['timestamp'] <= t_end)
    ]

    nearby_ships = []
    for mmsi, grp in window_df.groupby('mmsi'):
        for seg_id, seg in grp.groupby('segment_id'):
            if len(seg) < 5:
                continue
            mid_idx = len(seg) // 2
            d = haversine_nm(seg.iloc[mid_idx]['lat'], seg.iloc[mid_idx]['lon'],
                             seed['center_lat'], seed['center_lon'])
            if d <= radius:
                duration = seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]
                if duration >= min_coexist:
                    nearby_ships.append((mmsi, seg_id, d, seg))

    nearby_ships.sort(key=lambda x: x[2])
    if len(nearby_ships) > max_ships:
        nearby_ships = nearby_ships[:max_ships]

    if len(nearby_ships) < scenes_cfg.get('min_ships', 2):
        return None

    ship_dfs = [s[3] for s in nearby_ships]
    mmsi_list = [str(s[0]) for s in nearby_ships]

    seed_dfs = []
    for s in nearby_ships:
        if (s[0] == seed['mmsi_i'] and s[1] == seed['seg_i']) or \
           (s[0] == seed['mmsi_j'] and s[1] == seed['seg_j']):
            seed_dfs.append(s[3])
    if len(seed_dfs) < 2:
        seed_dfs = [s[3] for s in nearby_ships[:2]]

    all_t_start = max(df['timestamp'].iloc[0] for df in seed_dfs)
    all_t_end = min(df['timestamp'].iloc[-1] for df in seed_dfs)

    if all_t_end - all_t_start < min_coexist:
        return None

    min_scene_steps = scenes_cfg.get('min_scene_steps', 60)
    if (all_t_end - all_t_start) / interval < min_scene_steps:
        return None

    min_tc = scenes_cfg.get('min_temporal_coverage', 0.85)

    trajectories, t_grid, kept_indices = align_trajectories_to_grid(
        ship_dfs, all_t_start, all_t_end, interval_sec=interval,
        min_coverage=min_tc
    )
    if trajectories is None or trajectories.shape[0] < 2:
        return None

    mmsi_list = [mmsi_list[i] for i in kept_indices]

    trajectories = post_quality_check(
        trajectories,
        max_sog=config['preprocess'].get('max_sog_knots', 35),
        max_sog_change=config['preprocess'].get('max_sog_change_per_step', 9.0),
        timestamps=t_grid,
        sog_align_alpha=config['preprocess'].get('sog_position_align_alpha', 0.7)
    )

    coverage = scenes_cfg.get('coverage_ratio', 0.8)
    valid_ships = []
    for s in range(trajectories.shape[0]):
        sog = trajectories[s, :, 2]
        moving_ratio = np.mean(sog >= 0.5)
        if moving_ratio >= coverage:
            valid_ships.append(s)

    if len(valid_ships) < 2:
        return None

    trajectories = trajectories[valid_ships]
    mmsi_list = [mmsi_list[i] for i in valid_ships]

    adj = compute_mean_adjacency(trajectories, threshold_nm=3.0)

    return {
        'trajectories': trajectories.astype(np.float32),
        'timestamps': t_grid.astype(np.int64),
        'mmsi_list': np.array(mmsi_list),
        'adjacency': adj,
        'n_ships': len(valid_ships),
        'duration_sec': int(t_grid[-1] - t_grid[0]),
        'seed_dist_nm': float(seed['dist_min']),
        'center_lat': float(seed['center_lat']),
        'center_lon': float(seed['center_lon']),
    }


def merge_overlapping_scenes(scenes, merge_dist_nm=2.0):
    """Merge scenes with centers closer than merge_dist_nm. Keep the one with more ships."""
    if len(scenes) <= 1:
        return scenes

    keep = [True] * len(scenes)
    for i in range(len(scenes)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(scenes)):
            if not keep[j]:
                continue
            d = haversine_nm(scenes[i]['center_lat'], scenes[i]['center_lon'],
                             scenes[j]['center_lat'], scenes[j]['center_lon'])
            t_overlap = (min(scenes[i]['timestamps'][-1], scenes[j]['timestamps'][-1]) -
                         max(scenes[i]['timestamps'][0], scenes[j]['timestamps'][0]))
            if d < merge_dist_nm and t_overlap > 0:
                if scenes[j]['n_ships'] > scenes[i]['n_ships']:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
    return [s for s, k in zip(scenes, keep) if k]


def main():
    parser = argparse.ArgumentParser(description='Extract encounter-centric scenes')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    parser.add_argument('--single-day', type=str, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    proc_dir = base / config['preprocess']['processed_dir']
    scenes_dir = base / config['scenes']['scenes_dir']
    scenes_dir.mkdir(parents=True, exist_ok=True)

    if args.single_day:
        parquet_files = [Path(args.single_day)]
        if not parquet_files[0].exists():
            parquet_files = [proc_dir / args.single_day]
    else:
        parquet_files = sorted(proc_dir.glob('*_processed.parquet'))

    print(f"Processing {len(parquet_files)} files for scene extraction")
    total_scenes = 0
    scene_id = 0

    for pq_path in tqdm(parquet_files, desc="Extracting scenes"):
        day_df = pd.read_parquet(pq_path)
        if len(day_df) == 0:
            continue

        seeds = find_encounter_seeds(day_df, config)
        if not seeds:
            continue

        seeds.sort(key=lambda s: s['dist_min'])

        day_scenes = []
        for seed in seeds:
            scene = build_scene_from_seed(seed, day_df, config)
            if scene is not None:
                day_scenes.append(scene)

        day_scenes = merge_overlapping_scenes(
            day_scenes,
            merge_dist_nm=config['scenes'].get('merge_distance_nm', 2.0)
        )

        for sc in day_scenes:
            sc['scene_id'] = scene_id
            out_path = scenes_dir / f"scene_{scene_id:06d}.npz"
            np.savez_compressed(out_path, **sc)
            scene_id += 1

        total_scenes += len(day_scenes)
        if day_scenes:
            ships_per_scene = [s['n_ships'] for s in day_scenes]
            print(f"  {pq_path.stem}: {len(seeds)} seeds → {len(day_scenes)} scenes "
                  f"(ships/scene: {np.mean(ships_per_scene):.1f})")

    print(f"\nTotal scenes extracted: {total_scenes}")
    print(f"Output: {scenes_dir}")


if __name__ == '__main__':
    main()