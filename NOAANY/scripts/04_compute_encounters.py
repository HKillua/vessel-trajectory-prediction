"""
04_compute_encounters.py - Compute DCPA/TCPA/CRI and COLREGs encounter classification.
Reads scene NPZ files, adds encounter matrices, writes back.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, haversine_nm, compute_dcpa_tcpa, compute_cri, classify_encounter
)


def compute_bearing(lat1, lon1, lat2, lon2):
    cos_lat = np.cos(np.radians((lat1 + lat2) / 2))
    dx = (lon2 - lon1) * cos_lat
    dy = lat2 - lat1
    return np.degrees(np.arctan2(dx, dy)) % 360.0


def compute_dcpa_tcpa_vectorized(lat1, lon1, sog1, cog1, lat2, lon2, sog2, cog2, tcpa_max=1800):
    """Vectorized DCPA/TCPA over T timesteps. All inputs are 1-D arrays of length T."""
    mean_lat = (lat1 + lat2) / 2.0
    cos_lat = np.cos(np.radians(mean_lat))
    dx = (lon2 - lon1) * 60.0 * cos_lat
    dy = (lat2 - lat1) * 60.0

    cog1_r = np.radians(cog1)
    cog2_r = np.radians(cog2)
    vx1 = sog1 * np.sin(cog1_r)
    vy1 = sog1 * np.cos(cog1_r)
    vx2 = sog2 * np.sin(cog2_r)
    vy2 = sog2 * np.cos(cog2_r)

    dvx = vx2 - vx1
    dvy = vy2 - vy1
    dv2 = dvx * dvx + dvy * dvy

    dv2_safe = np.where(dv2 > 1e-10, dv2, 1.0)
    tcpa_h = np.where(dv2 > 1e-10, -(dx * dvx + dy * dvy) / dv2_safe, 0.0)
    tcpa_sec = np.clip(tcpa_h * 3600, -tcpa_max, tcpa_max)

    cx = dx + dvx * tcpa_h
    cy = dy + dvy * tcpa_h
    dcpa = np.sqrt(cx * cx + cy * cy)
    dcpa = np.minimum(dcpa, 10.0)

    return dcpa.astype(np.float32), tcpa_sec.astype(np.float32)


def compute_cri_vectorized(dcpa_nm, tcpa_sec, d0=1.0, t0=600.0):
    spatial = np.exp(-dcpa_nm / d0)
    temporal = np.where(tcpa_sec >= 0, np.exp(-tcpa_sec / t0), np.exp(tcpa_sec / t0))
    return (spatial * temporal).astype(np.float32)


def process_scene(scene_path, config):
    enc_cfg = config['encounters']
    data = dict(np.load(scene_path, allow_pickle=True))

    trajs = data['trajectories']
    n_ships, n_steps, _ = trajs.shape

    dcpa_mat = np.full((n_ships, n_ships, n_steps), 10.0, dtype=np.float32)
    tcpa_mat = np.zeros((n_ships, n_ships, n_steps), dtype=np.float32)
    cri_mat = np.zeros((n_ships, n_ships, n_steps), dtype=np.float32)
    enc_type = np.zeros((n_ships, n_ships, n_steps), dtype=np.int8)

    tcpa_max = enc_cfg.get('tcpa_max_sec', 1800)
    cri_d0 = enc_cfg.get('cri_d0_nm', 1.0)
    cri_t0 = enc_cfg.get('cri_t0_sec', 600.0)
    dcpa_thr = enc_cfg.get('dcpa_threshold_nm', 1.0)
    head_on_min = enc_cfg.get('head_on_min_diff', 170.0)
    stern_arc = enc_cfg.get('stern_half_arc', 112.5)

    for i in range(n_ships):
        for j in range(i + 1, n_ships):
            lat_i, lon_i, sog_i, cog_i = trajs[i, :, 0], trajs[i, :, 1], trajs[i, :, 2], trajs[i, :, 3]
            lat_j, lon_j, sog_j, cog_j = trajs[j, :, 0], trajs[j, :, 1], trajs[j, :, 2], trajs[j, :, 3]
            hdg_i, hdg_j = trajs[i, :, 4], trajs[j, :, 4]

            dcpa, tcpa = compute_dcpa_tcpa_vectorized(
                lat_i, lon_i, sog_i, cog_i,
                lat_j, lon_j, sog_j, cog_j,
                tcpa_max=tcpa_max
            )
            cri = compute_cri_vectorized(dcpa, tcpa, d0=cri_d0, t0=cri_t0)

            dcpa_mat[i, j] = dcpa_mat[j, i] = dcpa
            tcpa_mat[i, j] = tcpa
            tcpa_mat[j, i] = tcpa
            cri_mat[i, j] = cri_mat[j, i] = cri

            bearing_ij = compute_bearing(lat_i, lon_i, lat_j, lon_j)
            enc_ij, enc_ji = classify_encounter(
                bearing_ij, hdg_i, hdg_j, dcpa,
                dcpa_threshold=dcpa_thr,
                head_on_min_diff=head_on_min,
                stern_half_arc=stern_arc
            )
            enc_type[i, j] = enc_ij
            enc_type[j, i] = enc_ji

    min_dcpa_per_pair = dcpa_mat.min(axis=2)
    max_cri_per_pair = cri_mat.max(axis=2)

    dominant_enc = np.zeros((n_ships, n_ships), dtype=np.int8)
    for i in range(n_ships):
        for j in range(i + 1, n_ships):
            min_t = np.argmin(dcpa_mat[i, j])
            dominant_enc[i, j] = enc_type[i, j, min_t]
            dominant_enc[j, i] = enc_type[j, i, min_t]

    has_encounter = bool(np.any(min_dcpa_per_pair[min_dcpa_per_pair < 10] <
                                enc_cfg.get('dcpa_threshold_nm', 1.0)))

    data['dcpa_matrix'] = dcpa_mat
    data['tcpa_matrix'] = tcpa_mat
    data['cri_matrix'] = cri_mat
    data['encounter_type_temporal'] = enc_type
    data['min_dcpa'] = min_dcpa_per_pair
    data['max_cri'] = max_cri_per_pair
    data['encounter_type'] = dominant_enc
    data['has_encounter'] = has_encounter

    np.savez_compressed(scene_path, **data)
    return has_encounter, min_dcpa_per_pair.min()


def main():
    parser = argparse.ArgumentParser(description='Compute encounters for scenes')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    scenes_dir = base / config['scenes']['scenes_dir']

    scene_files = sorted(scenes_dir.glob('scene_*.npz'))
    print(f"Computing encounters for {len(scene_files)} scenes")

    n_with_enc = 0
    min_dcpas = []

    for sf in tqdm(scene_files, desc="Computing encounters"):
        has_enc, min_d = process_scene(sf, config)
        if has_enc:
            n_with_enc += 1
        min_dcpas.append(min_d)

    print(f"\nScenes with close encounters: {n_with_enc}/{len(scene_files)}")
    if min_dcpas:
        print(f"Min DCPA distribution: "
              f"mean={np.mean(min_dcpas):.2f}nm, "
              f"median={np.median(min_dcpas):.2f}nm, "
              f"min={np.min(min_dcpas):.3f}nm")


if __name__ == '__main__':
    main()
