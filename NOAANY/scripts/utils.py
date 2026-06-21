"""
Utility functions for NOAA NY AIS data processing pipeline.
Designed for 60-second sampling interval and encounter-centric extraction.
"""

import math
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Coordinate / distance helpers
# ---------------------------------------------------------------------------

NM_PER_DEG_LAT = 60.0
R_NM = 3440.065


def haversine_nm(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R_NM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def latlon_to_meters(lat, lon, ref_lat, ref_lon):
    dy = (lat - ref_lat) * 111320.0
    dx = (lon - ref_lon) * 111320.0 * np.cos(np.radians(ref_lat))
    return dx, dy


def angular_diff(a, b):
    d = (a - b) % 360.0
    return np.where(d > 180, d - 360, d)


def angular_interp(angles, kind='linear'):
    """Interpolate angles (degrees) via sin/cos decomposition."""
    rad = np.radians(angles)
    sin_v = np.sin(rad)
    cos_v = np.cos(rad)
    return sin_v, cos_v


def angular_from_sincos(sin_v, cos_v):
    return np.degrees(np.arctan2(sin_v, cos_v)) % 360.0


# ---------------------------------------------------------------------------
# Filtering functions
# ---------------------------------------------------------------------------

def filter_position_jumps(df, max_speed_kn=50, speed_factor=1.5):
    if len(df) < 2:
        return df
    keep = [True]
    last_valid = 0
    for i in range(1, len(df)):
        dt_h = (df.iloc[i]['timestamp'] - df.iloc[last_valid]['timestamp']) / 3600.0
        if dt_h <= 0:
            keep.append(False)
            continue
        dist = haversine_nm(
            df.iloc[last_valid]['lat'], df.iloc[last_valid]['lon'],
            df.iloc[i]['lat'], df.iloc[i]['lon']
        )
        implied_speed = dist / dt_h
        prev_sog = df.iloc[last_valid]['sog']
        relative_threshold = max(prev_sog * speed_factor, 15.0)
        if implied_speed > max_speed_kn and implied_speed > relative_threshold:
            keep.append(False)
        else:
            keep.append(True)
            last_valid = i
    return df[keep].reset_index(drop=True)


def filter_gps_freeze(df, window=4, sog_thr=5.0, ratio_thr=0.1):
    if len(df) < window:
        return df
    frozen = np.zeros(len(df), dtype=bool)
    for i in range(len(df) - window + 1):
        seg = df.iloc[i:i + window]
        mean_sog = seg['sog'].mean()
        if mean_sog < sog_thr:
            continue
        dt_h = (seg.iloc[-1]['timestamp'] - seg.iloc[0]['timestamp']) / 3600.0
        if dt_h <= 0:
            continue
        expected_nm = mean_sog * dt_h
        actual_nm = haversine_nm(
            seg.iloc[0]['lat'], seg.iloc[0]['lon'],
            seg.iloc[-1]['lat'], seg.iloc[-1]['lon']
        )
        if expected_nm > 0 and actual_nm / expected_nm < ratio_thr:
            frozen[i:i + window] = True
    return df[~frozen].reset_index(drop=True)


def remove_stationary_segments(df, sog_thr=0.5, max_duration=300):
    if len(df) < 2:
        return df
    is_moving = df['sog'].values >= sog_thr
    keep = np.ones(len(df), dtype=bool)
    i = 0
    while i < len(df):
        if not is_moving[i]:
            j = i
            while j < len(df) and not is_moving[j]:
                j += 1
            duration = df.iloc[min(j, len(df) - 1)]['timestamp'] - df.iloc[i]['timestamp']
            if duration > max_duration:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return df[keep].reset_index(drop=True)


def split_by_gap(df, gap_threshold=600):
    if len(df) < 2:
        return [df] if len(df) > 0 else []
    ts = df['timestamp'].values
    gaps = np.where(np.diff(ts) > gap_threshold)[0]
    if len(gaps) == 0:
        return [df]
    segments = []
    prev = 0
    for g in gaps:
        seg = df.iloc[prev:g + 1].reset_index(drop=True)
        if len(seg) >= 2:
            segments.append(seg)
        prev = g + 1
    seg = df.iloc[prev:].reset_index(drop=True)
    if len(seg) >= 2:
        segments.append(seg)
    return segments


# ---------------------------------------------------------------------------
# Interpolation and smoothing
# ---------------------------------------------------------------------------

def interpolate_trajectory(df, interval_sec=60):
    if len(df) < 2:
        return None
    ts = df['timestamp'].values
    t_start = ts[0]
    t_end = ts[-1]
    if t_end - t_start < interval_sec:
        return None
    t_new = np.arange(t_start, t_end + 1, interval_sec)
    if len(t_new) < 2:
        return None

    from scipy.interpolate import interp1d
    result = {'timestamp': t_new}
    for col in ['lat', 'lon', 'sog']:
        f = interp1d(ts, df[col].values, kind='linear', fill_value='extrapolate')
        result[col] = f(t_new)

    result['sog'] = np.clip(result['sog'], 0, None)

    for col in ['cog', 'heading']:
        vals = df[col].values
        rad = np.radians(vals)
        f_sin = interp1d(ts, np.sin(rad), kind='linear', fill_value='extrapolate')
        f_cos = interp1d(ts, np.cos(rad), kind='linear', fill_value='extrapolate')
        result[col] = np.degrees(np.arctan2(f_sin(t_new), f_cos(t_new))) % 360.0

    out = pd.DataFrame(result)
    if 'mmsi' in df.columns:
        out['mmsi'] = df['mmsi'].iloc[0]
    if 'vessel_type' in df.columns:
        out['vessel_type'] = df['vessel_type'].iloc[0]
    return out


def smooth_trajectory(df, sog_median_window=3, max_sog_change=9.0, min_sog_for_cog=1.0, max_sog_knots=None):
    df = df.copy()
    sog = df['sog'].values.copy()

    if len(sog) >= sog_median_window:
        from scipy.ndimage import median_filter
        sog = median_filter(sog, size=sog_median_window).astype(np.float64)

    for i in range(1, len(sog)):
        if sog[i] - sog[i - 1] > max_sog_change:
            sog[i] = sog[i - 1] + max_sog_change
        elif sog[i - 1] - sog[i] > max_sog_change:
            sog[i] = sog[i - 1] - max_sog_change
    for i in range(len(sog) - 2, -1, -1):
        if sog[i] - sog[i + 1] > max_sog_change:
            sog[i] = sog[i + 1] + max_sog_change
        elif sog[i + 1] - sog[i] > max_sog_change:
            sog[i] = sog[i + 1] - max_sog_change

    df['sog'] = np.clip(sog, 0, None)

    if max_sog_knots is not None:
        df['sog'] = np.clip(df['sog'], 0, max_sog_knots)

    cog = df['cog'].values.copy()
    last_valid_cog = cog[0]
    for i in range(len(cog)):
        if df['sog'].iloc[i] >= min_sog_for_cog:
            last_valid_cog = cog[i]
        elif df['sog'].iloc[i] < min_sog_for_cog * 0.5:
            cog[i] = last_valid_cog
        else:
            alpha = (df['sog'].iloc[i] - min_sog_for_cog * 0.5) / (min_sog_for_cog * 0.5)
            cog[i] = _blend_angles(last_valid_cog, cog[i], alpha)
    df['cog'] = cog % 360.0
    return df


def _blend_angles(a, b, alpha):
    a_r, b_r = math.radians(a), math.radians(b)
    sin_v = (1 - alpha) * math.sin(a_r) + alpha * math.sin(b_r)
    cos_v = (1 - alpha) * math.cos(a_r) + alpha * math.cos(b_r)
    return math.degrees(math.atan2(sin_v, cos_v)) % 360.0


def align_sog_to_position(df, alpha=0.7):
    """Blend reported SOG with position-implied speed to reduce inconsistency."""
    df = df.copy()
    lats = df['lat'].values
    lons = df['lon'].values
    ts = df['timestamp'].values
    sog = df['sog'].values.copy()
    n = len(df)
    if n < 3:
        return df

    implied = np.zeros(n, dtype=np.float64)
    for i in range(1, n - 1):
        dt_h = (ts[i + 1] - ts[i - 1]) / 3600.0
        if dt_h > 0:
            dist = haversine_nm(lats[i - 1], lons[i - 1], lats[i + 1], lons[i + 1])
            implied[i] = dist / dt_h

    dt0 = (ts[1] - ts[0]) / 3600.0
    if dt0 > 0:
        implied[0] = haversine_nm(lats[0], lons[0], lats[1], lons[1]) / dt0
    dt_last = (ts[-1] - ts[-2]) / 3600.0
    if dt_last > 0:
        implied[-1] = haversine_nm(lats[-2], lons[-2], lats[-1], lons[-1]) / dt_last

    df['sog'] = np.clip(alpha * sog + (1 - alpha) * implied, 0, None)
    return df


def repair_heading_cog(df, max_diff_deg=90, stuck_std_thr=0.5, min_sog_for_check=3.0):
    df = df.copy()
    heading = df['heading'].values.copy()
    cog = df['cog'].values.copy()
    sog = df['sog'].values.copy()

    high_speed = sog >= min_sog_for_check
    if high_speed.sum() > 5:
        hs_heading = heading[high_speed]
        if np.std(hs_heading) < stuck_std_thr and np.mean(sog[high_speed]) > min_sog_for_check:
            heading[high_speed] = cog[high_speed]

    for i in range(len(heading)):
        if sog[i] >= min_sog_for_check:
            diff = abs(angular_diff(heading[i], cog[i]))
            if diff > max_diff_deg:
                heading[i] = cog[i]

    df['heading'] = heading % 360.0
    return df


# ---------------------------------------------------------------------------
# Adjacency / interaction matrices
# ---------------------------------------------------------------------------

def compute_pairwise_distances(trajectories, step_idx):
    n = trajectories.shape[0]
    lats = trajectories[:, step_idx, 0]
    lons = trajectories[:, step_idx, 1]
    lat1 = lats[:, None]
    lat2 = lats[None, :]
    lon1 = lons[:, None]
    lon2 = lons[None, :]
    dist = haversine_nm(lat1, lon1, lat2, lon2)
    np.fill_diagonal(dist, 0.0)
    return dist.astype(np.float32)


def compute_adjacency_matrix(trajectories, step_idx, d0_nm=1.0):
    dist = compute_pairwise_distances(trajectories, step_idx)
    adj = np.exp(-dist / d0_nm)
    np.fill_diagonal(adj, 0)
    return adj.astype(np.float32)


def compute_mean_adjacency(trajectories, threshold_nm=3.0):
    n_ships, n_steps = trajectories.shape[0], trajectories.shape[1]
    d0 = threshold_nm / 3.0
    adj_sum = np.zeros((n_ships, n_ships), dtype=np.float64)
    for t in range(n_steps):
        adj_sum += compute_adjacency_matrix(trajectories, t, d0_nm=d0)
    return (adj_sum / n_steps).astype(np.float32)


# ---------------------------------------------------------------------------
# DCPA / TCPA / CRI
# ---------------------------------------------------------------------------

def compute_dcpa_tcpa(lat1, lon1, sog1, cog1, lat2, lon2, sog2, cog2, tcpa_max=1800):
    mean_lat = (lat1 + lat2) / 2.0
    cos_lat = np.cos(np.radians(mean_lat))
    dx = (lon2 - lon1) * NM_PER_DEG_LAT * cos_lat
    dy = (lat2 - lat1) * NM_PER_DEG_LAT

    cog1_r = np.radians(cog1)
    cog2_r = np.radians(cog2)
    vx1 = sog1 * np.sin(cog1_r)
    vy1 = sog1 * np.cos(cog1_r)
    vx2 = sog2 * np.sin(cog2_r)
    vy2 = sog2 * np.cos(cog2_r)

    dvx = vx2 - vx1
    dvy = vy2 - vy1
    dv2 = dvx * dvx + dvy * dvy

    if isinstance(dv2, np.ndarray):
        tcpa_h = np.where(dv2 > 1e-10, -(dx * dvx + dy * dvy) / dv2, 0.0)
    else:
        tcpa_h = -(dx * dvx + dy * dvy) / dv2 if dv2 > 1e-10 else 0.0

    tcpa_sec = np.clip(tcpa_h * 3600, -tcpa_max, tcpa_max)

    cx = dx + dvx * tcpa_h
    cy = dy + dvy * tcpa_h
    dcpa = np.sqrt(cx * cx + cy * cy)
    dcpa = np.minimum(dcpa, 10.0)

    return dcpa, tcpa_sec


def compute_cri(dcpa_nm, tcpa_sec, d0=1.0, t0=600.0):
    spatial = np.exp(-dcpa_nm / d0)
    temporal = np.where(tcpa_sec >= 0, np.exp(-tcpa_sec / t0), np.exp(tcpa_sec / t0))
    return spatial * temporal


def classify_encounter(bearing_ij, heading_i, heading_j, dcpa_nm,
                       dcpa_threshold=2.0, head_on_min_diff=170.0,
                       stern_half_arc=112.5):
    """
    Classify encounter type per COLREGs. Supports scalar or array inputs.
    Returns: enc_ij, enc_ji
      0=safe, 1=head_on, 2=crossing_give_way,
      3=crossing_stand_on, 4=overtaking, 5=being_overtaken
    """
    scalar = not isinstance(dcpa_nm, np.ndarray)
    if scalar:
        bearing_ij = np.atleast_1d(np.float64(bearing_ij))
        heading_i = np.atleast_1d(np.float64(heading_i))
        heading_j = np.atleast_1d(np.float64(heading_j))
        dcpa_nm = np.atleast_1d(np.float64(dcpa_nm))

    enc_ij = np.zeros_like(dcpa_nm, dtype=np.int8)
    enc_ji = np.zeros_like(dcpa_nm, dtype=np.int8)

    close = dcpa_nm <= dcpa_threshold
    if not np.any(close):
        if scalar:
            return int(enc_ij[0]), int(enc_ji[0])
        return enc_ij, enc_ji

    rel_bearing_i = (bearing_ij - heading_i) % 360
    heading_diff = np.abs(angular_diff(heading_i, heading_j))

    head_on = close & (heading_diff >= head_on_min_diff)
    enc_ij[head_on] = 1
    enc_ji[head_on] = 1

    remaining = close & ~head_on

    in_forward_arc_i = (rel_bearing_i > (360 - stern_half_arc)) | (rel_bearing_i < stern_half_arc)

    bearing_ji = (bearing_ij + 180) % 360
    rel_bearing_j = (bearing_ji - heading_j) % 360
    in_forward_arc_j = (rel_bearing_j > (360 - stern_half_arc)) | (rel_bearing_j < stern_half_arc)

    # j in i's stern AND j sees i ahead → i being overtaken by j
    in_stern_i = remaining & ~in_forward_arc_i
    ot_case1 = in_stern_i & in_forward_arc_j
    enc_ij[ot_case1] = 5
    enc_ji[ot_case1] = 4

    # i in j's stern AND i sees j ahead → j being overtaken by i
    in_stern_j = remaining & ~in_forward_arc_j
    ot_case2 = in_stern_j & in_forward_arc_i & ~ot_case1
    enc_ij[ot_case2] = 4
    enc_ji[ot_case2] = 5

    crossing = remaining & ~ot_case1 & ~ot_case2 & in_forward_arc_i
    starboard = crossing & (rel_bearing_i > 0) & (rel_bearing_i < 180)
    port = crossing & ~starboard
    enc_ij[starboard] = 2
    enc_ji[starboard] = 3
    enc_ij[port] = 3
    enc_ji[port] = 2

    if scalar:
        return int(enc_ij[0]), int(enc_ji[0])
    return enc_ij, enc_ji


# ---------------------------------------------------------------------------
# Vessel type mapping
# ---------------------------------------------------------------------------

VESSEL_TYPE_MAP = {
    range(20, 30): 'WIG',
    range(30, 40): 'Fishing/Towing/Dredging',
    range(40, 50): 'HSC',
    range(50, 60): 'Special',
    range(60, 70): 'Passenger',
    range(70, 80): 'Cargo',
    range(80, 90): 'Tanker',
    range(90, 100): 'Other',
}


def vessel_type_name(code):
    try:
        code = int(code)
    except (ValueError, TypeError):
        return 'Unknown'
    for rng, name in VESSEL_TYPE_MAP.items():
        if code in rng:
            return name
    return 'Unknown'


# ---------------------------------------------------------------------------
# Scene-level trajectory alignment
# ---------------------------------------------------------------------------

def align_trajectories_to_grid(ship_dfs, t_start, t_end, interval_sec=60, min_coverage=0.85):
    """
    Align multiple ship trajectories to a common time grid.
    Uses edge-value clamping for position/angles and SOG decay for
    timesteps outside a ship's actual data range.
    Returns: (trajectories [N, T, 5], t_grid [T], valid_ship_indices [N])
    """
    t_grid = np.arange(t_start, t_end + 1, interval_sec)
    n_steps = len(t_grid)
    if n_steps < 2:
        return None, None, []

    from scipy.interpolate import interp1d

    aligned = []
    valid_ships = []
    for idx, df in enumerate(ship_dfs):
        ts = df['timestamp'].values
        mask = (ts >= t_start) & (ts <= t_end)
        seg = df[mask]
        if len(seg) < 2:
            continue
        coverage = (seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]) / max(t_end - t_start, 1)
        if coverage < min_coverage:
            continue

        ts_seg = seg['timestamp'].values
        row = np.zeros((n_steps, 5), dtype=np.float32)

        for ci, col in enumerate(['lat', 'lon', 'sog']):
            vals = seg[col].values
            edge = (float(vals[0]), float(vals[-1]))
            f = interp1d(ts_seg, vals, kind='linear',
                         bounds_error=False, fill_value=edge)
            row[:, ci] = f(t_grid)
        row[:, 2] = np.clip(row[:, 2], 0, None)

        for ci, col in zip([3, 4], ['cog', 'heading']):
            vals = seg[col].values
            rad = np.radians(vals)
            sin_v = np.sin(rad)
            cos_v = np.cos(rad)
            f_s = interp1d(ts_seg, sin_v, kind='linear',
                           bounds_error=False,
                           fill_value=(float(sin_v[0]), float(sin_v[-1])))
            f_c = interp1d(ts_seg, cos_v, kind='linear',
                           bounds_error=False,
                           fill_value=(float(cos_v[0]), float(cos_v[-1])))
            row[:, ci] = np.degrees(np.arctan2(f_s(t_grid), f_c(t_grid))) % 360.0

        for i in range(n_steps):
            if t_grid[i] < ts_seg[0] or t_grid[i] > ts_seg[-1]:
                row[i, 2] = 0.0

        aligned.append(row)
        valid_ships.append(idx)

    if len(aligned) < 2:
        return None, None, []
    return np.stack(aligned, axis=0), t_grid, valid_ships


def post_quality_check(trajectories, max_sog=35, max_sog_change=9.0,
                       timestamps=None, sog_align_alpha=0.7):
    """
    Post-alignment quality check on [N_ships, T, 5] array.
    Clamps SOG spikes, then re-aligns SOG with position-implied speed.
    Returns cleaned array or None if unfixable.
    """
    n_ships, n_steps, _ = trajectories.shape
    result = trajectories.copy()

    for s in range(n_ships):
        sog = result[s, :, 2].copy()
        for i in range(1, n_steps):
            if sog[i] - sog[i - 1] > max_sog_change:
                sog[i] = sog[i - 1] + max_sog_change
            elif sog[i - 1] - sog[i] > max_sog_change:
                sog[i] = sog[i - 1] - max_sog_change
        for i in range(n_steps - 2, -1, -1):
            if sog[i] - sog[i + 1] > max_sog_change:
                sog[i] = sog[i + 1] + max_sog_change
            elif sog[i + 1] - sog[i] > max_sog_change:
                sog[i] = sog[i + 1] - max_sog_change
        result[s, :, 2] = np.clip(sog, 0, max_sog)

    if timestamps is not None and n_steps >= 3:
        for s in range(n_ships):
            lats = result[s, :, 0]
            lons = result[s, :, 1]
            sog = result[s, :, 2].copy()
            implied = np.zeros(n_steps, dtype=np.float64)
            for i in range(1, n_steps - 1):
                dt_h = (timestamps[i + 1] - timestamps[i - 1]) / 3600.0
                if dt_h > 0:
                    d = haversine_nm(lats[i - 1], lons[i - 1],
                                     lats[i + 1], lons[i + 1])
                    implied[i] = d / dt_h
            dt0 = (timestamps[1] - timestamps[0]) / 3600.0
            if dt0 > 0:
                implied[0] = haversine_nm(lats[0], lons[0], lats[1], lons[1]) / dt0
            dt_last = (timestamps[-1] - timestamps[-2]) / 3600.0
            if dt_last > 0:
                implied[-1] = haversine_nm(lats[-2], lons[-2],
                                           lats[-1], lons[-1]) / dt_last
            result[s, :, 2] = np.clip(
                sog_align_alpha * sog + (1 - sog_align_alpha) * implied,
                0, max_sog)

        for s in range(n_ships):
            sog = result[s, :, 2]
            for i in range(1, n_steps):
                if sog[i] - sog[i - 1] > max_sog_change:
                    sog[i] = sog[i - 1] + max_sog_change
                elif sog[i - 1] - sog[i] > max_sog_change:
                    sog[i] = sog[i - 1] - max_sog_change
            for i in range(n_steps - 2, -1, -1):
                if sog[i] - sog[i + 1] > max_sog_change:
                    sog[i] = sog[i + 1] + max_sog_change
                elif sog[i + 1] - sog[i] > max_sog_change:
                    sog[i] = sog[i + 1] - max_sog_change
            result[s, :, 2] = np.clip(sog, 0, max_sog)

    return result