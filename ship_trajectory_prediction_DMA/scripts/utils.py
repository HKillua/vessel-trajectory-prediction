"""共享工具函数 - 多船交互轨迹预测数据处理"""

import os
import yaml
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from geopy.distance import geodesic
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def get_data_path(stage, dataset_name=""):
    path = PROJECT_ROOT / "data" / stage / dataset_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def haversine_distance_nm(lat1, lon1, lat2, lon2):
    """计算两点间的海里距离"""
    R = 3440.065  # 地球半径（海里）
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def haversine_distance_km(lat1, lon1, lat2, lon2):
    """计算两点间的公里距离"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def interpolate_trajectory(df, interval_sec=20):
    """将不均匀采样的轨迹重采样为固定间隔

    Args:
        df: DataFrame with columns [timestamp, lat, lon, sog, cog, heading]
        interval_sec: 目标采样间隔（秒）

    Returns:
        重采样后的 DataFrame
    """
    if len(df) < 2:
        return None

    df = df.sort_values("timestamp").reset_index(drop=True)
    t = df["timestamp"].values.astype(float)
    t_start = t[0]
    t_end = t[-1]

    t_new = np.arange(t_start, t_end + 1, interval_sec)
    if len(t_new) < 2:
        return None

    result = {"timestamp": t_new}

    for col in ["lat", "lon", "sog"]:
        if col in df.columns:
            vals = df[col].values
            f = interp1d(t, vals, kind="linear", bounds_error=False,
                         fill_value=(vals[0], vals[-1]))
            result[col] = f(t_new)

    # COG 和 heading 需要角度插值（处理 0/360 跨越）
    for col in ["cog", "heading"]:
        if col in df.columns:
            angles = df[col].values.copy()
            # 处理残留 NaN：用 COG 填充 heading，或用前后值填充
            nan_mask = np.isnan(angles)
            if nan_mask.any():
                if col == "heading" and "cog" in df.columns:
                    angles[nan_mask] = df["cog"].values[nan_mask]                
                # 如果仍有 NaN（COG 也是 NaN），用前向填充
                nan_mask = np.isnan(angles)
                if nan_mask.any():
                    for i in range(len(angles)):
                        if nan_mask[i] and i > 0:
                            angles[i] = angles[i-1]
                    # 后向填充剩余开头的 NaN
                    for i in range(len(angles)-1, -1, -1):
                        if np.isnan(angles[i]) and i < len(angles)-1:
                            angles[i] = angles[i+1]
                    # 极端情况：全 NaN -> 填 0
                    if np.isnan(angles).all():
                        angles[:] = 0.0
            sin_vals = np.sin(np.radians(angles))
            cos_vals = np.cos(np.radians(angles))
            f_sin = interp1d(t, sin_vals, kind="linear", bounds_error=False,
                             fill_value=(sin_vals[0], sin_vals[-1]))
            f_cos = interp1d(t, cos_vals, kind="linear", bounds_error=False,
                             fill_value=(cos_vals[0], cos_vals[-1]))            
            interp_angles = np.degrees(np.arctan2(f_sin(t_new), f_cos(t_new))) % 360
            result[col] = interp_angles

    out = pd.DataFrame(result)
    # 插值后安全校验
    if "sog" in out.columns:
        out["sog"] = out["sog"].clip(lower=0.0)
    return out


def smooth_trajectory(df, config):
    """重采样后的轨迹平滑：SOG 中值滤波 + SOG 变化率剪切 + 低速 COG 渐进过渡填充"""
    prep = config["preprocessing"]
    median_window = prep.get("sog_median_window", 0)
    max_sog_change = prep.get("max_sog_change_per_step", 0)
    min_sog_for_cog = prep.get("min_sog_for_cog", 0)

    if len(df) < 3:
        return df

    df = df.copy()

    # 1. SOG 中值滤波
    if median_window > 0 and "sog" in df.columns:
        sog_vals = df["sog"].values.copy()
        half_w = median_window // 2
        smoothed = sog_vals.copy()
        for i in range(len(sog_vals)):
            start = max(0, i - half_w)
            end = min(len(sog_vals), i + half_w + 1)
            smoothed[i] = np.median(sog_vals[start:end])
        df["sog"] = smoothed

    # 2. SOG 变化率剪切（双向：前向+后向，取较保守值）
    if max_sog_change > 0 and "sog" in df.columns:
        sog_raw = df["sog"].values.copy()
        # 前向限幅
        fwd = sog_raw.copy()
        for i in range(1, len(fwd)):
            delta = fwd[i] - fwd[i-1]
            if abs(delta) > max_sog_change:
                fwd[i] = fwd[i-1] + np.sign(delta) * max_sog_change
        # 后向限幅
        bwd = sog_raw.copy()
        for i in range(len(bwd) - 2, -1, -1):
            delta = bwd[i] - bwd[i+1]
            if abs(delta) > max_sog_change:
                bwd[i] = bwd[i+1] + np.sign(delta) * max_sog_change
        # 取两个方向中更接近原始值的结果（较保守的限幅）
        sog_vals = np.where(np.abs(fwd - sog_raw) < np.abs(bwd - sog_raw), fwd, bwd)
        df["sog"] = np.clip(sog_vals, 0.0, None)

    # 3. 低速时 COG 渐进过渡填充
    if min_sog_for_cog > 0 and "sog" in df.columns and "cog" in df.columns:
        sog_vals = df["sog"].values
        cog_vals = df["cog"].values.copy()
        transition_upper = min_sog_for_cog * 2.0
        last_valid_cog = cog_vals[0]
        for i in range(len(sog_vals)):
            if sog_vals[i] >= transition_upper:
                last_valid_cog = cog_vals[i]
            elif sog_vals[i] < min_sog_for_cog:
                cog_vals[i] = last_valid_cog
            else:
                alpha = (sog_vals[i] - min_sog_for_cog) / (transition_upper - min_sog_for_cog)
                cog_vals[i] = _blend_angles(last_valid_cog, cog_vals[i], alpha)
        df["cog"] = cog_vals

    return df


def _blend_angles(a, b, alpha):
    """通过 sin/cos 加权平均混合两个角度，避免 0/360 断裂"""
    a_rad = np.radians(a)
    b_rad = np.radians(b)
    sin_blend = (1 - alpha) * np.sin(a_rad) + alpha * np.sin(b_rad)
    cos_blend = (1 - alpha) * np.cos(a_rad) + alpha * np.cos(b_rad)
    return np.degrees(np.arctan2(sin_blend, cos_blend)) % 360


def split_trajectory_by_gap(df, gap_threshold_sec=600):
    """按时间间隔切割轨迹段

    Args:
        df: 按时间排序的 DataFrame，需包含 timestamp 列
        gap_threshold_sec: 间隔阈值（秒）

    Returns:
        list of DataFrames
    """
    if len(df) < 2:
        return [df]

    df = df.sort_values("timestamp").reset_index(drop=True)
    time_diffs = df["timestamp"].diff()
    split_indices = time_diffs[time_diffs > gap_threshold_sec].index.tolist()

    segments = []
    prev_idx = 0
    for idx in split_indices:
        seg = df.iloc[prev_idx:idx].reset_index(drop=True)
        if len(seg) >= 2:
            segments.append(seg)
        prev_idx = idx

    last_seg = df.iloc[prev_idx:].reset_index(drop=True)
    if len(last_seg) >= 2:
        segments.append(last_seg)

    return segments


def filter_trajectory(df, config):
    """应用基本过滤规则"""
    prep = config["preprocessing"]

    # 坐标范围过滤
    mask = (
        (df["lat"] >= prep["lat_range"][0]) & (df["lat"] <= prep["lat_range"][1]) &
        (df["lon"] >= prep["lon_range"][0]) & (df["lon"] <= prep["lon_range"][1])
    )
    df = df[mask].copy()

    # 速度过滤（去除异常高速点）
    if "sog" in df.columns:
        df = df[df["sog"] <= prep["max_sog_knots"]].copy()

    return df


def filter_gps_freeze(df, config):
    """移除 GPS 冻结片段：SOG 正常报告但位置不更新
    
    策略：滑动窗口（默认 10 步 = 200s）检测。若窗口内平均 SOG > sog_thr
    但实际位移 < 预期位移的 ratio_thr（默认 10%），则标记窗口内所有点为冻结。
    """
    if len(df) < 5:
        return df

    prep = config["preprocessing"]
    window = prep.get("gps_freeze_window", 10)
    sog_thr = prep.get("gps_freeze_sog_thr", 3.0)
    ratio_thr = prep.get("gps_freeze_ratio_thr", 0.1)

    df = df.sort_values("timestamp").reset_index(drop=True)
    frozen_mask = np.zeros(len(df), dtype=bool)

    lats = df["lat"].values
    lons = df["lon"].values
    sogs = df["sog"].values
    ts = df["timestamp"].values.astype(float)

    for start in range(len(df) - window):
        end = start + window
        mean_sog = sogs[start:end].mean()
        if mean_sog < sog_thr:
            continue

        dt_sec = ts[end - 1] - ts[start]
        if dt_sec <= 0:
            continue

        expected_nm = mean_sog * (dt_sec / 3600.0)
        actual_nm = haversine_distance_nm(lats[start], lons[start], lats[end - 1], lons[end - 1])

        if expected_nm > 0.05 and actual_nm < expected_nm * ratio_thr:
            frozen_mask[start:end] = True

    removed = frozen_mask.sum()
    if removed > 0:
        df = df[~frozen_mask].reset_index(drop=True)

    return df


def filter_position_jumps(df, config):
    """移除物理上不可能的位置跳变点
    
    策略：从第一个点开始，维护“最后有效点”。对每个后续点，计算其与最后有效点之间的
    隐含速度（位移/时间差）。若隐含速度 > max_sog * jump_speed_factor，则标记为异常并跳过。
    否则更新“最后有效点”。这能正确处理连续漂移常数和“跳去又跳回”的模式。
    """
    if len(df) < 3:
        return df

    prep = config["preprocessing"]
    max_sog = prep["max_sog_knots"]
    factor = prep.get("jump_speed_factor", 3.0)
    speed_threshold_knots = max_sog * factor

    df = df.sort_values("timestamp").reset_index(drop=True)
    keep_mask = np.ones(len(df), dtype=bool)

    last_valid_idx = 0
    for i in range(1, len(df)):
        dt_sec = df.iloc[i]["timestamp"] - df.iloc[last_valid_idx]["timestamp"]
        if dt_sec <= 0:
            keep_mask[i] = False
            continue

        dist_nm = haversine_distance_nm(
            df.iloc[last_valid_idx]["lat"], df.iloc[last_valid_idx]["lon"],
            df.iloc[i]["lat"], df.iloc[i]["lon"]
        )
        implied_speed_knots = dist_nm / (dt_sec / 3600.0)

        if implied_speed_knots > speed_threshold_knots:
            keep_mask[i] = False
        else:
            last_valid_idx = i

    removed = (~keep_mask).sum()
    if removed > 0:
        df = df[keep_mask].reset_index(drop=True)

    return df


def remove_stationary_segments(df, config):
    """去除长时间静止的轨迹段（基于实际经过时间而非行数）"""
    prep = config["preprocessing"]
    min_sog = prep["min_sog_knots"]
    max_stationary_sec = prep.get("stationary_duration_min", 30) * 60

    if "sog" not in df.columns or "timestamp" not in df.columns or len(df) == 0:
        return df

    sog_vals = df["sog"].values
    ts_vals = df["timestamp"].values.astype(float)
    is_moving = sog_vals >= min_sog

    segments_to_keep = []
    stationary_start = None
    start_idx = 0

    for i in range(len(df)):
        if not is_moving[i]:
            if stationary_start is None:
                stationary_start = i
        else:
            if stationary_start is not None:
                elapsed = ts_vals[i - 1] - ts_vals[stationary_start]
                if elapsed > max_stationary_sec:
                    if start_idx < stationary_start:
                        segments_to_keep.append(df.iloc[start_idx:stationary_start])
                    start_idx = i
                stationary_start = None

    if stationary_start is not None:
        elapsed = ts_vals[len(df) - 1] - ts_vals[stationary_start]
        if elapsed > max_stationary_sec:
            if start_idx < stationary_start:
                segments_to_keep.append(df.iloc[start_idx:stationary_start])
        else:
            segments_to_keep.append(df.iloc[start_idx:])
    else:
        segments_to_keep.append(df.iloc[start_idx:])

    if segments_to_keep:
        return pd.concat(segments_to_keep, ignore_index=True)
    return pd.DataFrame(columns=df.columns)


def compute_adjacency_matrix(trajectories, step_idx, d0_nm=1.0):
    """计算某一时刻的连续权重邻接矩阵

    Args:
        trajectories: np.array [N_ships, T_steps, 5] (lat, lon, sog, cog, heading)
        step_idx: 时间步索引
        d0_nm: 衰减距离参数 (海里), W = exp(-d/d0)

    Returns:
        np.array [N_ships, N_ships] 连续权重邻接矩阵
    """
    n_ships = trajectories.shape[0]
    adj = np.zeros((n_ships, n_ships), dtype=np.float32)

    for i in range(n_ships):
        for j in range(i + 1, n_ships):
            lat_i, lon_i = trajectories[i, step_idx, 0], trajectories[i, step_idx, 1]
            lat_j, lon_j = trajectories[j, step_idx, 0], trajectories[j, step_idx, 1]
            dist = haversine_distance_nm(lat_i, lon_i, lat_j, lon_j)
            w = np.exp(-dist / d0_nm)
            adj[i, j] = w
            adj[j, i] = w

    return adj


def compute_mean_adjacency(trajectories, threshold_nm=3.0):
    """计算整个观测期间的时间平均连续权重邻接矩阵"""
    d0_nm = threshold_nm / 3.0
    n_steps = trajectories.shape[1]
    adj_sum = np.zeros((trajectories.shape[0], trajectories.shape[0]), dtype=np.float32)

    for t in range(n_steps):
        adj_sum += compute_adjacency_matrix(trajectories, t, d0_nm)

    return adj_sum / n_steps


# =======================================================================
# 场景插值后质量后处理（解决 SOG/COG/位置 跳变穿透问题）
# =======================================================================

def post_interpolation_quality_pass(aligned_trajs, config, dt=20.0):
    """对场景插值后的对齐轨迹做质量后处理

    Args:
        aligned_trajs: np.array [N_ships, T_steps, 5] (lat, lon, sog, cog, heading)
        config: 配置字典
        dt: 步长（秒）

    Returns:
        cleaned_trajs: 清洗后的轨迹，同 shape；若不可修复返回 None
    """
    trajs = aligned_trajs.copy()
    prep = config["preprocessing"]
    max_sog_change = prep.get("max_sog_change_per_step", 3.0)
    min_sog_for_cog = prep.get("min_sog_for_cog", 1.0)
    max_sog_knots = prep.get("max_sog_knots", 30.0)
    max_cog_change_deg = 15.0

    n_ships, n_steps, _ = trajs.shape

    for s in range(n_ships):
        # --- 1. SOG 变化率裁剪 ---
        sog = trajs[s, :, 2].copy()
        for t in range(1, n_steps):
            delta = sog[t] - sog[t - 1]
            if abs(delta) > max_sog_change:
                sog[t] = sog[t - 1] + np.sign(delta) * max_sog_change
        for t in range(n_steps - 2, -1, -1):
            delta = sog[t] - sog[t + 1]
            if abs(delta) > max_sog_change:
                sog[t] = sog[t + 1] + np.sign(delta) * max_sog_change
        trajs[s, :, 2] = np.clip(sog, 0.0, max_sog_knots)

        # --- 2. 位置-SOG 一致性修复 ---
        # 2a. 整船 GPS 冻结检测（后备检查, 主防线在 04_preprocess 的 filter_gps_freeze）
        mean_sog = trajs[s, :, 2].mean()
        if mean_sog > 3.0:
            cumul_dist = 0.0
            for t in range(1, n_steps):
                cumul_dist += haversine_distance_nm(
                    trajs[s, t - 1, 0], trajs[s, t - 1, 1],
                    trajs[s, t, 0], trajs[s, t, 1])
            expected_dist = mean_sog * (n_steps * dt / 3600.0)
            if expected_dist > 0.1 and cumul_dist < expected_dist * 0.1:
                return None

        # 2b. 逐步位移过快修正
        for t in range(1, n_steps):
            lat1, lon1 = trajs[s, t - 1, 0], trajs[s, t - 1, 1]
            lat2, lon2 = trajs[s, t, 0], trajs[s, t, 1]
            dist_nm = haversine_distance_nm(lat1, lon1, lat2, lon2)
            implied_speed = dist_nm / (dt / 3600.0)
            if implied_speed > max_sog_knots:
                # 用前一个有效点的 SOG/COG 外推位置
                sog_kn = trajs[s, t - 1, 2]
                cog_rad = np.radians(trajs[s, t - 1, 3])
                speed_nm_per_sec = sog_kn / 3600.0
                dx_nm = speed_nm_per_sec * dt * np.sin(cog_rad)
                dy_nm = speed_nm_per_sec * dt * np.cos(cog_rad)
                trajs[s, t, 0] = lat1 + dy_nm / 60.0
                trajs[s, t, 1] = lon1 + dx_nm / (60.0 * np.cos(np.radians(lat1)))

        # --- 3. COG 变化率裁剪（高速时） ---
        sog_vals = trajs[s, :, 2]
        cog_vals = trajs[s, :, 3].copy()
        for t in range(1, n_steps):
            if sog_vals[t] >= min_sog_for_cog and sog_vals[t - 1] >= min_sog_for_cog:
                diff = (cog_vals[t] - cog_vals[t - 1] + 180) % 360 - 180
                if abs(diff) > max_cog_change_deg:
                    cog_vals[t] = (cog_vals[t - 1] + np.sign(diff) * max_cog_change_deg) % 360
            elif sog_vals[t] < min_sog_for_cog:
                cog_vals[t] = cog_vals[t - 1]
        trajs[s, :, 3] = cog_vals

        # Heading 也做同样的限幅
        hdg_vals = trajs[s, :, 4].copy()
        for t in range(1, n_steps):
            if sog_vals[t] >= min_sog_for_cog and sog_vals[t - 1] >= min_sog_for_cog:
                diff = (hdg_vals[t] - hdg_vals[t - 1] + 180) % 360 - 180
                if abs(diff) > max_cog_change_deg:
                    hdg_vals[t] = (hdg_vals[t - 1] + np.sign(diff) * max_cog_change_deg) % 360
            elif sog_vals[t] < min_sog_for_cog:
                hdg_vals[t] = hdg_vals[t - 1]
        trajs[s, :, 4] = hdg_vals

        # --- 4. Heading-COG 一致性修复 ---
        heading_cog_max_diff = prep.get("heading_cog_max_diff_deg", 90.0)
        stuck_heading_thr = prep.get("stuck_heading_std_threshold", 0.5)

        sog_vals = trajs[s, :, 2]
        cog_vals = trajs[s, :, 3]
        hdg_vals = trajs[s, :, 4].copy()

        # 4a. 高速时 heading-COG 差 > 阈值 -> 用 COG 替代 heading
        for t in range(n_steps):
            if sog_vals[t] >= min_sog_for_cog:
                diff = abs((hdg_vals[t] - cog_vals[t] + 180) % 360 - 180)
                if diff > heading_cog_max_diff:
                    hdg_vals[t] = cog_vals[t]

        # 4b. 整段 heading 冻结（std < 阈值）且高速 -> 全部用 COG 替代
        if sog_vals.mean() > 3.0:
            sin_h = np.sin(np.radians(hdg_vals))
            cos_h = np.cos(np.radians(hdg_vals))
            mean_angle = np.degrees(np.arctan2(sin_h.mean(), cos_h.mean())) % 360
            angle_diffs = np.abs((hdg_vals - mean_angle + 180) % 360 - 180)
            if angle_diffs.std() < stuck_heading_thr:
                hdg_vals = cog_vals.copy()

        trajs[s, :, 4] = hdg_vals

    if np.isnan(trajs).any() or np.isinf(trajs).any():
        return None

    return trajs