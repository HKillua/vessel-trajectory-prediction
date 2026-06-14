"""提取多船交互场景

从预处理后的统一格式数据中，提取多船共存的交互场景。
每个场景包含多艘船在同一时空窗口内的完整轨迹。
"""

import sys
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import (
    load_config, get_data_path,
    haversine_distance_km, haversine_distance_nm,
    compute_mean_adjacency, post_interpolation_quality_pass
)


def build_spatial_temporal_index(df, spatial_km, temporal_sec):
    """构建时空索引：将轨迹点分配到网格中"""
    # 空间网格大小（度数近似）
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(df["lat"].mean()))
    grid_lat = spatial_km / km_per_deg_lat
    grid_lon = spatial_km / km_per_deg_lon

    df = df.copy()
    df["grid_lat"] = (df["lat"] / grid_lat).astype(int)
    df["grid_lon"] = (df["lon"] / grid_lon).astype(int)
    df["grid_time"] = (df["timestamp"] / temporal_sec).astype(int)

    return df


def find_interaction_scenes(df, config):
    """找出所有多船交互场景

    策略：取并集收集所有出现过的船舶，然后在 extract_scene_trajectories
    中按实际共存时间筛选。这样避免交集操作把多船场景缩为 2 船。
    """
    scene_config = config["scene_extraction"]
    spatial_km = scene_config["spatial_window_km"]
    temporal_min = scene_config["temporal_window_min"]
    temporal_sec = temporal_min * 60
    min_ships = scene_config["min_ships"]
    max_ships = scene_config["max_ships"]
    min_coexist_sec = scene_config["min_coexist_min"] * 60

    print(f"  参数: 空间窗口={spatial_km}km, 时间窗口={temporal_min}min, 最少{min_ships}船")

    df_indexed = build_spatial_temporal_index(df, spatial_km, temporal_sec)
    cell_ships = df_indexed.groupby(["grid_lat", "grid_lon", "grid_time"])["mmsi"].apply(set)
    multi_ship_cells = cell_ships[cell_ships.apply(len) >= min_ships]
    print(f"  多船时空格子数: {len(multi_ship_cells)}")

    if len(multi_ship_cells) == 0:
        return []

    scenes = []
    processed_keys = set()

    for (glat, glon, gtime), ships in tqdm(multi_ship_cells.items(), desc="构建场景"):
        key = (glat, glon, gtime)
        if key in processed_keys:
            continue

        scene_ships = set(ships)
        time_range = [gtime, gtime]

        # 前向扩展
        for dt in range(1, 100):
            next_key = (glat, glon, gtime + dt)
            if next_key in multi_ship_cells.index:
                next_ships = multi_ship_cells[next_key]
                scene_ships |= next_ships  # 取并集
                time_range[1] = gtime + dt
                processed_keys.add(next_key)
            else:
                break

        # 后向扩展
        for dt in range(1, 100):
            prev_key = (glat, glon, gtime - dt)
            if prev_key in multi_ship_cells.index:
                prev_ships = multi_ship_cells[prev_key]
                scene_ships |= prev_ships
                time_range[0] = gtime - dt
                processed_keys.add(prev_key)
            else:
                break

        processed_keys.add(key)

        duration_sec = (time_range[1] - time_range[0] + 1) * temporal_sec
        if duration_sec < min_coexist_sec:
            continue

        if len(scene_ships) > max_ships:
            center_lat = (glat + 0.5) * (spatial_km / 111.0)
            center_lon = (glon + 0.5) * (spatial_km / (111.0 * np.cos(np.radians(center_lat))))
            t_start = time_range[0] * temporal_sec
            t_end = (time_range[1] + 1) * temporal_sec
            ship_dists = []
            for mmsi in scene_ships:
                ship_pts = df_indexed[(df_indexed["mmsi"] == mmsi) &
                                      (df_indexed["timestamp"] >= t_start) &
                                      (df_indexed["timestamp"] <= t_end)]
                if len(ship_pts) == 0:
                    ship_dists.append((mmsi, float("inf")))
                    continue
                mean_lat = ship_pts["lat"].mean()
                mean_lon = ship_pts["lon"].mean()
                dist = haversine_distance_km(center_lat, center_lon, mean_lat, mean_lon)
                ship_dists.append((mmsi, dist))
            ship_dists.sort(key=lambda x: x[1])
            scene_ships = {mmsi for mmsi, _ in ship_dists[:max_ships]}

        scenes.append({
            "ships": list(scene_ships),
            "grid_lat": glat,
            "grid_lon": glon,
            "time_start": time_range[0] * temporal_sec,
            "time_end": (time_range[1] + 1) * temporal_sec,
            "n_ships": len(scene_ships),
        })

    print(f"  提取到 {len(scenes)} 个候选场景")
    return scenes


def extract_scene_trajectories(df, scene_info, config):
    """为一个场景提取所有船舶的对齐轨迹

    由于场景中的船舶来自时间并集，这里找出共存时间最长的船舶子集，
    确保至少 min_ships 艘船在同一时间段内共存。
    """
    sampling_interval = config["sampling_interval"]
    ships = scene_info["ships"]
    t_start = scene_info["time_start"]
    t_end = scene_info["time_end"]
    min_ships = config["scene_extraction"]["min_ships"]
    min_coexist_sec = config["scene_extraction"]["min_coexist_min"] * 60
    min_steps = config["preprocessing"]["min_trajectory_steps"]

    ship_time_ranges = []
    ship_data_map = {}
    for mmsi in ships:
        ship_data = df[(df["mmsi"] == mmsi) &
                       (df["timestamp"] >= t_start) &
                       (df["timestamp"] <= t_end)]
        if len(ship_data) < 10:
            continue
        ship_data = ship_data.sort_values("timestamp").reset_index(drop=True)
        ts_min = ship_data["timestamp"].iloc[0]
        ts_max = ship_data["timestamp"].iloc[-1]
        if ts_max - ts_min < min_coexist_sec:
            continue
        ship_time_ranges.append((mmsi, ts_min, ts_max))
        ship_data_map[mmsi] = ship_data

    if len(ship_time_ranges) < min_ships:
        return None

    ship_time_ranges.sort(key=lambda x: x[1])

    best_set = None
    best_duration = 0

    for i in range(len(ship_time_ranges)):
        anchor_mmsi, anchor_start, anchor_end = ship_time_ranges[i]
        coexist = [(anchor_mmsi, anchor_start, anchor_end)]

        for j in range(len(ship_time_ranges)):
            if i == j:
                continue
            other_mmsi, other_start, other_end = ship_time_ranges[j]
            overlap_start = max(anchor_start, other_start)
            overlap_end = min(anchor_end, other_end)
            if overlap_end - overlap_start >= min_coexist_sec:
                coexist.append((other_mmsi, other_start, other_end))

        if len(coexist) >= min_ships:
            common_start = max(r[1] for r in coexist)
            common_end = min(r[2] for r in coexist)
            duration = common_end - common_start
            if duration > best_duration:
                best_duration = duration
                best_set = coexist

    if best_set is None or best_duration < min_coexist_sec:
        return None

    common_start = max(r[1] for r in best_set)
    common_end = min(r[2] for r in best_set)
    t_unified = np.arange(common_start, common_end + 1, sampling_interval)
    if len(t_unified) < min_steps:
        return None

    selected_ships = [r[0] for r in best_set]

    # 数据覆盖率检查：拒绝实际数据不覆盖公共窗口大部分时段的船舶
    min_coverage = config["scene_extraction"].get("min_coverage_ratio", 0.8)
    common_duration = common_end - common_start
    if common_duration > 0 and min_coverage > 0:
        coverage_mask = []
        for mmsi, ts_min, ts_max in best_set:
            actual_start = max(ts_min, common_start)
            actual_end = min(ts_max, common_end)
            coverage = (actual_end - actual_start) / common_duration
            coverage_mask.append(coverage >= min_coverage)
        if sum(coverage_mask) < min_ships:
            return None
        if not all(coverage_mask):
            selected_ships = [s for s, ok in zip(selected_ships, coverage_mask) if ok]

    features = config["features"]
    aligned_trajs = np.zeros((len(selected_ships), len(t_unified), len(features)), dtype=np.float32)

    for i, mmsi in enumerate(selected_ships):
        ship_data = ship_data_map[mmsi]
        t_ship = ship_data["timestamp"].values.astype(float)

        for j, feat in enumerate(features):
            if feat not in ship_data.columns:
                continue
            vals = ship_data[feat].values.astype(float)
            if feat in ["cog", "heading"]:
                sin_vals = np.sin(np.radians(vals))
                cos_vals = np.cos(np.radians(vals))
                from scipy.interpolate import interp1d
                f_sin = interp1d(t_ship, sin_vals, kind="linear",
                                 bounds_error=False,
                                 fill_value=(sin_vals[0], sin_vals[-1]))
                f_cos = interp1d(t_ship, cos_vals, kind="linear",
                                 bounds_error=False,
                                 fill_value=(cos_vals[0], cos_vals[-1]))
                aligned_trajs[i, :, j] = np.degrees(
                    np.arctan2(f_sin(t_unified), f_cos(t_unified))) % 360
            else:
                from scipy.interpolate import interp1d
                f = interp1d(t_ship, vals, kind="linear",
                             bounds_error=False,
                             fill_value=(vals[0], vals[-1]))
                aligned_trajs[i, :, j] = f(t_unified)

    # 插值后校验：SOG ≥ 0，拒绝含 NaN 的场景
    aligned_trajs[:, :, 2] = np.clip(aligned_trajs[:, :, 2], 0.0, None)  # SOG ≥ 0
    aligned_trajs[:, :, 3] = aligned_trajs[:, :, 3] % 360  # COG ∈ [0, 360)
    aligned_trajs[:, :, 4] = aligned_trajs[:, :, 4] % 360  # Heading ∈ [0, 360)

    # 插值后质量后处理：修复 SOG 跳变 / 位置-SOG 不一致 / COG 高速跳变
    aligned_trajs = post_interpolation_quality_pass(aligned_trajs, config, dt=sampling_interval)
    if aligned_trajs is None:
        return None

    if np.isnan(aligned_trajs).any() or np.isinf(aligned_trajs).any():
        return None

    # 场景内静止船过滤：移除场景窗口内平均 SOG 过低的船舶
    min_mean_sog = config["scene_extraction"].get("min_mean_sog_knots", 0.0)
    if min_mean_sog > 0:
        mean_sog = aligned_trajs[:, :, 2].mean(axis=1)  # [N]
        moving_mask = mean_sog >= min_mean_sog
        if moving_mask.sum() < min_ships:
            return None
        if moving_mask.sum() < len(selected_ships):
            aligned_trajs = aligned_trajs[moving_mask]
            selected_ships = [s for s, m in zip(selected_ships, moving_mask) if m]

    threshold_nm = config["scene_extraction"]["interaction_distance_nm"]
    adj_matrix = compute_mean_adjacency(aligned_trajs, threshold_nm)

    # 场景空间紧致性检查：计算船舶平均位置的最大两两距离
    max_scene_diameter_nm = config["scene_extraction"].get("max_scene_diameter_nm", 0)
    if max_scene_diameter_nm > 0:
        mean_positions = aligned_trajs[:, :, :2].mean(axis=1)  # [N, 2] (lat, lon)
        max_dist = 0.0
        for i in range(len(selected_ships)):
            for j in range(i + 1, len(selected_ships)):
                d = haversine_distance_nm(
                    mean_positions[i, 0], mean_positions[i, 1],
                    mean_positions[j, 0], mean_positions[j, 1])
                if d > max_dist:
                    max_dist = d
        if max_dist > max_scene_diameter_nm:
            return None

    # 计算最小船对距离（用于后续会遇质量过滤）
    min_pairwise_dist = float("inf")
    n_final = len(selected_ships)
    for i in range(n_final):
        for j in range(i + 1, n_final):
            for t in range(0, len(t_unified), max(1, len(t_unified) // 20)):
                d = haversine_distance_nm(
                    aligned_trajs[i, t, 0], aligned_trajs[i, t, 1],
                    aligned_trajs[j, t, 0], aligned_trajs[j, t, 1])
                if d < min_pairwise_dist:
                    min_pairwise_dist = d

    return {
        "trajectories": aligned_trajs,
        "timestamps": t_unified,
        "mmsi_list": selected_ships,
        "adjacency": adj_matrix,
        "n_ships": len(selected_ships),
        "duration_sec": common_end - common_start,
        "min_pairwise_distance_nm": min_pairwise_dist,
        "heading_quality": np.array([
            ship_data_map[m]["heading_quality"].iloc[0]
            if "heading_quality" in ship_data_map[m].columns else 1.0
            for m in selected_ships
        ], dtype=np.float32),
    }


def extract_all_scenes():
    config = load_config()
    processed_dir = get_data_path("processed")
    output_dir = get_data_path("processed", "scenes")

    print("=" * 60)
    print("[场景提取] 从所有数据集中提取多船交互场景")
    print("=" * 60)

    datasets_config = config.get("datasets", {})
    dataset_names = list(datasets_config.keys()) if datasets_config else ["envship", "dma", "marinecadastre"]
    min_heading_quality = config.get("preprocessing", {}).get("min_heading_quality", 0.0)

    scene_id = 0
    all_scene_data = []

    # === 流式处理：逐数据集加载 → 提取场景 → 释放 ===
    # 内存策略：一次只持有一个区域的 DataFrame（1-5 GB），处理完立即释放
    # 而非把 3 个区域全部加载到 combined（峰值 20-30 GB）
    for dataset_name in dataset_names:
        data_dir = processed_dir / dataset_name
        if not data_dir.exists():
            continue
        parquet_files = list(data_dir.glob("*.parquet"))
        if not parquet_files:
            continue

        print(f"\n--- 处理数据集: {dataset_name} ---")

        # 逐文件加载（一个 dataset 通常只有 1 个 unified parquet）
        subset_parts = []
        for f in parquet_files:
            print(f"  加载: {dataset_name}/{f.name}")
            df = pd.read_parquet(f)
            df["dataset"] = dataset_name
            subset_parts.append(df)
        if len(subset_parts) > 1:
            subset = pd.concat(subset_parts, ignore_index=True)
        else:
            subset = subset_parts[0]
        del subset_parts
        gc.collect()

        print(f"  数据量: {len(subset):,} 行, {subset['mmsi'].nunique()} 艘船")

        # heading 质量过滤（局部）
        if "heading_quality" in subset.columns:
            if min_heading_quality > 0:
                heading_quality_map = subset.groupby("mmsi")["heading_quality"].first().to_dict()
                n_before = subset["mmsi"].nunique()
                low_q_ships = {m for m, q in heading_quality_map.items() if q < min_heading_quality}
                if low_q_ships:
                    subset = subset[~subset["mmsi"].isin(low_q_ships)].copy()
                    n_after = subset["mmsi"].nunique()
                    print(f"  Heading 质量过滤 (>={min_heading_quality:.0%}): {n_before} → {n_after} 船 "
                          f"(移除 {len(low_q_ships)} 艘低质量船)")
        else:
            print(f"  [WARN] parquet 中无 heading_quality 列，跳过质量过滤")

        # 提取场景
        scenes = find_interaction_scenes(subset, config)

        for scene_info in tqdm(scenes, desc=f"提取{dataset_name}场景轨迹"):
            result = extract_scene_trajectories(subset, scene_info, config)
            if result is None:
                continue

            result["scene_id"] = scene_id
            result["dataset"] = dataset_name
            all_scene_data.append(result)
            scene_id += 1

        # 释放该区域内存
        del subset
        gc.collect()

    print(f"\n{'=' * 60}")
    print(f"  去重前场景数: {len(all_scene_data)}")

    if not all_scene_data:
        print("  错误: 未提取到有效场景")
        return False

    # 场景去重：合并船舶集合高度重叠且时间接近的场景
    dedup_config = config.get("scene_dedup", {})
    jaccard_th = dedup_config.get("jaccard_threshold", 0.5)
    time_th = dedup_config.get("time_overlap_sec", 1800)

    all_scene_data.sort(key=lambda s: (s["dataset"], s["timestamps"][0]))
    keep_mask = [True] * len(all_scene_data)

    for i in range(len(all_scene_data)):
        if not keep_mask[i]:
            continue
        si = all_scene_data[i]
        ships_i = set(si["mmsi_list"])
        for j in range(i + 1, len(all_scene_data)):
            if not keep_mask[j]:
                continue
            sj = all_scene_data[j]
            if si["dataset"] != sj["dataset"]:
                break
            if sj["timestamps"][0] - si["timestamps"][-1] > time_th:
                break
            ships_j = set(sj["mmsi_list"])
            jaccard = len(ships_i & ships_j) / max(len(ships_i | ships_j), 1)
            if jaccard >= jaccard_th:
                if si["n_ships"] >= sj["n_ships"]:
                    keep_mask[j] = False
                else:
                    keep_mask[i] = False
                    break

    n_before_dedup = len(all_scene_data)
    all_scene_data = [s for s, k in zip(all_scene_data, keep_mask) if k]
    n_removed = n_before_dedup - len(all_scene_data)
    print(f"  去重: {n_before_dedup} → {len(all_scene_data)} 场景 (移除 {n_removed} 个重叠场景)")

    # 重新编号 scene_id
    for idx, scene in enumerate(all_scene_data):
        scene["scene_id"] = idx

    # 清理旧场景文件
    for old_file in output_dir.glob("scene_*.npz"):
        old_file.unlink()

    # 保存场景
    for scene in tqdm(all_scene_data, desc="保存场景"):
        sid = scene["scene_id"]
        ds = scene["dataset"]
        out_file = output_dir / f"scene_{sid:05d}_{ds}.npz"
        np.savez_compressed(
            out_file,
            trajectories=scene["trajectories"],
            timestamps=scene["timestamps"],
            mmsi_list=np.array(scene["mmsi_list"]),
            adjacency=scene["adjacency"],
            heading_quality=scene.get("heading_quality", np.ones(scene["n_ships"], dtype=np.float32)),
            min_pairwise_distance_nm=scene.get("min_pairwise_distance_nm", 999.0),
            n_ships=scene["n_ships"],
            duration_sec=scene["duration_sec"],
            dataset=ds,
            scene_id=sid,
        )

    # 统计
    print(f"\n  场景统计:")
    dataset_set = sorted(set(s["dataset"] for s in all_scene_data))
    for ds in dataset_set:
        ds_scenes = [s for s in all_scene_data if s["dataset"] == ds]
        if ds_scenes:
            avg_ships = np.mean([s["n_ships"] for s in ds_scenes])
            avg_dur = np.mean([s["duration_sec"] for s in ds_scenes]) / 60
            print(f"    {ds}: {len(ds_scenes)} 场景, 平均 {avg_ships:.1f} 船, 平均 {avg_dur:.0f} min")

    print(f"\n  保存至: {output_dir}/")
    print("[场景提取] 完成！")
    return True


if __name__ == "__main__":
    extract_all_scenes()