"""从 DMA 原始 AIS 数据中提取多个区域的轨迹

数据来源: data/raw/envship/DMA/data_raw/dma/incoming/2025-09/aisdk-2025-09-*.zip
区域: Great Belt / Kattegat / Oresund (配置在 config.yaml 的 datasets 字段)

处理流程 (每个区域独立):
  1. 从 zip 读取原始 CSV
  2. 按区域空间过滤
  3. 过滤异常值 + 去静止船
  4. 按船舶重采样为 20s 间隔
  5. 保存统一格式的 parquet
"""

import sys
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils import (
    load_config, get_data_path,
    filter_trajectory, filter_position_jumps, filter_gps_freeze,
    interpolate_trajectory,
    split_trajectory_by_gap, remove_stationary_segments,
    smooth_trajectory
)


def load_dma_zip(zip_path):
    """从 zip 文件加载原始 DMA CSV (不做区域过滤)"""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
        if not csv_names:
            return pd.DataFrame()
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, dtype={"MMSI": str}, low_memory=False)
            
    df = df.rename(columns={"# Timestamp": "timestamp_str", "Ship type": "ship_type"})
    for col in ["Latitude", "Longitude", "SOG", "COG", "Heading"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        
    df["timestamp"] = pd.to_datetime(df["timestamp_str"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].apply(lambda x: int(x.timestamp()))
    
    result = pd.DataFrame({
        "mmsi": df["MMSI"],
        "timestamp": df["timestamp"],
        "lat": df["Latitude"],
        "lon": df["Longitude"],
        "sog": df["SOG"],
        "cog": df["COG"],
        "heading": df["Heading"],
        "ship_type": df.get("ship_type", "unknown"),
        "vessel_length": pd.to_numeric(df.get("Length", 0), errors="coerce").fillna(0),
        "vessel_width": pd.to_numeric(df.get("Width", 0), errors="coerce").fillna(0),
        "mobile_type": df.get("Type of mobile", "unknown"),
    })
    result = result.dropna(subset=["mmsi", "timestamp", "lat", "lon", "sog", "cog"])
    
    # Heading NaN 用 COG 填充 (AIS 中 heading 经常缺失, COG 是最佳替代)
    heading_nan_mask = result["heading"].isna()
    result["heading_is_genuine"] = ~heading_nan_mask
    if heading_nan_mask.any():
        nan_count = heading_nan_mask.sum()
        total_count = len(result)
        result.loc[heading_nan_mask, "heading"] = result.loc[heading_nan_mask, "cog"]
        print(f"  Heading NaN 填充: {nan_count:,}/{total_count:,} ({nan_count/total_count*100:.1f}%) 用 COG 替代")
        
    # SOG clamp >= 0 (浮点精度可能产生 -0.0)
    result["sog"] = result["sog"].clip(lower=0.0).abs()
    
    # COG/Heading 规范到 [0, 360)
    result["cog"] = result["cog"] % 360
    result["heading"] = result["heading"] % 360
    
    config = load_config()
    if config.get("ais_filter", {}).get("class_a_only", False):
        before = len(result)
        result = result[result["mobile_type"].str.contains("Class A", case=False, na=False)]
        print(f"  Class A 过滤: {before:,} -> {len(result):,} 行 ({len(result)/max(before,1)*100:.0f}%)")
        
    return result


def process_region(all_raw, region_key, region_config, config):
    """处理单个区域: 空间过滤 -> 重采样 -> 保存"""
    output_dir = get_data_path("processed", region_key)
    sampling_interval = config["sampling_interval"]
    gap_threshold = config["preprocessing"]["gap_threshold_min"] * 60
    min_steps = config["preprocessing"]["min_trajectory_steps"]
    
    name = region_config["name"]
    lat_min, lat_max = region_config["lat_min"], region_config["lat_max"]
    lon_min, lon_max = region_config["lon_min"], region_config["lon_max"]
    
    print(f"\n" + "=" * 50)
    print(f"区域: {name} ({lat_min}~{lat_max}N, {lon_min}~{lon_max}E)")
    
    mask = (
        (all_raw["lat"] >= lat_min) & (all_raw["lat"] <= lat_max) &
        (all_raw["lon"] >= lon_min) & (all_raw["lon"] <= lon_max)
    )
    df = all_raw[mask].copy()
    
    # 船型白名单过滤
    whitelist = config.get("ais_filter", {}).get("ship_type_whitelist", [])
    if whitelist:
        before_type = df["mmsi"].nunique()
        ship_types = df.groupby("mmsi")["ship_type"].first()
        allowed_mmsi = ship_types[ship_types.apply(
            lambda t: any(w.lower() in str(t).lower() for w in whitelist)
        )].index
        excluded_types = ship_types[~ship_types.index.isin(allowed_mmsi)].value_counts()
        df = df[df["mmsi"].isin(allowed_mmsi)].copy()
        after_type = df["mmsi"].nunique()
        print(f"  船型过滤: {before_type} -> {after_type} 船 (排除 {before_type - after_type} 船)")
        if len(excluded_types) > 0:
            for et, cnt in excluded_types.items():
                print(f"    排除: {et} ({cnt})")
                
    df = filter_trajectory(df, config)
    print(f"  区域内运动点: {len(df):,} 行, {df['mmsi'].nunique()} 船")
    
    if len(df) == 0:
        print("  无数据, 跳过")
        return False
        
    processed_ships = []
    jump_removed_total = 0
    for mmsi, ship_df in tqdm(df.groupby("mmsi"), desc=f"重采样 {name}"):
        ship_df = ship_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)
        if len(ship_df) < 10:
            continue
            
        # 位置跳变过滤: 移除 GPS 异常/MMSI 冲突导致的不可能位移 (先于静止段去除, 避免跳变掩盖静止)
        before_jump = len(ship_df)
        ship_df = filter_position_jumps(ship_df, config)
        jump_removed_total += before_jump - len(ship_df)
        if len(ship_df) < 10:
            continue
            
        # GPS 冻结检测: 移除 SOG 正常但位置不更新的段 (位置跳变的反向问题)
        before_freeze = len(ship_df)
        ship_df = filter_gps_freeze(ship_df, config)
        freeze_removed = before_freeze - len(ship_df)
        if freeze_removed > 0:
            jump_removed_total += freeze_removed
        if len(ship_df) < 10:
            continue
            
        ship_df = remove_stationary_segments(ship_df, config)
        if len(ship_df) < 10:
            continue
            
        segments = split_trajectory_by_gap(ship_df, gap_threshold)
        for seg in segments:
            # 计算该段的 heading 真实率 (而非整船, 确保每段质量独立评估)
            if "heading_is_genuine" in seg.columns:
                heading_quality = seg["heading_is_genuine"].mean()
            else:
                heading_quality = 1.0                
            resampled = interpolate_trajectory(seg, interval_sec=sampling_interval)
            if resampled is None or len(resampled) < min_steps:
                continue                
            # 重采样后平滑: SOG 中值滤波 + 变化率裁剪 + 低速 COG 前向填充
            resampled = smooth_trajectory(resampled, config)
            resampled["mmsi"] = mmsi
            resampled["heading_quality"] = heading_quality
            resampled["ship_type"] = ship_df["ship_type"].iloc[0] if "ship_type" in ship_df.columns else "unknown"
            resampled["vessel_length"] = ship_df["vessel_length"].iloc[0] if "vessel_length" in ship_df.columns else 0
            resampled["vessel_width"] = ship_df["vessel_width"].iloc[0] if "vessel_width" in ship_df.columns else 0
            resampled["dataset"] = region_key
            processed_ships.append(resampled)
            
    if not processed_ships:
        print("  重采样后无有效轨迹")
        return False
        
    result = pd.concat(processed_ships, ignore_index=True)
    for col in ["cog", "heading"]:
        if col in result.columns:
            result[col] = result[col] % 360            
    output_file = output_dir / f"{region_key}_unified.parquet"
    result.to_parquet(output_file, index=False)
    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  结果: {len(result):,} 行, {len(processed_ships)} 轨迹段, {result['mmsi'].nunique()} 船")
    if jump_removed_total > 0:
        print(f"  位置跳变移除: {jump_removed_total} 个异常点")
    if "heading_quality" in result.columns:
        hq = result.groupby("mmsi")["heading_quality"].first()
        print(f"  Heading 质量: mean={hq.mean():.1%}, median={hq.median():.1%}, "
              f"<=60%={ (hq <= 0.6).sum() } 船, >=60%={ (hq >= 0.6).sum() } 船")
    print(f"  保存: {output_file.name} ({size_mb:.1f} MB)")
    return True


def merge_zip_parts(raw_dir):
    """合并分片 zip 文件"""
    part_groups = {}
    for f in raw_dir.glob("*.zip.part-*"):
        base_name = f.name.rsplit(".part-", 1)[0]
        if base_name not in part_groups:
            part_groups[base_name] = []
        part_groups[base_name].append(f)
        
    for base_name, parts in part_groups.items():
        output_path = raw_dir / base_name
        if output_path.exists():
            continue
        parts.sort(key=lambda x: x.name)
        print(f"  合并 {len(parts)} 个分片 -> {base_name}")
        with open(output_path, "wb") as out:
            for part in parts:
                with open(part, "rb") as inp:
                    out.write(inp.read())
        try:
            with zipfile.ZipFile(output_path, 'r') as zf:
                zf.testzip()
        except Exception as e:
            print(f"  zip 损坏: {e}")
            output_path.unlink()


def process_all_regions():
    config = load_config()
    # 优先使用新的多月份数据目录
    if "dma_download" in config and (Path(__file__).parent.parent / config["dma_download"]["output_dir"]).exists():
        raw_dir = Path(__file__).parent.parent / config["dma_download"]["output_dir"]
        print(f"  使用多月份数据目录: {raw_dir}")
    else:
        raw_dir = Path(config["dma_raw"]["zip_dir"])
        if not raw_dir.is_absolute():
            raw_dir = Path(__file__).parent.parent / raw_dir
            
    print("=" * 60)
    print("[多区域预处理] 从 DMA 原始数据提取 3 个区域")
    print("=" * 60)
    
    if not raw_dir.exists():
        print(f"  错误: 未找到原始数据 {raw_dir}")
        return False
        
    merge_zip_parts(raw_dir)
    zip_files = sorted(raw_dir.glob("aisdk-*.zip"))
    print(f"  找到 {len(zip_files)} 个 zip 文件")
    
    print("\n[1/2] 加载原始数据...")
    all_raw = []
    for zf in zip_files:
        print(f"  读取: {zf.name}")
        try:
            df = load_dma_zip(zf)
        except Exception as e:
            print(f"    跳过: 无法读取 {zf.name} ({e})")
            continue
        print(f"    ({len(df):,} 行)")
        if len(df) > 0:
            all_raw.append(df)
            
    if not all_raw:
        print("  错误: 无数据")
        return False
        
    combined = pd.concat(all_raw, ignore_index=True)
    print(f"  合计: {len(combined):,} 行, {combined['mmsi'].nunique()} 船")
    
    print("\n[2/2] 按区域提取...")
    datasets = config["datasets"]
    success_count = 0
    for region_key, region_config in datasets.items():
        if process_region(combined, region_key, region_config, config):
            success_count += 1
            
    print(f"\n" + "=" * 60)
    print(f"[多区域预处理] 完成! 成功处理 {success_count}/{len(datasets)} 个区域")
    return success_count > 0


if __name__ == "__main__":
    process_all_regions()