"""
02_preprocess.py - Core AIS preprocessing pipeline for NOAA NY data.
Per-day streaming → per-MMSI cleaning → gap splitting → interpolation → smoothing.
"""

import os
import sys
import argparse
import zipfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    load_config, filter_position_jumps, filter_gps_freeze,
    remove_stationary_segments, split_by_gap, interpolate_trajectory,
    smooth_trajectory, repair_heading_cog, angular_from_sincos,
    align_sog_to_position
)


COLUMNS_KEEP = ['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG',
                'Heading', 'VesselType', 'Status', 'TransceiverClass']


def load_day_csv(csv_path, config):
    bbox = config['region']['bbox']
    prep = config['preprocess']

    try:
        df = pd.read_csv(csv_path, usecols=COLUMNS_KEEP, low_memory=False)
    except Exception as e:
        print(f"  Error reading {csv_path}: {e}")
        return None

    n_raw = len(df)

    df = df[
        (df['LAT'] >= bbox['lat_min']) & (df['LAT'] <= bbox['lat_max']) &
        (df['LON'] >= bbox['lon_min']) & (df['LON'] <= bbox['lon_max'])
    ]

    if prep.get('heading_filter', True):
        df = df[df['Heading'] != 511]

    if prep.get('status_filter'):
        df = df[~df['Status'].isin(prep['status_filter'])]

    df = df[df['TransceiverClass'] == 'A']

    df = df.dropna(subset=['MMSI', 'BaseDateTime', 'LAT', 'LON', 'SOG', 'COG', 'Heading'])

    df['timestamp'] = (pd.to_datetime(df['BaseDateTime']) - pd.Timestamp('1970-01-01')).dt.total_seconds().astype(np.int64)
    df = df.rename(columns={
        'MMSI': 'mmsi', 'LAT': 'lat', 'LON': 'lon',
        'SOG': 'sog', 'COG': 'cog', 'Heading': 'heading',
        'VesselType': 'vessel_type'
    })
    df = df[['mmsi', 'timestamp', 'lat', 'lon', 'sog', 'cog', 'heading', 'vessel_type']]
    df = df.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)

    n_filtered = len(df)
    n_ships = df['mmsi'].nunique()
    print(f"  {Path(csv_path).stem}: {n_raw:,} raw → {n_filtered:,} filtered ({n_ships} ships)")
    return df


def process_ship(ship_df, config):
    prep = config['preprocess']
    interval = prep.get('sampling_interval', 60)

    ship_df = ship_df.sort_values('timestamp').reset_index(drop=True)

    # NOAA data has multi-station duplicates: aggregate within each minute
    ship_df['time_bin'] = (ship_df['timestamp'] // interval) * interval
    ship_df['cog_sin'] = np.sin(np.radians(ship_df['cog']))
    ship_df['cog_cos'] = np.cos(np.radians(ship_df['cog']))
    ship_df['hdg_sin'] = np.sin(np.radians(ship_df['heading']))
    ship_df['hdg_cos'] = np.cos(np.radians(ship_df['heading']))
    agg = ship_df.groupby('time_bin').agg({
        'timestamp': 'median',
        'lat': 'median',
        'lon': 'median',
        'sog': 'median',
        'cog_sin': 'mean',
        'cog_cos': 'mean',
        'hdg_sin': 'mean',
        'hdg_cos': 'mean',
        'vessel_type': 'first',
        'mmsi': 'first',
    }).reset_index(drop=True)
    agg['timestamp'] = agg['timestamp'].astype(np.int64)
    agg['cog'] = angular_from_sincos(agg.pop('cog_sin').values, agg.pop('cog_cos').values)
    agg['heading'] = angular_from_sincos(agg.pop('hdg_sin').values, agg.pop('hdg_cos').values)
    ship_df = agg

    ship_df = filter_position_jumps(
        ship_df,
        max_speed_kn=prep.get('position_jump_speed', 50),
        speed_factor=prep.get('jump_speed_factor', 1.5)
    )
    if len(ship_df) < 2:
        return []

    ship_df = filter_gps_freeze(
        ship_df,
        window=prep.get('gps_freeze_window', 4),
        sog_thr=prep.get('gps_freeze_sog_thr', 5.0)
    )
    if len(ship_df) < 2:
        return []

    ship_df = remove_stationary_segments(
        ship_df,
        sog_thr=prep.get('stationary_sog_threshold', 0.5),
        max_duration=prep.get('stationary_max_duration', 300)
    )
    if len(ship_df) < 2:
        return []

    segments = split_by_gap(ship_df, gap_threshold=prep.get('gap_split_threshold', 600))

    interval = prep.get('sampling_interval', 60)
    min_steps = prep.get('min_trajectory_steps', 60)
    results = []

    for seg in segments:
        interp = interpolate_trajectory(seg, interval_sec=interval)
        if interp is None or len(interp) < min_steps:
            continue
        sog_alpha = prep.get('sog_position_align_alpha', None)
        if sog_alpha is not None:
            interp = align_sog_to_position(interp, alpha=sog_alpha)
        interp = smooth_trajectory(
            interp,
            sog_median_window=prep.get('sog_median_window', 3),
            max_sog_change=prep.get('max_sog_change_per_step', 9.0),
            min_sog_for_cog=prep.get('min_sog_for_cog', 1.0),
            max_sog_knots=prep.get('max_sog_knots', None)
        )
        interp = repair_heading_cog(
            interp,
            max_diff_deg=prep.get('heading_cog_max_diff_deg', 90)
        )
        if interp['sog'].mean() < prep.get('min_mean_sog', 0.5):
            continue
        results.append(interp)

    return results


def process_day(csv_path, config):
    df = load_day_csv(csv_path, config)
    if df is None or len(df) == 0:
        return pd.DataFrame()

    all_segments = []
    seg_id = 0

    for mmsi, group in df.groupby('mmsi'):
        segments = process_ship(group, config)
        for seg in segments:
            seg['segment_id'] = seg_id
            all_segments.append(seg)
            seg_id += 1

    if not all_segments:
        return pd.DataFrame()

    result = pd.concat(all_segments, ignore_index=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='Preprocess NOAA AIS data')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    parser.add_argument('--single-day', type=str, default=None,
                        help='Process single CSV file for testing')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    raw_dir = base / config['download']['raw_dir']
    proc_dir = base / config['preprocess']['processed_dir']
    proc_dir.mkdir(parents=True, exist_ok=True)

    if args.single_day:
        csv_path = Path(args.single_day)
        if not csv_path.exists() and (raw_dir / args.single_day).exists():
            csv_path = raw_dir / args.single_day
        if csv_path.suffix == '.zip':
            print(f"Extracting {csv_path}...")
            with zipfile.ZipFile(csv_path, 'r') as zf:
                csv_name = [n for n in zf.namelist() if n.endswith('.csv')][0]
                zf.extract(csv_name, csv_path.parent)
                csv_path = csv_path.parent / csv_name

        print(f"Processing {csv_path}...")
        result = process_day(str(csv_path), config)
        if len(result) > 0:
            out_path = proc_dir / f"{csv_path.stem}_processed.parquet"
            result.to_parquet(out_path, index=False)
            n_ships = result['mmsi'].nunique()
            n_segs = result['segment_id'].nunique()
            print(f"\nOutput: {out_path}")
            print(f"  Ships: {n_ships}, Segments: {n_segs}, Records: {len(result):,}")

            ts_diff = result.groupby('segment_id')['timestamp'].diff().dropna()
            print(f"  Sampling interval: median={ts_diff.median():.0f}s, "
                  f"min={ts_diff.min():.0f}s, max={ts_diff.max():.0f}s")

            seg_lengths = result.groupby('segment_id').size()
            print(f"  Segment lengths: min={seg_lengths.min()}, "
                  f"median={seg_lengths.median():.0f}, max={seg_lengths.max()}")
        else:
            print("No valid trajectories after preprocessing.")
        return

    zip_files = sorted(raw_dir.glob('AIS_2024_*.zip'))
    existing_parquets = {f.stem.replace('_processed', '') for f in proc_dir.glob('*_processed.parquet')}

    print(f"Streaming {len(zip_files)} ZIP files (extract -> process -> delete CSV)...")
    all_results = []
    total_ships = set()
    total_segs = 0
    skipped = 0

    for zp in tqdm(zip_files, desc="Processing"):
        day_name = zp.stem
        out_path = proc_dir / f"{day_name}_processed.parquet"
        if out_path.exists() or day_name in existing_parquets:
            all_results.append(out_path)
            skipped += 1
            continue

        # Extract single zip
        with zipfile.ZipFile(zp, 'r') as zf:
            csv_names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not csv_names:
                continue
            zf.extract(csv_names[0], raw_dir)
            csv_path = raw_dir / csv_names[0]

        # Process
        result = process_day(str(csv_path), config)
        if len(result) > 0:
            result.to_parquet(out_path, index=False)
            total_ships.update(result['mmsi'].unique())
            total_segs += result['segment_id'].nunique()
            all_results.append(out_path)

        # Delete CSV to save space
        csv_path.unlink(missing_ok=True)

    print(f"\nTotal: {len(all_results)} days processed ({skipped} skipped)")
    print(f"  Unique ships: {len(total_ships)}")
    print(f"  Total segments: {total_segs}")
    print(f"  Output dir: {proc_dir}")


if __name__ == '__main__':
    main() 