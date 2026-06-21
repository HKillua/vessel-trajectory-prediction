"""
01_download.py - Download NOAA MarineCadastre AIS data.
Multi-threaded with resume support.
"""

import os
import sys
import argparse
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config


def generate_urls(base_url, start_date, end_date):
    urls = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current <= end:
        fname = f"AIS_{current.strftime('%Y_%m_%d')}.zip"
        urls.append((f"{base_url}/{fname}", fname))
        current += timedelta(days=1)
    return urls


def download_file(url, dest_path, min_size=1_000_000, max_retries=3):
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path)
        if size > min_size:
            return 'skipped', os.path.basename(dest_path)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, stream=True, timeout=300)
            resp.raise_for_status()
            total = int(resp.headers.get('content-length', 0))

            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            tmp_path = dest_path + '.tmp'
            with open(tmp_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            if total > 0 and os.path.getsize(tmp_path) < total * 0.9:
                os.remove(tmp_path)
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                return 'failed', f"{os.path.basename(dest_path)} (incomplete)"

            os.rename(tmp_path, dest_path)
            return 'success', os.path.basename(dest_path)
        except Exception as e:
            if attempt < max_retries - 1:
                import time
                time.sleep(2 ** attempt)
                continue
            return 'failed', f"{os.path.basename(dest_path)} ({e})"


def main():
    parser = argparse.ArgumentParser(description='Download NOAA AIS data')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    dl = config['download']
    raw_dir = base / dl['raw_dir']
    raw_dir.mkdir(parents=True, exist_ok=True)

    urls = generate_urls(dl['base_url'], dl['date_range']['start'], dl['date_range']['end'])
    workers = args.workers or dl.get('max_workers', 4)

    print(f"Downloading {len(urls)} files to {raw_dir} with {workers} workers")

    stats = {'success': 0, 'skipped': 0, 'failed': 0}
    failed = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_file, url, str(raw_dir / fname)): fname
            for url, fname in urls
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            status, info = future.result()
            stats[status] += 1
            if status == 'failed':
                failed.append(info)

    print(f"\nDone: {stats['success']} downloaded, {stats['skipped']} skipped, {stats['failed']} failed")
    if failed:
        print("Failed files:")
        for f in failed:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
