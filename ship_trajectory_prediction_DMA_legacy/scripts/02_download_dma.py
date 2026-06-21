"""下载 DMA (Danish Maritime Authority) AIS 数据

来源: https://web.ais.dk/aisdata/
格式: ZIP 压缩的 CSV, 按天提供
支持多月份日期范围下载, 断点续传
"""

import sys
import requests
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from utils import load_config

PROJECT_ROOT = Path(__file__).parent.parent


def download_file(url, output_path, chunk_size=8192):
    if output_path.exists():
        print(f"  已存在, 跳过: {output_path.name}")
        return True
        
    try:
        resp = requests.get(url, stream=True, timeout=120)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        with open(output_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  下载失败 {url}: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def download_dma():
    config = load_config()
    dl_config = config["dma_download"]
    
    base_url = dl_config["base_url"]
    output_dir = PROJECT_ROOT / dl_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    date_ranges = dl_config["date_ranges"]
    
    print("=" * 60)
    print("[DMA] 下载丹麦海事局 AIS 数据 (多月份) ")
    print(f"输出目录: {output_dir}")
    for dr in date_ranges:
        print(f"  日期范围: {dr['start']} ~ {dr['end']}")
    print("=" * 60)
    
    all_dates = []
    for dr in date_ranges:
        start = datetime.strptime(dr["start"], "%Y-%m-%d")
        end = datetime.strptime(dr["end"], "%Y-%m-%d")
        current = start
        while current <= end:
            all_dates.append(current)
            current += timedelta(days=1)
            
    print(f"\n共 {len(all_dates)} 天待下载")
    
    downloaded = 0
    failed = 0
    
    for date in all_dates:
        date_str = date.strftime("%Y-%m-%d")
        
        url_patterns = [
            f"{base_url}aisdk-{date_str}.zip",
            f"{base_url}aisdk-{date_str}.csv",
        ]
        
        success = False
        for url in url_patterns:
            filename = url.split("/")[-1]
            output_path = output_dir / filename
            if download_file(url, output_path):
                downloaded += 1
                success = True
                break
                
        if not success:
            failed += 1
            print(f"  {date_str}: 下载失败")
            
    print(f"\n" + "=" * 60)
    print(f"[DMA] 下载完成: {downloaded} 成功, {failed} 失败")
    print(f"  文件位置: {output_dir}")
    
    if failed > 0:
        print(f"\n  手动下载地址: {base_url}")
        print(f"  文件命名: aisdk-YYYY-MM-DD.zip")
        
    return downloaded > 0


if __name__ == "__main__":
    download_dma()