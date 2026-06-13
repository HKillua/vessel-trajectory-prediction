"""数据统计与质量报告

统计各区域场景数、会遇分布、DCPA/TCPA 分布、
train/val/test 样本数，估算训练时间。
"""

import sys
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_data_path


ENCOUNTER_NAMES = {
    0: "safe",
    1: "head_on",
    2: "crossing_give_way",
    3: "crossing_stand_on",
    4: "overtaking",
    5: "being_overtaken",
}


def report_scenes():
    """统计场景数据"""
    config = load_config()
    scene_dir = get_data_path("processed", "scenes")
    npz_files = sorted(scene_dir.glob("scene_*.npz"))

    if not npz_files:
        print("  未找到场景文件")
        return

    print(f"\n{'='*60}")
    print(f"  场景统计 ({len(npz_files)} 个场景)")
    print(f"{'='*60}")

    dataset_stats = defaultdict(lambda: {
        "count": 0, "ships": [], "duration": [],
        "encounter_count": 0, "min_dcpa": [],
        "enc_types": Counter(),
    })

    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        ds = str(data["dataset"])
        stats = dataset_stats[ds]
        stats["count"] += 1
        stats["ships"].append(int(data["n_ships"]))
        stats["duration"].append(float(data["duration_sec"]) / 60)

        if "has_encounter" in data.files:
            has_enc = bool(data["has_encounter"])
            if has_enc:
                stats["encounter_count"] += 1
            stats["min_dcpa"].append(float(data["min_dcpa"]))

        if "encounter_type" in data.files:
            enc = data["encounter_type"]
            for et in range(6):
                stats["enc_types"][et] += int(np.sum(enc == et))

    for ds, stats in sorted(dataset_stats.items()):
        print(f"\n  --- {ds} ---")
        print(f"  场景数: {stats['count']}")
        ships = stats["ships"]
        print(f"  船舶数: min={min(ships)}, max={max(ships)}, mean={np.mean(ships):.1f}")
        dur = stats["duration"]
        print(f"  时长(min): min={min(dur):.0f}, max={max(dur):.0f}, mean={np.mean(dur):.0f}")

        if stats["min_dcpa"]:
            dcpa = stats["min_dcpa"]
            enc_pct = stats["encounter_count"] / max(stats["count"], 1) * 100
            print(f"  会遇场景: {stats['encounter_count']}/{stats['count']} ({enc_pct:.1f}%)")
            print(f"  最小DCPA(nm): min={min(dcpa):.3f}, median={np.median(dcpa):.2f}, mean={np.mean(dcpa):.2f}")

        if stats["enc_types"]:
            total = sum(stats["enc_types"].values())
            print(f"  会遇类型分布 (ship-pair-timestep level):")
            for et in range(6):
                cnt = stats["enc_types"][et]
                pct = cnt / max(total, 1) * 100
                print(f"    {ENCOUNTER_NAMES[et]:25s}: {cnt:>10,} ({pct:5.1f}%)")


def report_splits():
    """统计 train/val/test 样本数"""
    final_dir = get_data_path("final")
    variant_dir = final_dir / "obs10_pred10"

    if not variant_dir.exists():
        print("\n  未找到最终数据集")
        return

    print(f"\n{'='*60}")
    print(f"  数据集划分统计 (obs10_pred10)")
    print(f"{'='*60}")

    total = 0
    for split in ["train", "val", "test"]:
        index_file = variant_dir / split / f"{split}_index.npz"
        if not index_file.exists():
            print(f"  {split}: 不存在")
            continue

        data = np.load(index_file)
        n = int(data["n_samples"])
        ns = data["n_ships"]
        total += n
        print(f"  {split:5s}: {n:>8,} 样本 | ships [{ns.min()}-{ns.max()}] mean={ns.mean():.1f}")

        # 抽样检查样本级会遇质量
        split_dir = variant_dir / split
        sample_files = sorted(split_dir.glob("sample_*.npz"))
        if sample_files:
            n_check = min(100, len(sample_files))
            indices = np.linspace(0, len(sample_files)-1, n_check, dtype=int)
            enc_counts = Counter()
            scene_ids = set()
            for idx in indices:
                d = np.load(sample_files[idx], allow_pickle=True)
                scene_ids.add(int(d["scene_id"]))
                if "encounter_type" in d.files:
                    enc = d["encounter_type"]
                    for et in range(6):
                        enc_counts[et] += int(np.sum(enc == et))

            total_enc = sum(enc_counts.values())
            non_safe_pct = (total_enc - enc_counts[0]) / max(total_enc, 1) * 100
            print(f"         独立场景: {len(scene_ids)} | 非safe会遇占比: {non_safe_pct:.1f}%")
            for et in range(1, 6):
                pct = enc_counts[et] / max(total_enc, 1) * 100
                print(f"           {ENCOUNTER_NAMES[et]:20s}: {pct:.1f}%")

    print(f"  {'total':5s}: {total:>8,} 样本")


def estimate_training_time(n_samples, batch_size=32, model_params_m=12, epochs=100):
    """估算 RTX 3090 训练时间"""
    iters_per_epoch = n_samples // batch_size
    ms_per_iter = 60 + model_params_m * 3
    time_per_epoch_sec = iters_per_epoch * ms_per_iter / 1000
    total_hours = time_per_epoch_sec * epochs / 3600

    print(f"\n{'='*60}")
    print(f"  训练时间估算 (RTX 3090)")
    print(f"{'='*60}")
    print(f"  模型参数: ~{model_params_m}M")
    print(f"  训练样本: {n_samples:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Iters/epoch: {iters_per_epoch:,}")
    print(f"  预估 iter 耗时: ~{ms_per_iter}ms")
    print(f"  预估 epoch 耗时: {time_per_epoch_sec/60:.1f} min")
    print(f"  预估总训练时间: {total_hours:.1f} hours")

    if total_hours > 8:
        print(f"  ⚠ 超过 8 小时限制，建议减少 epochs 至 {int(8 / (time_per_epoch_sec / 3600))}")


def report_preprocessed():
    """统计预处理后的数据"""
    processed_dir = get_data_path("processed")
    config = load_config()

    print(f"\n{'='*60}")
    print(f"  预处理数据统计")
    print(f"{'='*60}")

    for ds_name in config.get("datasets", {}).keys():
        ds_dir = processed_dir / ds_name
        if not ds_dir.exists():
            continue
        for f in ds_dir.glob("*.parquet"):
            try:
                import pandas as pd
                df = pd.read_parquet(f)
                n_ships = df["mmsi"].nunique()
                duration_days = (df["timestamp"].max() - df["timestamp"].min()) / 86400
                print(f"  {ds_name}: {len(df):>10,} 行 | {n_ships:>5} 船 | {duration_days:.1f} 天")

                # 船型分布
                ship_types = df.groupby("mmsi")["ship_type"].first().value_counts()
                print(f"    船型分布:")
                for st, cnt in ship_types.items():
                    print(f"      {st}: {cnt}")

                # Heading 质量
                if "heading_quality" in df.columns:
                    hq = df.groupby("mmsi")["heading_quality"].first()
                    print(f"    Heading 质量: mean={hq.mean():.1%}, >=60%={int((hq>=0.6).sum())}/{n_ships}")

                # 缺失 vessel_length
                if "vessel_length" in df.columns:
                    vl = df.groupby("mmsi")["vessel_length"].first()
                    print(f"    Vessel length=0 (缺失): {int((vl==0).sum())}/{n_ships}")
            except Exception as e:
                print(f"  {ds_name}: 读取失败 ({e})")


def main():
    print("=" * 60)
    print("  MambaDiff-ECR 数据质量报告")
    print("=" * 60)

    report_preprocessed()
    report_scenes()
    report_splits()

    final_dir = get_data_path("final")
    train_index = final_dir / "obs10_pred10" / "train" / "train_index.npz"
    if train_index.exists():
        data = np.load(train_index)
        n_train = int(data["n_samples"])
        estimate_training_time(n_train)

    print(f"\n{'='*60}")
    print("  报告完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
