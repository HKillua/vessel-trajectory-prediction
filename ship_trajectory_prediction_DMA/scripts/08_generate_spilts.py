"""生成最终数据集：时间分割 + 会遇过滤 + 滑动窗口切分

按时间段划分 train/val/test（防止 vessel-level 数据泄漏）。
仅保留包含真实会遇（DCPA < threshold）的场景。

输出格式 (NPZ):
- obs: [N_ships, obs_steps, 5]  (lat, lon, sog, cog, heading)
- pred: [N_ships, pred_steps, 5]
- adj_matrix: [N_ships, N_ships]
- dcpa_matrix: [N_ships, N_ships]
- tcpa_matrix: [N_ships, N_ships]
- cri_matrix: [N_ships, N_ships]
- encounter_type: [N_ships, N_ships]
- target_ship_idx: int
- n_ships: int
"""

import sys
import gc
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_data_path, compute_mean_adjacency, haversine_distance_nm


def date_to_timestamp(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def build_temporal_split_map(config):
    """根据 config 中的日期范围构建时间段→split 映射"""
    split_config = config["split"]
    ranges = []

    for dr in split_config["train_date_ranges"]:
        ranges.append((date_to_timestamp(dr["start"]),
                        date_to_timestamp(dr["end"]) + 86400, "train"))

    for dr in split_config["val_date_ranges"]:
        ranges.append((date_to_timestamp(dr["start"]),
                        date_to_timestamp(dr["end"]) + 86400, "val"))

    for dr in split_config["test_date_ranges"]:
        ranges.append((date_to_timestamp(dr["start"]),
                        date_to_timestamp(dr["end"]) + 86400, "test"))

    return ranges


def classify_scene_split(scene_start_ts, ranges):
    """根据场景起始时间戳判断属于哪个 split"""
    for ts_start, ts_end, split_name in ranges:
        if ts_start <= scene_start_ts < ts_end:
            return split_name
    return None


def load_scenes(scenes_dir, config):
    """加载场景文件，按会遇质量过滤"""
    require_encounter = config["split"].get("require_encounter", True)
    min_dcpa_for_scene = config.get("encounter", {}).get("min_dcpa_for_scene_nm", 2.0)

    scene_files = sorted(scenes_dir.glob("scene_*.npz"))
    scenes = []
    skipped_no_enc = 0
    skipped_dcpa = 0

    for f in scene_files:
        data = np.load(f, allow_pickle=True)

        if require_encounter:
            has_enc = bool(data["has_encounter"]) if "has_encounter" in data.files else False
            if not has_enc:
                skipped_no_enc += 1
                continue

            # 收紧过滤：检查 min_dcpa（来自 11_compute_encounters）或 min_pairwise_distance_nm（来自 07）
            scene_min_dcpa = float("inf")
            if "min_dcpa" in data.files:
                scene_min_dcpa = float(data["min_dcpa"])
            elif "min_pairwise_distance_nm" in data.files:
                scene_min_dcpa = float(data["min_pairwise_distance_nm"])
            if scene_min_dcpa > min_dcpa_for_scene:
                skipped_dcpa += 1
                continue

        scene = {
            "trajectories": data["trajectories"],
            "timestamps": data["timestamps"],
            "adjacency": data["adjacency"],
            "n_ships": int(data["n_ships"]),
            "dataset": str(data["dataset"]),
            "scene_id": int(data["scene_id"]),
            "file": f,
        }

        if "dcpa_matrix" in data.files:
            scene["dcpa_matrix"] = data["dcpa_matrix"]
            scene["tcpa_matrix"] = data["tcpa_matrix"]
            scene["cri_matrix"] = data["cri_matrix"]
            scene["encounter_type"] = data["encounter_type"]

        scenes.append(scene)

    if require_encounter:
        print(f"  会遇过滤: 保留 {len(scenes)}, 跳过 {skipped_no_enc} 无会遇 + {skipped_dcpa} 个 DCPA>{min_dcpa_for_scene}nm")

    return scenes


def summarize_encounters(obs_enc_type):
    """从观测窗口的 encounter_type [N,N,T] 中提取每对船的主要会遇类型（成对一致）"""
    n_ships = obs_enc_type.shape[0]
    dominant = np.zeros((n_ships, n_ships), dtype=np.int8)
    for i in range(n_ships):
        for j in range(i + 1, n_ships):
            pairs = list(zip(obs_enc_type[i, j, :], obs_enc_type[j, i, :]))
            nonzero_pairs = [(a, b) for a, b in pairs if a != 0 or b != 0]
            if nonzero_pairs:
                from collections import Counter
                (best_ij, best_ji), _ = Counter(nonzero_pairs).most_common(1)[0]
                dominant[i, j] = best_ij
                dominant[j, i] = best_ji
    return dominant


def generate_samples_from_scene(scene, obs_steps, pred_steps, stride, min_pred_mean_sog=0, config=None):
    """从单个场景中用滑动窗口生成样本（含会遇特征）"""
    trajs = scene["trajectories"]
    n_ships, total_steps, n_feat = trajs.shape
    window_size = obs_steps + pred_steps

    if total_steps < window_size:
        return []

    has_encounter_data = "dcpa_matrix" in scene
    sog_idx = 2  # SOG 在特征维度中的索引

    max_sog_change = 3.0
    dcpa_threshold = 2.0
    min_obs_tail_sog = 0
    obs_tail_steps = 5
    if config is not None:
        max_sog_change = config.get("preprocessing", {}).get("max_sog_change_per_step", 3.0)
        dcpa_threshold = config.get("encounter", {}).get("dcpa_threshold_nm", 2.0)
        min_obs_tail_sog = config.get("split", {}).get("min_obs_tail_sog", 0)

    samples = []
    for start in range(0, total_steps - window_size + 1, stride):
        obs = trajs[:, start:start + obs_steps, :]
        pred = trajs[:, start + obs_steps:start + window_size, :]

        # 质量过滤：pred 窗口中任意船平均 SOG 过低 → 跳过（突然停船不可预测）
        if min_pred_mean_sog > 0:
            pred_mean_sog = pred[:, :, sog_idx].mean(axis=1)  # [N_ships]
            if (pred_mean_sog < min_pred_mean_sog).any():
                continue

        # 质量过滤：obs 窗口尾部任意船平均 SOG 过低 → 跳过（从停泊起航不可预测）
        if min_obs_tail_sog > 0:
            obs_tail_mean_sog = obs[:, -obs_tail_steps:, sog_idx].mean(axis=1)
            if (obs_tail_mean_sog < min_obs_tail_sog).any():
                continue

        # Fix 3: obs→pred SOG 边界连续性检查
        boundary_ok = True
        for s in range(n_ships):
            sog_jump = abs(float(pred[s, 0, sog_idx]) - float(obs[s, -1, sog_idx]))
            if sog_jump > max_sog_change:
                boundary_ok = False
                break
        if not boundary_ok:
            continue

        # Fix M20: sample-level GPS freeze check
        dt = config.get("sampling_interval", 20) if config else 20
        freeze_sog_thr = config.get("preprocessing", {}).get("gps_freeze_sog_thr", 3.0) if config else 3.0
        freeze_ratio_thr = config.get("preprocessing", {}).get("gps_freeze_ratio_thr", 0.1) if config else 0.1
        window_trajs = np.concatenate([obs, pred], axis=1)  # [N_ships, window_size, 5]
        gps_freeze_found = False
        for s in range(n_ships):
            mean_sog = float(window_trajs[s, :, sog_idx].mean())
            if mean_sog < freeze_sog_thr:
                continue
            cumul_dist = 0.0
            for t in range(1, window_size):
                cumul_dist += haversine_distance_nm(
                    window_trajs[s, t-1, 0], window_trajs[s, t-1, 1],
                    window_trajs[s, t, 0], window_trajs[s, t, 1])
            expected_dist = mean_sog * (window_size * dt / 3600.0)
            if expected_dist > 0.1 and cumul_dist < expected_dist * freeze_ratio_thr:
                gps_freeze_found = True
                break
        if gps_freeze_found:
            continue

        adj = compute_mean_adjacency(obs, threshold_nm=config.get("scene_extraction", {}).get("interaction_distance_nm", 3.0) if config else 3.0)

        # Fix 5: 全零 adj_matrix 过滤
        if adj.max() < 1e-6:
            continue

        sample = {
            "obs": obs.astype(np.float32),
            "pred": pred.astype(np.float32),
            "adj_matrix": adj.astype(np.float32),
            "n_ships": np.int32(n_ships),
            "scene_id": np.int32(scene["scene_id"]),
        }

        if has_encounter_data:
            obs_dcpa = scene["dcpa_matrix"][:, :, start:start + obs_steps]
            obs_tcpa = scene["tcpa_matrix"][:, :, start:start + obs_steps]
            obs_cri = scene["cri_matrix"][:, :, start:start + obs_steps]
            obs_enc = scene["encounter_type"][:, :, start:start + obs_steps]

            sample["dcpa_matrix"] = np.min(obs_dcpa, axis=2).astype(np.float32)
            sample["cri_matrix"] = np.max(obs_cri, axis=2).astype(np.float32)
            sample["encounter_type"] = summarize_encounters(obs_enc).astype(np.int8)

            # TCPA: 取 DCPA 最小时刻对应的 TCPA
            sample["tcpa_matrix"] = np.zeros((n_ships, n_ships), dtype=np.float32)
            for ii in range(n_ships):
                for jj in range(n_ships):
                    min_t = np.argmin(obs_dcpa[ii, jj, :])
                    sample["tcpa_matrix"][ii, jj] = obs_tcpa[ii, jj, min_t]

            # Fix 2: obs 窗口无近距离交互 → encounter_type 置 safe
            obs_min_dcpa = sample["dcpa_matrix"].copy()
            np.fill_diagonal(obs_min_dcpa, np.inf)
            if n_ships > 1 and obs_min_dcpa.min() > dcpa_threshold:
                sample["encounter_type"] = np.zeros_like(sample["encounter_type"])

        # 不再对每船轮转生成样本，只保存 1 个样本，target 在训练时随机采样
        sample["target_ship_idx"] = np.int32(0)  # 占位，Dataloader 会随机替换
        samples.append(sample)

    return samples


def prepare_split_dir(output_dir, split_name):
    """流式写盘前清理 split 目录。返回已就绪的目录路径"""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for old_file in split_dir.glob("sample_*.npz"):
        old_file.unlink()
    old_index = split_dir / f"{split_name}_index.npz"
    if old_index.exists():
        old_index.unlink()
    return split_dir


def save_split_index(split_dir, metas, split_name):
    """根据已写入 sample 的 metadata 列表生成 index NPZ"""
    if not metas:
        return
    index = {
        "target_ship_idx": np.array([m["target_ship_idx"] for m in metas]),
        "scene_id": np.array([m["scene_id"] for m in metas]),
        "n_ships": np.array([m["n_ships"] for m in metas]),
        "n_samples": np.int32(len(metas)),
    }
    np.savez_compressed(split_dir / f"{split_name}_index.npz", **index)


def save_split(samples, output_dir, split_name):
    """保存一个 split 的所有样本（兼容入口；新流程请用 prepare_split_dir + 流式写盘 + save_split_index）"""
    split_dir = prepare_split_dir(output_dir, split_name)

    for i, sample in enumerate(samples):
        np.savez_compressed(split_dir / f"sample_{i:06d}.npz", **sample)

    metas = [
        {
            "target_ship_idx": int(s["target_ship_idx"]),
            "scene_id": int(s["scene_id"]),
            "n_ships": int(s["n_ships"]),
        }
        for s in samples
    ]
    save_split_index(split_dir, metas, split_name)
    return len(samples)


def generate_dataset():
    config = load_config()
    scenes_dir = get_data_path("processed", "scenes")
    final_dir = get_data_path("final")
    split_config = config["split"]

    print("=" * 60)
    print("[数据集生成] 时间分割 + 会遇过滤 + 滑动窗口")
    print("=" * 60)

    require_encounter = split_config.get("require_encounter", True)
    scenes = load_scenes(scenes_dir, config)
    if not scenes:
        print("  错误: 未找到场景文件")
        return False

    print(f"  加载 {len(scenes)} 个场景")

    time_ranges = build_temporal_split_map(config)
    print(f"  时间分割:")
    for ts_start, ts_end, name in time_ranges:
        d1 = datetime.fromtimestamp(ts_start, tz=timezone.utc).strftime("%Y-%m-%d")
        d2 = datetime.fromtimestamp(ts_end, tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"    {name}: {d1} ~ {d2}")

    split_scenes = {"train": [], "val": [], "test": []}
    unassigned = 0
    for scene in scenes:
        ts = scene["timestamps"][0]
        split_name = classify_scene_split(ts, time_ranges)
        if split_name:
            split_scenes[split_name].append(scene)
        else:
            unassigned += 1

    for name, sc_list in split_scenes.items():
        print(f"  {name}: {len(sc_list)} 场景")
    if unassigned:
        print(f"  未分配（时间范围外）: {unassigned} 场景")

    obs_steps = config["observation"]["steps"]

    # 生成多个预测窗口变体（pred10 + pred20）
    pred_variants = ["pred10", "pred20"]
    for variant_name in pred_variants:
        if variant_name not in config["prediction_variants"]:
            print(f"\n  跳过 {variant_name}（未在 config 中定义）")
            continue

        pred_config = config["prediction_variants"][variant_name]
        pred_steps = pred_config["steps"]
        stride = max(1, int(pred_steps * split_config.get("stride_ratio", 2.0)))
        max_samples_per_scene = split_config.get("max_samples_per_scene", 0)
        output_dir = final_dir / f"obs10_{variant_name}"

        print(f"\n{'─' * 40}")
        print(f"  变体: obs10_{variant_name}")
        print(f"  窗口参数: obs={obs_steps}, pred={pred_steps}, stride={stride}")
        if max_samples_per_scene > 0:
            print(f"  每场景样本上限: {max_samples_per_scene}")

        min_pred_mean_sog = split_config.get("min_pred_mean_sog", 0)
        min_obs_tail_sog = split_config.get("min_obs_tail_sog", 0)
        if min_pred_mean_sog > 0:
            print(f"  pred 窗口最低平均 SOG: {min_pred_mean_sog} kn")
        if min_obs_tail_sog > 0:
            print(f"  obs 窗口尾部最低平均 SOG: {min_obs_tail_sog} kn")

        for split_name in ["train", "val", "test"]:
            # 流式写盘：处理一个 scene 立即写入 sample，不再累积 all_samples
            # 内存峰值从 N_samples × 50KB（数 GB）降到单个 scene 的 samples（数 MB）
            split_dir = prepare_split_dir(output_dir, split_name)
            metas = []
            sample_idx = 0
            for scene in tqdm(split_scenes[split_name], desc=f"生成 {split_name} ({variant_name})"):
                samples = generate_samples_from_scene(scene, obs_steps, pred_steps, stride, min_pred_mean_sog, config)
                if max_samples_per_scene > 0 and len(samples) > max_samples_per_scene:
                    np.random.seed(scene["scene_id"])
                    indices = np.random.choice(len(samples), max_samples_per_scene, replace=False)
                    samples = [samples[i] for i in sorted(indices)]

                # 立即落盘 + 收集轻量 metadata（int），原始 sample 字典随循环结束被回收
                for s in samples:
                    np.savez_compressed(split_dir / f"sample_{sample_idx:06d}.npz", **s)
                    metas.append({
                        "target_ship_idx": int(s["target_ship_idx"]),
                        "scene_id": int(s["scene_id"]),
                        "n_ships": int(s["n_ships"]),
                    })
                    sample_idx += 1
                del samples

            # 该 split 全部 sample 落盘后写 index
            save_split_index(split_dir, metas, split_name)
            print(f"  {split_name}: {sample_idx} 样本")
            del metas
            gc.collect()

        print(f"\n  {'=' * 40}")
        for split_name in ["train", "val", "test"]:
            index_file = output_dir / split_name / f"{split_name}_index.npz"
            if index_file.exists():
                data = np.load(index_file)
                n = int(data["n_samples"])
                ns = data["n_ships"]
                print(f"  {split_name}: {n} 样本, ships=[{ns.min()}-{ns.max()}], mean={ns.mean():.1f}")

        print(f"  输出: {output_dir}")
    print("[数据集生成] 完成！")
    return True


if __name__ == "__main__":
    generate_dataset()