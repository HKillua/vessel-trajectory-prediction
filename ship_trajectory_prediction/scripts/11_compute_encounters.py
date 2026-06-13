"""计算会遇特征: DCPA/TCPA/CRI + COLREGs 会遇分类

在 07_extract_scenes.py 输出的场景 NPZ 基础上，为每对船舶每个时间步计算：
- DCPA (Distance to Closest Point of Approach, nm)
- TCPA (Time to CPA, seconds)
- CRI (Collision Risk Index)
- encounter_type (COLREGs Rule 13/14/15 会遇分类)

输出：增强的 NPZ 文件，新增上述矩阵字段
"""

import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from utils import load_config, get_data_path, haversine_distance_nm

KNOTS_TO_MS = 0.514444
NM_TO_M = 1852.0

DCPA_CLAMP_NM = 10.0  # DCPA 上限：超过 10nm 视为无交互风险

ENCOUNTER_SAFE = 0
ENCOUNTER_HEAD_ON = 1
ENCOUNTER_CROSSING_GIVE_WAY = 2
ENCOUNTER_CROSSING_STAND_ON = 3
ENCOUNTER_OVERTAKING = 4
ENCOUNTER_BEING_OVERTAKEN = 5


def compute_dcpa_tcpa(lat1, lon1, sog1, cog1, lat2, lon2, sog2, cog2):
    """计算两船的 DCPA 和 TCPA
    
    Args:
        lat1, lon1: 本船位置 (度)
        sog1, cog1: 本船 SOG(knots) 和 COG(度)
        lat2, lon2: 目标船位置 (度)
        sog2, cog2: 目标船 SOG(knots) 和 COG(度)
        
    Returns:
        dcpa_nm: DCPA (海里)
        tcpa_sec: TCPA (秒), 负值表示已过 CPA
    """
    cos_lat = np.cos(np.radians((lat1 + lat2) / 2.0))
    dx = (lon2 - lon1) * cos_lat * 60.0
    dy = (lat2 - lat1) * 60.0
    
    vx1 = sog1 * np.sin(np.radians(cog1))
    vy1 = sog1 * np.cos(np.radians(cog1))
    vx2 = sog2 * np.sin(np.radians(cog2))
    vy2 = sog2 * np.cos(np.radians(cog2))
    
    dvx = vx2 - vx1
    dvy = vy2 - vy1
    
    dv_sq = dvx * dvx + dvy * dvy
    if dv_sq < 1e-10:
        # 相对速度近似为零 (平行同速), DCPA = 当前距离
        dcpa_nm = np.sqrt(dx * dx + dy * dy)
        return min(dcpa_nm, DCPA_CLAMP_NM), 0.0
        
    tcpa_hr = -(dx * dvx + dy * dvy) / dv_sq
    tcpa_sec = tcpa_hr * 3600.0
    
    dcpa_x = dx + dvx * tcpa_hr
    dcpa_y = dy + dvy * tcpa_hr
    dcpa_nm = np.sqrt(dcpa_x * dcpa_x + dcpa_y * dcpa_y)
    
    # Clamp DCPA to prevent inf/EXTREME values
    dcpa_nm = min(dcpa_nm, DCPA_CLAMP_NM)
    
    return dcpa_nm, tcpa_sec


def compute_cri(dcpa_nm, tcpa_sec, d0_nm=1.0, t0_sec=600.0):
    """CRI = exp(-DCPA/d0) * exp(-|TCPA|/t0)

    TCPA < 0 (已过 CPA) 时保留空间风险分量并施加时间衰减，
    而非直接返回 0, 避免丢失远距离离航时的风险信号。
    """
    spatial_risk = np.exp(-dcpa_nm / d0_nm)
    if tcpa_sec < 0:
        return spatial_risk * np.exp(tcpa_sec / t0_sec)
    return spatial_risk * np.exp(-tcpa_sec / t0_sec)


def compute_bearing(lat1, lon1, lat2, lon2):
    """计算从船1到船2的方位角 (度, 0-360 北为0顺时针)"""
    cos_lat = np.cos(np.radians((lat1 + lat2) / 2.0))
    dx = (lon2 - lon1) * cos_lat
    dy = lat2 - lat1
    bearing = np.degrees(np.arctan2(dx, dy)) % 360
    return bearing


def classify_encounter_pair(bearing_ij, bearing_ji, heading_i, heading_j,
                            dcpa_nm, dcpa_threshold=2.0, sog_i=None, sog_j=None,
                            head_on_min_diff=170.0, ahead_sector=15.0,
                            stern_half_arc=112.5):
    """根据 COLREGs Rule 13/14/15 成对分类会遇类型
    
    Args:
        head_on_min_diff: 对遇航向差下限 (默认 170° 即 ±10°)
        ahead_sector: 对遇前方扇区半角 (默认 15° 即 ±15°)
        stern_half_arc: 船尾扇区半角 (默认 112.5° COLREGs Rule 13 定义)
        
    返回 (enc_ij, enc_ji) 元组, 保证成对一致性:
    - head_on: (1, 1) 对称
    - crossing: (2, 3) 或 (3, 2) 互补
    - overtaking: (4, 5) 或 (5, 4) 追越/被追越
    - safe: (0, 0)
    """
    SAFE = (ENCOUNTER_SAFE, ENCOUNTER_SAFE)
    
    if dcpa_nm > dcpa_threshold:
        return SAFE
        
    if sog_i is not None and sog_j is not None:
        if sog_i < 0.5 and sog_j < 0.5:
            return SAFE
            
    heading_diff = abs((heading_j - heading_i + 180) % 360 - 180)
    
    rel_bearing_ij = (bearing_ij - heading_i) % 360
    rel_bearing_ji = (bearing_ji - heading_j) % 360
    
    i_sees_j_ahead = rel_bearing_ij < ahead_sector or rel_bearing_ij > (360 - ahead_sector)
    j_sees_i_ahead = rel_bearing_ji < ahead_sector or rel_bearing_ji > (360 - ahead_sector)
    
    # Rule 14: 对遇 - 航向近乎相反且互相正前方
    if heading_diff > head_on_min_diff and i_sees_j_ahead and j_sees_i_ahead:
        return (ENCOUNTER_HEAD_ON, ENCOUNTER_HEAD_ON)
        
    stern_lo = stern_half_arc
    stern_hi = 360.0 - stern_half_arc
    i_in_j_stern = stern_lo < rel_bearing_ji < stern_hi
    j_in_i_stern = stern_lo < rel_bearing_ij < stern_hi
    
    # Rule 13: 追越 - 从另一船船尾扇区接近, 航向差 < 90°
    if heading_diff < 90.0:
        if i_in_j_stern and not j_in_i_stern:
            return (ENCOUNTER_BEING_OVERTAKEN, ENCOUNTER_OVERTAKING)
        if j_in_i_stern and not i_in_j_stern:
            return (ENCOUNTER_OVERTAKING, ENCOUNTER_BEING_OVERTAKEN)
        if i_in_j_stern and j_in_i_stern:
            if sog_i is not None and sog_j is not None and abs(sog_i - sog_j) > 1.0:
                if sog_i > sog_j:
                    return (ENCOUNTER_OVERTAKING, ENCOUNTER_BEING_OVERTAKEN)
                else:
                    return (ENCOUNTER_BEING_OVERTAKEN, ENCOUNTER_OVERTAKING)
                
    # Rule 15: 交叉 - 按右舷优先原则判定让路/直航
    if 0 < rel_bearing_ij <= stern_lo:
        return (ENCOUNTER_CROSSING_GIVE_WAY, ENCOUNTER_CROSSING_STAND_ON)
    if stern_hi <= rel_bearing_ij < 360:
        return (ENCOUNTER_CROSSING_STAND_ON, ENCOUNTER_CROSSING_GIVE_WAY)
        
    # rel_bearing_ij == 0 (正前方) 但不满足对遇条件 -> safe
    if rel_bearing_ij == 0:
        return SAFE
        
    # 船尾扇区但 heading_diff >= 90° -> 仍视为交叉
    if rel_bearing_ij <= 180:
        return (ENCOUNTER_CROSSING_GIVE_WAY, ENCOUNTER_CROSSING_STAND_ON)
    else:
        return (ENCOUNTER_CROSSING_STAND_ON, ENCOUNTER_CROSSING_GIVE_WAY)


def process_scene(npz_path, config):
    """为一个场景计算所有会遇特征"""
    enc_config = config["encounter"]
    dcpa_threshold = enc_config["dcpa_threshold_nm"]
    tcpa_max = enc_config["tcpa_max_sec"]
    cri_d0 = enc_config["cri_d0_nm"]
    cri_t0 = enc_config["cri_t0_sec"]
    head_on_range = enc_config.get("head_on_angle_range", [170, 190])
    head_on_min_diff = head_on_range[0]
    ahead_sector = enc_config.get("ahead_sector_deg", 15.0)
    stern_half_arc = enc_config.get("overtaking_bearing_threshold", 112.5)
    
    data = np.load(npz_path, allow_pickle=True)
    trajs = data["trajectories"]
    n_ships, n_steps, n_feat = trajs.shape
    
    dcpa_matrix = np.full((n_ships, n_ships, n_steps), DCPA_CLAMP_NM, dtype=np.float32)
    tcpa_matrix = np.zeros((n_ships, n_ships, n_steps), dtype=np.float32)
    cri_matrix = np.zeros((n_ships, n_ships, n_steps), dtype=np.float32)
    encounter_matrix = np.zeros((n_ships, n_ships, n_steps), dtype=np.int8)
    
    # 自身配对: DCPA=0, CRI=0 (而非 DCPA_CLAMP_NM)
    for i in range(n_ships):
        dcpa_matrix[i, i, :] = 0.0
        
    feat_idx = {"lat": 0, "lon": 1, "sog": 2, "cog": 3, "heading": 4}
    
    for t in range(n_steps):
        for i in range(n_ships):
            for j in range(i + 1, n_ships):
                lat_i = trajs[i, t, feat_idx["lat"]]
                lon_i = trajs[i, t, feat_idx["lon"]]
                sog_i = trajs[i, t, feat_idx["sog"]]
                cog_i = trajs[i, t, feat_idx["cog"]]
                hdg_i = trajs[i, t, feat_idx["heading"]]
                
                lat_j = trajs[j, t, feat_idx["lat"]]
                lon_j = trajs[j, t, feat_idx["lon"]]
                sog_j = trajs[j, t, feat_idx["sog"]]
                cog_j = trajs[j, t, feat_idx["cog"]]
                hdg_j = trajs[j, t, feat_idx["heading"]]
                
                dcpa, tcpa = compute_dcpa_tcpa(
                    lat_i, lon_i, sog_i, cog_i,
                    lat_j, lon_j, sog_j, cog_j
                )               
                tcpa = np.clip(tcpa, -tcpa_max, tcpa_max)
                cri = compute_cri(dcpa, tcpa, cri_d0, cri_t0)
                
                dcpa_matrix[i, j, t] = dcpa
                dcpa_matrix[j, i, t] = dcpa
                tcpa_matrix[i, j, t] = tcpa
                tcpa_matrix[j, i, t] = tcpa
                cri_matrix[i, j, t] = cri
                cri_matrix[j, i, t] = cri
                
                bearing_ij = compute_bearing(lat_i, lon_i, lat_j, lon_j)
                bearing_ji = compute_bearing(lat_j, lon_j, lat_i, lon_i)
                enc_ij, enc_ji = classify_encounter_pair(
                    bearing_ij, bearing_ji, hdg_i, hdg_j,
                    dcpa, dcpa_threshold, sog_i, sog_j,
                    head_on_min_diff, ahead_sector, stern_half_arc
                )                
                encounter_matrix[i, j, t] = enc_ij
                encounter_matrix[j, i, t] = enc_ji
                
    off_diag_mask = ~np.eye(n_ships, dtype=bool)
    off_diag_dcpa = dcpa_matrix[off_diag_mask]
    min_dcpa = float(np.min(off_diag_dcpa)) if off_diag_dcpa.size > 0 else np.inf
    has_encounter = min_dcpa < dcpa_threshold
    
    save_dict = {k: data[k] for k in data.files}
    save_dict.update({
        "dcpa_matrix": dcpa_matrix,
        "tcpa_matrix": tcpa_matrix,
        "cri_matrix": cri_matrix,
        "encounter_type": encounter_matrix,
        "has_encounter": has_encounter,
        "min_dcpa": min_dcpa,
    })
    
    np.savez_compressed(npz_path, **save_dict)
    return has_encounter, min_dcpa


def compute_all_encounters():
    config = load_config()
    scene_dir = get_data_path("processed", "scenes")
    
    print("=" * 60)
    print("[会遇计算] DCPA/TCPA/CRI + COLREGs 会遇分类")
    print("=" * 60)
    
    npz_files = sorted(scene_dir.glob("scene_*.npz"))
    if not npz_files:
        print(" 错误：未找到场景文件，请先运行 07_extract_scenes.py")
        return False
        
    print(f" 找到 {len(npz_files)} 个场景文件")
    
    encounter_count = 0
    enc_type_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    min_dcpa_list = []
    
    for npz_path in tqdm(npz_files, desc="计算会遇特征"):
        has_enc, min_dcpa = process_scene(npz_path, config)
        if has_enc:
            encounter_count += 1
        min_dcpa_list.append(min_dcpa)
        
        data = np.load(npz_path)
        enc_types = data["encounter_type"]
        for et in range(6):
            enc_type_counts[et] += np.sum(enc_types == et)
            
    enc_names = ["safe", "head_on", "crossing_give_way", "crossing_stand_on", "overtaking", "being_overtaken"]
    total_pairs = sum(enc_type_counts.values())
    
    print(f"\n" + "=" * 60)
    print(f" 会遇场景数: {encounter_count}/{len(npz_files)} ({encounter_count/max(len(npz_files),1)*100:.1f}%)")
    print(f" 最小 DCPA 分布: mean={np.mean(min_dcpa_list):.2f}nm, "
          f"median={np.median(min_dcpa_list):.2f}nm, min={np.min(min_dcpa_list):.3f}nm")
    print(f"\n 会遇类型分布:")
    for et, name in enumerate(enc_names):
        pct = enc_type_counts[et] / max(total_pairs, 1) * 100
        print(f"  {name}: {enc_type_counts[et]:,} ({pct:.1f}%)")
        
    print(f"\n[会遇计算] 完成！")
    return True


if __name__ == "__main__":
    compute_all_encounters()