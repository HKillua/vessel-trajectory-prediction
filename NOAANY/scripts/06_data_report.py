"""
06_data_report.py - Generate quality statistics report for the processed dataset.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_config, vessel_type_name

ENCOUNTER_NAMES = {0: 'safe', 1: 'head_on', 2: 'crossing_give_way',
                   3: 'crossing_stand_on', 4: 'overtaking', 5: 'being_overtaken'}


def main():
    parser = argparse.ArgumentParser(description='Generate data quality report')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    config = load_config(base / args.config)
    scenes_dir = base / config['scenes']['scenes_dir']
    output_dir = base / config['splits']['output_dir']

    report_lines = ["# NOAA NY AIS Dataset Quality Report\n"]
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    scene_files = sorted(scenes_dir.glob('scene_*.npz'))
    report_lines.append(f"## Scenes\n")
    report_lines.append(f"- Total scenes: {len(scene_files)}")

    if scene_files:
        ships_counts = []
        durations = []
        min_dcpas = []
        enc_counter = Counter()

        for sf in scene_files:
            data = np.load(sf, allow_pickle=True)
            ships_counts.append(int(data.get('n_ships', 0)))
            durations.append(int(data.get('duration_sec', 0)))
            if 'min_dcpa' in data:
                md = data['min_dcpa']
                np.fill_diagonal(md, 999)
                min_dcpas.append(md.min())
            if 'encounter_type' in data:
                et = data['encounter_type']
                for i in range(et.shape[0]):
                    for j in range(i + 1, et.shape[1]):
                        enc_counter[int(et[i, j])] += 1

        report_lines.append(f"- Ships per scene: min={min(ships_counts)}, "
                          f"mean={np.mean(ships_counts):.1f}, max={max(ships_counts)}")
        report_lines.append(f"- Duration (min): min={min(durations)//60}, "
                          f"mean={np.mean(durations)/60:.1f}, max={max(durations)//60}")
        if min_dcpas:
            report_lines.append(f"- Min DCPA (nm): mean={np.mean(min_dcpas):.2f}, "
                              f"median={np.median(min_dcpas):.2f}, min={np.min(min_dcpas):.3f}")

        if enc_counter:
            report_lines.append(f"\n## Encounter Types\n")
            report_lines.append("| Type | Count | % |")
            report_lines.append("|------|-------|---|")
            total_enc = sum(enc_counter.values())
            for etype in sorted(enc_counter.keys()):
                name = ENCOUNTER_NAMES.get(etype, f'type_{etype}')
                count = enc_counter[etype]
                pct = count / total_enc * 100
                report_lines.append(f"| {name} | {count} | {pct:.1f}% |")

        if enc_counter:
            enc_cfg = config.get('encounters', {})
            dcpa_thr = enc_cfg.get('dcpa_threshold_nm', 1.0)
            tcpa_thr_min = enc_cfg.get('tcpa_threshold_min', 20)
            cri_thr = enc_cfg.get('cri_threshold', 0.5)

            report_lines.append(f"\n## Risk Assessment (threshold-based)\n")
            report_lines.append(f"- DCPA threshold: {dcpa_thr} nm | "
                              f"TCPA threshold: {tcpa_thr_min} min | "
                              f"CRI threshold: {cri_thr}")

            n_high_cri = 0
            n_close_dcpa = 0
            n_urgent_tcpa = 0
            n_total_pairs = 0

            for sf2 in scene_files:
                d2 = np.load(sf2, allow_pickle=True)
                if 'min_dcpa' not in d2 or 'max_cri' not in d2:
                    continue
                md2 = d2['min_dcpa']
                mc2 = d2['max_cri']
                n = md2.shape[0]
                for i in range(n):
                    for j in range(i + 1, n):
                        n_total_pairs += 1
                        if md2[i, j] < dcpa_thr:
                            n_close_dcpa += 1
                        if mc2[i, j] > cri_thr:
                            n_high_cri += 1
                if 'tcpa_matrix' in d2:
                    tm = d2['tcpa_matrix']
                    for i in range(n):
                        for j in range(i + 1, n):
                            min_tcpa = np.min(np.abs(tm[i, j]))
                            if min_tcpa < tcpa_thr_min * 60:
                                n_urgent_tcpa += 1

            if n_total_pairs > 0:
                report_lines.append(f"- Total ship pairs across all scenes: {n_total_pairs}")
                report_lines.append(f"- Close encounters (DCPA < {dcpa_thr}nm): "
                                  f"{n_close_dcpa} ({n_close_dcpa/n_total_pairs*100:.1f}%)")
                report_lines.append(f"- Urgent encounters (TCPA < {tcpa_thr_min}min): "
                                  f"{n_urgent_tcpa} ({n_urgent_tcpa/n_total_pairs*100:.1f}%)")
                report_lines.append(f"- High-risk encounters (CRI > {cri_thr}): "
                                  f"{n_high_cri} ({n_high_cri/n_total_pairs*100:.1f}%)")

    report_lines.append(f"\n## Samples\n")
    total_samples = 0
    for pv in config['splits']['pred_variants']:
        pred_dir = output_dir / f"pred{pv}"
        if not pred_dir.exists():
            continue
        for split in ['train', 'val', 'test']:
            split_dir = pred_dir / split
            if not split_dir.exists():
                continue
            samples = list(split_dir.glob('sample_*.npz'))
            count = len(samples)
            total_samples += count
            if count > 0:
                s = np.load(samples[0], allow_pickle=True)
                n_ships = int(s.get('n_ships', 0))
                obs_shape = s['obs'].shape if 'obs' in s else 'N/A'
                pred_shape = s['pred'].shape if 'pred' in s else 'N/A'
                report_lines.append(f"- pred{pv}/{split}: {count} samples "
                                  f"(obs={obs_shape}, pred={pred_shape})")
            else:
                report_lines.append(f"- pred{pv}/{split}: 0 samples")

    report_lines.append(f"\n- **Total samples: {total_samples}**")

    report_lines.append(f"\n## Config Summary\n")
    report_lines.append(f"- Region: {config['region']['name']}")
    bbox = config['region']['bbox']
    report_lines.append(f"- BBox: [{bbox['lat_min']},{bbox['lat_max']}] x "
                       f"[{bbox['lon_min']},{bbox['lon_max']}]")
    report_lines.append(f"- Sampling: {config['preprocess']['sampling_interval']}s")
    report_lines.append(f"- Obs/Pred: {config['splits']['obs_steps']} / "
                       f"{config['splits']['pred_variants']}")
    report_lines.append(f"- Scene radius: {config['scenes']['radius_nm']}nm (encounter-centric)")
    report_lines.append(f"- Min trajectory: {config['preprocess']['min_trajectory_steps']} steps")

    report_text = "\n".join(report_lines)
    report_path = base / "data" / "data_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
