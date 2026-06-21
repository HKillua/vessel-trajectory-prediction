"""
run_all.py - Full pipeline orchestration.
Runs steps 01-06 in sequence with --start-from support.
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path


STEPS = [
    ('01_download', '01_download.py', 'Download NOAA AIS data'),
    ('02_preprocess', '02_preprocess.py', 'Preprocess trajectories'),
    ('03_extract', '03_extract_scenes.py', 'Extract encounter-centric scenes'),
    ('04_encounters', '04_compute_encounters.py', 'Compute DCPA/TCPA/CRI'),
    ('05_splits', '05_generate_splits.py', 'Generate train/val/test samples'),
    ('06_report', '06_data_report.py', 'Generate data quality report'),
]


def main():
    parser = argparse.ArgumentParser(description='Run full NOAA NY pipeline')
    parser.add_argument('--config', default='configs/config_noaa_ny.yaml')
    parser.add_argument('--start-from', type=int, default=1,
                        help='Start from step N (1-6)')
    parser.add_argument('--single-day', type=str, default=None,
                        help='Pass --single-day to steps 02 and 03')
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent

    print("=" * 60)
    print("NOAA NY AIS Data Pipeline")
    print("=" * 60)

    for i, (name, script, desc) in enumerate(STEPS, 1):
        if i < args.start_from:
            print(f"\n[{i}/6] {desc} — SKIPPED")
            continue

        print(f"\n{'=' * 60}")
        print(f"[{i}/6] {desc}")
        print(f"{'=' * 60}")

        cmd = [sys.executable, str(scripts_dir / script), '--config', args.config]

        if args.single_day and script in ('02_preprocess.py', '03_extract_scenes.py'):
            cmd.extend(['--single-day', args.single_day])

        t0 = time.time()
        result = subprocess.run(cmd, cwd=str(scripts_dir.parent))
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\nStep {i} FAILED (exit code {result.returncode})")
            print("Pipeline stopped.")
            sys.exit(1)

        print(f"\nStep {i} completed in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
