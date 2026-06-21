#!/bin/bash
source /home/wangguangjie/anaconda3/etc/profile.d/conda.sh
conda activate ship-traj-pred
cd /home/wangguangjie/djs/vessel-trajectory-prediction/NOAANY

echo "========================================"
echo "  Step 02: Preprocessing"
echo "========================================"
python scripts/02_preprocess.py --config configs/config_noaa_ny.yaml 2>&1 | tee logs/02_preprocess.log

echo ""
echo "========================================"
echo "  Step 03: Extract Scenes"
echo "========================================"
python scripts/03_extract_scenes.py --config configs/config_noaa_ny.yaml 2>&1 | tee logs/03_extract_scenes.log

echo ""
echo "========================================"
echo "  Step 04: Compute Encounters"
echo "========================================"
python scripts/04_compute_encounters.py --config configs/config_noaa_ny.yaml 2>&1 | tee logs/04_compute_encounters.log

echo ""
echo "========================================"
echo "  Step 05: Generate Splits"
echo "========================================"
python scripts/05_generate_splits.py --config configs/config_noaa_ny.yaml 2>&1 | tee logs/05_generate_splits.log

echo ""
echo "========================================"
echo "  Step 06: Data Report"
echo "========================================"
python scripts/06_data_report.py --config configs/config_noaa_ny.yaml 2>&1 | tee logs/06_data_report.log

echo ""
echo "========================================"
echo "  Pipeline Complete!"
echo "========================================"
