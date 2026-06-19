#!/bin/bash
# Run LLM4STP baseline on GPU 3
set -e

PYTHON=~/anaconda3/envs/ship-traj-pred/bin/python
BASE_DIR=/home/wangguangjie/djs/baseline/LLM4STP
cd $BASE_DIR

export LLM4STP_GPU=3
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "============================================"
echo " LLM4STP Baseline Training"
echo " GPU: 3 | $(date)"
echo "============================================"

mkdir -p checkpoints_STPGeo temp_save_dir/ship_traj/train temp_save_dir/ship_traj/test temp_save_dir/ship_traj/val

$PYTHON mainSTP_Geo.py 2>&1 | tee llm4stp_train.log

echo ""
echo "=== Training Complete $(date) ==="
echo "Results in: checkpoints_STPGeo/"
echo "Log: llm4stp_train.log"
