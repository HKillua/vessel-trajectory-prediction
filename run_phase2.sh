#!/bin/bash
set -e
PYTHON=~/anaconda3/envs/ship-traj-pred/bin/python
REPO_DIR=/home/wangguangjie/djs/vessel-trajectory-prediction
cd $REPO_DIR
LOG_DIR=results/transformer_ecr

echo "=== Phase 2: End-to-End Training ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -m mambadiff.train \
    --cfg mambadiff/configs/mambadiff_ecr.yaml \
    --device cuda:0 \
    --phase train \
    --pretrained $LOG_DIR/checkpoint_phase1_best.pt \
    --log_dir $LOG_DIR \
    --no_report 2>&1 | tee -a $LOG_DIR/full_run.log

echo ""
echo "=== Testing ==="
CUDA_VISIBLE_DEVICES=0 $PYTHON -m mambadiff.train \
    --cfg mambadiff/configs/mambadiff_ecr.yaml \
    --device cuda:0 \
    --phase test \
    --checkpoint $LOG_DIR/checkpoint_best.pt \
    --log_dir $LOG_DIR \
    --guidance_scale 0.5 \
    --no_report 2>&1 | tee -a $LOG_DIR/full_run.log

echo ""
echo "=== ALL DONE $(date) ==="
