#!/bin/bash
# Run Transformer-based MambaDiff-ECR training
set -e

PYTHON=~/anaconda3/envs/ship-traj-pred/bin/python
REPO_DIR=/home/wangguangjie/djs/vessel-trajectory-prediction
cd $REPO_DIR

GPU=${1:-0}
LOG_DIR=results/transformer_ecr

mkdir -p $LOG_DIR

echo "============================================"
echo " Transformer MambaDiff-ECR Training"
echo " GPU: $GPU | $(date)"
echo "============================================"

# Phase 1: Pretrain denoiser (50 epochs)
echo "=== Phase 1: Pretrain Denoiser ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mambadiff.train \
    --cfg mambadiff/configs/mambadiff_ecr.yaml \
    --device cuda:0 \
    --phase pretrain \
    --log_dir $LOG_DIR \
    --pretrain_epochs 50 \
    --no_report 2>&1 | tee $LOG_DIR/full_run.log

# Phase 2: End-to-end training
echo ""
echo "=== Phase 2: End-to-End Training ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mambadiff.train \
    --cfg mambadiff/configs/mambadiff_ecr.yaml \
    --device cuda:0 \
    --phase train \
    --pretrained $LOG_DIR/checkpoint_phase1_best.pt \
    --log_dir $LOG_DIR \
    --no_report 2>&1 | tee -a $LOG_DIR/full_run.log

# Test
echo ""
echo "=== Testing ==="
CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m mambadiff.train \
    --cfg mambadiff/configs/mambadiff_ecr.yaml \
    --device cuda:0 \
    --phase test \
    --checkpoint $LOG_DIR/checkpoint_best.pt \
    --log_dir $LOG_DIR \
    --guidance_scale 0.5 \
    --no_report 2>&1 | tee -a $LOG_DIR/full_run.log

echo ""
echo "=== ALL DONE $(date) ==="
