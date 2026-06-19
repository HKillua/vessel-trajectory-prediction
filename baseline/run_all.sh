#!/bin/bash
# 统一 Baseline 训练启动脚本
# 用法: bash run_all.sh [gpu_id]

set -e
PYTHON=~/anaconda3/envs/ship-traj-pred/bin/python
GPU=${1:-1}
EPOCHS=${2:-100}
BATCH=${3:-128}

echo "=========================================="
echo "  Baseline Training Pipeline"
echo "  GPU: $GPU | Epochs: $EPOCHS | Batch: $BATCH"
echo "=========================================="

cd "$(dirname "$0")"

# --- 简单模型 (单船，无交互) ---
SIMPLE_MODELS="lstm gru bilstm bigru seq2seq_lstm seq2seq_gru transformer mamba itransformer"

for MODEL in $SIMPLE_MODELS; do
    echo ""
    echo ">>> Training $MODEL on GPU $GPU ..."
    $PYTHON -m unified.train \
        --model $MODEL --gpu $GPU \
        --epochs $EPOCHS --batch_size $BATCH \
        --lr 1e-3 --patience 20 \
        2>&1 | tee "../results/${MODEL}.log"
done

# --- 交互模型 (多船) ---
# Social-LSTM
echo ""
echo ">>> Training social_lstm on GPU $GPU ..."
$PYTHON -m unified.train \
    --model social_lstm --gpu $GPU \
    --epochs $EPOCHS --batch_size 64 \
    --lr 1e-3 --patience 20 \
    2>&1 | tee "../results/social_lstm.log"

echo ""
echo "=========================================="
echo "  All baselines done!"
echo "=========================================="

# 汇总结果
echo ""
echo "=== Results Summary ==="
for f in ../results/*/results.json; do
    if [ -f "$f" ]; then
        model=$(basename $(dirname "$f"))
        ade=$($PYTHON -c "import json; d=json.load(open('$f')); print(f'{d[\"test_ade_nm\"]:.4f}')")
        fde=$($PYTHON -c "import json; d=json.load(open('$f')); print(f'{d[\"test_fde_nm\"]:.4f}')")
        echo "  $model: ADE=${ade}nm  FDE=${fde}nm"
    fi
done
