#!/bin/bash
# 在 NOAANY 数据集上批量训练所有 baseline 模型
# 用法: bash run_noaa_baselines.sh [pred_variant]
# pred_variant: pred10 (默认), pred20, pred30

source /home/wangguangjie/anaconda3/etc/profile.d/conda.sh
conda activate ship-traj-pred

PRED_VARIANT=${1:-pred10}
DATA_ROOT="/home/wangguangjie/djs/vessel-trajectory-prediction/NOAANY/data/final/${PRED_VARIANT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/../results/NOAANY/obs30_${PRED_VARIANT}"
LOG_DIR="${SCRIPT_DIR}/../logs/noaa_obs30_${PRED_VARIANT}"

mkdir -p "$RESULTS_BASE" "$LOG_DIR"

echo "=============================================="
echo "  NOAANY Baseline Training: ${PRED_VARIANT}"
echo "  Data: ${DATA_ROOT}"
echo "  Results: ${RESULTS_BASE}"
echo "=============================================="

# 模型列表 (10 个主模型)
MODELS=(lstm gru bilstm bigru seq2seq_lstm seq2seq_gru transformer mamba itransformer social_lstm)

# GPU 分配 (4 GPU, 循环分配)
GPUS=(0 1 2 3)

PIDS=()
for i in "${!MODELS[@]}"; do
    model=${MODELS[$i]}
    gpu=${GPUS[$((i % 4))]}
    results_dir="${RESULTS_BASE}/${model}"

    echo "[${model}] Starting on GPU ${gpu}..."
    python -u train.py \
        --model "$model" \
        --data_root "$DATA_ROOT" \
        --results_dir "$RESULTS_BASE" \
        --gpu "$gpu" \
        --epochs 100 \
        --batch_size 128 \
        --lr 1e-3 \
        --patience 20 \
        --num_workers 4 \
        2>&1 | tee "${LOG_DIR}/${model}.log" &
    PIDS+=($!)

    # 每4个模型等一波 (避免GPU抢占)
    if (( (i + 1) % 4 == 0 )); then
        echo "Waiting for batch $((i/4 + 1)) to finish..."
        for pid in "${PIDS[@]}"; do
            wait $pid
        done
        PIDS=()
    fi
done

# 等待剩余任务
for pid in "${PIDS[@]}"; do
    wait $pid
done

echo ""
echo "=============================================="
echo "  All 10 models training complete!"
echo "  Running Social-STGCNN..."
echo "=============================================="

# Social-STGCNN (独立脚本)
python -u train_social_stgcnn.py \
    --data_root "$DATA_ROOT" \
    --gpu 0 \
    2>&1 | tee "${LOG_DIR}/social_stgcnn.log"

echo ""
echo "=============================================="
echo "  All baselines done! Summary:"
echo "=============================================="

for model in "${MODELS[@]}" social_stgcnn; do
    rfile="${RESULTS_BASE}/${model}/results.json"
    if [ -f "$rfile" ]; then
        python3 -c "
import json
with open('$rfile') as f:
    r = json.load(f)
m = r.get('model', '$model')
ade = r.get('test_ade_nm', -1)
fde = r.get('test_fde_nm', -1)
print(f'  {m:20s}  ADE={ade:.4f}nm  FDE={fde:.4f}nm')
"
    else
        echo "  ${model}: results NOT found"
    fi
done
