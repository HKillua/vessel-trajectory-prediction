#!/bin/bash
# setup_baselines.sh - 一键克隆所有 baseline 依赖仓库
# 用法: cd baseline && bash setup_baselines.sh

set -e
echo "=== 克隆 Baseline 依赖仓库 ==="

# Social-STGCNN (图卷积基线 + 模型代码依赖)
if [ ! -d "social-stgcnn" ]; then
    echo "Cloning Social-STGCNN..."
    git clone https://github.com/abduallahmohamed/Social-STGCNN.git social-stgcnn
else
    echo "social-stgcnn already exists, skipping"
fi

# LLM4STP (GPT-2 基线 + 模型代码依赖)
if [ ! -d "LLM4STP" ]; then
    echo "Cloning LLM4STP..."
    git clone https://github.com/Joker-hang/LLM4STP.git LLM4STP
else
    echo "LLM4STP already exists, skipping"
fi

# iTransformer (变量级注意力基线)
if [ ! -d "itransformer" ]; then
    echo "Cloning iTransformer..."
    git clone https://github.com/thuml/iTransformer.git itransformer
else
    echo "itransformer already exists, skipping"
fi

# Maritime-Autonomy (LSTM/GRU/BiLSTM/BiGRU/Seq2Seq/Transformer 参考)
if [ ! -d "maritime-autonomy" ]; then
    echo "Cloning Maritime-Autonomy..."
    git clone https://github.com/Maritime-Autonomy/Multi-factor-influence-based-ship-trajectory-prediction-analysis-via-deep-learning.git maritime-autonomy
else
    echo "maritime-autonomy already exists, skipping"
fi

# Social-LSTM (社交池化基线)
if [ ! -d "social-lstm" ]; then
    echo "Cloning Social-LSTM..."
    git clone https://github.com/An-Yuhang-ace/MultiShipPrediction.git social-lstm
else
    echo "social-lstm already exists, skipping"
fi

# STAR (时空注意力基线 - 未使用, 备用)
if [ ! -d "star" ]; then
    echo "Cloning STAR..."
    git clone https://github.com/cunjunyu/STAR.git star
else
    echo "star already exists, skipping"
fi

# GPT-2 pretrained model (LLM4STP 依赖)
if [ ! -d "gpt2_model" ]; then
    echo "Downloading GPT-2 model..."
    mkdir -p gpt2_model
    # 使用 HuggingFace mirror (国内)
    export HF_ENDPOINT=https://hf-mirror.com
    python3 -c "
from transformers import GPT2Model, GPT2Tokenizer
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.save_pretrained('gpt2_model')
tokenizer.save_pretrained('gpt2_model')
print('GPT-2 model saved to gpt2_model/')
"
else
    echo "gpt2_model already exists, skipping"
fi

echo ""
echo "=== 所有依赖仓库克隆完成 ==="
echo "接下来可以运行 baseline 训练:"
echo "  python unified/train.py --model lstm --gpu 0"
echo "  python unified/train.py --model transformer --gpu 1"
echo "  ..."
