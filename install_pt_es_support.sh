#!/bin/bash
# GPT-SoVITS 西班牙语和葡萄牙语支持安装脚本
# 作者: Augment Agent
# 版本: 1.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

echo -e "${BLUE}========== GPT-SoVITS 西班牙语和葡萄牙语支持安装 ==========${NC}"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python命令${NC}"
    exit 1
fi

# 检查GPT-SoVITS目录结构
if [ ! -d "GPT_SoVITS" ]; then
    echo -e "${RED}错误: 未找到GPT_SoVITS目录，请确保在GPT-SoVITS项目根目录下运行此脚本${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
python -c "import torch" || { echo -e "${RED}错误: 未安装PyTorch${NC}"; exit 1; }
python -c "import numpy" || { echo -e "${RED}错误: 未安装NumPy${NC}"; exit 1; }
python -c "import librosa" || { echo -e "${YELLOW}警告: 未安装librosa，将尝试安装${NC}"; pip install librosa; }
python -c "import soundfile" || { echo -e "${YELLOW}警告: 未安装soundfile，将尝试安装${NC}"; pip install soundfile; }
python -c "import requests" || { echo -e "${YELLOW}警告: 未安装requests，将尝试安装${NC}"; pip install requests; }
python -c "import tqdm" || { echo -e "${YELLOW}警告: 未安装tqdm，将尝试安装${NC}"; pip install tqdm; }

# 创建必要的目录
echo -e "${YELLOW}创建必要的目录...${NC}"
mkdir -p data/es_corpus/raw
mkdir -p data/es_corpus/processed
mkdir -p data/es_corpus/prepared
mkdir -p data/pt_corpus/raw
mkdir -p data/pt_corpus/processed
mkdir -p data/pt_corpus/prepared
mkdir -p checkpoints/es_gpt
mkdir -p checkpoints/es_sovits
mkdir -p checkpoints/pt_gpt
mkdir -p checkpoints/pt_sovits
mkdir -p pretrained_models
mkdir -p logs

# 设置文件权限
echo -e "${YELLOW}设置文件权限...${NC}"
chmod +x pt_es_pipeline.sh
chmod +x download_es_data.py
chmod +x download_pt_data.py
chmod +x download_bert_model.py
chmod +x preprocess_data.py
chmod +x train_gpt_model.py
chmod +x train_sovits_model.py
chmod +x inference.py
chmod +x test_pt_es_support.py

# 检查BERT模型
echo -e "${YELLOW}检查BERT模型...${NC}"
if [ ! -d "pretrained_models/spanish-bert" ]; then
    echo -e "${YELLOW}未找到西班牙语BERT模型，将尝试下载...${NC}"
    python download_bert_model.py --language es
else
    echo -e "${GREEN}已找到西班牙语BERT模型${NC}"
fi

if [ ! -d "pretrained_models/portuguese-bert" ]; then
    echo -e "${YELLOW}未找到葡萄牙语BERT模型，将尝试下载...${NC}"
    python download_bert_model.py --language pt
else
    echo -e "${GREEN}已找到葡萄牙语BERT模型${NC}"
fi

# 运行测试
echo -e "${YELLOW}运行测试...${NC}"
python test_pt_es_support.py --test-compatibility

echo -e "${GREEN}安装完成！${NC}"
echo -e "${YELLOW}使用以下命令开始训练流程:${NC}"
echo -e "bash pt_es_pipeline.sh --language es --dataset common_voice"
echo -e "或"
echo -e "bash pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian"
echo ""
echo -e "${YELLOW}查看README_PT_ES.md获取更多信息${NC}"
