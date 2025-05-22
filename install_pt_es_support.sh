#!/bin/bash
# GPT-SoVITS 西班牙语和葡萄牙语支持安装脚本
# 版本: 2.0

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

# 检查Python版本
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo -e "${RED}错误: Python版本必须是3.8或更高, 当前版本: $PYTHON_VERSION${NC}"
    exit 1
fi

# 检查GPT-SoVITS目录结构
if [ ! -d "GPT_SoVITS" ]; then
    echo -e "${RED}错误: 未找到GPT_SoVITS目录，请确保在GPT-SoVITS项目根目录下运行此脚本${NC}"
    exit 1
fi

# 检查依赖
echo -e "${YELLOW}检查依赖...${NC}"
MISSING_DEPS=()

python -c "import torch" &> /dev/null || MISSING_DEPS+=("torch")
python -c "import numpy" &> /dev/null || MISSING_DEPS+=("numpy")
python -c "import librosa" &> /dev/null || MISSING_DEPS+=("librosa")
python -c "import soundfile" &> /dev/null || MISSING_DEPS+=("soundfile")
python -c "import requests" &> /dev/null || MISSING_DEPS+=("requests")
python -c "import tqdm" &> /dev/null || MISSING_DEPS+=("tqdm")
python -c "import matplotlib" &> /dev/null || MISSING_DEPS+=("matplotlib")
python -c "import scipy" &> /dev/null || MISSING_DEPS+=("scipy")
python -c "import transformers" &> /dev/null || MISSING_DEPS+=("transformers")

# 安装缺失的依赖
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}安装缺失的依赖: ${MISSING_DEPS[*]}${NC}"
    pip install ${MISSING_DEPS[*]} -q
    echo -e "${GREEN}依赖安装完成${NC}"
else
    echo -e "${GREEN}所有依赖已安装${NC}"
fi

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
mkdir -p text/LangSegmenter

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

# 检查HuBERT模型
echo -e "${YELLOW}检查HuBERT模型...${NC}"
if [ ! -d "GPT_SoVITS/pretrained_models/chinese-hubert-base" ]; then
    echo -e "${YELLOW}未找到HuBERT模型，请确保已下载chinese-hubert-base模型${NC}"
    echo -e "${YELLOW}这是一个必需的模型，通常在运行GPT-SoVITS的初始安装脚本时下载${NC}"
    echo -e "${YELLOW}请参考GPT-SoVITS的安装说明获取此模型${NC}"
fi

# 检查其他预训练模型
echo -e "${YELLOW}检查SoVITS预训练模型...${NC}"
if [ ! -f "GPT_SoVITS/pretrained_models/s2G488k.pth" ]; then
    echo -e "${YELLOW}未找到SoVITS预训练模型，请确保已下载s2G488k.pth模型${NC}"
    echo -e "${YELLOW}这是一个必需的模型，通常在运行GPT-SoVITS的初始安装脚本时下载${NC}"
    echo -e "${YELLOW}请参考GPT-SoVITS的安装说明获取此模型${NC}"
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
