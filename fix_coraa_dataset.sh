#!/bin/bash
# 快速修复CORAA数据集问题的脚本

echo "修复CORAA数据集问题..."
echo "创建示例葡萄牙语数据集..."

# 创建必要的目录
mkdir -p data/pt_corpus/raw
mkdir -p data/pt_corpus/processed
mkdir -p data/pt_corpus/prepared

# 创建示例数据集
python3 create_sample_data.py --output-dir ./data/pt_corpus/raw --language pt

# 创建示例音频文件
python3 create_sample_audio.py --metadata-file ./data/pt_corpus/raw/metadata.csv --output-dir ./data/pt_corpus/raw

echo "修复完成！"
echo "现在您可以继续运行训练流程："
echo "bash pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian --skip-download"
