#!/bin/bash
# 快速测试脚本 - 测试简化版本的葡萄牙语训练流程

echo "开始快速测试..."

# 设置权限
chmod +x preprocess_data_simple.py
chmod +x train_gpt_simple.py
chmod +x train_sovits_simple.py

# 检查是否有示例数据
if [ ! -f "data/pt_corpus/raw/metadata.csv" ]; then
    echo "创建示例数据..."
    python3 create_sample_data.py --output-dir ./data/pt_corpus/raw --language pt
    python3 create_sample_audio.py --metadata-file ./data/pt_corpus/raw/metadata.csv --output-dir ./data/pt_corpus/raw
fi

echo "运行简化版本的训练流程..."

# 使用简化版本运行测试
./pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian --skip-download --batch-size 2 --max-epochs 2

echo "测试完成！"
