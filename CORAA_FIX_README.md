# CORAA数据集问题解决方案

## 问题描述
CORAA数据集的GitHub链接只包含项目代码，不包含实际的音频数据。真实的CORAA数据集需要从官方网站单独申请和下载。

## 快速解决方案

### 方法1: 使用修复脚本（推荐）
```bash
# 运行修复脚本
./fix_coraa_dataset.sh

# 然后继续训练流程（跳过下载步骤）
./pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian --skip-download
```

### 方法2: 手动修复
```bash
# 1. 创建示例数据集
python3 create_sample_data.py --output-dir ./data/pt_corpus/raw --language pt

# 2. 创建示例音频文件
python3 create_sample_audio.py --metadata-file ./data/pt_corpus/raw/metadata.csv --output-dir ./data/pt_corpus/raw

# 3. 继续训练流程
./pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian --skip-download
```

### 方法3: 使用其他数据集
```bash
# 使用MLS数据集（推荐用于实际训练）
./pt_es_pipeline.sh --language pt --dataset mls --dialect brazilian

# 使用Common Voice数据集
./pt_es_pipeline.sh --language pt --dataset common_voice --dialect brazilian
```

## 关于示例数据集

创建的示例数据集包含：
- 10个葡萄牙语文本样本
- 对应的合成音频文件（正弦波，不同频率代表不同说话者）
- 正确格式的metadata.csv文件

**注意：**
- 示例数据集仅用于测试流程，不适合实际的语音合成训练
- 如需进行真实训练，请：
  1. 使用MLS或Common Voice数据集
  2. 或获取真实的CORAA数据集
  3. 或提供您自己的音频数据

## 数据集说明

### MLS数据集（推荐）
- 完全开放，自动下载和处理
- 高质量的有声读物录音
- 适合训练语音合成模型

### Common Voice数据集
- Mozilla开源项目
- 包含多种方言的葡萄牙语
- 社区贡献的录音

### CORAA数据集
- 巴西葡萄牙语专用数据集
- 需要从官方网站申请
- 包含日常对话和演讲

## 故障排除

如果遇到其他问题，请检查：
1. Python环境是否正确安装
2. 所需的Python包是否已安装
3. 文件权限是否正确
4. 磁盘空间是否充足

## 联系和支持

如果您需要使用真实的CORAA数据集，请：
1. 访问CORAA官方网站
2. 按照要求申请数据集访问权限
3. 下载后替换示例数据集
