# GPT-SoVITS 西班牙语和葡萄牙语支持

本项目提供了一套完整的脚本，用于自动化GPT-SoVITS西班牙语和葡萄牙语的数据下载、处理、训练和推理流程。

## 目录

1. [简介](#简介)
2. [环境准备](#环境准备)
3. [脚本说明](#脚本说明)
4. [使用流程](#使用流程)
5. [数据集](#数据集)
6. [训练参数](#训练参数)
7. [推理](#推理)
8. [常见问题](#常见问题)

## 简介

GPT-SoVITS是一个先进的文本到语音(TTS)系统，结合了GPT（生成式预训练变换器）和SoVITS（声音向量化即时文本到语音）技术。本项目扩展了GPT-SoVITS的功能，提供了对西班牙语和葡萄牙语的完整支持，包括数据下载、预处理、模型训练和推理。

### 主要特点

- **零样本TTS**：只需5秒的声音样本，即可实现文本到语音转换
- **少样本TTS**：仅需1分钟的训练数据即可微调模型，提升声音相似度和真实感
- **跨语言支持**：支持与训练数据集不同语言的推理
- **多语言混合**：支持在同一文本中混合使用多种语言

## 环境准备

### 系统要求

- **操作系统**：Windows 10/11、Linux或macOS
- **CPU训练**：至少16GB RAM
- **GPU训练**：
  - NVIDIA GPU，至少8GB显存
  - CUDA 11.7或更高版本
  - cuDNN 8.0或更高版本

### 安装依赖

确保已安装GPT-SoVITS的所有依赖项：

```bash
pip install -r requirements.txt
```

此外，还需要安装以下依赖：

```bash
pip install requests tqdm librosa soundfile
```

## 脚本说明

本项目包含以下脚本：

1. **pt_es_pipeline.sh**：主控脚本，控制整个流程的执行
2. **download_es_data.py**：下载西班牙语数据集
3. **download_pt_data.py**：下载葡萄牙语数据集
4. **download_bert_model.py**：下载西班牙语和葡萄牙语BERT模型
5. **preprocess_data.py**：数据预处理脚本
6. **train_gpt_model.py**：GPT模型训练脚本
7. **train_sovits_model.py**：SoVITS模型训练脚本
8. **inference.py**：推理脚本
9. **test_pt_es_support.py**：测试脚本

## 使用流程

### 1. 运行主控脚本

最简单的方式是使用主控脚本，它会自动执行整个流程：

```bash
bash pt_es_pipeline.sh --language es --dataset common_voice
```

或者：

```bash
bash pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian
```

### 2. 分步执行

如果您想分步执行流程，可以按照以下步骤操作：

#### 2.1 下载数据

```bash
# 下载西班牙语数据
python download_es_data.py --dataset-type common_voice --data-dir ./data

# 下载葡萄牙语数据
python download_pt_data.py --dataset-type coraa --data-dir ./data --dialect brazilian
```

#### 2.2 下载BERT模型

```bash
# 下载西班牙语BERT模型
python download_bert_model.py --language es

# 下载葡萄牙语BERT模型
python download_bert_model.py --language pt
```

#### 2.3 预处理数据

```bash
# 预处理西班牙语数据
python preprocess_data.py --language es --data-dir ./data --sample-rate 24000

# 预处理葡萄牙语数据
python preprocess_data.py --language pt --data-dir ./data --sample-rate 24000 --dialect brazilian
```

#### 2.4 训练模型

```bash
# 训练西班牙语GPT模型
python train_gpt_model.py --language es --data-dir ./data/es_corpus/prepared --batch-size 16 --max-epochs 100

# 训练西班牙语SoVITS模型
python train_sovits_model.py --language es --data-dir ./data/es_corpus/prepared --batch-size 16 --max-epochs 100
```

#### 2.5 测试支持

```bash
# 测试西班牙语和葡萄牙语支持
python test_pt_es_support.py --test-all
```

## 数据集

本项目支持以下数据集：

### 西班牙语数据集

- **Common Voice**：Mozilla的多语言语音数据集
- **CIEMPIESS**：墨西哥西班牙语数据集
- **MLS**：Multilingual LibriSpeech西班牙语部分
- **CSS10**：CSS10西班牙语数据集

### 葡萄牙语数据集

- **Common Voice**：Mozilla的多语言语音数据集
- **CORAA**：巴西葡萄牙语数据集
- **MLS**：Multilingual LibriSpeech葡萄牙语部分
- **VoxForge**：VoxForge葡萄牙语数据集

## 训练参数

### GPT模型训练参数

- **batch_size**：批处理大小，默认16
- **max_epochs**：最大训练轮数，默认100
- **device**：训练设备，默认cuda
- **bert_model**：BERT模型路径

### SoVITS模型训练参数

- **batch_size**：批处理大小，默认16
- **max_epochs**：最大训练轮数，默认100
- **device**：训练设备，默认cuda

## 推理

训练完成后，可以使用以下命令进行推理：

```bash
python inference.py \
  --text "¡Hola! ¿Cómo estás?" \
  --language es \
  --reference-audio path/to/reference.wav \
  --gpt-model checkpoints/es_gpt/best_model.pth \
  --sovits-model checkpoints/es_sovits/best_model.pth \
  --output-path output.wav
```

或者：

```bash
python inference.py \
  --text "Olá, como está você?" \
  --language pt \
  --reference-audio path/to/reference.wav \
  --prompt-text "Eu sou um assistente de voz" \
  --gpt-model checkpoints/pt_gpt/best_model.pth \
  --sovits-model checkpoints/pt_sovits/best_model.pth \
  --output-path output.wav
```

## 常见问题

### 1. 训练相关问题

**Q: 训练时显存不足怎么办？**

A: 可以尝试以下方法：
- 减小batch_size
- 使用梯度累积
- 使用混合精度训练
- 使用较小的模型配置

**Q: 训练时出现NaN损失怎么办？**

A: 可能的解决方法：
- 降低学习率
- 检查数据集是否有异常值
- 使用梯度裁剪
- 从较早的检查点恢复训练

### 2. 推理相关问题

**Q: 生成的语音有口音或发音不准确怎么办？**

A: 可能的解决方法：
- 使用更多的训练数据
- 确保训练数据发音准确
- 调整G2P规则以适应特定方言
- 使用特定方言的预训练模型

**Q: 如何提高生成语音的自然度？**

A: 可以尝试：
- 调整temperature参数（0.6-0.8通常效果较好）
- 调整top_k和top_p参数
- 使用更高质量的参考音频
- 微调模型以适应特定说话风格

### 3. 语言相关问题

**Q: 如何处理葡萄牙语或西班牙语的方言差异？**

A: 本项目支持方言处理：
- 在训练时指定方言参数：`--dialect brazilian`或`--dialect european`（葡萄牙语）
- 在训练时指定方言参数：`--dialect latin_american`或`--dialect european`（西班牙语）
- 使用特定方言的训练数据
- 调整G2P规则以适应特定方言

## 许可证

本项目遵循MIT许可证。详见[LICENSE](LICENSE)文件。

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)项目团队
- Mozilla Common Voice项目
- CORAA和CIEMPIESS数据集团队
- 所有贡献者和测试者
