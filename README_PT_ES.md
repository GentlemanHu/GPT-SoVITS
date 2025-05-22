# GPT-SoVITS 西班牙语和葡萄牙语完整支持

本项目提供了GPT-SoVITS对西班牙语(es)和葡萄牙语(pt)的完整支持，包括自动数据下载、处理、训练和推理的全流程解决方案。

## 目录

1. [概述](#概述)
2. [快速开始](#快速开始)
3. [安装依赖](#安装依赖)
4. [数据集](#数据集)
5. [自动化流程](#自动化流程)
6. [模型训练](#模型训练)
7. [推理](#推理)
8. [高级用法](#高级用法)
9. [常见问题](#常见问题)
10. [故障排除](#故障排除)

## 概述

GPT-SoVITS是一个先进的文本到语音(TTS)系统，结合了GPT（生成式预训练变换器）和SoVITS（声音向量化即时文本到语音）技术。我们对GPT-SoVITS进行了扩展，添加了对西班牙语和葡萄牙语的完整支持，使其能够生成高质量的西班牙语和葡萄牙语语音。

### 主要特点

- **自动化流程**：提供从数据下载到模型训练的完整自动化流程
- **多方言支持**：同时支持欧洲西班牙语、拉丁美洲西班牙语、巴西葡萄牙语和欧洲葡萄牙语
- **零样本TTS**：只需5秒的参考音频即可合成新的语音
- **少样本训练**：仅需10分钟的训练数据即可得到高质量的个性化声音
- **跨语言支持**：支持与训练数据不同语言的推理
- **多语言混合**：在同一文本中混合使用多种语言

## 快速开始

最简单的使用方式是运行自动化安装和流程脚本：

```bash
# 1. 安装西班牙语和葡萄牙语支持
./install_pt_es_support.sh

# 2. 运行西班牙语自动化流程
./pt_es_pipeline.sh --language es --dataset common_voice

# 或运行葡萄牙语自动化流程
./pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian
```

## 安装依赖

在开始之前，您需要确保已经正确安装了GPT-SoVITS的基本环境。然后运行我们的安装脚本来添加西班牙语和葡萄牙语支持：

```bash
# 安装西班牙语和葡萄牙语支持
./install_pt_es_support.sh
```

安装脚本会：
- 检查并安装所需的Python依赖
- 创建必要的目录结构
- 下载西班牙语和葡萄牙语的BERT模型
- 检查其他必要的预训练模型
- 测试西班牙语和葡萄牙语支持

## 数据集

本项目支持以下数据集：

### 西班牙语数据集

| 数据集名称 | 类型 | 大小 | 方言 | 描述 |
|-----------|------|------|------|------|
| Common Voice | 开放 | ~50GB | 混合 | Mozilla的多语言语音数据集，包含多种西班牙语方言 |
| CIEMPIESS | 开放 | ~4GB | 墨西哥 | 墨西哥西班牙语数据集，专注于新闻播报音频 |
| MLS | 开放 | ~20GB | 混合 | Multilingual LibriSpeech西班牙语部分，包含有声读物 |
| CSS10 | 开放 | ~5GB | 欧洲 | CSS10西班牙语数据集，包含单一说话人的高质量录音 |

### 葡萄牙语数据集

| 数据集名称 | 类型 | 大小 | 方言 | 描述 |
|-----------|------|------|------|------|
| Common Voice | 开放 | ~30GB | 混合 | Mozilla的多语言语音数据集，包含巴西和欧洲葡萄牙语 |
| CORAA | 开放 | ~10GB | 巴西 | 巴西葡萄牙语数据集，包含日常对话和演讲 |
| MLS | 开放 | ~15GB | 混合 | Multilingual LibriSpeech葡萄牙语部分，包含有声读物 |
| VoxForge | 开放 | ~2GB | 混合 | VoxForge葡萄牙语数据集，包含众包录音 |

### 数据集下载

您可以使用我们提供的脚本自动下载数据集：

```bash
# 下载西班牙语Common Voice数据集
python download_es_data.py --dataset-type common_voice --data-dir ./data

# 下载葡萄牙语CORAA数据集
python download_pt_data.py --dataset-type coraa --data-dir ./data --dialect brazilian
```

## 自动化流程

### 主流程脚本

`pt_es_pipeline.sh`脚本提供了从数据下载到模型训练的完整自动化流程。以下是常用的参数：

```bash
./pt_es_pipeline.sh [参数]
```

主要参数：
- `-l, --language LANG`：设置语言(es: 西班牙语, pt: 葡萄牙语)
- `-d, --dataset TYPE`：设置数据集类型(common_voice, mls, ciempiess, coraa, custom)
- `--data-dir DIR`：设置数据目录(默认: ./data)
- `-b, --batch-size SIZE`：设置批处理大小(默认: 16)
- `-e, --max-epochs NUM`：设置最大训练轮数(默认: 100)
- `--dialect DIALECT`：设置方言(brazilian, european, latin_american)
- `--device DEVICE`：设置使用的设备(cuda, cpu)
- `--skip-download`：跳过数据下载步骤
- `--skip-preprocess`：跳过数据预处理步骤
- `--skip-training`：跳过模型训练步骤
- `-h, --help`：显示帮助信息

示例：
```bash
# 使用欧洲西班牙语，从Common Voice数据集训练
./pt_es_pipeline.sh --language es --dataset common_voice --dialect european

# 使用巴西葡萄牙语，从CORAA数据集训练，减小批处理大小
./pt_es_pipeline.sh --language pt --dataset coraa --dialect brazilian --batch-size 8
```

### 分步执行

如果您想分步执行流程，可以按照以下步骤操作：

1. **下载BERT模型**
```bash
python download_bert_model.py --language es  # 西班牙语
python download_bert_model.py --language pt  # 葡萄牙语
```

2. **下载和处理数据**
```bash
python download_es_data.py --dataset-type common_voice
python preprocess_data.py --language es --data-dir ./data
```

3. **训练模型**
```bash
python train_gpt_model.py --language es --data-dir ./data/es_corpus/prepared
python train_sovits_model.py --language es --data-dir ./data/es_corpus/prepared
```

## 模型训练

### GPT模型训练

GPT模型负责将文本转换为语义表示。以下是训练参数：

```bash
python train_gpt_model.py [参数]
```

主要参数：
- `--language`：语言代码(es: 西班牙语, pt: 葡萄牙语)
- `--data-dir`：数据目录，包含预处理后的数据
- `--batch-size`：批处理大小(默认: 16)
- `--max-epochs`：最大训练轮数(默认: 100)
- `--device`：训练设备(默认: cuda)
- `--dialect`：方言类型
- `--bert-model`：BERT模型路径

### SoVITS模型训练

SoVITS模型负责将语义表示转换为声音。以下是训练参数：

```bash
python train_sovits_model.py [参数]
```

主要参数：
- `--language`：语言代码(es: 西班牙语, pt: 葡萄牙语)
- `--data-dir`：数据目录，包含预处理后的数据
- `--batch-size`：批处理大小(默认: 16)
- `--max-epochs`：最大训练轮数(默认: 100)
- `--device`：训练设备(默认: cuda)
- `--dialect`：方言类型

### 训练技巧

- 对于NVIDIA RTX 3090等高端GPU，可以使用较大的批处理大小(16-32)
- 对于中低端GPU(如GTX 1660)，建议减小批处理大小(4-8)
- 如果遇到内存不足错误，尝试减小批处理大小或使用梯度累积
- 训练时间通常需要几小时到数天，取决于数据集大小和硬件配置
- 建议先训练50个epoch，然后根据损失情况决定是否继续训练

## 推理

训练完成后，可以使用以下命令进行推理：

```bash
python inference.py [参数]
```

主要参数：
- `--text`：要合成的文本
- `--language`：语言代码(es: 西班牙语, pt: 葡萄牙语)
- `--reference-audio`：参考音频路径
- `--prompt-text`：参考音频的文本(可选)
- `--gpt-model`：GPT模型路径
- `--sovits-model`：SoVITS模型路径
- `--output-path`：输出音频路径(默认: output.wav)
- `--top-k`：Top-K采样参数(默认: 5)
- `--top-p`：Top-P采样参数(默认: 0.7)
- `--temperature`：温度参数(默认: 0.7)
- `--speed`：语速(默认: 1.0)

示例：
```bash
# 西班牙语推理
python inference.py \
  --text "¡Hola! ¿Cómo estás?" \
  --language es \
  --reference-audio path/to/reference.wav \
  --gpt-model checkpoints/es_gpt/best_model.pth \
  --sovits-model checkpoints/es_sovits/best_model.pth \
  --output-path output_es.wav

# 葡萄牙语推理
python inference.py \
  --text "Olá, como está você?" \
  --language pt \
  --reference-audio path/to/reference.wav \
  --prompt-text "Eu sou um assistente de voz" \
  --gpt-model checkpoints/pt_gpt/best_model.pth \
  --sovits-model checkpoints/pt_sovits/best_model.pth \
  --output-path output_pt.wav
```

### 多语言混合推理

您可以在一段文本中混合使用不同的语言：

```bash
python inference.py \
  --text "Hello, me llamo Carlos. I speak both español and English." \
  --language en \
  --reference-audio path/to/reference.wav \
  --gpt-model checkpoints/multi_gpt/best_model.pth \
  --sovits-model checkpoints/multi_sovits/best_model.pth \
  --output-path output_multi.wav
```

## 高级用法

### 自定义数据集

如果您有自己的数据集，可以按照以下格式准备：

1. 创建一个包含音频文件的目录
2. 创建metadata.csv文件，格式如下：
```
file_path|speaker|text|language
audio1.wav|speaker1|这是第一个音频的文本|es
audio2.wav|speaker1|这是第二个音频的文本|es
...
```

然后使用以下命令处理数据：
```bash
python preprocess_data.py --language es --data-dir ./your_data_dir
```

### 模型微调

如果您有一个预训练模型，想要在新的数据上进行微调：

```bash
python train_gpt_model.py \
  --language es \
  --data-dir ./new_data \
  --batch-size 8 \
  --max-epochs 50 \
  --checkpoint path/to/pretrained_model.pth
```

### 混合方言训练

如果您想训练一个支持多种方言的模型：

```bash
# 准备包含不同方言的数据
python preprocess_data.py --language es --data-dir ./data_european --dialect european
python preprocess_data.py --language es --data-dir ./data_latin --dialect latin_american

# 合并数据
# (手动合并prepared目录下的文件)

# 训练混合方言模型
python train_gpt_model.py --language es --data-dir ./combined_data
```

## 常见问题

### 1. 训练相关问题

**Q: 训练时显存不足怎么办？**
A: 尝试以下方法：
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

## 故障排除

### 数据集下载问题

如果数据集下载失败，尝试以下解决方法：
- 检查网络连接
- 使用VPN（某些数据集可能需要）
- 手动下载数据集，然后放到相应目录
- 尝试使用不同的数据集

### 训练问题

如果训练过程中遇到问题：
- 检查GPU内存使用情况
- 减小批处理大小
- 确保已正确安装所有依赖
- 查看日志文件获取详细错误信息

### 推理问题

如果推理时遇到问题：
- 确保模型文件存在并且完整
- 检查参考音频是否有效
- 调整推理参数
- 尝试使用不同的输入文本

## 许可证

本项目遵循GPT-SoVITS原有的MIT许可证。详见[LICENSE](LICENSE)文件。

## 致谢

- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)项目团队
- Mozilla Common Voice项目
- CORAA和CIEMPIESS数据集团队
- 所有贡献者和测试者

---

如果您有任何问题或建议，请提交Issue或Pull Request。
