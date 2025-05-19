# GPT-SoVITS 葡萄牙语和西班牙语支持使用指南

## 目录

1. [简介](#简介)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [训练流程](#训练流程)
5. [推理与使用](#推理与使用)
6. [常见问题解答](#常见问题解答)
7. [进阶优化](#进阶优化)
8. [附录](#附录)

## 简介

GPT-SoVITS是一个先进的文本到语音(TTS)系统，结合了GPT（生成式预训练变换器）和SoVITS（声音向量化即时文本到语音）技术。本文档详细介绍了如何使用GPT-SoVITS的葡萄牙语和西班牙语支持功能，包括环境配置、数据准备、模型训练和推理使用等方面。

### 主要特点

- **零样本TTS**：只需5秒的声音样本，即可实现文本到语音转换
- **少样本TTS**：仅需1分钟的训练数据即可微调模型，提升声音相似度和真实感
- **跨语言支持**：支持与训练数据集不同语言的推理
- **多语言混合**：支持在同一文本中混合使用多种语言

### 支持的语言

GPT-SoVITS现已支持以下语言：
- 中文(zh)
- 英语(en)
- 日语(ja)
- 韩语(ko)
- 粤语(yue)
- 葡萄牙语(pt)
- 西班牙语(es)

## 环境准备

### 系统要求

- **操作系统**：Windows 10/11、Linux或macOS
- **CPU训练**：至少16GB RAM
- **GPU训练**：
  - NVIDIA GPU，至少8GB显存
  - CUDA 11.7或更高版本
  - cuDNN 8.0或更高版本

### 安装步骤

1. **克隆仓库**：
   ```bash
   git clone https://github.com/GentlemanHu/GPT-SoVITS.git
   cd GPT-SoVITS
   ```

2. **创建虚拟环境**：
   ```bash
   # 使用conda
   conda create -n gpt-sovits python=3.10
   conda activate gpt-sovits
   
   # 或使用venv
   python -m venv menv
   # Windows
   menv\Scripts\activate
   # Linux/macOS
   source menv/bin/activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **下载预训练模型**：
   ```bash
   # 创建预训练模型目录
   mkdir -p pretrained_models
   
   # 下载基础模型
   # 可以从项目发布页或Hugging Face下载预训练模型
   # 将下载的模型放入pretrained_models目录
   ```

## 数据准备

### 数据集结构

为了训练葡萄牙语或西班牙语模型，您需要准备以下数据：

```
data/
├── pt_corpus/  # 葡萄牙语数据集
│   ├── speaker1/
│   │   ├── audio1.wav
│   │   ├── audio2.wav
│   │   └── ...
│   └── metadata.csv
│
└── es_corpus/  # 西班牙语数据集
    ├── speaker1/
    │   ├── audio1.wav
    │   ├── audio2.wav
    │   └── ...
    └── metadata.csv
```

### metadata.csv格式

```
file_path|speaker|text|language
speaker1/audio1.wav|speaker1|葡萄牙语文本内容|pt
speaker1/audio2.wav|speaker1|葡萄牙语文本内容|pt
```

### 数据集收集

1. **公开数据集**：
   - 葡萄牙语：
     - [Common Voice](https://commonvoice.mozilla.org/zh-CN/datasets)
     - [CORAA](https://github.com/nilc-nlp/CORAA)（巴西葡萄牙语）
     - [Multilingual LibriSpeech](https://www.openslr.org/94/)
   
   - 西班牙语：
     - [Common Voice](https://commonvoice.mozilla.org/zh-CN/datasets)
     - [CIEMPIESS](http://www.ciempiess.org/downloads)（墨西哥西班牙语）
     - [Crowdsourced Spanish Speech](https://www.openslr.org/74/)

2. **自建数据集**：
   - 使用录音设备录制高质量的葡萄牙语或西班牙语语音
   - 确保录音环境安静，避免背景噪音
   - 每个音频文件建议长度在5-15秒之间

### 数据预处理

1. **音频预处理**：
   ```bash
   # 处理音频文件（重采样、降噪等）
   python tools/process_audio.py --input_dir data/pt_corpus/raw --output_dir data/pt_corpus/processed --sample_rate 24000
   ```

2. **数据集准备**：
   ```bash
   # 准备葡萄牙语数据集
   python GPT_SoVITS/prepare_datasets.py --language pt --input_dir data/pt_corpus/processed --output_dir data/pt_corpus/prepared
   
   # 准备西班牙语数据集
   python GPT_SoVITS/prepare_datasets.py --language es --input_dir data/es_corpus/processed --output_dir data/es_corpus/prepared
   ```

3. **自动标注**（如果没有文本标注）：
   ```bash
   # 葡萄牙语自动标注
   python GPT_SoVITS/tools/auto_labeling.py --language pt --input_dir data/pt_corpus/prepared
   
   # 西班牙语自动标注
   python GPT_SoVITS/tools/auto_labeling.py --language es --input_dir data/es_corpus/prepared
   ```

## 训练流程

GPT-SoVITS的训练分为两个阶段：GPT模型训练和SoVITS模型训练。

### 阶段1：GPT模型训练

```bash
# 葡萄牙语GPT模型训练
python GPT_SoVITS/train_stage1.py \
  --config GPT_SoVITS/configs/train_config.json \
  --train_data_dir data/pt_corpus/prepared \
  --language pt \
  --batch_size 16 \
  --max_epochs 100 \
  --save_dir checkpoints/pt_gpt

# 西班牙语GPT模型训练
python GPT_SoVITS/train_stage1.py \
  --config GPT_SoVITS/configs/train_config.json \
  --train_data_dir data/es_corpus/prepared \
  --language es \
  --batch_size 16 \
  --max_epochs 100 \
  --save_dir checkpoints/es_gpt
```

训练参数说明：
- `--config`：配置文件路径
- `--train_data_dir`：训练数据目录
- `--language`：训练语言（pt或es）
- `--batch_size`：批处理大小（根据GPU内存调整）
- `--max_epochs`：最大训练轮数
- `--save_dir`：模型保存目录

### 阶段2：SoVITS模型训练

```bash
# 葡萄牙语SoVITS模型训练
python GPT_SoVITS/train_stage2.py \
  --config GPT_SoVITS/configs/train_config.json \
  --train_data_dir data/pt_corpus/prepared \
  --language pt \
  --batch_size 16 \
  --max_epochs 100 \
  --save_dir checkpoints/pt_sovits

# 西班牙语SoVITS模型训练
python GPT_SoVITS/train_stage2.py \
  --config GPT_SoVITS/configs/train_config.json \
  --train_data_dir data/es_corpus/prepared \
  --language es \
  --batch_size 16 \
  --max_epochs 100 \
  --save_dir checkpoints/es_sovits
```

### 训练监控

1. **使用TensorBoard监控训练进度**：
   ```bash
   tensorboard --logdir=checkpoints
   ```

2. **训练指标**：
   - 损失函数值（loss）
   - 梯度范数（grad_norm）
   - 学习率（learning_rate）
   - 验证集性能（validation_loss）

### 训练技巧

1. **学习率调整**：
   - 初始学习率设置为1e-4
   - 使用学习率衰减策略，如余弦退火（cosine annealing）

2. **梯度累积**：
   - 如果GPU内存不足，可以使用梯度累积
   - 设置`--gradient_accumulation_steps 2`或更高

3. **混合精度训练**：
   - 使用FP16混合精度训练加速
   - 设置`--use_fp16 true`

4. **断点续训**：
   - 使用`--resume_checkpoint path/to/checkpoint`从断点继续训练

## 推理与使用

### 使用WebUI

GPT-SoVITS提供了用户友好的WebUI界面，可以轻松进行文本到语音的转换：

```bash
# 启动WebUI
python GPT_SoVITS/webui.py
```

在WebUI中：
1. 选择语言为"葡萄牙语"或"西班牙语"
2. 输入葡萄牙语或西班牙语文本
3. 选择参考音频（可以使用任何语言的参考音频）
4. 设置生成参数（如top_k、top_p、temperature等）
5. 点击"生成"按钮

### 使用API

GPT-SoVITS也提供了API接口，可以通过HTTP请求进行文本到语音的转换：

```bash
# 启动API服务
python api.py --port 8080
```

API请求示例：
```bash
# 葡萄牙语示例
curl -X POST "http://localhost:8080/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Olá, como está você?",
    "language": "葡萄牙语",
    "reference_audio_path": "path/to/reference.wav",
    "top_k": 5,
    "top_p": 0.7,
    "temperature": 0.7
  }'

# 西班牙语示例
curl -X POST "http://localhost:8080/tts" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "¡Hola! ¿Cómo estás?",
    "language": "西班牙语",
    "reference_audio_path": "path/to/reference.wav",
    "top_k": 5,
    "top_p": 0.7,
    "temperature": 0.7
  }'
```

### 命令行使用

也可以通过命令行直接使用GPT-SoVITS：

```bash
# 葡萄牙语示例
python GPT_SoVITS/inference.py \
  --text "Olá, como está você?" \
  --language pt \
  --reference_audio path/to/reference.wav \
  --gpt_model checkpoints/pt_gpt/best_model.pth \
  --sovits_model checkpoints/pt_sovits/best_model.pth \
  --output_path outputs/pt_output.wav

# 西班牙语示例
python GPT_SoVITS/inference.py \
  --text "¡Hola! ¿Cómo estás?" \
  --language es \
  --reference_audio path/to/reference.wav \
  --gpt_model checkpoints/es_gpt/best_model.pth \
  --sovits_model checkpoints/es_sovits/best_model.pth \
  --output_path outputs/es_output.wav
```

### 混合语言使用

GPT-SoVITS支持在同一文本中混合使用多种语言：

```bash
# 混合语言示例（葡萄牙语和英语）
python GPT_SoVITS/inference.py \
  --text "Hello, meu nome é João. I speak both português and English." \
  --language auto \
  --reference_audio path/to/reference.wav \
  --output_path outputs/mixed_pt_en.wav

# 混合语言示例（西班牙语和英语）
python GPT_SoVITS/inference.py \
  --text "Hello, mi nombre es Carlos. I speak both español and English." \
  --language auto \
  --reference_audio path/to/reference.wav \
  --output_path outputs/mixed_es_en.wav
```
## 常见问题解答

### 1. 训练相关问题

**Q: 训练时显存不足怎么办？**

A: 可以尝试以下方法：
- 减小batch_size
- 使用梯度累积（设置`--gradient_accumulation_steps`）
- 使用混合精度训练（设置`--use_fp16 true`）
- 使用较小的模型配置

**Q: 训练时出现NaN损失怎么办？**

A: 可能的解决方法：
- 降低学习率
- 检查数据集是否有异常值
- 使用梯度裁剪（设置`--clip_grad_norm`）
- 从较早的检查点恢复训练

**Q: 训练多久才能得到好的模型？**

A: 这取决于数据集大小和质量：
- 小数据集（<1小时）：约50-100轮
- 中等数据集（1-5小时）：约30-50轮
- 大数据集（>5小时）：约20-30轮

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

**Q: 生成速度慢怎么办？**

A: 可以尝试：
- 使用GPU进行推理
- 减小模型大小
- 使用批处理模式处理多个文本
- 使用FP16或INT8量化模型

### 3. 语言相关问题

**Q: 如何处理葡萄牙语或西班牙语的方言差异？**

A: GPT-SoVITS支持方言处理：
- 在训练时指定方言参数：`--dialect brazilian`或`--dialect european`（葡萄牙语）
- 在训练时指定方言参数：`--dialect latin_american`或`--dialect european`（西班牙语）
- 使用特定方言的训练数据
- 调整G2P规则以适应特定方言

**Q: 如何处理带有重音的字符？**

A: GPT-SoVITS已经内置了对重音字符的支持：
- 葡萄牙语：á, é, í, ó, ú, â, ê, ô, ã, õ等
- 西班牙语：á, é, í, ó, ú, ü, ñ等
- 无需特殊处理，直接在文本中使用这些字符即可

## 进阶优化

### 1. BERT模型集成

为了提高语义理解能力，可以集成专门的葡萄牙语和西班牙语BERT模型：

```bash
# 下载葡萄牙语BERT模型
mkdir -p pretrained_models/portuguese-bert
wget https://huggingface.co/neuralmind/bert-base-portuguese-cased/resolve/main/pytorch_model.bin -O pretrained_models/portuguese-bert/pytorch_model.bin
wget https://huggingface.co/neuralmind/bert-base-portuguese-cased/resolve/main/config.json -O pretrained_models/portuguese-bert/config.json
wget https://huggingface.co/neuralmind/bert-base-portuguese-cased/resolve/main/vocab.txt -O pretrained_models/portuguese-bert/vocab.txt

# 下载西班牙语BERT模型
mkdir -p pretrained_models/spanish-bert
wget https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/resolve/main/pytorch_model.bin -O pretrained_models/spanish-bert/pytorch_model.bin
wget https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/resolve/main/config.json -O pretrained_models/spanish-bert/config.json
wget https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/resolve/main/vocab.txt -O pretrained_models/spanish-bert/vocab.txt
```

然后在训练时指定BERT模型：
```bash
python GPT_SoVITS/train_stage1.py \
  --config GPT_SoVITS/configs/train_config.json \
  --train_data_dir data/pt_corpus/prepared \
  --language pt \
  --bert_model pretrained_models/portuguese-bert \
  --save_dir checkpoints/pt_gpt
```

### 2. 发音词典优化

为了处理特殊发音词汇，可以创建自定义发音词典：

```python
# 葡萄牙语发音词典
pt_pronunciation_dict = {
    "auxiliar": ["a", "u", "x", "i", "l", "i", "a", "r"],
    "exceção": ["e", "x", "c", "e", "ç", "ã", "o"],
    # 更多特殊词汇...
}

# 西班牙语发音词典
es_pronunciation_dict = {
    "México": ["m", "e", "j", "i", "k", "o"],
    "guitarra": ["g", "i", "t", "a", "rr", "a"],
    # 更多特殊词汇...
}
```

将词典保存为JSON文件：
```bash
# 保存葡萄牙语发音词典
python -c "import json; json.dump(pt_pronunciation_dict, open('data/pt_pronunciation_dict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"

# 保存西班牙语发音词典
python -c "import json; json.dump(es_pronunciation_dict, open('data/es_pronunciation_dict.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)"
```

然后在训练或推理时指定发音词典：
```bash
python GPT_SoVITS/inference.py \
  --text "México es un país hermoso." \
  --language es \
  --pronunciation_dict data/es_pronunciation_dict.json \
  --reference_audio path/to/reference.wav \
  --output_path outputs/es_output.wav
```
### 3. 用户界面本地化

为了提供更好的用户体验，可以为葡萄牙语和西班牙语用户提供本地化的用户界面：

```bash
# 创建葡萄牙语翻译文件
mkdir -p GPT_SoVITS/tools/i18n/locale/pt_BR
mkdir -p GPT_SoVITS/tools/i18n/locale/pt_PT

# 创建西班牙语翻译文件
mkdir -p GPT_SoVITS/tools/i18n/locale/es_ES
mkdir -p GPT_SoVITS/tools/i18n/locale/es_MX
```

编写翻译JSON文件：
```json
// GPT_SoVITS/tools/i18n/locale/pt_BR.json
{
  "语言": "Idioma",
  "文本": "Texto",
  "参考音频": "Áudio de referência",
  "生成": "Gerar",
  "停止": "Parar",
  "模型": "Modelo",
  "设置": "Configurações",
  "帮助": "Ajuda",
  "关于": "Sobre"
}

// GPT_SoVITS/tools/i18n/locale/es_ES.json
{
  "语言": "Idioma",
  "文本": "Texto",
  "参考音频": "Audio de referencia",
  "生成": "Generar",
  "停止": "Detener",
  "模型": "Modelo",
  "设置": "Configuración",
  "帮助": "Ayuda",
  "关于": "Acerca de"
}
```

## 附录

### 葡萄牙语和西班牙语音素表

#### 葡萄牙语音素
```
元音: a, e, i, o, u, á, é, í, ó, ú, â, ê, ô, ã, õ
辅音: b, d, f, g, k, l, m, n, p, r, s, t, v, z, ç, j, x, h, rr, nh, lh, ch
组合: br, cr, dr, fr, gr, pr, tr, vr
```

#### 西班牙语音素
```
元音: a, e, i, o, u, á, é, í, ó, ú, ü
辅音: b, c, d, f, g, h, j, k, l, m, n, ñ, p, q, r, s, t, v, w, x, y, z
组合: ch, ll, rr, th, fl, fr, pl, pr, bl, br, cl, cr, gl, gr, dr, tr
```

### 常用测试文本

#### 葡萄牙语测试文本
```
1. "Olá, como está você? Eu estou bem, obrigado!"
2. "A raposa marrom rápida pula sobre o cão preguiçoso."
3. "Todos os seres humanos nascem livres e iguais em dignidade e direitos."
4. "O Brasil é um país de dimensões continentais com uma rica diversidade cultural."
5. "Amanhã vou ao cinema com meus amigos para assistir ao novo filme."
```

#### 西班牙语测试文本
```
1. "¡Hola! ¿Cómo estás? ¡Yo estoy bien, gracias!"
2. "El zorro marrón rápido salta sobre el perro perezoso."
3. "Todos los seres humanos nacen libres e iguales en dignidad y derechos."
4. "España es un país con una rica historia y diversidad cultural."
5. "Mañana iré al cine con mis amigos para ver la nueva película."
```

### 参考资源

1. [葡萄牙语语音学习资源](https://forvo.com/languages/pt/)
2. [西班牙语语音学习资源](https://forvo.com/languages/es/)
3. [葡萄牙语语言学资料](https://www.ethnologue.com/language/por)
4. [西班牙语语言学资料](https://www.ethnologue.com/language/spa)
5. [Common Voice数据集](https://commonvoice.mozilla.org/zh-CN/datasets)
6. [Hugging Face预训练模型](https://huggingface.co/models)
