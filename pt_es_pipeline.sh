#!/bin/bash
# GPT-SoVITS 自动化流程脚本 - 支持葡萄牙语和西班牙语
# 作者: Agent
# 版本: 2.0

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 无颜色

# 默认参数
LANGUAGE="es"  # 默认语言: es (西班牙语) 或 pt (葡萄牙语)
DATASET_TYPE="common_voice"  # 默认数据集类型
DATA_DIR="./data"  # 数据目录
BATCH_SIZE=16  # 批处理大小
MAX_EPOCHS=100  # 最大训练轮数
DOWNLOAD_BERT=true  # 是否下载BERT模型
SKIP_DOWNLOAD=false  # 是否跳过数据下载
SKIP_PREPROCESS=false  # 是否跳过预处理
SKIP_TRAINING=false  # 是否跳过训练
SKIP_TEST=false  # 是否跳过测试
DEVICE="cuda"  # 使用的设备
SAMPLE_RATE=24000  # 音频采样率
DIALECT=""  # 方言 (巴西葡萄牙语: brazilian, 欧洲葡萄牙语: european, 拉丁美洲西班牙语: latin_american, 欧洲西班牙语: european)
CLEANUP=false  # 是否清理临时文件
# 显示帮助信息
show_help() {
    echo -e "${BLUE}GPT-SoVITS 自动化流程脚本 - 支持葡萄牙语和西班牙语${NC}"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -l, --language LANG       设置语言 (es: 西班牙语, pt: 葡萄牙语)"
    echo "  -d, --dataset TYPE        设置数据集类型 (common_voice, mls, ciempiess, coraa, custom)"
    echo "  --data-dir DIR            设置数据目录 (默认: ./data)"
    echo "  -b, --batch-size SIZE     设置批处理大小 (默认: 16)"
    echo "  -e, --max-epochs NUM      设置最大训练轮数 (默认: 100)"
    echo "  --no-bert                 不下载BERT模型"
    echo "  --skip-download           跳过数据下载步骤"
    echo "  --skip-preprocess         跳过数据预处理步骤"
    echo "  --skip-training           跳过模型训练步骤"
    echo "  --skip-test               跳过测试步骤"
    echo "  --device DEVICE           设置使用的设备 (cuda, cpu)"
    echo "  --sample-rate RATE        设置音频采样率 (默认: 24000)"
    echo "  --dialect DIALECT         设置方言 (brazilian, european, latin_american)"
    echo "  --cleanup                 清理临时文件"
    echo "  -h, --help                显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 --language es --dataset common_voice"
    echo "  $0 --language pt --dataset coraa --dialect brazilian --batch-size 8"
    echo "  $0 --language es --skip-download --skip-preprocess"
    echo ""
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--language)
            LANGUAGE="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET_TYPE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -e|--max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --no-bert)
            DOWNLOAD_BERT=false
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-test)
            SKIP_TEST=true
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        --dialect)
            DIALECT="$2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done
# 验证参数
if [[ "$LANGUAGE" != "es" && "$LANGUAGE" != "pt" ]]; then
    echo -e "${RED}错误: 语言必须是 'es' (西班牙语) 或 'pt' (葡萄牙语)${NC}"
    exit 1
fi

if [[ "$DIALECT" != "" && "$DIALECT" != "brazilian" && "$DIALECT" != "european" && "$DIALECT" != "latin_american" ]]; then
    echo -e "${RED}错误: 方言必须是 'brazilian', 'european' 或 'latin_american'${NC}"
    exit 1
fi

# 设置语言相关变量
if [[ "$LANGUAGE" == "es" ]]; then
    LANGUAGE_FULL="spanish"
    LANGUAGE_NAME="西班牙语"
    if [[ "$DIALECT" == "" ]]; then
        DIALECT="european"  # 默认欧洲西班牙语
    fi
else
    LANGUAGE_FULL="portuguese"
    LANGUAGE_NAME="葡萄牙语"
    if [[ "$DIALECT" == "" ]]; then
        DIALECT="brazilian"  # 默认巴西葡萄牙语
    fi
fi

# 检查系统和依赖项
echo -e "${YELLOW}检查系统和依赖项...${NC}"

# 检查Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python命令${NC}"
    echo -e "请安装Python 3.8或更高版本"
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

# 检查PyTorch
echo -e "${YELLOW}检查PyTorch...${NC}"
if ! python -c "import torch" &> /dev/null; then
    echo -e "${RED}错误: 未找到PyTorch${NC}"
    echo -e "请安装PyTorch: pip install torch"
    exit 1
fi

# 检查CUDA是否可用
if [[ "$DEVICE" == "cuda" ]]; then
    echo -e "${YELLOW}检查CUDA可用性...${NC}"
    if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        echo -e "${YELLOW}警告: CUDA不可用, 将使用CPU${NC}"
        DEVICE="cpu"
    fi
fi

# 创建必要的目录
echo -e "${YELLOW}创建必要的目录...${NC}"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/${LANGUAGE}_corpus/raw"
mkdir -p "$DATA_DIR/${LANGUAGE}_corpus/processed"
mkdir -p "$DATA_DIR/${LANGUAGE}_corpus/prepared"
mkdir -p "checkpoints/${LANGUAGE}_gpt"
mkdir -p "checkpoints/${LANGUAGE}_sovits"
mkdir -p "logs"
# 显示配置信息
echo -e "${BLUE}========== GPT-SoVITS ${LANGUAGE_NAME}训练流程 ==========${NC}"
echo -e "${GREEN}语言:${NC} $LANGUAGE_NAME ($LANGUAGE)"
echo -e "${GREEN}方言:${NC} $DIALECT"
echo -e "${GREEN}数据集类型:${NC} $DATASET_TYPE"
echo -e "${GREEN}数据目录:${NC} $DATA_DIR"
echo -e "${GREEN}批处理大小:${NC} $BATCH_SIZE"
echo -e "${GREEN}最大训练轮数:${NC} $MAX_EPOCHS"
echo -e "${GREEN}设备:${NC} $DEVICE"
echo -e "${GREEN}音频采样率:${NC} $SAMPLE_RATE"
echo -e "${GREEN}清理临时文件:${NC} $CLEANUP"
echo ""

# 检查GPT-SoVITS目录结构
if [ ! -d "GPT_SoVITS" ]; then
    echo -e "${RED}错误: 未找到GPT_SoVITS目录，请确保在GPT-SoVITS项目根目录下运行此脚本${NC}"
    exit 1
fi

# 创建日志目录和文件
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/pipeline_${LANGUAGE}_$(date +'%Y%m%d_%H%M%S').log"
echo -e "所有日志将记录到: $LOG_FILE\n"

# 定义记录日志的函数
log() {
    local message="$1"
    local timestamp=$(date +'%Y-%m-%d %H:%M:%S')
    echo -e "$timestamp - $message" | tee -a "$LOG_FILE"
}
# 执行数据下载脚本
if [ "$SKIP_DOWNLOAD" = false ]; then
    log "${YELLOW}开始下载${LANGUAGE_NAME}数据集...${NC}"
    python download_${LANGUAGE}_data.py --dataset-type "$DATASET_TYPE" --data-dir "$DATA_DIR" --dialect "$DIALECT" 2>&1 | tee -a "$LOG_FILE"
    DOWNLOAD_STATUS=${PIPESTATUS[0]}
    if [ $DOWNLOAD_STATUS -ne 0 ]; then
        log "${RED}数据集下载失败, 错误代码: $DOWNLOAD_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}数据集下载完成${NC}"
    
    # 对MLS数据集进行额外检查，确保metadata.csv已生成
    if [ "$DATASET_TYPE" = "mls" ] && [ "$LANGUAGE" = "pt" ]; then
        METADATA_FILE="$DATA_DIR/${LANGUAGE}_corpus/raw/metadata.csv"
        if [ ! -f "$METADATA_FILE" ] || [ $(wc -l < "$METADATA_FILE") -le 1 ]; then
            log "${YELLOW}MLS数据集metadata.csv不存在或为空，创建中...${NC}"
            
            # 使用Python脚本处理MLS数据集
            MLS_DIR="$DATA_DIR/${LANGUAGE}_corpus/raw/mls_extracted/mls_portuguese"
            OUTPUT_DIR="$DATA_DIR/${LANGUAGE}_corpus/raw"
            
            if [ -f "process_mls.py" ]; then
                log "运行process_mls.py脚本处理MLS数据集..."
                python process_mls.py --mls-dir "$MLS_DIR" --output-dir "$OUTPUT_DIR" --language "$LANGUAGE" --sample-rate "$SAMPLE_RATE" 2>&1 | tee -a "$LOG_FILE"
                if [ ${PIPESTATUS[0]} -ne 0 ]; then
                    log "${RED}MLS数据处理失败${NC}"
                    exit 1
                fi
            else
                log "${YELLOW}未找到process_mls.py脚本，使用内置处理方式...${NC}"
                
                # 创建metadata.csv文件
                echo "file_path|speaker|text|language" > "$METADATA_FILE"
                
                # 处理训练集
                TRAIN_AUDIO_DIR="$MLS_DIR/train/audio"
                TRAIN_TRANS_FILE="$MLS_DIR/train/transcripts.txt"
                
                if [ -f "$TRAIN_TRANS_FILE" ]; then
                    log "处理训练集转录文件..."
                    # 读取前1000行转录，仅处理前500个
                    count=0
                    head -n 1000 "$TRAIN_TRANS_FILE" | while IFS=$'\t' read -r file_id text; do
                        if [ -n "$file_id" ] && [ -n "$text" ]; then
                            # 限制处理数量
                            count=$((count + 1))
                            if [ $count -gt 500 ]; then
                                break
                            fi
                            
                            # 查找对应的音频文件
                            flac_file=$(find "$TRAIN_AUDIO_DIR" -name "${file_id}.flac" -type f 2>/dev/null | head -n 1)
                            if [ -n "$flac_file" ]; then
                                # 提取说话者ID (第一级子目录)
                                rel_path=$(dirname "$flac_file")
                                rel_path=${rel_path#$TRAIN_AUDIO_DIR/}
                                speaker_id=$(echo "$rel_path" | cut -d'/' -f1)
                                
                                # 输出WAV文件名
                                wav_filename="${file_id}.wav"
                                wav_path="$OUTPUT_DIR/$wav_filename"
                                
                                # 使用ffmpeg转换flac到wav
                                if [ ! -f "$wav_path" ]; then
                                    ffmpeg -i "$flac_file" -ar "$SAMPLE_RATE" -ac 1 -hide_banner -loglevel error "$wav_path"
                                fi
                                
                                # 写入metadata
                                echo "${wav_filename}|speaker_${speaker_id}|${text}|${LANGUAGE}" >> "$METADATA_FILE"
                                
                                # 每处理50个样本，打印一次进度
                                if [ $((count % 50)) -eq 0 ]; then
                                    log "已处理 $count 个样本"
                                fi
                            fi
                        fi
                    done
                fi
            fi
            
            # 检查metadata.csv是否成功创建
            if [ -f "$METADATA_FILE" ] && [ $(wc -l < "$METADATA_FILE") -gt 1 ]; then
                log "${GREEN}metadata.csv创建成功，共 $(( $(wc -l < "$METADATA_FILE") - 1 )) 条记录${NC}"
            else
                log "${RED}metadata.csv创建失败${NC}"
                exit 1
            fi
        else
            log "${GREEN}MLS数据集metadata.csv已存在${NC}"
        fi
    fi
else
    log "${YELLOW}跳过数据下载步骤${NC}"
fi
# 下载BERT模型
if [ "$DOWNLOAD_BERT" = true ]; then
    log "${YELLOW}开始下载${LANGUAGE_NAME} BERT模型...${NC}"
    python download_bert_model.py --language "$LANGUAGE" 2>&1 | tee -a "$LOG_FILE"
    BERT_STATUS=${PIPESTATUS[0]}
    if [ $BERT_STATUS -ne 0 ]; then
        log "${RED}BERT模型下载失败, 错误代码: $BERT_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}BERT模型下载完成${NC}"
else
    log "${YELLOW}跳过BERT模型下载${NC}"
fi

# 执行数据预处理脚本
if [ "$SKIP_PREPROCESS" = false ]; then
    log "${YELLOW}开始预处理${LANGUAGE_NAME}数据...${NC}"
    python preprocess_data.py --language "$LANGUAGE" --data-dir "$DATA_DIR" --sample-rate "$SAMPLE_RATE" --dialect "$DIALECT" 2>&1 | tee -a "$LOG_FILE"
    PREPROCESS_STATUS=${PIPESTATUS[0]}
    if [ $PREPROCESS_STATUS -ne 0 ]; then
        log "${RED}数据预处理失败, 错误代码: $PREPROCESS_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}数据预处理完成${NC}"
else
    log "${YELLOW}跳过数据预处理步骤${NC}"
fi
# 执行模型训练脚本
if [ "$SKIP_TRAINING" = false ]; then
    # 训练GPT模型
    log "${YELLOW}开始训练${LANGUAGE_NAME} GPT模型...${NC}"
    python train_gpt_model.py --language "$LANGUAGE" --data-dir "$DATA_DIR/${LANGUAGE}_corpus/prepared" --batch-size "$BATCH_SIZE" --max-epochs "$MAX_EPOCHS" --device "$DEVICE" --dialect "$DIALECT" 2>&1 | tee -a "$LOG_FILE"
    GPT_STATUS=${PIPESTATUS[0]}
    if [ $GPT_STATUS -ne 0 ]; then
        log "${RED}GPT模型训练失败, 错误代码: $GPT_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}GPT模型训练完成${NC}"
    
    # 训练SoVITS模型
    log "${YELLOW}开始训练${LANGUAGE_NAME} SoVITS模型...${NC}"
    python train_sovits_model.py --language "$LANGUAGE" --data-dir "$DATA_DIR/${LANGUAGE}_corpus/prepared" --batch-size "$BATCH_SIZE" --max-epochs "$MAX_EPOCHS" --device "$DEVICE" --dialect "$DIALECT" 2>&1 | tee -a "$LOG_FILE"
    SOVITS_STATUS=${PIPESTATUS[0]}
    if [ $SOVITS_STATUS -ne 0 ]; then
        log "${RED}SoVITS模型训练失败, 错误代码: $SOVITS_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}SoVITS模型训练完成${NC}"
else
    log "${YELLOW}跳过模型训练步骤${NC}"
fi

# 执行测试脚本
if [ "$SKIP_TEST" = false ]; then
    log "${YELLOW}开始测试${LANGUAGE_NAME}支持...${NC}"
    if [ "$LANGUAGE" = "es" ]; then
        python test_pt_es_support.py --test-es 2>&1 | tee -a "$LOG_FILE"
    else
        python test_pt_es_support.py --test-pt 2>&1 | tee -a "$LOG_FILE"
    fi
    TEST_STATUS=${PIPESTATUS[0]}
    if [ $TEST_STATUS -ne 0 ]; then
        log "${RED}测试失败, 错误代码: $TEST_STATUS${NC}"
        exit 1
    fi
    log "${GREEN}测试完成${NC}"
else
    log "${YELLOW}跳过测试步骤${NC}"
fi
# 清理临时文件
if [ "$CLEANUP" = true ]; then
    log "${YELLOW}清理临时文件...${NC}"
    # 删除原始数据集
    if [ -d "$DATA_DIR/${LANGUAGE}_corpus/raw" ]; then
        rm -rf "$DATA_DIR/${LANGUAGE}_corpus/raw/*_extracted"
        find "$DATA_DIR/${LANGUAGE}_corpus/raw" -name "*.tar.gz" -delete
        find "$DATA_DIR/${LANGUAGE}_corpus/raw" -name "*.zip" -delete
    fi
    log "${GREEN}清理完成${NC}"
fi

log "${BLUE}========== 流程完成 ==========${NC}"
log "${GREEN}GPT模型保存在:${NC} checkpoints/${LANGUAGE}_gpt/"
log "${GREEN}SoVITS模型保存在:${NC} checkpoints/${LANGUAGE}_sovits/"
log ""
log "${YELLOW}要使用训练好的模型进行推理，请运行:${NC}"
log "python inference.py --text \"你的文本\" --language $LANGUAGE --reference-audio path/to/reference.wav --gpt-model checkpoints/${LANGUAGE}_gpt/best_model.pth --sovits-model checkpoints/${LANGUAGE}_sovits/best_model.pth --output-path output.wav"
log ""

# 提供示例文本
if [ "$LANGUAGE" = "es" ]; then
    log "${YELLOW}西班牙语示例文本:${NC}"
    log "¡Hola! ¿Cómo estás? Me llamo Claude y soy un asistente virtual."
    log "El cielo es azul y los pájaros cantan en los árboles."
else
    log "${YELLOW}葡萄牙语示例文本:${NC}"
    log "Olá! Como está você? Meu nome é Claude e sou um assistente virtual."
    log "O céu é azul e os pássaros cantam nas árvores."
fi