#!/bin/bash
# GPT-SoVITS 西班牙语和葡萄牙语训练流程自动化脚本
# 作者: Augment Agent
# 版本: 1.0

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
DEVICE="cuda"  # 使用的设备
SAMPLE_RATE=24000  # 音频采样率
DIALECT=""  # 方言 (巴西葡萄牙语: brazilian, 欧洲葡萄牙语: european, 拉丁美洲西班牙语: latin_american, 欧洲西班牙语: european)

# 显示帮助信息
show_help() {
    echo -e "${BLUE}GPT-SoVITS 西班牙语和葡萄牙语训练流程自动化脚本${NC}"
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
    echo "  --device DEVICE           设置使用的设备 (cuda, cpu)"
    echo "  --sample-rate RATE        设置音频采样率 (默认: 24000)"
    echo "  --dialect DIALECT         设置方言 (brazilian, european, latin_american)"
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

# 创建必要的目录
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
echo ""

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

# 执行数据下载脚本
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "${YELLOW}开始下载${LANGUAGE_NAME}数据集...${NC}"
    python download_${LANGUAGE}_data.py --dataset-type "$DATASET_TYPE" --data-dir "$DATA_DIR" --dialect "$DIALECT"
    echo -e "${GREEN}数据集下载完成${NC}"
else
    echo -e "${YELLOW}跳过数据下载步骤${NC}"
fi

# 下载BERT模型
if [ "$DOWNLOAD_BERT" = true ]; then
    echo -e "${YELLOW}开始下载${LANGUAGE_NAME} BERT模型...${NC}"
    python download_bert_model.py --language "$LANGUAGE"
    echo -e "${GREEN}BERT模型下载完成${NC}"
else
    echo -e "${YELLOW}跳过BERT模型下载${NC}"
fi

# 执行数据预处理脚本
if [ "$SKIP_PREPROCESS" = false ]; then
    echo -e "${YELLOW}开始预处理${LANGUAGE_NAME}数据...${NC}"
    python preprocess_data.py --language "$LANGUAGE" --data-dir "$DATA_DIR" --sample-rate "$SAMPLE_RATE" --dialect "$DIALECT"
    echo -e "${GREEN}数据预处理完成${NC}"
else
    echo -e "${YELLOW}跳过数据预处理步骤${NC}"
fi

# 执行模型训练脚本
if [ "$SKIP_TRAINING" = false ]; then
    # 训练GPT模型
    echo -e "${YELLOW}开始训练${LANGUAGE_NAME} GPT模型...${NC}"
    python train_gpt_model.py --language "$LANGUAGE" --data-dir "$DATA_DIR/${LANGUAGE}_corpus/prepared" --batch-size "$BATCH_SIZE" --max-epochs "$MAX_EPOCHS" --device "$DEVICE" --dialect "$DIALECT"
    echo -e "${GREEN}GPT模型训练完成${NC}"
    
    # 训练SoVITS模型
    echo -e "${YELLOW}开始训练${LANGUAGE_NAME} SoVITS模型...${NC}"
    python train_sovits_model.py --language "$LANGUAGE" --data-dir "$DATA_DIR/${LANGUAGE}_corpus/prepared" --batch-size "$BATCH_SIZE" --max-epochs "$MAX_EPOCHS" --device "$DEVICE" --dialect "$DIALECT"
    echo -e "${GREEN}SoVITS模型训练完成${NC}"
else
    echo -e "${YELLOW}跳过模型训练步骤${NC}"
fi

echo -e "${BLUE}========== 流程完成 ==========${NC}"
echo -e "${GREEN}GPT模型保存在:${NC} checkpoints/${LANGUAGE}_gpt/"
echo -e "${GREEN}SoVITS模型保存在:${NC} checkpoints/${LANGUAGE}_sovits/"
echo ""
echo -e "${YELLOW}要使用训练好的模型进行推理，请运行:${NC}"
echo "python inference.py --text \"你的文本\" --language $LANGUAGE --reference-audio path/to/reference.wav --gpt-model checkpoints/${LANGUAGE}_gpt/best_model.pth --sovits-model checkpoints/${LANGUAGE}_sovits/best_model.pth"
echo ""
