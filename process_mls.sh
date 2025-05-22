#!/bin/bash
# 处理MLS数据集并生成metadata.csv

echo "开始处理MLS葡萄牙语数据集..."

# 设置路径
MLS_DIR="data/pt_corpus/raw/mls_extracted/mls_portuguese"
OUTPUT_DIR="data/pt_corpus/raw"
METADATA_FILE="${OUTPUT_DIR}/metadata.csv"

# 创建metadata.csv文件
echo "file_path|speaker|text|language" > "${METADATA_FILE}"

# 查找并处理转录文件
process_transcripts() {
    local audio_dir="$1"
    local trans_file="$2"
    
    echo "处理转录文件: ${trans_file}"
    
    # 读取转录文件
    while IFS=$'\t' read -r file_id text; do
        # 查找对应的音频文件
        flac_file=$(find "${audio_dir}" -name "${file_id}.flac" 2>/dev/null)
        if [ -n "${flac_file}" ]; then
            # 提取说话者ID (第一级子目录)
            rel_path=$(realpath --relative-to="${audio_dir}" "$(dirname "${flac_file}")")
            speaker_id=$(echo "${rel_path}" | cut -d'/' -f1)
            
            # 输出WAV文件名
            wav_filename="${file_id}.wav"
            wav_path="${OUTPUT_DIR}/${wav_filename}"
            
            # 使用ffmpeg转换flac到wav
            if [ ! -f "${wav_path}" ]; then
                ffmpeg -i "${flac_file}" -ar 24000 -ac 1 -hide_banner -loglevel error "${wav_path}"
            fi
            
            # 写入metadata
            echo "${wav_filename}|speaker_${speaker_id}|${text}|pt" >> "${METADATA_FILE}"
        fi
    done < "${trans_file}"
}

# 处理训练集
TRAIN_AUDIO_DIR="${MLS_DIR}/train/audio"
TRAIN_TRANS_FILE="${MLS_DIR}/train/transcripts.txt"

if [ -f "${TRAIN_TRANS_FILE}" ]; then
    process_transcripts "${TRAIN_AUDIO_DIR}" "${TRAIN_TRANS_FILE}"
fi

# 处理验证集
DEV_AUDIO_DIR="${MLS_DIR}/dev/audio"
DEV_TRANS_FILE="${MLS_DIR}/dev/transcripts.txt"

if [ -f "${DEV_TRANS_FILE}" ]; then
    process_transcripts "${DEV_AUDIO_DIR}" "${DEV_TRANS_FILE}"
fi

echo "处理完成！"
