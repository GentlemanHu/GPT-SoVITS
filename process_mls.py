#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接处理MLS数据集并生成metadata.csv
"""

import os
import sys
import argparse
import logging
import subprocess
from pathlib import Path
import glob
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('process_mls')

def find_flac_file(audio_dir, file_id):
    """
    查找对应的FLAC文件
    """
    # 使用glob快速查找文件
    pattern = os.path.join(audio_dir, "**", f"{file_id}.flac")
    matching_files = glob.glob(pattern, recursive=True)
    if matching_files:
        return matching_files[0]
    return None

def process_mls(mls_dir, output_dir, language="pt", max_samples=1000, sample_rate=24000):
    """
    处理MLS数据集
    """
    logger.info(f"处理MLS数据集: {mls_dir}")
    
    # 检查MLS目录
    if not os.path.exists(mls_dir):
        logger.error(f"MLS目录不存在: {mls_dir}")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理训练集
        train_dir = os.path.join(mls_dir, 'train')
        train_audio_dir = os.path.join(train_dir, 'audio')
        train_trans_file = os.path.join(train_dir, 'transcripts.txt')
        
        if not os.path.exists(train_trans_file):
            logger.error(f"转录文件不存在: {train_trans_file}")
            return False
        
        # 索引音频文件
        logger.info("索引音频文件...")
        audio_files = {}
        for root, dirs, files in os.walk(train_audio_dir):
            for file in files:
                if file.endswith('.flac'):
                    file_id = os.path.splitext(file)[0]
                    audio_files[file_id] = os.path.join(root, file)
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 读取转录文件
        logger.info(f"读取转录文件: {train_trans_file}")
        processed_count = 0
        
        with open(train_trans_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    file_id, text = parts
                    
                    # 查找音频文件
                    flac_path = audio_files.get(file_id)
                    if not flac_path:
                        continue
                    
                    # 提取说话者ID
                    rel_path = os.path.relpath(os.path.dirname(flac_path), train_audio_dir)
                    speaker_id = rel_path.split(os.path.sep)[0]
                    
                    # 输出WAV文件名
                    wav_filename = f"{file_id}.wav"
                    wav_path = os.path.join(output_dir, wav_filename)
                    
                    # 使用ffmpeg转换flac到wav
                    subprocess.run([
                        'ffmpeg', '-i', flac_path, '-ar', str(sample_rate), '-ac', '1',
                        '-hide_banner', '-loglevel', 'error', wav_path
                    ], check=False)
                    
                    # 写入metadata
                    metadata_file.write(f"{wav_filename}|speaker_{speaker_id}|{text}|{language}\n")
                    processed_count += 1
                    
                    # 定期报告进度
                    if processed_count % 10 == 0:
                        logger.info(f"已处理 {processed_count} 个样本")
    
    logger.info(f"完成处理，共处理 {processed_count} 个样本")
    return True

def main():
    parser = argparse.ArgumentParser(description='处理MLS数据集并生成metadata.csv')
    parser.add_argument('--mls-dir', type=str, default='data/pt_corpus/raw/mls_extracted/mls_portuguese',
                        help='MLS数据集目录 (默认: data/pt_corpus/raw/mls_extracted/mls_portuguese)')
    parser.add_argument('--output-dir', type=str, default='data/pt_corpus/raw',
                        help='输出目录 (默认: data/pt_corpus/raw)')
    parser.add_argument('--language', type=str, default='pt', choices=['pt', 'es'],
                        help='语言 (默认: pt)')
    parser.add_argument('--max-samples', type=int, default=1000,
                        help='最大处理样本数 (默认: 1000)')
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='音频采样率 (默认: 24000)')
    
    args = parser.parse_args()
    
    # 开始处理
    start_time = time.time()
    logger.info("开始处理MLS数据集...")
    
    success = process_mls(
        args.mls_dir,
        args.output_dir,
        args.language,
        args.max_samples,
        args.sample_rate
    )
    
    if success:
        elapsed_time = time.time() - start_time
        logger.info(f"处理完成，耗时: {elapsed_time:.2f}秒")
    else:
        logger.error("处理失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
