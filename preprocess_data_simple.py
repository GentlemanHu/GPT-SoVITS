#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的数据预处理脚本 - 避免复杂依赖
"""

import os
import sys
import argparse
import logging
import subprocess
import torch
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('preprocess_data_simple')

def process_audio(audio_path, output_path, sample_rate):
    """
    处理音频文件：重采样、转换为单声道、标准化
    """
    try:
        # 使用ffmpeg处理音频
        subprocess.run([
            'ffmpeg', '-i', audio_path, 
            '-ar', str(sample_rate), 
            '-ac', '1', 
            '-hide_banner', 
            '-loglevel', 'error',
            '-y',  # 覆盖输出文件
            output_path
        ], check=True)
        return True
    except Exception as e:
        logger.error(f"处理音频 {audio_path} 失败: {e}")
        return False

def prepare_metadata(metadata_path, output_dir, language):
    """
    准备训练所需的元数据文件
    """
    try:
        # 读取原始metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 跳过标题行
        if lines[0].startswith('file_path'):
            lines = lines[1:]
        
        # 创建输出文件
        name2text_path = os.path.join(output_dir, '2-name2text.txt')
        
        with open(name2text_path, 'w', encoding='utf-8') as f_text:
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) < 4:
                    continue
                
                filename = parts[0]
                speaker = parts[1]
                text = parts[2]
                lang = parts[3]
                
                # 检查语言
                if lang != language:
                    continue
                
                # 获取文件名（不含扩展名）
                basename = os.path.splitext(filename)[0]
                
                # 写入name2text (简化格式)
                f_text.write(f"{basename}|{speaker}|{lang}|{text}\n")
        
        logger.info(f"已创建 {name2text_path}")
        
        # 创建简化的语义标记文件
        name2semantic_path = os.path.join(output_dir, '6-name2semantic.tsv')
        with open(name2semantic_path, 'w', encoding='utf-8') as f_semantic:
            for line in lines:
                parts = line.strip().split('|')
                if len(parts) < 4:
                    continue
                
                filename = parts[0]
                lang = parts[3]
                
                if lang != language:
                    continue
                
                basename = os.path.splitext(filename)[0]
                
                # 创建虚拟的语义标记（用于测试）
                dummy_semantic = " ".join([str(i % 100) for i in range(20)])
                f_semantic.write(f"{basename}\t{dummy_semantic}\n")
        
        logger.info(f"已创建 {name2semantic_path}")
        
        return True
    except Exception as e:
        logger.error(f"准备元数据失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='简化的数据预处理')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='音频采样率 (默认: 24000)')
    parser.add_argument('--dialect', type=str, default=None,
                        help='方言 (brazilian, european, latin_american)')
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / f"{args.language}_corpus/raw"
    processed_dir = data_dir / f"{args.language}_corpus/processed"
    prepared_dir = data_dir / f"{args.language}_corpus/prepared"
    
    # 创建目录
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查metadata.csv
    metadata_path = raw_dir / 'metadata.csv'
    if not metadata_path.exists():
        logger.error(f"未找到metadata.csv: {metadata_path}")
        sys.exit(1)
    
    # 1. 处理音频文件
    logger.info("开始处理音频文件...")
    
    # 获取所有音频文件
    audio_files = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        # 跳过标题行
        next(f)
        for line in f:
            parts = line.strip().split('|')
            if len(parts) < 4:
                continue
            
            filename = parts[0]
            audio_path = raw_dir / filename
            
            if audio_path.exists():
                audio_files.append(audio_path)
    
    # 处理音频文件
    success_count = 0
    for audio_path in tqdm(audio_files, desc="处理音频"):
        output_path = processed_dir / audio_path.name
        if process_audio(str(audio_path), str(output_path), args.sample_rate):
            success_count += 1
    
    logger.info(f"音频处理完成，成功处理 {success_count}/{len(audio_files)} 个文件")
    
    # 2. 准备元数据
    logger.info("准备元数据...")
    prepare_metadata(metadata_path, prepared_dir, args.language)
    
    logger.info("简化预处理完成")
    logger.info("注意: 此预处理使用简化流程，适用于测试")
    logger.info("对于生产环境，建议使用完整的预处理流程")

if __name__ == '__main__':
    main()
