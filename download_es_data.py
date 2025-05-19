#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语数据下载脚本
"""

import os
import sys
import argparse
import subprocess
import shutil
import zipfile
import tarfile
import logging
import requests
from tqdm import tqdm
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('download_es_data')

# 数据集URL
DATASETS = {
    'common_voice': {
        'url': 'https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-es.tar.gz',
        'file_type': 'tar.gz',
        'description': 'Mozilla Common Voice 西班牙语数据集'
    },
    'ciempiess': {
        'url': 'http://www.ciempiess.org/downloads/CIEMPIESS_v1.0.zip',
        'file_type': 'zip',
        'description': 'CIEMPIESS 墨西哥西班牙语数据集'
    },
    'mls': {
        'url': 'https://dl.fbaipublicfiles.com/mls/mls_spanish.tar.gz',
        'file_type': 'tar.gz',
        'description': 'Multilingual LibriSpeech 西班牙语数据集'
    },
    'css10': {
        'url': 'https://github.com/Kyubyong/css10/raw/master/es.zip',
        'file_type': 'zip',
        'description': 'CSS10 西班牙语数据集'
    }
}

def download_file(url, destination, desc=None):
    """
    下载文件并显示进度条
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    
    if desc is None:
        desc = f"下载 {url.split('/')[-1]}"
    
    with open(destination, 'wb') as file, tqdm(
            desc=desc,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)

def extract_archive(archive_path, extract_to, file_type):
    """
    解压缩文件
    """
    logger.info(f"正在解压 {archive_path} 到 {extract_to}")
    
    if file_type == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif file_type == 'tar.gz':
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    else:
        logger.error(f"不支持的文件类型: {file_type}")
        sys.exit(1)
    
    logger.info(f"解压完成")

def process_common_voice(extract_dir, output_dir, dialect):
    """
    处理Common Voice数据集
    """
    logger.info("处理Common Voice数据集...")
    
    # 找到clips目录
    clips_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if 'clips' in dirs:
            clips_dir = os.path.join(root, 'clips')
            break
    
    if not clips_dir:
        logger.error("未找到clips目录")
        return
    
    # 找到TSV文件
    tsv_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith('.tsv'):
                tsv_files.append(os.path.join(root, file))
    
    if not tsv_files:
        logger.error("未找到TSV文件")
        return
    
    # 优先使用validated.tsv
    validated_tsv = None
    for tsv in tsv_files:
        if 'validated.tsv' in tsv:
            validated_tsv = tsv
            break
    
    if not validated_tsv:
        logger.warning("未找到validated.tsv，使用第一个找到的TSV文件")
        validated_tsv = tsv_files[0]
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(validated_tsv, 'r', encoding='utf-8') as tsv_file, \
         open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        
        # 跳过标题行
        next(tsv_file)
        
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理每一行
        count = 0
        for line in tsv_file:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            
            filename = parts[1]
            text = parts[2]
            
            # 跳过过短的文本
            if len(text) < 5:
                continue
            
            # 构建文件路径
            mp3_path = os.path.join(clips_dir, filename)
            if not os.path.exists(mp3_path):
                continue
            
            # 复制音频文件
            wav_filename = filename.replace('.mp3', '.wav')
            wav_path = os.path.join(output_dir, wav_filename)
            
            # 使用ffmpeg转换mp3到wav
            subprocess.run([
                'ffmpeg', '-i', mp3_path, '-ar', '24000', '-ac', '1', 
                '-hide_banner', '-loglevel', 'error', wav_path
            ])
            
            # 提取说话者ID
            speaker = f"speaker_{filename.split('_')[0]}"
            
            # 写入metadata
            metadata_file.write(f"{wav_filename}|{speaker}|{text}|es\n")
            count += 1
            
            # 限制数量，避免数据集过大
            if count >= 10000:
                break
    
    logger.info(f"处理了 {count} 个音频文件")

def process_ciempiess(extract_dir, output_dir, dialect):
    """
    处理CIEMPIESS数据集
    """
    logger.info("处理CIEMPIESS数据集...")
    
    # 找到音频目录和转录文件
    audio_dir = None
    transcript_file = None
    
    for root, dirs, files in os.walk(extract_dir):
        for dir in dirs:
            if 'audio' in dir.lower():
                audio_dir = os.path.join(root, dir)
        
        for file in files:
            if 'transcript' in file.lower() and file.endswith('.txt'):
                transcript_file = os.path.join(root, file)
    
    if not audio_dir or not transcript_file:
        logger.error(f"未找到音频目录或转录文件: audio_dir={audio_dir}, transcript_file={transcript_file}")
        return
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(transcript_file, 'r', encoding='utf-8') as trans_file, \
         open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理每一行
        count = 0
        for line in trans_file:
            if not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            filename = parts[0]
            text = ' '.join(parts[1:])
            
            # 查找音频文件
            audio_path = None
            for root, dirs, files in os.walk(audio_dir):
                for file in files:
                    if filename in file:
                        audio_path = os.path.join(root, file)
                        break
                if audio_path:
                    break
            
            if not audio_path:
                continue
            
            # 复制并转换音频文件
            wav_filename = f"{filename}.wav"
            wav_path = os.path.join(output_dir, wav_filename)
            
            # 使用ffmpeg转换到wav
            subprocess.run([
                'ffmpeg', '-i', audio_path, '-ar', '24000', '-ac', '1',
                '-hide_banner', '-loglevel', 'error', wav_path
            ])
            
            # 提取说话者ID
            speaker = f"speaker_{filename.split('_')[0]}"
            
            # 写入metadata
            metadata_file.write(f"{wav_filename}|{speaker}|{text}|es\n")
            count += 1
            
            # 限制数量
            if count >= 10000:
                break
    
    logger.info(f"处理了 {count} 个音频文件")

def main():
    parser = argparse.ArgumentParser(description='下载西班牙语语音数据集')
    parser.add_argument('--dataset-type', type=str, default='common_voice',
                        choices=DATASETS.keys(),
                        help='数据集类型 (默认: common_voice)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--dialect', type=str, default='european',
                        choices=['european', 'latin_american'],
                        help='西班牙语方言 (默认: european)')
    
    args = parser.parse_args()
    
    # 创建目录
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / 'es_corpus' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取数据集信息
    dataset_info = DATASETS.get(args.dataset_type)
    if not dataset_info:
        logger.error(f"未知的数据集类型: {args.dataset_type}")
        sys.exit(1)
    
    logger.info(f"开始下载 {dataset_info['description']}")
    
    # 下载数据集
    archive_name = dataset_info['url'].split('/')[-1]
    archive_path = raw_dir / archive_name
    
    if not archive_path.exists():
        download_file(dataset_info['url'], archive_path, f"下载 {args.dataset_type} 数据集")
    else:
        logger.info(f"文件已存在: {archive_path}")
    
    # 解压数据集
    extract_dir = raw_dir / f"{args.dataset_type}_extracted"
    extract_dir.mkdir(exist_ok=True)
    
    extract_archive(archive_path, extract_dir, dataset_info['file_type'])
    
    # 处理数据集
    if args.dataset_type == 'common_voice':
        process_common_voice(extract_dir, raw_dir, args.dialect)
    elif args.dataset_type == 'ciempiess':
        process_ciempiess(extract_dir, raw_dir, args.dialect)
    else:
        logger.warning(f"暂不支持自动处理 {args.dataset_type} 数据集，请手动处理")
    
    logger.info("数据下载和处理完成")

if __name__ == '__main__':
    main()
