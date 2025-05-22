#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 葡萄牙语数据下载脚本
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
logger = logging.getLogger('download_pt_data')

# 数据集URL
DATASETS = {
    'common_voice': {
        'url': 'https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-10.0-2022-07-04/cv-corpus-10.0-2022-07-04-pt.tar.gz',
        'file_type': 'tar.gz',
        'description': 'Mozilla Common Voice 葡萄牙语数据集'
    },
    'coraa': {
        'url': 'https://github.com/nilc-nlp/CORAA/archive/refs/heads/main.zip',
        'file_type': 'zip',
        'description': 'CORAA 巴西葡萄牙语数据集'
    },
    'mls': {
        'url': 'https://dl.fbaipublicfiles.com/mls/mls_portuguese.tar.gz',
        'file_type': 'tar.gz',
        'description': 'Multilingual LibriSpeech 葡萄牙语数据集'
    },
    'voxforge': {
        'url': 'http://www.repository.voxforge1.org/downloads/pt/Trunk/Audio/Main/16kHz_16bit.zip',
        'file_type': 'zip',
        'description': 'VoxForge 葡萄牙语数据集'
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
            
            # 根据方言过滤
            if dialect == 'brazilian' and 'brazil' not in parts[0].lower():
                continue
            if dialect == 'european' and 'portugal' not in parts[0].lower():
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
            metadata_file.write(f"{wav_filename}|{speaker}|{text}|pt\n")
            count += 1
            
            # 限制数量，避免数据集过大
            if count >= 10000:
                break
    
    logger.info(f"处理了 {count} 个音频文件")

def process_coraa(extract_dir, output_dir, dialect):
    """
    处理CORAA数据集 (巴西葡萄牙语)
    注意: CORAA数据集需要单独申请下载
    """
    logger.info("处理CORAA数据集...")
    
    # 检查是否存在CORAA数据集的实际音频文件
    # CORAA数据集通常包含以下结构:
    # - audio/ 目录包含音频文件
    # - transcriptions/ 目录包含转录文件
    
    # 查找可能的目录结构
    possible_dirs = []
    for root, dirs, files in os.walk(extract_dir):
        for d in dirs:
            if any(keyword in d.lower() for keyword in ['audio', 'wav', 'speech', 'coraa']):
                possible_dirs.append(os.path.join(root, d))
    
    logger.info(f"找到可能的音频目录: {possible_dirs}")
    
    # 查找转录文件
    transcript_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            if file.endswith(('.txt', '.csv', '.tsv')) and any(keyword in file.lower() for keyword in ['transcript', 'text', 'label']):
                transcript_files.append(os.path.join(root, file))
    
    logger.info(f"找到可能的转录文件: {transcript_files}")
    
    # 如果没有找到音频和转录文件，说明这不是完整的CORAA数据集
    if not possible_dirs or not transcript_files:
        logger.warning("未找到CORAA数据集的音频文件和转录文件")
        logger.warning("CORAA数据集需要从官方网站单独申请和下载")
        logger.warning("正在创建示例数据集以便继续流程...")
        
        # 创建示例数据集
        subprocess.run([
            sys.executable, 'create_sample_data.py', 
            '--output-dir', output_dir, 
            '--language', 'pt'
        ])
        return
    
    # 如果找到了文件，继续处理
    audio_dirs = possible_dirs
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理每个转录文件
        count = 0
        for transcript_file in transcript_files:
            logger.info(f"处理转录文件: {transcript_file}")
            
            try:
                with open(transcript_file, 'r', encoding='utf-8') as trans_file:
                    for line in trans_file:
                        if not line.strip() or line.startswith('#'):
                            continue
                        
                        # 尝试不同的分隔符
                        parts = None
                        if '\t' in line:
                            parts = line.strip().split('\t')
                        elif '|' in line:
                            parts = line.strip().split('|')
                        elif ',' in line:
                            parts = line.strip().split(',')
                        else:
                            # 尝试空格分隔
                            parts = line.strip().split(' ', 1)
                        
                        if not parts or len(parts) < 2:
                            continue
                        
                        filename = parts[0]
                        text = parts[1]
                        
                        # 查找音频文件
                        audio_path = None
                        for audio_dir in audio_dirs:
                            for root, dirs, files in os.walk(audio_dir):
                                for file in files:
                                    if filename in file or file.startswith(filename):
                                        audio_path = os.path.join(root, file)
                                        break
                                if audio_path:
                                    break
                            if audio_path:
                                break
                        
                        if not audio_path:
                            continue
                        
                        # 复制并转换音频文件
                        wav_filename = f"{filename}.wav"
                        wav_path = os.path.join(output_dir, wav_filename)
                        
                        # 使用ffmpeg转换到wav
                        try:
                            subprocess.run([
                                'ffmpeg', '-i', audio_path, '-ar', '24000', '-ac', '1',
                                '-hide_banner', '-loglevel', 'error', wav_path
                            ], check=True)
                        except subprocess.CalledProcessError:
                            logger.warning(f"转换音频文件失败: {audio_path}")
                            continue
                        
                        # 提取说话者ID
                        speaker = f"speaker_{filename.split('_')[0] if '_' in filename else 'default'}"
                        
                        # 写入metadata
                        metadata_file.write(f"{wav_filename}|{speaker}|{text}|pt\n")
                        count += 1
                        
                        # 限制数量
                        if count >= 100:  # 限制为100个样本用于测试
                            break
            
            except Exception as e:
                logger.error(f"处理转录文件 {transcript_file} 时出错: {e}")
                continue
            
            if count >= 100:
                break
        
        # 如果没有成功处理任何文件，创建示例数据集
        if count == 0:
            logger.warning("未能处理任何音频文件，创建示例数据集...")
            
            # 创建示例数据集
            subprocess.run([
                sys.executable, 'create_sample_data.py', 
                '--output-dir', output_dir, 
                '--language', 'pt'
            ], check=False)
    
    logger.info(f"处理了 {count} 个音频文件")

def process_mls(extract_dir, output_dir, dialect):
    """
    处理MLS数据集
    """
    logger.info("处理MLS数据集...")
    
    # 寻找MLS数据集目录
    mls_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if 'mls_portuguese' in dirs:
            mls_dir = os.path.join(root, 'mls_portuguese')
            break
    
    if not mls_dir:
        logger.error("未找到MLS葡萄牙语数据集目录")
        return
    
    # 查找音频文件和转录文件
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        # 写入标题
        metadata_file.write("file_path|speaker|text|language\n")
        
        # 处理训练集
        train_dir = os.path.join(mls_dir, 'train')
        train_audio_dir = os.path.join(train_dir, 'audio')
        train_trans_file = os.path.join(train_dir, 'transcripts.txt')
        
        if os.path.exists(train_trans_file):
            process_mls_split(train_audio_dir, train_trans_file, output_dir, metadata_file, dialect)
        
        # 处理验证集
        dev_dir = os.path.join(mls_dir, 'dev')
        dev_audio_dir = os.path.join(dev_dir, 'audio')
        dev_trans_file = os.path.join(dev_dir, 'transcripts.txt')
        
        if os.path.exists(dev_trans_file):
            process_mls_split(dev_audio_dir, dev_trans_file, output_dir, metadata_file, dialect)
    
    logger.info(f"metadata.csv创建完成")

def process_mls_split(audio_dir, trans_file, output_dir, metadata_file, dialect):
    """
    处理MLS数据集的一个分割
    """
    # 读取转录文件
    transcripts = {}
    count = 0
    
    logger.info(f"读取转录文件: {trans_file}")
    with open(trans_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                file_id, text = parts
                transcripts[file_id] = text
                count += 1
                if count >= 1000:  # 限制处理的样本数量
                    break
    
    logger.info(f"读取了 {count} 条转录")
    
    # 查找音频文件并处理
    processed_count = 0
    for file_id, text in transcripts.items():
        # 查找对应的音频文件
        flac_files = []
        for root, dirs, files in os.walk(audio_dir):
            for file in files:
                if file == f"{file_id}.flac":
                    flac_files.append(os.path.join(root, file))
        
        for flac_path in flac_files:
            # 提取子目录作为说话者ID
            rel_path = os.path.relpath(os.path.dirname(flac_path), audio_dir)
            speaker_id = rel_path.split(os.path.sep)[0]
            
            # 输出WAV文件名
            wav_filename = f"{file_id}.wav"
            wav_path = os.path.join(output_dir, wav_filename)
            
            # 使用ffmpeg转换flac到wav
            subprocess.run([
                'ffmpeg', '-i', flac_path, '-ar', '24000', '-ac', '1',
                '-hide_banner', '-loglevel', 'error', wav_path
            ])
            
            # 写入metadata
            metadata_file.write(f"{wav_filename}|speaker_{speaker_id}|{text}|pt\n")
            processed_count += 1
            
            # 每处理10个音频文件，记录一次日志
            if processed_count % 10 == 0:
                logger.info(f"已处理 {processed_count} 个音频文件")
            
            # 限制数量，避免数据集过大
            if processed_count >= 500:
                logger.info(f"已达到处理上限，共处理 {processed_count} 个音频文件")
                return
    
    logger.info(f"处理了 {processed_count} 个音频文件")

def main():
    parser = argparse.ArgumentParser(description='下载葡萄牙语语音数据集')
    parser.add_argument('--dataset-type', type=str, default='common_voice',
                        choices=DATASETS.keys(),
                        help='数据集类型 (默认: common_voice)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--dialect', type=str, default='brazilian',
                        choices=['brazilian', 'european'],
                        help='葡萄牙语方言 (默认: brazilian)')
    
    args = parser.parse_args()
    
    # 创建目录
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / 'pt_corpus' / 'raw'
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
    elif args.dataset_type == 'coraa':
        process_coraa(extract_dir, raw_dir, args.dialect)
    elif args.dataset_type == 'mls':
        process_mls(extract_dir, raw_dir, args.dialect)
    else:
        logger.warning(f"暂不支持自动处理 {args.dataset_type} 数据集，请手动处理")
    
    logger.info("数据下载和处理完成")

if __name__ == '__main__':
    main()
