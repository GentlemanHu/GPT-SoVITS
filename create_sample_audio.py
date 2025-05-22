#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建示例音频文件用于测试 - 简化版本
"""

import os
import sys
import numpy as np
import wave
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('create_sample_audio')

def generate_sine_wave(frequency, duration, sample_rate=24000, amplitude=0.3):
    """
    生成正弦波音频
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    # 生成正弦波
    wave_data = amplitude * np.sin(2 * np.pi * frequency * t)
    
    # 添加一些变化以模拟语音
    # 频率调制
    modulation = 0.1 * np.sin(2 * np.pi * 5 * t)
    wave_data = wave_data * (1 + modulation)
    
    # 添加包络以模拟语音的音量变化
    envelope = np.exp(-0.5 * t) * (1 - np.exp(-10 * t))
    wave_data = wave_data * envelope
    
    # 转换为16位整数
    wave_data = (wave_data * 32767).astype(np.int16)
    
    return wave_data

def save_wav_file(audio_data, filename, sample_rate=24000):
    """
    保存WAV文件
    """
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

def create_sample_audio_files(metadata_file, output_dir):
    """
    为metadata.csv中的每个条目创建示例音频文件
    """
    logger.info(f"从 {metadata_file} 读取音频文件列表")
    
    if not os.path.exists(metadata_file):
        logger.error(f"metadata文件不存在: {metadata_file}")
        return False
    
    # 读取metadata.csv
    with open(metadata_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过标题行
    if lines[0].startswith('file_path'):
        lines = lines[1:]
    
    logger.info(f"找到 {len(lines)} 个音频文件需要创建")
    
    for i, line in enumerate(lines):
        parts = line.strip().split('|')
        if len(parts) < 4:
            continue
        
        filename = parts[0]
        speaker = parts[1]
        text = parts[2]
        language = parts[3]
        
        # 生成音频文件路径
        audio_path = os.path.join(output_dir, filename)
        
        # 如果文件已存在，跳过
        if os.path.exists(audio_path):
            logger.info(f"音频文件已存在，跳过: {filename}")
            continue
        
        # 根据说话者生成不同频率的音频
        speaker_id = i % 3  # 3个不同的说话者
        base_frequencies = [220, 330, 440]  # 不同的基频
        frequency = base_frequencies[speaker_id]
        
        # 根据文本长度确定音频时长
        duration = max(1.0, min(5.0, len(text) * 0.1))  # 1-5秒
        
        # 生成正弦波音频
        audio = generate_sine_wave(frequency, duration)
        
        # 保存音频文件
        save_wav_file(audio, audio_path)
        logger.info(f"创建音频文件: {filename} (时长: {duration:.1f}s, 频率: {frequency}Hz)")
    
    logger.info("所有示例音频文件创建完成")
    logger.info("注意: 这些是用于测试的合成音频，不能用于实际的语音合成训练")
    logger.info("如需进行真实训练，请替换为真实的语音数据")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='创建示例音频文件')
    parser.add_argument('--metadata-file', type=str, default='./data/pt_corpus/raw/metadata.csv',
                        help='metadata.csv文件路径 (默认: ./data/pt_corpus/raw/metadata.csv)')
    parser.add_argument('--output-dir', type=str, default='./data/pt_corpus/raw',
                        help='输出目录 (默认: ./data/pt_corpus/raw)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = create_sample_audio_files(args.metadata_file, args.output_dir)
    
    if success:
        logger.info("示例音频文件创建完成")
    else:
        logger.error("示例音频文件创建失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
