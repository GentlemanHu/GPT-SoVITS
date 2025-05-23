#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语数据预处理脚本
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
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加GPT-SoVITS路径到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'GPT_SoVITS'))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('preprocess_data')

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
            output_path
        ], check=True)
        return True
    except Exception as e:
        logger.error(f"处理音频 {audio_path} 失败: {e}")
        return False

def extract_features(audio_path, output_dir, device):
    """
    提取音频特征
    """
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        
        # 转换为16kHz (用于HuBERT特征提取)
        if sr != 16000:
            audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = audio
        
        # 转换为torch tensor
        audio_tensor = torch.FloatTensor(audio_16k).unsqueeze(0).to(device)
        
        # 尝试导入GPT-SoVITS的特征提取模块
        try:
            from GPT_SoVITS.feature_extractor import cnhubert
        except ImportError:
            try:
                import GPT_SoVITS.feature_extractor.cnhubert as cnhubert
            except ImportError:
                logger.warning("HuBERT特征提取器不可用，跳过特征提取")
                return True
        
        # 加载HuBERT模型
        hubert_model = cnhubert.CNHubert(base_path="GPT_SoVITS/pretrained_models/chinese-hubert-base")
        hubert_model = hubert_model.to(device)
        
        # 提取特征
        with torch.no_grad():
            hubert_features = hubert_model(audio_tensor)
        
        # 保存特征
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        feature_path = os.path.join(output_dir, f"{basename}.pt")
        torch.save(hubert_features, feature_path)
        
        return True
    except Exception as e:
        logger.error(f"提取特征 {audio_path} 失败: {e}")
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
        name2semantic_path = os.path.join(output_dir, '6-name2semantic.tsv')
        
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
                
                # 写入name2text
                f_text.write(f"{basename}|{speaker}|{lang}|{text}\n")
        
        logger.info(f"已创建 {name2text_path}")
        
        # name2semantic将在特征提取后创建
        logger.info(f"name2semantic将在特征提取后创建: {name2semantic_path}")
        
        return True
    except Exception as e:
        logger.error(f"准备元数据失败: {e}")
        return False

def extract_semantic_tokens(hubert_dir, output_dir):
    """
    提取语义标记
    """
    try:
        # 尝试导入GPT-SoVITS模型
        try:
            from GPT_SoVITS.module.models import SynthesizerTrn
        except ImportError:
            try:
                from module.models import SynthesizerTrn
            except ImportError:
                logger.warning("SynthesizerTrn模型不可用，跳过语义标记提取")
                return True
        
        import json
        
        # 加载配置
        config_path = "GPT_SoVITS/configs/s2.json"
        if not os.path.exists(config_path):
            logger.warning(f"配置文件不存在: {config_path}，跳过语义标记提取")
            return True
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载模型
        model = SynthesizerTrn(
            config['data']['filter_length'] // 2 + 1,
            config['train']['segment_size'] // config['data']['hop_length'],
            n_speakers=config['data']['n_speakers'],
            **config['model']
        )
        
        # 检查预训练权重
        model_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
        if not os.path.exists(model_path):
            logger.warning(f"预训练模型不存在: {model_path}，跳过语义标记提取")
            return True
            
        # 加载预训练权重
        model.load_state_dict(torch.load(model_path, map_location="cpu")["model"])
        
        # 设置为评估模式
        model.eval()
        
        # 创建name2semantic文件
        name2semantic_path = os.path.join(output_dir, '6-name2semantic.tsv')
        
        # 检查hubert目录是否存在
        if not os.path.exists(hubert_dir):
            logger.warning(f"HuBERT特征目录不存在: {hubert_dir}，跳过语义标记提取")
            return True
        
        hubert_files = [f for f in os.listdir(hubert_dir) if f.endswith('.pt')]
        if not hubert_files:
            logger.warning("未找到HuBERT特征文件，跳过语义标记提取")
            return True
        
        with open(name2semantic_path, 'w', encoding='utf-8') as f_semantic:
            # 处理每个HuBERT特征文件
            for feature_file in tqdm(hubert_files[:100], desc="提取语义标记"):  # 限制处理数量
                # 获取文件名（不含扩展名）
                basename = os.path.splitext(feature_file)[0]
                
                # 加载HuBERT特征
                feature_path = os.path.join(hubert_dir, feature_file)
                try:
                    ssl_content = torch.load(feature_path, map_location="cpu")
                    
                    # 提取语义标记
                    with torch.no_grad():
                        codes = model.extract_latent(ssl_content)
                        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
                    
                    # 写入name2semantic
                    f_semantic.write(f"{basename}\t{semantic}\n")
                except Exception as e:
                    logger.warning(f"处理特征文件 {feature_file} 失败: {e}")
                    continue
        
        logger.info(f"已创建 {name2semantic_path}")
        
        return True
    except Exception as e:
        logger.error(f"提取语义标记失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='预处理西班牙语和葡萄牙语数据')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='数据目录 (默认: ./data)')
    parser.add_argument('--sample-rate', type=int, default=24000,
                        help='音频采样率 (默认: 24000)')
    parser.add_argument('--dialect', type=str, default=None,
                        help='方言 (brazilian, european, latin_american)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='并行处理的工作线程数 (默认: 4)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备 (默认: cuda 如果可用，否则 cpu)')
    
    args = parser.parse_args()
    
    # 设置路径
    data_dir = Path(args.data_dir)
    raw_dir = data_dir / f"{args.language}_corpus/raw"
    processed_dir = data_dir / f"{args.language}_corpus/processed"
    prepared_dir = data_dir / f"{args.language}_corpus/prepared"
    
    # 创建目录
    processed_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建特征目录
    hubert_dir = prepared_dir / "3-hubert"
    hubert_dir.mkdir(exist_ok=True)
    
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
    
    # 并行处理音频
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for audio_path in audio_files:
            output_path = processed_dir / audio_path.name
            futures.append(
                executor.submit(process_audio, str(audio_path), str(output_path), args.sample_rate)
            )
        
        # 显示进度
        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), desc="处理音频")):
            pass
    
    logger.info(f"音频处理完成，共 {len(audio_files)} 个文件")
    
    # 2. 准备元数据
    logger.info("准备元数据...")
    prepare_metadata(metadata_path, prepared_dir, args.language)
    
    # 3. 提取特征
    logger.info("开始提取特征...")
    
    # 获取处理后的音频文件
    processed_audio_files = list(processed_dir.glob('*.wav'))
    
    # 提取特征
    device = torch.device(args.device)
    for audio_path in tqdm(processed_audio_files, desc="提取特征"):
        extract_features(str(audio_path), str(hubert_dir), device)
    
    logger.info(f"特征提取完成，共 {len(processed_audio_files)} 个文件")
    
    # 4. 提取语义标记
    logger.info("开始提取语义标记...")
    extract_semantic_tokens(hubert_dir, prepared_dir)
    
    logger.info("数据预处理完成")

if __name__ == '__main__':
    main()
