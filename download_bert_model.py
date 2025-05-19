#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语BERT模型下载脚本
"""

import os
import sys
import argparse
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
logger = logging.getLogger('download_bert_model')

# BERT模型信息
BERT_MODELS = {
    'es': {
        'name': 'spanish-bert',
        'base_url': 'https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased/resolve/main',
        'files': [
            'pytorch_model.bin',
            'config.json',
            'vocab.txt'
        ],
        'description': '西班牙语BERT模型'
    },
    'pt': {
        'name': 'portuguese-bert',
        'base_url': 'https://huggingface.co/neuralmind/bert-base-portuguese-cased/resolve/main',
        'files': [
            'pytorch_model.bin',
            'config.json',
            'vocab.txt'
        ],
        'description': '葡萄牙语BERT模型'
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

def download_bert_model(language):
    """
    下载指定语言的BERT模型
    """
    if language not in BERT_MODELS:
        logger.error(f"不支持的语言: {language}")
        return False
    
    model_info = BERT_MODELS[language]
    logger.info(f"开始下载 {model_info['description']}")
    
    # 创建目录
    model_dir = Path('pretrained_models') / model_info['name']
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # 下载文件
    for file in model_info['files']:
        file_url = f"{model_info['base_url']}/{file}"
        file_path = model_dir / file
        
        if file_path.exists():
            logger.info(f"文件已存在: {file_path}")
            continue
        
        logger.info(f"下载 {file}")
        download_file(file_url, file_path, f"下载 {file}")
    
    # 创建符号链接到GPT_SoVITS目录
    gpt_sovits_model_dir = Path('GPT_SoVITS/pretrained_models') / model_info['name']
    if not gpt_sovits_model_dir.exists():
        # 创建目录
        gpt_sovits_model_dir.mkdir(parents=True, exist_ok=True)
        
        # 复制文件
        for file in model_info['files']:
            src_file = model_dir / file
            dst_file = gpt_sovits_model_dir / file
            
            if not dst_file.exists():
                logger.info(f"复制 {src_file} 到 {dst_file}")
                import shutil
                shutil.copy2(src_file, dst_file)
    
    logger.info(f"{model_info['description']}下载完成")
    return True

def main():
    parser = argparse.ArgumentParser(description='下载西班牙语和葡萄牙语BERT模型')
    parser.add_argument('--language', type=str, required=True,
                        choices=['es', 'pt'],
                        help='语言 (es: 西班牙语, pt: 葡萄牙语)')
    
    args = parser.parse_args()
    
    # 下载BERT模型
    success = download_bert_model(args.language)
    
    if success:
        logger.info("BERT模型下载完成")
        
        # 显示使用说明
        language_full = "西班牙语" if args.language == "es" else "葡萄牙语"
        model_name = BERT_MODELS[args.language]['name']
        
        logger.info(f"\n使用说明:")
        logger.info(f"在训练时指定BERT模型:")
        logger.info(f"python GPT_SoVITS/train_stage1.py \\")
        logger.info(f"  --config GPT_SoVITS/configs/train_config.json \\")
        logger.info(f"  --train_data_dir data/{args.language}_corpus/prepared \\")
        logger.info(f"  --language {args.language} \\")
        logger.info(f"  --bert_model pretrained_models/{model_name} \\")
        logger.info(f"  --save_dir checkpoints/{args.language}_gpt")
    else:
        logger.error("BERT模型下载失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
