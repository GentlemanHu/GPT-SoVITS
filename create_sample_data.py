#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建示例葡萄牙语数据集用于测试
"""

import os
import sys
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('create_sample_data')

def create_sample_dataset(output_dir, language='pt'):
    """
    创建示例数据集用于测试
    """
    logger.info(f"创建示例{language}数据集...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 示例葡萄牙语文本
    if language == 'pt':
        sample_texts = [
            "Olá, como você está hoje?",
            "Meu nome é Claude e sou um assistente virtual.",
            "O céu está azul e o sol está brilhando.",
            "Eu gosto de aprender novos idiomas.",
            "A música brasileira é muito bonita.",
            "Vamos estudar português juntos.",
            "O Brasil é um país muito grande.",
            "Adoro falar em português.",
            "A comida brasileira é deliciosa.",
            "Muito obrigado pela sua ajuda."
        ]
    else:  # 西班牙语
        sample_texts = [
            "Hola, ¿cómo estás hoy?",
            "Mi nombre es Claude y soy un asistente virtual.",
            "El cielo está azul y el sol está brillando.",
            "Me gusta aprender nuevos idiomas.",
            "La música española es muy hermosa.",
            "Vamos a estudiar español juntos.",
            "España es un país muy bonito.",
            "Me encanta hablar en español.",
            "La comida española es deliciosa.",
            "Muchas gracias por tu ayuda."
        ]
    
    # 创建metadata.csv
    metadata_path = os.path.join(output_dir, 'metadata.csv')
    logger.info(f"创建metadata.csv: {metadata_path}")
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        # 写入标题
        f.write("file_path|speaker|text|language\n")
        
        # 写入示例数据
        for i, text in enumerate(sample_texts):
            filename = f"sample_{i:03d}.wav"
            speaker = f"speaker_{i % 3}"  # 3个不同的说话者
            f.write(f"{filename}|{speaker}|{text}|{language}\n")
    
    logger.info(f"已创建包含 {len(sample_texts)} 个样本的metadata.csv")
    logger.info("注意: 这是一个示例数据集，您需要提供真实的音频文件来进行训练")
    logger.info("请将对应的WAV音频文件放置在相同目录下，文件名需要与metadata.csv中的file_path匹配")
    
    return True

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='创建示例数据集')
    parser.add_argument('--output-dir', type=str, default='./data/pt_corpus/raw',
                        help='输出目录 (默认: ./data/pt_corpus/raw)')
    parser.add_argument('--language', type=str, default='pt', choices=['pt', 'es'],
                        help='语言 (默认: pt)')
    
    args = parser.parse_args()
    
    success = create_sample_dataset(args.output_dir, args.language)
    
    if success:
        logger.info("示例数据集创建完成")
    else:
        logger.error("示例数据集创建失败")
        sys.exit(1)

if __name__ == '__main__':
    main()
