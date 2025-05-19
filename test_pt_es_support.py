#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-SoVITS 西班牙语和葡萄牙语支持测试脚本
"""

import os
import sys
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('test_pt_es_support')

def test_portuguese():
    """
    测试葡萄牙语支持
    """
    logger.info("测试葡萄牙语支持...")

    # 导入葡萄牙语模块
    from text.portuguese import g2p as pt_g2p, normalize_text as pt_normalize
    
    # 测试文本规范化
    test_texts = [
        "Olá, como está você?",
        "Eu tenho 25 anos e moro no Brasil.",
        "A temperatura hoje é de 30°C.",
        "O sr. João chegou às 15:30.",
        "Obrigado pela sua ajuda!"
    ]

    logger.info("\n葡萄牙语文本规范化测试:")
    for text in test_texts:
        normalized = pt_normalize(text)
        logger.info(f"原文: {text}")
        logger.info(f"规范化: {normalized}")
        logger.info("-" * 50)

    # 测试G2P转换
    logger.info("\n葡萄牙语G2P转换测试:")
    for text in test_texts:
        normalized = pt_normalize(text)
        phonemes, word2ph = pt_g2p(normalized)
        logger.info(f"原文: {text}")
        logger.info(f"音素: {phonemes}")
        logger.info(f"词到音素映射: {word2ph}")
        logger.info(f"音素总数: {len(phonemes)}")
        logger.info(f"词到音素映射总数: {sum(word2ph)}")
        logger.info("-" * 50)

def test_spanish():
    """
    测试西班牙语支持
    """
    logger.info("测试西班牙语支持...")

    # 导入西班牙语模块
    from text.spanish import g2p as es_g2p, normalize_text as es_normalize
    
    # 测试文本规范化
    test_texts = [
        "¡Hola! ¿Cómo estás?",
        "Tengo 30 años y vivo en España.",
        "La temperatura hoy es de 25°C.",
        "El Sr. García llegó a las 14:45.",
        "¡Gracias por tu ayuda!"
    ]

    logger.info("\n西班牙语文本规范化测试:")
    for text in test_texts:
        normalized = es_normalize(text)
        logger.info(f"原文: {text}")
        logger.info(f"规范化: {normalized}")
        logger.info("-" * 50)

    # 测试G2P转换
    logger.info("\n西班牙语G2P转换测试:")
    for text in test_texts:
        normalized = es_normalize(text)
        phonemes, word2ph = es_g2p(normalized)
        logger.info(f"原文: {text}")
        logger.info(f"音素: {phonemes}")
        logger.info(f"词到音素映射: {word2ph}")
        logger.info(f"音素总数: {len(phonemes)}")
        logger.info(f"词到音素映射总数: {sum(word2ph)}")
        logger.info("-" * 50)

def test_language_segmenter():
    """
    测试语言分割器
    """
    logger.info("测试语言分割器...")
    
    # 导入语言分割器
    from text.LangSegmenter.langsegmenter import LangSegmenter
    
    # 测试混合语言文本
    test_texts = [
        "Hello, mi nombre es Carlos. I speak both español and English.",
        "Olá, my name is João. I can speak português and English.",
        "This is a mixed text with some español words and some português palavras."
    ]
    
    for text in test_texts:
        logger.info(f"混合文本: {text}")
        segments = LangSegmenter.getTexts(text)
        for segment in segments:
            logger.info(f"语言: {segment['lang']}, 文本: {segment['text']}")
        logger.info("-" * 50)

def test_cleaner():
    """
    测试文本清理器
    """
    logger.info("测试文本清理器...")
    
    # 导入文本清理器
    from text.cleaner import clean_text
    
    # 测试文本
    test_texts = [
        {"text": "¡Hola! ¿Cómo estás?", "language": "es"},
        {"text": "Olá, como está você?", "language": "pt"},
        {"text": "Hello, how are you?", "language": "en"},
    ]
    
    for item in test_texts:
        text = item["text"]
        language = item["language"]
        
        logger.info(f"语言: {language}, 文本: {text}")
        phones, word2ph, norm_text = clean_text(text, language)
        logger.info(f"规范化文本: {norm_text}")
        logger.info(f"音素: {phones}")
        logger.info(f"词到音素映射: {word2ph}")
        logger.info("-" * 50)

def test_model_compatibility():
    """
    测试模型兼容性
    """
    logger.info("测试模型兼容性...")
    
    # 检查GPT-SoVITS目录结构
    if not os.path.exists("GPT_SoVITS"):
        logger.error("未找到GPT_SoVITS目录，请确保在GPT-SoVITS项目根目录下运行此脚本")
        return
    
    # 检查配置文件
    config_files = [
        "GPT_SoVITS/configs/s1longer.yaml",
        "GPT_SoVITS/configs/s2.json"
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            logger.info(f"找到配置文件: {config_file}")
        else:
            logger.warning(f"未找到配置文件: {config_file}")
    
    # 检查预训练模型
    pretrained_models = [
        "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        "GPT_SoVITS/pretrained_models/s2G488k.pth"
    ]
    
    for model_path in pretrained_models:
        if os.path.exists(model_path):
            logger.info(f"找到预训练模型: {model_path}")
        else:
            logger.warning(f"未找到预训练模型: {model_path}")
    
    # 检查BERT模型
    bert_models = [
        "pretrained_models/spanish-bert",
        "pretrained_models/portuguese-bert"
    ]
    
    for model_path in bert_models:
        if os.path.exists(model_path):
            logger.info(f"找到BERT模型: {model_path}")
        else:
            logger.warning(f"未找到BERT模型: {model_path}，可以使用download_bert_model.py下载")

def main():
    parser = argparse.ArgumentParser(description='测试GPT-SoVITS的西班牙语和葡萄牙语支持')
    parser.add_argument('--test-pt', action='store_true',
                        help='测试葡萄牙语支持')
    parser.add_argument('--test-es', action='store_true',
                        help='测试西班牙语支持')
    parser.add_argument('--test-segmenter', action='store_true',
                        help='测试语言分割器')
    parser.add_argument('--test-cleaner', action='store_true',
                        help='测试文本清理器')
    parser.add_argument('--test-compatibility', action='store_true',
                        help='测试模型兼容性')
    parser.add_argument('--test-all', action='store_true',
                        help='测试所有功能')
    
    args = parser.parse_args()
    
    # 如果没有指定测试项，则测试所有功能
    if not (args.test_pt or args.test_es or args.test_segmenter or args.test_cleaner or args.test_compatibility or args.test_all):
        args.test_all = True
    
    # 设置工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 添加当前目录到Python路径
    sys.path.insert(0, script_dir)
    
    logger.info("=" * 80)
    logger.info("GPT-SoVITS 葡萄牙语和西班牙语支持测试")
    logger.info("=" * 80)
    
    # 运行测试
    if args.test_pt or args.test_all:
        test_portuguese()
    
    if args.test_es or args.test_all:
        test_spanish()
    
    if args.test_segmenter or args.test_all:
        test_language_segmenter()
    
    if args.test_cleaner or args.test_all:
        test_cleaner()
    
    if args.test_compatibility or args.test_all:
        test_model_compatibility()
    
    logger.info("测试完成")

if __name__ == '__main__':
    main()
