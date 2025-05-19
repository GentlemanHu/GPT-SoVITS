import os
import sys
import torch
import numpy as np

# 添加当前目录到Python路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# 直接导入模块，不使用GPT_SoVITS前缀
from text.portuguese import g2p as pt_g2p, normalize_text as pt_normalize
from text.spanish import g2p as es_g2p, normalize_text as es_normalize
from text.LangSegmenter.langsegmenter import LangSegmenter

def test_portuguese():
    print("测试葡萄牙语支持...")

    # 测试文本规范化
    test_texts = [
        "Olá, como está você?",
        "Eu tenho 25 anos e moro no Brasil.",
        "A temperatura hoje é de 30°C.",
        "O sr. João chegou às 15:30.",
        "Obrigado pela sua ajuda!"
    ]

    print("\n葡萄牙语文本规范化测试:")
    for text in test_texts:
        normalized = pt_normalize(text)
        print(f"原文: {text}")
        print(f"规范化: {normalized}")
        print("-" * 50)

    # 测试G2P转换
    print("\n葡萄牙语G2P转换测试:")
    for text in test_texts:
        normalized = pt_normalize(text)
        phonemes, word2ph = pt_g2p(normalized)
        print(f"原文: {text}")
        print(f"音素: {phonemes}")
        print(f"词到音素映射: {word2ph}")
        print(f"音素总数: {len(phonemes)}")
        print(f"词到音素映射总数: {sum(word2ph)}")
        print("-" * 50)

    # 测试语言识别
    mixed_text = "Hello, meu nome é João. I speak both português and English."
    print("\n混合语言识别测试:")
    print(f"混合文本: {mixed_text}")
    lang_segments = LangSegmenter.getTexts(mixed_text)
    for segment in lang_segments:
        print(f"语言: {segment['lang']}, 文本: {segment['text']}")
    print("-" * 50)

def test_spanish():
    print("测试西班牙语支持...")

    # 测试文本规范化
    test_texts = [
        "¡Hola! ¿Cómo estás?",
        "Tengo 30 años y vivo en España.",
        "La temperatura hoy es de 25°C.",
        "El Sr. García llegó a las 14:45.",
        "¡Gracias por tu ayuda!"
    ]

    print("\n西班牙语文本规范化测试:")
    for text in test_texts:
        normalized = es_normalize(text)
        print(f"原文: {text}")
        print(f"规范化: {normalized}")
        print("-" * 50)

    # 测试G2P转换
    print("\n西班牙语G2P转换测试:")
    for text in test_texts:
        normalized = es_normalize(text)
        phonemes, word2ph = es_g2p(normalized)
        print(f"原文: {text}")
        print(f"音素: {phonemes}")
        print(f"词到音素映射: {word2ph}")
        print(f"音素总数: {len(phonemes)}")
        print(f"词到音素映射总数: {sum(word2ph)}")
        print("-" * 50)

    # 测试语言识别
    mixed_text = "Hello, mi nombre es Carlos. I speak both español and English."
    print("\n混合语言识别测试:")
    print(f"混合文本: {mixed_text}")
    lang_segments = LangSegmenter.getTexts(mixed_text)
    for segment in lang_segments:
        print(f"语言: {segment['lang']}, 文本: {segment['text']}")
    print("-" * 50)

if __name__ == "__main__":
    # 设置工作目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 添加当前目录到Python路径
    sys.path.append(script_dir)

    print("=" * 80)
    print("GPT-SoVITS 葡萄牙语和西班牙语支持测试")
    print("=" * 80)

    test_portuguese()
    print("\n")
    test_spanish()
