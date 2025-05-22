"""
文本清理器 - 负责处理不同语言的文本
"""

import unicodedata
from typing import List, Tuple, Optional

# 导入语言处理模块
from text.english import g2p as en_g2p, normalize_text as en_normalize
from text.chinese import g2p as zh_g2p, normalize_text as zh_normalize
from text.japanese import g2p as ja_g2p, normalize_text as ja_normalize
from text.korean import g2p as ko_g2p, normalize_text as ko_normalize
from text.spanish import g2p as es_g2p, normalize_text as es_normalize
from text.portuguese import g2p as pt_g2p, normalize_text as pt_normalize

def clean_text(text: str, language: str) -> Tuple[List[str], List[int], str]:
    """
    清理并处理文本，转换为音素

    Args:
        text: 输入文本
        language: 语言代码 (en, zh, ja, ko, es, pt)

    Returns:
        Tuple[List[str], List[int], str]: (音素列表, 词到音素的映射, 规范化后的文本)
    """
    # 去除控制字符
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    
    # 规范化文本
    if language == 'en':
        normalized_text = en_normalize(text)
        phones, word2ph = en_g2p(normalized_text)
    elif language == 'zh':
        normalized_text = zh_normalize(text)
        phones, word2ph = zh_g2p(normalized_text)
    elif language == 'ja':
        normalized_text = ja_normalize(text)
        phones, word2ph = ja_g2p(normalized_text)
    elif language == 'ko':
        normalized_text = ko_normalize(text)
        phones, word2ph = ko_g2p(normalized_text)
    elif language == 'es':
        normalized_text = es_normalize(text)
        phones, word2ph = es_g2p(normalized_text)
    elif language == 'pt':
        normalized_text = pt_normalize(text)
        phones, word2ph = pt_g2p(normalized_text)
    else:
        # 默认使用英语
        normalized_text = en_normalize(text)
        phones, word2ph = en_g2p(normalized_text)
    
    return phones, word2ph, normalized_text

def clean_text_bert(text: str, language: str) -> str:
    """
    清理文本用于BERT输入

    Args:
        text: 输入文本
        language: 语言代码 (en, zh, ja, ko, es, pt)

    Returns:
        清理后的文本
    """
    # 去除控制字符
    text = ''.join(char for char in text if not unicodedata.category(char).startswith('C'))
    
    # 规范化文本
    if language == 'en':
        normalized_text = en_normalize(text)
    elif language == 'zh':
        normalized_text = zh_normalize(text)
    elif language == 'ja':
        normalized_text = ja_normalize(text)
    elif language == 'ko':
        normalized_text = ko_normalize(text)
    elif language == 'es':
        normalized_text = es_normalize(text)
    elif language == 'pt':
        normalized_text = pt_normalize(text)
    else:
        # 默认使用英语
        normalized_text = en_normalize(text)
    
    return normalized_text

def text_to_sequence(text: str, language: str) -> List[int]:
    """
    将文本转换为音素序列ID

    Args:
        text: 输入文本
        language: 语言代码 (en, zh, ja, ko, es, pt)

    Returns:
        音素ID序列
    """
    phones, _, _ = clean_text(text, language)
    
    # 获取音素映射
    if language == 'en':
        from text.english import _phoneme_to_id
        phoneme_to_id = _phoneme_to_id
    elif language == 'zh':
        from text.chinese import _symbol_to_id
        phoneme_to_id = _symbol_to_id
    elif language == 'ja':
        from text.japanese import _symbol_to_id
        phoneme_to_id = _symbol_to_id
    elif language == 'ko':
        from text.korean import _symbol_to_id
        phoneme_to_id = _symbol_to_id
    elif language == 'es':
        from text.spanish import pt_phoneme_to_letter as phoneme_to_id  # 使用映射为ID
    elif language == 'pt':
        from text.portuguese import pt_phoneme_to_letter as phoneme_to_id  # 使用映射为ID
    else:
        # 默认使用英语
        from text.english import _phoneme_to_id
        phoneme_to_id = _phoneme_to_id
    
    # 将音素转换为ID
    sequence = [phoneme_to_id.get(p, 0) for p in phones]
    return sequence

if __name__ == "__main__":
    # 测试
    test_texts = [
        ("Hello, how are you?", "en"),
        ("你好，你好吗？", "zh"),
        ("こんにちは、お元気ですか？", "ja"),
        ("안녕하세요, 어떻게 지내세요?", "ko"),
        ("¡Hola! ¿Cómo estás?", "es"),
        ("Olá, como está você?", "pt"),
    ]
    
    for text, lang in test_texts:
        print(f"语言: {lang}, 文本: {text}")
        phones, word2ph, norm_text = clean_text(text, lang)
        print(f"规范化文本: {norm_text}")
        print(f"音素: {phones}")
        print(f"词到音素映射: {word2ph}")
        print()
