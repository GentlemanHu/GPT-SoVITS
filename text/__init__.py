"""
GPT-SoVITS 语言模块 - 支持多语言文本处理
包括英语、中文、日语、韩语、西班牙语和葡萄牙语
"""

from text.cleaner import clean_text, clean_text_bert, text_to_sequence
from text.LangSegmenter.langsegmenter import LangSegmenter
from text.spanish import g2p as es_g2p, normalize_text as es_normalize
from text.portuguese import g2p as pt_g2p, normalize_text as pt_normalize

__all__ = [
    'clean_text', 
    'clean_text_bert', 
    'text_to_sequence', 
    'LangSegmenter',
    'es_g2p', 
    'es_normalize',
    'pt_g2p', 
    'pt_normalize'
]
