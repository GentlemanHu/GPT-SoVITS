"""
多语言分割器，支持英语、中文、日语、韩语、西班牙语和葡萄牙语
"""

import re
import unicodedata
from typing import List, Dict, Any

class LangSegmenter:
    """
    语言分割器类，用于分割不同语言的文本
    """
    
    @staticmethod
    def is_chinese(char: str) -> bool:
        """
        判断字符是否为中文
        """
        if not char:
            return False
        return '\u4e00' <= char <= '\u9fff'
    
    @staticmethod
    def is_japanese(char: str) -> bool:
        """
        判断字符是否为日文
        """
        if not char:
            return False
        # 日文平假名
        if '\u3040' <= char <= '\u309f':
            return True
        # 日文片假名
        if '\u30a0' <= char <= '\u30ff':
            return True
        # 其他日文字符
        if '\u31f0' <= char <= '\u31ff':
            return True
        return False
    
    @staticmethod
    def is_korean(char: str) -> bool:
        """
        判断字符是否为韩文
        """
        if not char:
            return False
        # 韩文字符
        if '\uac00' <= char <= '\ud7a3':
            return True
        # 韩文字母
        if '\u1100' <= char <= '\u11ff':
            return True
        return False
    
    @staticmethod
    def is_spanish(text: str) -> bool:
        """
        判断文本是否含有西班牙语特有字符
        """
        if not text:
            return False
        
        # 西班牙语特有字符
        spanish_chars = set(['á', 'é', 'í', 'ó', 'ú', 'ü', 'ñ', '¿', '¡'])
        # 西班牙语常见单词
        spanish_common_words = set(['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 
                                    'y', 'o', 'pero', 'porque', 'como', 'qué', 'quién', 
                                    'dónde', 'cuándo', 'por', 'para', 'es', 'son', 'está', 'están'])
        
        # 检查特有字符
        for char in text:
            if char.lower() in spanish_chars:
                return True
        
        # 检查常见单词
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in spanish_common_words:
                return True
        
        return False
    
    @staticmethod
    def is_portuguese(text: str) -> bool:
        """
        判断文本是否含有葡萄牙语特有字符
        """
        if not text:
            return False
        
        # 葡萄牙语特有字符
        portuguese_chars = set(['á', 'â', 'ã', 'à', 'é', 'ê', 'í', 'ó', 'ô', 'õ', 'ú', 'ç'])
        # 葡萄牙语常见单词
        portuguese_common_words = set(['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 
                                       'e', 'ou', 'mas', 'porque', 'como', 'que', 'quem', 
                                       'onde', 'quando', 'por', 'para', 'é', 'são', 'está', 'estão'])
        
        # 检查特有字符
        for char in text:
            if char.lower() in portuguese_chars:
                return True
        
        # 检查常见单词
        words = re.findall(r'\b\w+\b', text.lower())
        for word in words:
            if word in portuguese_common_words and word not in {'a', 'e', 'o', 'é'}:  # 避免与西班牙语混淆
                return True
        
        return False
    
    @staticmethod
    def is_english(char: str) -> bool:
        """
        判断字符是否为英文
        """
        if not char:
            return False
        try:
            return 'LATIN' in unicodedata.name(char) or char.isascii()
        except ValueError:
            return False
    
    @staticmethod
    def getTexts(text: str) -> List[Dict[str, Any]]:
        """
        分割不同语言的文本
        
        Args:
            text: 输入文本
            
        Returns:
            分割后的文本片段列表，每个片段包含语言类型和文本内容
        """
        if not text:
            return []
        
        language_pattern = r'[\u4e00-\u9fff]+|[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff]+|[\uac00-\ud7a3\u1100-\u11ff]+|[a-zA-Z\s.,!?;:\'\"()0-9]+'
        matches = re.finditer(language_pattern, text)
        
        segments = []
        for match in matches:
            segment_text = match.group(0).strip()
            if not segment_text:
                continue
            
            # 判断语言类型
            if any(LangSegmenter.is_chinese(char) for char in segment_text):
                lang = 'zh'
            elif any(LangSegmenter.is_japanese(char) for char in segment_text):
                lang = 'ja'
            elif any(LangSegmenter.is_korean(char) for char in segment_text):
                lang = 'ko'
            elif LangSegmenter.is_spanish(segment_text):
                lang = 'es'
            elif LangSegmenter.is_portuguese(segment_text):
                lang = 'pt'
            else:
                lang = 'en'  # 默认为英语
            
            segments.append({
                'lang': lang,
                'text': segment_text,
                'start': match.start(),
                'end': match.end()
            })
        
        # 合并相同语言的相邻片段
        merged_segments = []
        for segment in segments:
            if merged_segments and merged_segments[-1]['lang'] == segment['lang']:
                merged_segments[-1]['text'] += ' ' + segment['text']
                merged_segments[-1]['end'] = segment['end']
            else:
                merged_segments.append(segment)
        
        return merged_segments

if __name__ == "__main__":
    # 测试
    test_texts = [
        "Hello, mi nombre es Carlos. I speak both español and English.",
        "Olá, my name is João. I can speak português and English.",
        "This is a mixed text with some español words and some português palavras.",
        "你好，我是张三。I can speak Chinese and English。",
        "こんにちは、私の名前は田中です。",
        "안녕하세요, 제 이름은 김입니다."
    ]
    
    for test_text in test_texts:
        print(f"原文: {test_text}")
        segments = LangSegmenter.getTexts(test_text)
        for i, segment in enumerate(segments):
            print(f"片段 {i+1}: {segment['lang']} - {segment['text']}")
        print()
