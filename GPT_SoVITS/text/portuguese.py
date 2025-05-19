import re
import os
from typing import Dict, List, Union, Tuple

# 葡萄牙语音素集 - 扩展版本
_portuguese_phonemes = [
    # 元音
    'a', 'e', 'i', 'o', 'u', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô', 'ã', 'õ',
    # 辅音
    'b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'z',
    'ç', 'j', 'x', 'h', 'rr', 'nh', 'lh', 'ch', 'br', 'cr', 'dr', 'fr', 'gr', 'pr', 'tr', 'vr',
    # 特殊符号
    ' ', '.', ',', '!', '?', ';', ':', '-'
]

# 葡萄牙语字形到音素的映射 - 扩展版本
_g2p_map = {
    'a': 'a',
    'á': 'á',
    'à': 'a',
    'â': 'â',
    'ã': 'ã',
    'b': 'b',
    'c': 'k',  # 在e,i前发/s/，其他情况发/k/
    'ç': 'ç',
    'd': 'd',
    'e': 'e',
    'é': 'é',
    'ê': 'ê',
    'f': 'f',
    'g': 'g',  # 在e,i前发/ʒ/，其他情况发/g/
    'h': 'h',  # 葡萄牙语中h通常不发音，但在组合中有作用
    'i': 'i',
    'í': 'í',
    'j': 'j',
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'o': 'o',
    'ó': 'ó',
    'ô': 'ô',
    'õ': 'õ',
    'p': 'p',
    'q': 'k',  # 通常与u连用，qu发/k/
    'r': 'r',
    's': 's',  # 在词尾或辅音前发/sh/，在元音间发/z/
    't': 't',
    'u': 'u',
    'ú': 'ú',
    'v': 'v',
    'w': 'v',  # 外来词中使用
    'x': 'x',  # 发音多变，可能是/sh/、/ks/、/z/或/s/
    'y': 'i',  # 外来词中使用
    'z': 'z',  # 在词尾发/s/或/sh/
    ' ': ' ',
    '.': '.',
    ',': ',',
    '!': '!',
    '?': '?',
    ';': ';',
    ':': ':',
    '-': '-'
}

# 特殊组合音素映射 - 扩展版本
_special_combinations = {
    # 辅音组合
    'ch': 'ch',  # 发/sh/
    'lh': 'lh',  # 发/ly/
    'nh': 'nh',  # 发/ny/
    'rr': 'rr',  # 发/h/或/R/
    'qu': 'k',   # 在e,i前发/k/
    'gu': 'g',   # 在e,i前发/g/
    'ce': 'se',  # 发/se/
    'ci': 'si',  # 发/si/
    'ge': 'je',  # 发/je/
    'gi': 'ji',  # 发/ji/
    'sc': 's',   # 在e,i前发/s/
    'sç': 's',   # 发/s/
    'xc': 's',   # 在e,i前发/s/

    # 元音组合
    'ão': 'ão',  # 鼻化元音
    'ãe': 'ãe',  # 鼻化元音
    'õe': 'õe',  # 鼻化元音
    'am': 'ã',   # 词尾发/ã/
    'an': 'ã',   # 词尾发/ã/
    'em': 'ẽ',   # 词尾发/ẽ/
    'en': 'ẽ',   # 词尾发/ẽ/
    'im': 'ĩ',   # 词尾发/ĩ/
    'in': 'ĩ',   # 词尾发/ĩ/
    'om': 'õ',   # 词尾发/õ/
    'on': 'õ',   # 词尾发/õ/
    'um': 'ũ',   # 词尾发/ũ/
    'un': 'ũ',   # 词尾发/ũ/

    # 辅音+r组合
    'br': 'br',
    'cr': 'cr',
    'dr': 'dr',
    'fr': 'fr',
    'gr': 'gr',
    'pr': 'pr',
    'tr': 'tr',
    'vr': 'vr',
}

# 葡萄牙语重音规则
_stress_rules = {
    # 重音规则将在text_normalize中应用
}

# 标点符号映射
rep_map = {
    "：": ":",
    "；": ";",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": "…",
    "$": ".",
    "/": ",",
    "—": "-",
    "~": "…",
    "～": "…",
}

def replace_punctuation(text):
    pattern = re.compile("|".join(re.escape(p) for p in rep_map.keys()))
    return pattern.sub(lambda x: rep_map[x.group()], text)

def replace_consecutive_punctuation(text):
    punctuations = ",.!?;:"
    pattern = f"([{re.escape(punctuations)}])\\1+"
    result = re.sub(pattern, r"\1", text)
    return result

def normalize_text(text: str) -> str:
    """葡萄牙语文本规范化"""
    # 转换为小写
    text = text.lower()

    # 替换数字
    text = re.sub(r'\d+', lambda m: _convert_number(m.group(0)), text)

    # 处理标点符号
    text = replace_punctuation(text)

    # 处理多余的空格
    text = re.sub(r'\s+', ' ', text)

    # 避免重复标点引起的参考泄露
    text = replace_consecutive_punctuation(text)

    # 处理常见缩写
    text = _expand_abbreviations(text)

    # 处理特殊字符
    text = _handle_special_chars(text)

    return text.strip()

def _convert_number(number_str: str) -> str:
    """将数字转换为葡萄牙语单词 - 增强版"""
    try:
        num = int(number_str)
        if num == 0:
            return "zero"
        elif 1 <= num <= 19:
            return _convert_small_number(num)
        elif 20 <= num <= 99:
            tens = num // 10
            units = num % 10
            if units == 0:
                return _get_tens(tens)
            else:
                return f"{_get_tens(tens)} e {_convert_small_number(units)}"
        elif 100 <= num <= 999:
            hundreds = num // 100
            remainder = num % 100
            if remainder == 0:
                return _get_hundreds(hundreds)
            else:
                return f"{_get_hundreds(hundreds)} e {_convert_number(str(remainder))}"
        elif 1000 <= num <= 999999:
            thousands = num // 1000
            remainder = num % 1000
            if thousands == 1:
                thousand_text = "mil"
            else:
                thousand_text = f"{_convert_number(str(thousands))} mil"

            if remainder == 0:
                return thousand_text
            else:
                return f"{thousand_text} e {_convert_number(str(remainder))}"
        else:
            # 对于更大的数字，简单返回数字串
            return " ".join(number_str)
    except ValueError:
        # 如果转换失败，返回原始字符串
        return number_str

def _convert_small_number(num: int) -> str:
    """转换1-19的小数字"""
    small_numbers = {
        1: "um", 2: "dois", 3: "três", 4: "quatro", 5: "cinco",
        6: "seis", 7: "sete", 8: "oito", 9: "nove", 10: "dez",
        11: "onze", 12: "doze", 13: "treze", 14: "catorze", 15: "quinze",
        16: "dezesseis", 17: "dezessete", 18: "dezoito", 19: "dezenove"
    }
    return small_numbers.get(num, "")

def _get_tens(tens: int) -> str:
    """获取十位数的名称"""
    tens_names = {
        2: "vinte", 3: "trinta", 4: "quarenta", 5: "cinquenta",
        6: "sessenta", 7: "setenta", 8: "oitenta", 9: "noventa"
    }
    return tens_names.get(tens, "")

def _get_hundreds(hundreds: int) -> str:
    """获取百位数的名称"""
    hundreds_names = {
        1: "cem", 2: "duzentos", 3: "trezentos", 4: "quatrocentos", 5: "quinhentos",
        6: "seiscentos", 7: "setecentos", 8: "oitocentos", 9: "novecentos"
    }
    return hundreds_names.get(hundreds, "")

def _expand_abbreviations(text: str) -> str:
    """展开葡萄牙语常见缩写"""
    abbreviations = {
        r'\bsr\b': 'senhor',
        r'\bsra\b': 'senhora',
        r'\bdr\b': 'doutor',
        r'\bdra\b': 'doutora',
        r'\bprof\b': 'professor',
        r'\bprofa\b': 'professora',
        r'\bavg\b': 'avenida',
        r'\betc\b': 'etcetera',
        r'\bex\b': 'exemplo',
    }

    for abbr, expansion in abbreviations.items():
        text = re.sub(abbr, expansion, text, flags=re.IGNORECASE)

    return text

def _handle_special_chars(text: str) -> str:
    """处理特殊字符"""
    # 替换特殊字符
    special_chars = {
        'æ': 'ae',
        'œ': 'oe',
        'ü': 'u',
        'ö': 'o',
        'ä': 'a',
    }

    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)

    return text

def _number_to_words(number: str) -> str:
    """将单个数字转换为葡萄牙语单词"""
    number_words = {
        '0': 'zero',
        '1': 'um',
        '2': 'dois',
        '3': 'três',
        '4': 'quatro',
        '5': 'cinco',
        '6': 'seis',
        '7': 'sete',
        '8': 'oito',
        '9': 'nove'
    }
    return number_words.get(number, '')

def _get_context_dependent_pronunciation(char: str, prev_char: str, next_char: str) -> str:
    """根据上下文获取字符的发音"""
    # 处理上下文相关的发音规则
    if char == 's':
        if next_char in ['a', 'e', 'i', 'o', 'u', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô', 'ã', 'õ'] and prev_char in ['a', 'e', 'i', 'o', 'u', 'á', 'é', 'í', 'ó', 'ú', 'â', 'ê', 'ô', 'ã', 'õ']:
            return 'z'  # 元音间的s发/z/
        elif next_char in [' ', '.', ',', '!', '?', ';', ':', '-'] or next_char == '':
            return 'sh'  # 词尾的s发/sh/

    elif char == 'c':
        if next_char in ['e', 'i', 'é', 'í', 'ê']:
            return 's'  # c在e,i前发/s/

    elif char == 'g':
        if next_char in ['e', 'i', 'é', 'í', 'ê']:
            return 'j'  # g在e,i前发/ʒ/

    # 默认返回基本映射
    return _g2p_map.get(char, char)

def g2p(text: str) -> Tuple[List[str], List[int]]:
    """葡萄牙语字形到音素转换 - 增强版"""
    text = normalize_text(text)

    phonemes = []
    word2ph = []

    i = 0
    while i < len(text):
        # 检查特殊组合
        found_special = False
        for combo, phoneme in _special_combinations.items():
            if i + len(combo) <= len(text) and text[i:i+len(combo)] == combo:
                phonemes.append(phoneme)
                word2ph.append(len(combo))
                i += len(combo)
                found_special = True
                break

        if not found_special:
            # 处理单个字符，考虑上下文
            char = text[i]
            prev_char = text[i-1] if i > 0 else ''
            next_char = text[i+1] if i < len(text)-1 else ''

            # 获取上下文相关的发音
            phoneme = _get_context_dependent_pronunciation(char, prev_char, next_char)

            if phoneme:
                phonemes.append(phoneme)
                word2ph.append(1)
            else:
                # 对于未知字符，使用默认音素或跳过
                pass
            i += 1

    return phonemes, word2ph

def text_to_sequence(text: str) -> List[int]:
    """将文本转换为音素序列ID"""
    phonemes, _ = g2p(text)
    # 这里应该有一个映射函数将音素转换为ID
    # 暂时返回音素列表
    return phonemes
