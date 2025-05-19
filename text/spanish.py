import re
from typing import Dict, List, Union, Tuple

# 西班牙语音素集 - 扩展版本
_spanish_phonemes = [
    # 元音
    'a', 'e', 'i', 'o', 'u', 'á', 'é', 'í', 'ó', 'ú', 'ü',
    # 辅音
    'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'ñ', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z',
    'ch', 'll', 'rr', 'th', 'fl', 'fr', 'pl', 'pr', 'bl', 'br', 'cl', 'cr', 'gl', 'gr', 'dr', 'tr',
    # 特殊符号
    ' ', '.', ',', '!', '?', ';', ':', '-'
]

# 西班牙语字形到音素的映射 - 扩展版本
_g2p_map = {
    'a': 'a',
    'á': 'á',
    'b': 'b',  # 在词首和m,n后发/b/，其他情况发/β/
    'c': 'k',  # 在e,i前发/s/或/θ/，其他情况发/k/
    'd': 'd',  # 在词首和l,n后发/d/，其他情况发/ð/
    'e': 'e',
    'é': 'é',
    'f': 'f',
    'g': 'g',  # 在e,i前发/x/，其他情况发/g/或/ɣ/
    'h': '',   # 西班牙语中h通常不发音
    'i': 'i',
    'í': 'í',
    'j': 'j',  # 发/x/
    'k': 'k',
    'l': 'l',
    'm': 'm',
    'n': 'n',
    'ñ': 'ñ',  # 发/ɲ/
    'o': 'o',
    'ó': 'ó',
    'p': 'p',
    'q': 'k',  # 通常与u连用，qu发/k/
    'r': 'r',  # 单r发/ɾ/，词首r发/r/
    's': 's',  # 在清辅音前或词尾可能发/s/或/h/
    't': 't',
    'u': 'u',  # 在q,g后通常不发音
    'ú': 'ú',
    'ü': 'ü',  # 在g后发/w/
    'v': 'b',  # 在西班牙语中，v发音与b相同
    'w': 'w',  # 外来词中使用
    'x': 'x',  # 发/ks/，在词首可能发/s/
    'y': 'y',  # 作为辅音时发/j/，作为元音时发/i/
    'z': 'z',  # 发/s/或/θ/
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
    'ch': 'ch',  # 发/tʃ/
    'll': 'll',  # 发/ʎ/或/j/
    'rr': 'rr',  # 发/r/
    'qu': 'k',   # 在e,i前发/k/
    'gu': 'g',   # 在e,i前发/g/
    'gü': 'gü',  # 发/gw/
    'ce': 'se',  # 发/se/或/θe/
    'ci': 'si',  # 发/si/或/θi/
    'ge': 'je',  # 发/xe/
    'gi': 'ji',  # 发/xi/
    'za': 'sa',  # 发/sa/或/θa/
    'ze': 'se',  # 发/se/或/θe/
    'zi': 'si',  # 发/si/或/θi/
    'zo': 'so',  # 发/so/或/θo/
    'zu': 'su',  # 发/su/或/θu/
    
    # 元音组合
    'ue': 'we',  # 发/we/
    'ua': 'wa',  # 发/wa/
    'uo': 'wo',  # 发/wo/
    'ui': 'wi',  # 发/wi/
    'ie': 'ie',  # 发/ie/
    'ia': 'ia',  # 发/ia/
    'io': 'io',  # 发/io/
    'iu': 'iu',  # 发/iu/
    'ei': 'ei',  # 发/ei/
    'ai': 'ai',  # 发/ai/
    'oi': 'oi',  # 发/oi/
    'au': 'au',  # 发/au/
    'eu': 'eu',  # 发/eu/
    'ou': 'ou',  # 发/ou/
    
    # 辅音+l/r组合
    'fl': 'fl',
    'fr': 'fr',
    'pl': 'pl',
    'pr': 'pr',
    'bl': 'bl',
    'br': 'br',
    'cl': 'cl',
    'cr': 'cr',
    'gl': 'gl',
    'gr': 'gr',
    'dr': 'dr',
    'tr': 'tr',
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

def text_normalize(text: str) -> str:
    """西班牙语文本规范化"""
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
    """将数字转换为西班牙语单词 - 增强版"""
    try:
        num = int(number_str)
        if num == 0:
            return "cero"
        elif 1 <= num <= 15:
            return _convert_small_number(num)
        elif 16 <= num <= 19:
            return f"dieci{_convert_small_number(num - 10)}"
        elif 20 <= num <= 29:
            if num == 20:
                return "veinte"
            else:
                return f"veinti{_convert_small_number(num - 20)}"
        elif 30 <= num <= 99:
            tens = num // 10
            units = num % 10
            if units == 0:
                return _get_tens(tens)
            else:
                return f"{_get_tens(tens)} y {_convert_small_number(units)}"
        elif 100 <= num <= 999:
            hundreds = num // 100
            remainder = num % 100
            if remainder == 0:
                return _get_hundreds(hundreds)
            else:
                return f"{_get_hundreds(hundreds)} {_convert_number(str(remainder))}"
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
                return f"{thousand_text} {_convert_number(str(remainder))}"
        else:
            # 对于更大的数字，简单返回数字串
            return " ".join(number_str)
    except ValueError:
        # 如果转换失败，返回原始字符串
        return number_str

def _convert_small_number(num: int) -> str:
    """转换1-15的小数字"""
    small_numbers = {
        1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
        6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
        11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince"
    }
    return small_numbers.get(num, "")

def _get_tens(tens: int) -> str:
    """获取十位数的名称"""
    tens_names = {
        3: "treinta", 4: "cuarenta", 5: "cincuenta",
        6: "sesenta", 7: "setenta", 8: "ochenta", 9: "noventa"
    }
    return tens_names.get(tens, "")

def _get_hundreds(hundreds: int) -> str:
    """获取百位数的名称"""
    hundreds_names = {
        1: "cien", 2: "doscientos", 3: "trescientos", 4: "cuatrocientos", 5: "quinientos",
        6: "seiscientos", 7: "setecientos", 8: "ochocientos", 9: "novecientos"
    }
    # 特殊情况：如果后面有其他数字，"cien"变为"ciento"
    if hundreds == 1:
        return "ciento"
    return hundreds_names.get(hundreds, "")

def _expand_abbreviations(text: str) -> str:
    """展开西班牙语常见缩写"""
    abbreviations = {
        r'\bsr\b': 'señor',
        r'\bsra\b': 'señora',
        r'\bdr\b': 'doctor',
        r'\bdra\b': 'doctora',
        r'\bprof\b': 'profesor',
        r'\bprofa\b': 'profesora',
        r'\bavda\b': 'avenida',
        r'\betc\b': 'etcétera',
        r'\bej\b': 'ejemplo',
        r'\bpág\b': 'página',
        r'\btel\b': 'teléfono',
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
        'ö': 'o',
        'ä': 'a',
        'ë': 'e',
        'ï': 'i',
    }
    
    for char, replacement in special_chars.items():
        text = text.replace(char, replacement)
    
    return text

def _get_context_dependent_pronunciation(char: str, prev_char: str, next_char: str, word_start: bool = False) -> str:
    """根据上下文获取字符的发音"""
    # 处理上下文相关的发音规则
    if char == 'c':
        if next_char in ['e', 'i', 'é', 'í']:
            return 's'  # 在拉丁美洲西班牙语中，c在e,i前发/s/
    
    elif char == 'g':
        if next_char in ['e', 'i', 'é', 'í']:
            return 'j'  # g在e,i前发/x/
    
    elif char == 'r':
        if word_start or prev_char in [' ', '.', ',', '!', '?', ';', ':', '-', 'n', 'l', 's']:
            return 'rr'  # 词首的r发/r/
    
    elif char == 'y':
        if next_char in [' ', '.', ',', '!', '?', ';', ':', '-'] or next_char == '':
            return 'i'  # 词尾的y作为元音发/i/
    
    elif char == 'z':
        return 's'  # 在拉丁美洲西班牙语中，z发/s/
    
    # 默认返回基本映射
    return _g2p_map.get(char, char)

def g2p(text: str) -> Tuple[List[str], List[int]]:
    """西班牙语字形到音素转换 - 增强版"""
    text = text_normalize(text)
    
    phonemes = []
    word2ph = []
    
    words = text.split()
    for word_idx, word in enumerate(words):
        word_phonemes = []
        word_word2ph = []
        
        i = 0
        while i < len(word):
            # 检查特殊组合
            found_special = False
            for combo, phoneme in _special_combinations.items():
                if i + len(combo) <= len(word) and word[i:i+len(combo)] == combo:
                    word_phonemes.append(phoneme)
                    word_word2ph.append(len(combo))
                    i += len(combo)
                    found_special = True
                    break
            
            if not found_special:
                # 处理单个字符，考虑上下文
                char = word[i]
                prev_char = word[i-1] if i > 0 else ' '
                next_char = word[i+1] if i < len(word)-1 else ' '
                word_start = (i == 0)
                
                # 获取上下文相关的发音
                phoneme = _get_context_dependent_pronunciation(char, prev_char, next_char, word_start)
                
                if phoneme:  # 跳过空音素（如h）
                    word_phonemes.append(phoneme)
                    word_word2ph.append(1)
                else:
                    word_word2ph.append(1)  # 对于不发音的字符，仍然需要记录word2ph
                i += 1
        
        # 添加单词的音素和word2ph
        phonemes.extend(word_phonemes)
        word2ph.extend(word_word2ph)
        
        # 如果不是最后一个单词，添加空格
        if word_idx < len(words) - 1:
            phonemes.append(' ')
            word2ph.append(1)
    
    return phonemes, word2ph
