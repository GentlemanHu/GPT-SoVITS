"""
西班牙语文本处理模块 - Spanish text processing module
"""
import re
import unicodedata
from typing import List, Tuple
import string

# 西班牙语正则表达式模式
_whitespace_re = re.compile(r'\s+')
_abbreviations = [(re.compile('\\b%s\\.' % x), '%s' % x) for x in ['Sr', 'Sra', 'Srta', 'Dr', 'Dra', 'Prof']]
_number_re = re.compile(r'[0-9]+')
_currency_re = re.compile(r'([0-9]+)\s?(€|EUR|USD|\$)')
_time_re = re.compile(r'([0-9]{1,2}):([0-9]{2})')
_ordinal_re = re.compile(r'([0-9]+)[°º]')
_date_re = re.compile(r'([0-9]{1,2})/([0-9]{1,2})/([0-9]{2,4})')
_temperature_re = re.compile(r'([0-9]+)°C')

# 西班牙语数字到文字的映射
_num2words = {
    '0': 'cero', '1': 'uno', '2': 'dos', '3': 'tres', '4': 'cuatro', '5': 'cinco',
    '6': 'seis', '7': 'siete', '8': 'ocho', '9': 'nueve', '10': 'diez',
    '11': 'once', '12': 'doce', '13': 'trece', '14': 'catorce', '15': 'quince',
    '16': 'dieciséis', '17': 'diecisiete', '18': 'dieciocho', '19': 'diecinueve',
    '20': 'veinte', '30': 'treinta', '40': 'cuarenta', '50': 'cincuenta',
    '60': 'sesenta', '70': 'setenta', '80': 'ochenta', '90': 'noventa',
}

# 西班牙语音素集
es_phonemes = {
    'consonants': [
        'b', 'd', 'f', 'g', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'x', 'B', 'D', 'G', 'L', 'R', 'T', 'N', 'J',
    ],
    'vowels': [
        'a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U',
    ],
    'special': [
        'h', 'j', 'w', 'y', 'z',
    ],
}

# 西班牙语音素到字母的映射
es_phoneme_to_letter = {
    'a': 'a', 'e': 'e', 'i': 'i', 'o': 'o', 'u': 'u',
    'A': 'á', 'E': 'é', 'I': 'í', 'O': 'ó', 'U': 'ú',
    'b': 'b', 'd': 'd', 'f': 'f', 'g': 'g', 'k': 'c',
    'l': 'l', 'm': 'm', 'n': 'n', 'p': 'p', 'r': 'r', 's': 's', 't': 't',
    'B': 'v', 'D': 'dh', 'G': 'gh', 'L': 'll', 'R': 'rr', 'T': 'ch', 'N': 'ñ', 'J': 'j',
    'h': 'h', 'j': 'j', 'w': 'w', 'y': 'y', 'z': 'z',
    'x': 'x',
}

def normalize_text(text: str) -> str:
    """
    规范化西班牙语文本

    Args:
        text: 输入文本

    Returns:
        规范化后的文本
    """
    # 转换为小写并去除两端空白
    text = text.lower().strip()
    
    # 将所有的标点符号替换为空格，但保留句号和逗号
    punctuation = set(char for char in string.punctuation) - {'.', ','}
    for char in punctuation:
        text = text.replace(char, ' ')
    
    # 将重复的空格替换为单个空格
    text = _whitespace_re.sub(' ', text)
    
    # 处理缩写
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    
    # 处理货币
    def _currency_helper(match):
        amount, currency = match.groups()
        amount_text = ' '.join([_num2words.get(n, n) for n in amount])
        if currency in ['€', 'EUR']:
            return f"{amount_text} euros"
        else:
            return f"{amount_text} dólares"

    text = _currency_re.sub(_currency_helper, text)
    
    # 处理时间
    def _time_helper(match):
        hour, minute = match.groups()
        hour_text = _num2words.get(hour, hour)
        if minute == '00':
            return f"{hour_text} en punto"
        minute_text = _num2words.get(minute, minute)
        return f"{hour_text} y {minute_text}"

    text = _time_re.sub(_time_helper, text)
    
    # 处理序数
    def _ordinal_helper(match):
        number = match.group(1)
        if number == '1':
            return "primero"
        elif number == '2':
            return "segundo"
        elif number == '3':
            return "tercero"
        else:
            return _num2words.get(number, number)

    text = _ordinal_re.sub(_ordinal_helper, text)
    
    # 处理日期
    def _date_helper(match):
        day, month, year = match.groups()
        day_text = _num2words.get(day, day)
        
        months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
                 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
        try:
            month_text = months[int(month) - 1]
        except (ValueError, IndexError):
            month_text = month
        
        year_text = ''
        for digit in year:
            year_text += _num2words.get(digit, digit) + ' '
        
        return f"{day_text} de {month_text} de {year_text.strip()}"

    text = _date_re.sub(_date_helper, text)
    
    # 处理温度
    def _temperature_helper(match):
        temp = match.group(1)
        temp_text = _num2words.get(temp, temp)
        return f"{temp_text} grados celsius"

    text = _temperature_re.sub(_temperature_helper, text)
    
    # 处理普通数字
    def _number_helper(match):
        number = match.group(0)
        if len(number) == 1:
            return _num2words.get(number, number)
        elif len(number) == 2:
            if number in _num2words:
                return _num2words[number]
            tens, ones = number
            if ones == '0':
                return _num2words.get(number, number)
            tens_text = _num2words.get(tens + '0', '')
            ones_text = _num2words.get(ones, ones)
            return f"{tens_text} y {ones_text}"
        else:
            # 对于更大的数字，转换每一位
            return ' '.join([_num2words.get(n, n) for n in number])

    text = _number_re.sub(_number_helper, text)
    
    return text

def _get_spanish_phonemes(word: str) -> List[str]:
    """
    获取西班牙语单词的音素表示
    这是一个简化的音素转换，真实的系统会更复杂

    Args:
        word: 输入单词

    Returns:
        音素列表
    """
    phonemes = []
    skip_next = False
    for i, char in enumerate(word):
        if skip_next:
            skip_next = False
            continue
            
        # 处理重音符号
        if unicodedata.category(char).startswith('M'):
            continue
        
        char_lower = unicodedata.normalize('NFD', char.lower()).replace('\u0301', '')
        
        # 元音音素
        if char_lower in 'aeiou':
            # 检查是否带有重音符号
            if i > 0 and unicodedata.category(word[i-1]).startswith('M'):
                phonemes.append(char_lower.upper())  # 重音元音
            else:
                phonemes.append(char_lower)
        
        # 辅音音素
        elif char_lower == 'b' or char_lower == 'v':
            phonemes.append('b')
        elif char_lower == 'c':
            if i + 1 < len(word) and word[i + 1].lower() in 'ei':
                phonemes.append('s')
            elif i + 1 < len(word) and word[i + 1].lower() == 'h':
                phonemes.append('T')  # ch
                skip_next = True
            else:
                phonemes.append('k')
        elif char_lower == 'd':
            phonemes.append('d')
        elif char_lower == 'f':
            phonemes.append('f')
        elif char_lower == 'g':
            if i + 1 < len(word) and word[i + 1].lower() in 'ei':
                phonemes.append('J')  # j sound
            else:
                phonemes.append('g')
        elif char_lower == 'h':
            # 西班牙语中h通常不发音，忽略
            pass
        elif char_lower == 'j':
            phonemes.append('J')
        elif char_lower == 'k':
            phonemes.append('k')
        elif char_lower == 'l':
            if i + 1 < len(word) and word[i + 1].lower() == 'l':
                phonemes.append('L')  # ll
                skip_next = True
            else:
                phonemes.append('l')
        elif char_lower == 'm':
            phonemes.append('m')
        elif char_lower == 'n':
            if i + 1 < len(word) and word[i + 1].lower() == 'y':
                phonemes.append('N')  # ñ
                skip_next = True
            else:
                phonemes.append('n')
        elif char_lower == 'ñ':
            phonemes.append('N')
        elif char_lower == 'p':
            phonemes.append('p')
        elif char_lower == 'q':
            if i + 1 < len(word) and word[i + 1].lower() == 'u':
                if i + 2 < len(word) and word[i + 2].lower() in 'ei':
                    phonemes.append('k')
                    skip_next = True
                else:
                    phonemes.append('k')
                    if i + 2 < len(word) and word[i + 2].lower() not in 'aeiou':
                        skip_next = True  # Skip 'u' if it's not pronounced
            else:
                phonemes.append('k')
        elif char_lower == 'r':
            if i == 0 or (i > 0 and word[i - 1] in 'nlms'):
                phonemes.append('R')  # Strong r at the beginning or after n, l, m, s
            elif i + 1 < len(word) and word[i + 1].lower() == 'r':
                phonemes.append('R')  # rr
                skip_next = True
            else:
                phonemes.append('r')
        elif char_lower == 's':
            phonemes.append('s')
        elif char_lower == 't':
            phonemes.append('t')
        elif char_lower == 'w':
            phonemes.append('w')
        elif char_lower == 'x':
            phonemes.append('k')
            phonemes.append('s')
        elif char_lower == 'y':
            if i == 0 or word[i - 1] in ' .,:;!?':
                phonemes.append('y')  # Consonant y at the beginning
            else:
                phonemes.append('i')  # Vowel y elsewhere
        elif char_lower == 'z':
            phonemes.append('s')  # In European Spanish, would be 'T' (like 'th' in "think")
        
        # 其他特殊字符可能会被忽略
    
    return phonemes

def g2p(text: str) -> Tuple[List[str], List[int]]:
    """
    将西班牙语文本转换为音素序列和词到音素的映射

    Args:
        text: 输入文本

    Returns:
        (音素列表, 词到音素的映射列表)
    """
    # 将文本按空格分割成单词
    words = text.split()
    
    # 获取每个单词的音素
    all_phonemes = []
    word2ph = []
    
    for word in words:
        # 获取单词的音素
        phonemes = _get_spanish_phonemes(word)
        
        if phonemes:  # 如果音素不为空
            all_phonemes.extend(phonemes)
            word2ph.append(len(phonemes))
        else:
            # 如果没有音素（例如标点符号等），跳过
            continue
    
    return all_phonemes, word2ph
