            if i + 1 < len(word) and word[i + 1].lower() in 'mn':
                if char_lower == 'a':
                    phonemes.append('W')  # ã
                elif char_lower == 'o':
                    phonemes.append('Y')  # õ
                else:
                    phonemes.append(char_lower)
                if i + 2 < len(word) and word[i + 2].lower() not in 'aeiou':
                    skip_next = True  # Skip next consonant after nasal vowel if not followed by vowel
            else:
                phonemes.append(char_lower)
        
        # 辅音音素
        elif char_lower == 'b':
            phonemes.append('b')
        elif char_lower == 'c':
            if i + 1 < len(word) and word[i + 1].lower() in 'ei':
                phonemes.append('s')
            elif i + 1 < len(word) and word[i + 1].lower() == 'h':
                phonemes.append('S')  # ch
                skip_next = True
            elif i + 1 < len(word) and word[i + 1].lower() == 'ç':
                phonemes.append('s')
                skip_next = True
            else:
                phonemes.append('k')
        elif char_lower == 'ç':
            phonemes.append('s')
        elif char_lower == 'd':
            phonemes.append('d')
        elif char_lower == 'f':
            phonemes.append('f')
        elif char_lower == 'g':
            if i + 1 < len(word) and word[i + 1].lower() in 'ei':
                phonemes.append('Z')  # j sound in Brazilian Portuguese
            else:
                phonemes.append('g')
        elif char_lower == 'h':
            # 葡萄牙语中h通常不发音，忽略
            pass
        elif char_lower == 'j':
            phonemes.append('Z')  # j sound in Brazilian Portuguese
        elif char_lower == 'k':
            phonemes.append('k')
        elif char_lower == 'l':
            if i + 1 < len(word) and word[i + 1].lower() == 'h':
                phonemes.append('L')  # lh
                skip_next = True
            else:
                phonemes.append('l')
        elif char_lower == 'm':
            phonemes.append('m')
        elif char_lower == 'n':
            if i + 1 < len(word) and word[i + 1].lower() == 'h':
                phonemes.append('N')  # nh
                skip_next = True
            else:
                phonemes.append('n')
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
            if i + 1 == len(word) or (i + 1 < len(word) and word[i + 1] not in 'aeiou'):
                phonemes.append('s')  # Final s or s before consonant
            else:
                phonemes.append('z')  # s between vowels in Brazilian Portuguese
        elif char_lower == 't':
            phonemes.append('t')
        elif char_lower == 'v':
            phonemes.append('v')
        elif char_lower == 'w':
            phonemes.append('w')
        elif char_lower == 'x':
            # x has multiple pronunciations in Portuguese
            if i == 0:
                phonemes.append('S')  # initial x as in "xícara"
            elif i > 0 and word[i - 1] in 'aeiou' and i + 1 < len(word) and word[i + 1] in 'aeiou':
                phonemes.append('z')  # x between vowels as in "exame"
            else:
                phonemes.append('S')  # other cases, simplification
        elif char_lower == 'y':
            phonemes.append('i')  # y is rarely used in Portuguese, typically treated as 'i'
        elif char_lower == 'z':
            if i + 1 == len(word):
                phonemes.append('s')  # Final z
            else:
                phonemes.append('z')
    
    return phonemes

def g2p(text: str) -> Tuple[List[str], List[int]]:
    """
    将葡萄牙语文本转换为音素序列和词到音素的映射

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
        phonemes = _get_portuguese_phonemes(word)
        
        if phonemes:  # 如果音素不为空
            all_phonemes.extend(phonemes)
            word2ph.append(len(phonemes))
        else:
            # 如果没有音素（例如标点符号等），跳过
            continue
    
    return all_phonemes, word2ph
