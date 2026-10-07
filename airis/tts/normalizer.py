"""
Airis TTS Normalization Module.
Provides Russian numeral-to-words expansion and phonetic Cyrillic text cleaning.
"""

import re


def num_to_words_ru(n: int) -> str:
    """Expands integer numbers into Russian words without chunk index overflow."""
    if n == 0:
        return "ноль"
    if n < 0:
        return f"минус {num_to_words_ru(-n)}"

    units_m = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    units_f = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
    teens = [
        "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать",
        "пятнадцать", "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"
    ]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    orders = [
        ("", "", "", False),                           # 10^0 (units)
        ("тысяча", "тысячи", "тысяч", True),            # 10^3 (thousands)
        ("миллион", "миллиона", "миллионов", False),    # 10^6 (millions)
        ("миллиард", "миллиарда", "миллиардов", False), # 10^9 (billions)
        ("триллион", "триллиона", "триллионов", False), # 10^12 (trillions)
    ]

    def parse_chunk(val: int, is_feminine: bool) -> str:
        val = val % 1000
        if val == 0:
            return ""
        parts = []
        h = val // 100
        t = (val % 100) // 10
        u = val % 10
        if h > 0:
            parts.append(hundreds[h])
        if t == 1:
            parts.append(teens[u])
        else:
            if t > 1:
                parts.append(tens[t])
            if u > 0:
                parts.append(units_f[u] if is_feminine else units_m[u])
        return " ".join(parts)

    def get_order_form(val: int, s1: str, s24: str, s5: str) -> str:
        rem100 = val % 100
        rem10 = val % 10
        if 11 <= rem100 <= 19:
            return s5
        if rem10 == 1:
            return s1
        if 2 <= rem10 <= 4:
            return s24
        return s5

    chunks = []
    temp = n
    while temp > 0:
        chunks.append(temp % 1000)
        temp //= 1000

    if len(chunks) > len(orders):
        digit_names = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
        return " ".join(digit_names[int(d)] for d in str(n))

    result_parts = []
    for idx in range(len(chunks) - 1, -1, -1):
        chunk_val = chunks[idx]
        if chunk_val == 0:
            continue
        s1, s24, s5, is_fem = orders[idx]
        text_chunk = parse_chunk(chunk_val, is_fem)
        if idx == 0:
            result_parts.append(text_chunk)
        else:
            order_name = get_order_form(chunk_val, s1, s24, s5)
            result_parts.append(f"{text_chunk} {order_name}".strip())

    return " ".join(result_parts).strip()


def clean_and_normalize_tts_text(text: str) -> str:
    """
    Cleans raw LLM response text into phonetic Cyrillic speech input:
    - Strips markdown formatting and URLs.
    - Transliterates Latin names (l3wsha -> Лёша, Airis -> Айрис).
    - Expands abbreviations (и т.д. -> и так далее, т.е. -> то есть, руб. -> рублей).
    - Expands percentages and decimal numbers into Russian words.
    - Expands integers (0 to 1,000,000).
    - Filters emojis, symbols, and non-Cyrillic characters.
    - Returns empty string if no Cyrillic phonemes remain.
    """
    if not text:
        return ""
    # Strip markdown bold, italics, headers, code backticks
    text = re.sub(r'[*_#`~>\[\]\(\)]', '', text)
    # Strip URLs
    text = re.sub(r'https?://\S+', '', text)
    # Transliterate known Latin names
    text = re.sub(r'\bl3wsha\b', 'Лёша', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAiris\b', 'Айрис', text, flags=re.IGNORECASE)
    # Expand Russian abbreviations
    text = re.sub(r'\bи\s+т\.д\.?', 'и так далее', text, flags=re.IGNORECASE)
    text = re.sub(r'\bт\.е\.?', 'то есть', text, flags=re.IGNORECASE)
    text = re.sub(r'\bруб\b\.?', 'рублей', text, flags=re.IGNORECASE)
    # Expand percentages
    text = re.sub(r'(\d+)%', lambda m: f"{num_to_words_ru(int(m.group(1)))} процентов", text)
    # Expand decimals (e.g. 3.14 -> три и четырнадцать)
    text = re.sub(
        r'(\d+)\.(\d+)',
        lambda m: f"{num_to_words_ru(int(m.group(1)))} и {num_to_words_ru(int(m.group(2)))}",
        text
    )
    # Expand standalone integers
    text = re.sub(r'\b\d+\b', lambda m: num_to_words_ru(int(m.group(0))), text)
    # Strip emojis and symbols (keep Russian letters, standard punctuation, whitespace)
    text = re.sub(r'[^А-Яа-яЁёA-Za-z\s.,!?:;\-]', '', text)
    # Normalize multiple punctuation / spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Check for presence of Cyrillic phonemes
    if not re.search(r'[А-Яа-яЁё]', text):
        return ""
    return text
