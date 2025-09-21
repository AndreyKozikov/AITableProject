import csv
import re
import pandas as pd

import csv
import re
from src.utils.config import HEADER_ANCHORS


def is_header_row(rows):
    """
    Определяет, содержит ли CSV заголовок в первой строке.
    Использует csv.Sniffer + эвристики + поиск по ключевым словам.
    Возвращает True/False.
    """

    print(f"{rows = }")
    sample = "\n".join([",".join(r) for r in rows])

    if len(rows) < 2:
        return False  # слишком мало строк, чтобы судить

    # --- 1. Попробуем встроенный Sniffer ---
    try:
        has_header = csv.Sniffer().has_header(sample)
        if has_header is not None:
            return has_header
    except Exception:
        pass

    # --- 2. Эвристики ---
    first_row = [cell.strip().lower() for cell in rows[0] if cell.strip() != ""]
    data_rows = rows[1:]

    def looks_like_number(s):
        return bool(re.match(r"^-?\d+(\.\d+)?$", s.strip()))

    # --- Проверка по ключевым словам ---
    header_keywords = [kw for _, keys in HEADER_ANCHORS for kw in keys]
    matches = 0
    for cell in first_row:
        for kw in header_keywords:
            if cell.startswith(kw) or kw in cell:
                matches += 1
                break  # одно совпадение на ячейку достаточно

    if matches >= 2:  # минимум два совпадения в строке
        return True

    # --- Остальные эвристики ---
    # если вся первая строка числа → явно не заголовок
    numbers_in_header = sum(looks_like_number(cell) for cell in first_row)
    if numbers_in_header == len(first_row):
        return False

    # если в первой строке есть повторы → маловероятно, что это заголовки
    if len(set(first_row)) < len(first_row):
        return False

    # сравнение первой и второй строки по "числовитости"
    row2 = data_rows[0]
    row2_numbers = sum(looks_like_number(cell) for cell in row2)
    if numbers_in_header < row2_numbers:
        return True

    # если большинство ячеек первой строки — текст
    text_like = sum(not looks_like_number(cell) for cell in first_row)
    if text_like / len(first_row) > 0.6:
        return True

    return False


def write_to_json(path: str, df):
    if is_header_row(df):
        columns, data = df[0], df[1:]
    else:
        columns = [""] * len(df[0])
        data = df
    columns = [
        col if col.strip() != "" else f"Unnamed_{i}"
        for i, col in enumerate(columns)
    ]
    df = pd.DataFrame(data, columns=columns)
    df.to_json(path, orient="records", force_ascii=False, indent=2)


import language_tool_python

# расширенный маппинг латиница → кириллица
latin_to_cyrillic = {
    "A": "А", "a": "а",
    "B": "В", "E": "Е", "e": "е",
    "K": "К", "M": "М",
    "H": "Н", "O": "О", "o": "о",
    "P": "Р", "C": "С", "c": "с",
    "T": "Т", "X": "Х", "x": "х",
    "Y": "У", "y": "у",
    "u": "и", "n": "н", "N": "Н",
    "r": "г", "R": "Г"
}

tool = language_tool_python.LanguageTool('ru-RU')


def normalize_text(text: str) -> str:
    """Жёсткая нормализация: всё, что похоже на кириллицу, переводим в кириллицу"""
    if not isinstance(text, str):
        return text
    return "".join(latin_to_cyrillic.get(ch, ch) for ch in text)


def fix_with_languagetool(text: str) -> str:
    if not isinstance(text, str):
        return text
    matches = tool.check(text)
    return language_tool_python.utils.correct(text, matches)


def clean_dataframe(df, use_languagetool=True):
    def process_cell(x):
        original = x
        after_norm = normalize_text(x)
        lt_result = after_norm
        if use_languagetool:
            lt_result = fix_with_languagetool(after_norm)
        print(f"[OCR] {original!r} -> [norm] {after_norm!r} -> [LT] {lt_result!r}")
        return lt_result

    return df.applymap(process_cell)
