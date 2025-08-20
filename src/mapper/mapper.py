import pandas as pd
from src.mapper.model_answer import ask_qwen, tokens_count
from src.utils.config import PARSING_DIR, MODEL_DIR
import csv
import re


def is_header_row(file_path, sample_size=5, encoding="utf-8"):
    """
    Определяет, содержит ли CSV заголовок в первой строке.
    Использует csv.Sniffer + эвристики.
    Возвращает True/False.
    """
    with open(file_path, "r", encoding=encoding) as f:
        sample = f.read(2048)
        f.seek(0)
        reader = csv.reader(f)
        rows = [row for _, row in zip(range(sample_size + 1), reader)]

    if len(rows) < 2:
        return False  # слишком мало строк, чтобы судить

    # --- 1. Попробуем встроенный Sniffer ---
    try:
        has_header = csv.Sniffer().has_header(sample)
        if has_header is not None:
            return has_header
    except Exception:
        pass  # если Sniffer не справился — пойдём по эвристикам

    # --- 2. Эвристики ---
    first_row = rows[0]
    data_rows = rows[1:]

    def looks_like_number(s):
        return bool(re.match(r"^-?\d+(\.\d+)?$", s.strip()))

    # если вся первая строка числа → явно не заголовок
    numbers_in_header = sum(looks_like_number(cell) for cell in first_row)
    if numbers_in_header == len(first_row):
        return False

    # если в первой строке есть повторы → маловероятно, что это заголовки
    if len(set(first_row)) < len(first_row):
        return False

    # если большинство ячеек первой строки — текст
    text_like = sum(not looks_like_number(cell) for cell in first_row)
    if text_like / len(first_row) > 0.6:
        return True

    # сравнение первой и второй строки по "числовитости"
    row2 = data_rows[0]
    row2_numbers = sum(looks_like_number(cell) for cell in row2)
    if numbers_in_header < row2_numbers:
        return True

    return False


def load_tables_from_csv(file):
    is_header = is_header_row(file)
    if is_header:
        df = pd.read_csv(file, delimiter=';', header=0)
    else:
        df = pd.read_csv(file, delimiter=';', header=None)
    return df, is_header


def mapper(files, extended=False):
    results = []
    if extended:
        pass
    else:
        file_path_header = MODEL_DIR / "simplified.csv"
        header = pd.read_csv(file_path_header).columns
        header_text = ", ".join(header)
        # files = PARSING_DIR.glob("*.csv")
        question = (
            f"Преобразовать её в таблицу с заданными столбцами. Во входной таблице колонки в строках разделены ';'")
        max_new_tokens = 4000
        for file in files:
            csv_file = PARSING_DIR / file
            df, is_header = load_tables_from_csv(csv_file)[:6]
            start = 1 if is_header else 0
            text = df.iloc[1].to_csv(index=False)
            t_count = tokens_count(text)
            chanks = max(5, int(max_new_tokens / t_count) - 10)
            for i in range(start, len(df), chanks):
                if i > len(df):
                    break
                batch = df.iloc[i:i + chanks]
                tables_text = batch.to_csv(index=False, header=False)
                result = ask_qwen(image_path=None,
                                  question=question,
                                  tables_text=tables_text,
                                  header=header_text,
                                  max_new_tokens=max_new_tokens)
                results.append(result.strip())
        final_text = "\n".join(results)
        print(final_text)
        out_file = PARSING_DIR / "output.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_text)
    return final_text, header
