import pandas as pd
from src.mapper.ask_qwen2 import ask_qwen2, tokens_count
from src.mapper.ask_qwen3 import ask_qwen3, tokens_count3
from src.mapper.ask_llama2 import ask_llama2, tokens_count_llama
from src.utils.config import PARSING_DIR, MODEL_DIR
from src.utils.df_utils import is_header_row
import json
from src.utils.config import PROMPT_TEMPLATE

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # список словарей
    return data


def chunk_data(data, chunk_size=20):
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def load_tables_from_csv(file):
    is_header = is_header_row(file)
    if is_header:
        df = pd.read_csv(file, delimiter=';', header=0)
    else:
        df = pd.read_csv(file, delimiter=';', header=None)
    return df


def mapper(files, extended=False):
    max_new_tokens = 4000
    final_text = None
    results = []
    if extended:
        print("Умное распределение")
        file_path_header = MODEL_DIR / "extended.csv"
    else:
        print("Упрощенное распределение")
        file_path_header = MODEL_DIR / "simplified.csv"

    header = list(pd.read_csv(file_path_header, nrows=0).columns)
    header_str = ", ".join(header)

    for file in files:
        file_path = PARSING_DIR / file
        data = load_json(file_path)
        for chunk in chunk_data(data, chunk_size=5):
            tables_text = json.dumps(chunk, ensure_ascii=False, indent=2)
            prompt = PROMPT_TEMPLATE.format(header=header_str, tables_text=tables_text)
            print(f"{tables_text = }")

            # result = ask_qwen2(prompt=prompt,
            #                    max_new_tokens=max_new_tokens)

            result = ask_qwen3(prompt=prompt,
                               max_new_tokens=max_new_tokens)

            # result = ask_llama2(prompt=prompt,
            #                     max_new_tokens=max_new_tokens)

            results.append(result)
        final_text = "\n".join(results)
        out_file = PARSING_DIR / "output.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_text)
    return final_text, header
