from parsers.pdf_parser import parse_pdf
from parsers.txt_parser import load_txt_file
from parsers.img_parser import image_ocr_dd, parse_images_ai, image_ocr
import parsers.openai_parser
from mapper.mapper import mapper
from src.utils.registry import PARSERS


from src.utils.config import OUT_DIR, PARSING_DIR
import pandas as pd


def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def parsing_files(files):
    results = {}
    parsing_files_name = []

    for file in files:
        suffix = file.suffix.lower()
        file_name = file.name
        handler = PARSERS.get(suffix)
        if handler:
            files_list = ensure_list(handler(file))
            if files_list is not None:
                parsing_files_name.extend(files_list)
                results[file_name] = "Файл успешно обработан"
        else:
            results[file_name] = "❌ Неподдерживаемый формат"
    return parsing_files_name


def mapping_files(files, extended):
    return mapper(files, extended)


def parse_pipe_table(text):
    rows = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('|') and line.endswith('|'):
            line = line[1:-1]
        cells = [c.strip() for c in line.split('|')]
        rows.append(cells)
        print(rows)
    return rows


def save_to_xlsx(rows, header):
    max_cols = max(len(row) for row in rows)
    # Если в данных больше колонок, чем в header — добавляем виртуальные имена
    if len(header) < max_cols:
        extra_cols = [f"extra_{i}" for i in range(1, max_cols - len(header) + 1)]
        header = header + extra_cols

    df = pd.DataFrame(rows, columns=header)
    df.insert(0, 'N п.п.', range(1, len(df) + 1))
    file_path = OUT_DIR / 'result.xlsx'
    df.to_excel(file_path, index=False)
    return file_path


def process_files(files, extended=False, remote_model=False):
    if remote_model:
        handler = PARSERS.get("openai")
        rows, header = handler(files)
    else:
        files_list_csv = parsing_files(files)
        text, header = mapping_files(files_list_csv, extended=extended)
        rows = parse_pipe_table(text)
    file = save_to_xlsx(rows, header)
    return file
    # return True
