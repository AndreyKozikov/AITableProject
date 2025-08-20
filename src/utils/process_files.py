from parsers.pdf_parser import parse_pdf
from mapper.mapper import mapper
from src.utils.config import OUT_DIR, PARSING_DIR
import pandas as pd

def parsing_files(files):
    results = {}
    parsing_files_name = []
    for file in files:
        suffix = file.suffix.lower()
        file_name = file.name
        if suffix in (".pdf",):
            files_list = parse_pdf(file)
            if files_list is not None:
                parsing_files_name += files_list
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
            print("Line: ", line)
        cells = [c.strip() for c in line.split('|')]
        rows.append(cells)
    return rows


def save_to_xlsx(rows, header):
    df = pd.DataFrame(rows, columns=header)
    df.insert(0, 'N п.п.', range(1, len(df) + 1))
    file_path = OUT_DIR / 'result.xlsx'
    df.to_excel(file_path, index=False)
    return file_path


def load_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def process_files(files):
    files_list_csv = parsing_files(files)

    text, header = mapping_files(files_list_csv, extended=False)

    text = load_text_from_txt(PARSING_DIR / 'output.txt')

    rows = parse_pipe_table(text)

    file = save_to_xlsx(rows, header)

    return file