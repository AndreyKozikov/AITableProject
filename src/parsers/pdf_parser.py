from operator import truediv
from pathlib import Path
import sys
import re
import pdfplumber as pdf
import math
import csv
from src.utils.config import HEADER_ANCHORS, PARSING_DIR


def histogram_words(page, lines, header,
                    n_lines=15):
    start = header["top_idx"]
    end = min(len(lines), header["bottom_idx"] + 1 + n_lines)
    roi_words = [w for ln in lines[start:end] for w in ln["words"]]
    segments = [{"x0": float(w["x0"]), "x1": float(w["x1"])} for w in roi_words]

    y_min = min(w["top"] for w in roi_words)
    y_max = max(w["bottom"] for w in roi_words)

    roi_images = []
    if getattr(page, "images", []):
        for img in getattr(page, "images", []):
            img_top, img_bottom = float(img["top"]), float(img["bottom"])
            if not (img_bottom <= y_min or img_top >= y_max):
                roi_images.append({"x0": float(img["x0"]), "x1": float(img["x1"]),
                                   "top": img_top, "bottom": img_bottom})
        segments += [{"x0": img["x0"], "x1": img["x1"]} for img in roi_images if roi_images]

    x_min = min(s["x0"] for s in segments)
    x_max = max(s["x1"] for s in segments)

    x_left = math.floor(x_min)
    x_right = math.ceil(x_max)
    width = x_right - x_left

    occ = [0] * width
    for w in segments:
        i0 = max(0, int(math.floor(w["x0"]) - x_left))
        i1 = min(width, int(math.ceil(w["x1"]) - x_left))  # half-open
        for i in range(i0, i1):
            occ[i] += 1
    occ = [0 if x <= 1 else x for x in occ]

    zero_bounds = []
    in_zero = False
    start = None

    for i, val in enumerate(occ):
        if val == 0 and not in_zero:
            in_zero = True
            start = 0 if i == 0 else i
        elif val != 0 and in_zero:
            in_zero = False
            end = i - 1
            mid = 0 if start == 0 else (start + end) / 2
            zero_bounds.append(x_left + mid)

    # на случай, если нулевая зона тянется до конца
    if in_zero:
        end = len(occ) - 1
        mid = end
        zero_bounds.append(x_left + mid)
    zero_bounds.append(x_right)

    n_cols = len(zero_bounds) - 1
    columns = [[] for _ in range(n_cols)]

    for line in header["words"]:
        # создаём временный список для одной строки
        row_parts = [""] * n_cols

        for w in line["words"]:
            x_center = (w["x0"] + w["x1"]) / 2
            for col_idx in range(n_cols):
                if zero_bounds[col_idx] <= x_center < zero_bounds[col_idx + 1]:
                    if row_parts[col_idx]:
                        row_parts[col_idx] += " " + w["text"]
                    else:
                        row_parts[col_idx] = w["text"]
                    break
        # добавляем строку в каждую колонку
        for col_idx in range(n_cols):
            columns[col_idx].append(row_parts[col_idx])

    for i, column in enumerate(columns):
        columns[i] = (" ".join(column)).strip()
    return zero_bounds, columns


def norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s-]", " ", s)  # уберём пунктуацию
    s = " ".join(s.split())
    return s


def group_words_into_lines(words, y_tol=3.0):
    if not words:
        return []

    words = sorted(words, key=lambda w: (w["top"], w["x0"]))  # упорядочиваем слова по вертикали,
    # затем в пределах строки по горизонтали
    lines = []
    cur = {"top": words[0]["top"], "words": []}  # берем вертикальную координату первого слова
    for w in words:
        if abs(w["top"] - cur["top"]) < y_tol:  #
            cur["words"].append(w)
        else:
            cur["words"].sort(key=lambda ww: ww["x0"])  # упорядочиваем слова в строке слева направо
            cur["text"] = " ".join(ww["text"] for ww in cur["words"])
            lines.append(cur)
            cur = {"top": w["top"], "words": [w]}
    cur["words"].sort(key=lambda ww: ww["x0"])
    cur["text"] = " ".join(ww["text"] for ww in cur["words"])
    lines.append(cur)
    return lines


def select_table(table):
    row = table[0]
    score = 0
    for col in row:
        if col is None:
            continue
        s = norm(col)
        for _, keys in HEADER_ANCHORS:
            if any(k in s for k in keys):
                score += 1
        if score > 2:
            return True
    return False


def find_header_line_idx(lines):
    best_i, best_score = None, 0
    for i, ln in enumerate(lines):
        s = norm(ln["text"])
        score = 0
        for _, keys in HEADER_ANCHORS:
            if any(k in s for k in keys):
                score += 1
        if score > best_score:
            best_i, best_score = i, score
    if best_score >= 2:
        return best_i
    else:
        return None


def find_header(hdr_idx: int, lines, page, eps=0.5):
    hdr_line = lines[hdr_idx]
    y_top = min(w["top"] for w in hdr_line["words"])
    y_bottom = max(w["bottom"] for w in hdr_line["words"])

    horizontal_lines = [l for l in page.lines if abs(l["y0"] - l["y1"]) < eps]

    top_line = max((l for l in horizontal_lines if l["top"] < y_top),
                   key=lambda l: l["top"], default=None)

    bottom_line = min((l for l in horizontal_lines if l["top"] > y_bottom),
                      key=lambda l: l["top"], default=None)

    start_idx = None
    end_idx = None
    header_words = []
    for i, ln in enumerate(lines):
        for w in ln["words"]:
            if top_line["top"] < w["top"] < bottom_line["top"]:
                header_words.append(w)
                if start_idx is None:
                    start_idx = i
                else:
                    end_idx = i
    header_words = group_words_into_lines(header_words)
    return {"words": header_words, "top_idx": start_idx, "bottom_idx": end_idx}


def cut_line_into_columns(line: dict, bounds: list[float]) -> list[str]:
    n_cols = max(0, len(bounds) - 1)
    cols = [""] * n_cols
    for w in line.get("words", []):
        xc = (w["x0"] + w["x1"]) / 2.0
        for i in range(n_cols):
            if bounds[i] <= xc < bounds[i + 1]:
                cols[i] = (cols[i] + " " + (w.get("text", "") or "").strip()).strip()
                break
    return cols


def cut_block_into_matrix(lines: list[dict], bounds: list[float],
                          start_idx: int, end_idx: int | None = None) -> list[list[str]]:
    if end_idx is None:
        end_idx = len(lines)
    end_idx = max(start_idx, min(end_idx, len(lines)))

    rows = []
    for i in range(start_idx, end_idx):
        row = cut_line_into_columns(lines[i], bounds)
        rows.append(row)
    return rows


def detect_pdf_kind(path: str):
    p = Path(path)
    if not p.exists():
        print("File not found")
        sys.exit(1)

    with pdf.open(path) as f:
        kinds = []
        for i, page in enumerate(f.pages, start=1):
            try:
                has_chars = bool(getattr(page, 'chars', []))
                has_text = bool((page.extract_text() or "").strip())
                has_images = bool(getattr(page, "images", []))

                if has_chars or has_text:
                    kind = "text"
                elif has_images:
                    kind = "scan"
                else:
                    kind = "unknown"

                kinds.append((i, kind, has_images))
            except Exception as e:
                kinds.append((i, "error", False))
    total = len(kinds)
    text_pages = sum(1 for _, k, _ in kinds if k == "text")
    scan_pages = sum(1 for _, k, _ in kinds if k == "scan")
    unknown_pages = sum(1 for _, k, _ in kinds if k == "unknown")
    error_pages = sum(1 for _, k, _ in kinds if k == "error")

    print(f"Файл: {p.name}")
    print(
        f"Всего страниц: {total} | текстовых: {text_pages} | сканов: {scan_pages} | неизвестно: {unknown_pages} | ошибок: {error_pages}\n")
    for idx, kind, has_img in kinds:
        extra = " (есть изображения)" if has_img else ""
        print(f"стр. {idx:>3}: {kind}{extra}")
    return kinds


def extract_tables_on_page(page):
    tables = page.extract_tables() or []

    if tables:
        return tables

    tables = page.extract_tables(table_settings={
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_tolerance": 2,
    }) or []

    if tables:
        return tables

    tables = page.extract_tables(table_settings={
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "text_tolerance": 3,
    }) or []

    if tables:
        return tables


def parse_table_block(page, lines, header: dict, bounds: list[float],
                      n_rows: int | None = None) -> dict:
    """
    Возвращает чистые «сырые» данные после шапки:
      {
        "bounds": [...],
        "start_idx": int,
        "end_idx": int,
        "raw_rows": [ [col0, col1, ...], ... ]
      }
    """
    start = header["bottom_idx"] + 1
    end = len(lines) if n_rows is None else min(len(lines), start + n_rows)

    raw_rows = cut_block_into_matrix(lines, bounds, start, end)
    return {
        "bounds": bounds[:],
        "start_idx": start,
        "end_idx": end,
        "raw_rows": raw_rows,
    }


def normalize_table(table):
    for i, row in enumerate(table):
        for j, col in enumerate(row):
            if col is None:
                row[j] = ""
                continue
                # Оставляем только буквы, цифры, пробелы и подчёркивания
            cleaned_col = re.sub(r'[^\w\s]', '', col, flags=re.UNICODE)
            # Нормализуем пробелы
            cleaned_col = re.sub(r'\s+', ' ', cleaned_col, flags=re.UNICODE).strip()
            row[j] = cleaned_col
        table[i] = row
    return table


def parsing_tables(path: str, kinds, file_name: str):
    output_path = PARSING_DIR
    text_page_indexes = [i for (i, k, _) in kinds if k == "text"]
    if not text_page_indexes:
        print("\n(Текстовых страниц нет — искать таблицы бессмысленно)")
        return
    found_any = False

    print("\n=== Поиск таблиц на текстовых страницах ===")

    with pdf.open(path) as f:
        for i in text_page_indexes:
            page = f.pages[i - 1]
            tables = extract_tables_on_page(page) or []
            for i, table in enumerate(tables):
                table = normalize_table(table)

                if select_table(table):
                    output_path = output_path / f"{file_name}_table{i}.csv"
                    write_to_csv(output_path, table)

            if not tables:
                continue

            found_any = True

    if found_any:
        return True

    # Если таблицы не найдены стандартным способом, пытаемся установить границы таблицы
    first_page = text_page_indexes[0]
    with pdf.open(path) as f:
        page = f.pages[first_page - 1]
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
        lines = group_words_into_lines(words, y_tol=3.0)
        hdr_idx = find_header_line_idx(lines)
        header = find_header(hdr_idx, lines, page)
        bounds, labels = histogram_words(page, lines, header, n_lines=30)
        if not bounds:
            print("Не удалось определить границы колонок")
            return
        raw_rows = [labels]
        for i in text_page_indexes:
            page = f.pages[i - 1]
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
            lines = group_words_into_lines(words, y_tol=3.0)
            result = parse_table_block(page, lines, header, bounds, n_rows=None)
            raw_rows += result["raw_rows"]
        output_path = output_path / f"{file_name}.csv"
        write_to_csv(output_path, raw_rows)
        return True
    return False


def write_to_csv(path: str, df):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, delimeter=";")
        writer.writerows(df)


def parse_pdf(path):
    file_path = Path(path)
    file_name = file_path.stem
    kinds = detect_pdf_kind(path)
    result = parsing_tables(path, kinds, file_name)
    return True if result else False
