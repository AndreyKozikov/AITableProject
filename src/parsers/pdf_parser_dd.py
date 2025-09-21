import os

import pandas as pd
import deepdoctection as dd
from deepdoctection.analyzer.factory import ServiceFactory
from deepdoctection.utils.metacfg import set_config_by_yaml




def table_to_dataframe(table) -> pd.DataFrame:
    """
    Конвертация DeepDoctection Table -> pandas.DataFrame
    с поддержкой row_span и col_span.
    """
    cells_by_coord = {}
    max_row, max_col = 0, 0

    for cell in table.cells:
        # координаты
        row_id = (int(cell.sub_categories["row_number"].category_id) - 1
                  if "row_number" in cell.sub_categories else None)
        col_id = (int(cell.sub_categories["column_number"].category_id) - 1
                  if "column_number" in cell.sub_categories else None)
        if row_id is None or col_id is None:
            continue

        # span
        row_span = int(cell.sub_categories["row_span"].category_id) if "row_span" in cell.sub_categories else 1
        col_span = int(cell.sub_categories["column_span"].category_id) if "column_span" in cell.sub_categories else 1

        # текст
        text = getattr(cell, "text", "") or ""
        if not text and cell.relationships and "child" in cell.relationships:
            child_texts = [ch.text for ch in cell.relationships["child"] if hasattr(ch, "text")]
            text = " ".join(filter(None, child_texts))

        cells_by_coord[(row_id, col_id)] = (text, row_span, col_span)
        max_row = max(max_row, row_id + row_span)
        max_col = max(max_col, col_id + col_span)

    # создаём матрицу
    matrix = [["" for _ in range(max_col)] for _ in range(max_row)]
    for (r, c), (text, rs, cs) in cells_by_coord.items():
        for rr in range(r, r + rs):
            for cc in range(c, c + cs):
                matrix[rr][cc] = text

    return pd.DataFrame(matrix)


if __name__ == "__main__":


    print('DD_USE_TORCH=', os.getenv('DD_USE_TORCH'))
    # Загружаем конфиг явно
    cfg = set_config_by_yaml(
        r"D:\Andrew\GeekBrains\Python\AITableProject\.venv\Lib\site-packages\deepdoctection\configs\conf_dd_one.yaml"
    )

    # Собираем пайплайн
    analyzer = ServiceFactory.build_analyzer(cfg)

    # Анализируем PDF
    df = analyzer.analyze(path=r"/inbox/2409038В КП ОСНАСТИК.pdf")

    # Сохраняем все таблицы
    for page_idx, page in enumerate(df, start=1):
        for tbl_idx, table in enumerate(page.tables, start=1):
            print(f"\n--- Страница {page_idx}, таблица {tbl_idx} ---")
            df_table = table_to_dataframe(table)
            print(df_table.head())
            df_table.to_csv(f"table_{page_idx}_{tbl_idx}.csv", index=False, encoding="utf-8-sig")
