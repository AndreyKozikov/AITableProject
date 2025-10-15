"""Модуль утилит для работы с DataFrame.

Этот модуль предоставляет утилиты для обработки DataFrame, обнаружения заголовков,
очистки данных и файловых операций для проекта AITableProject.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from src.utils.config import HEADER_ANCHORS
from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)


def is_header_row_semantic(
    rows: Union[List[List[str]], Path, str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.75
) -> bool:
    """Определяет, содержит ли CSV заголовок, используя семантические эмбеддинги.
    
    Использует sentence-transformers для кодирования первой строки и сравнения
    с эталонным набором известных заголовков с помощью косинусного сходства.
    
    Args:
        rows: Либо список строк, либо путь к CSV файлу.
        model_name: Название модели sentence-transformers для использования.
        similarity_threshold: Минимальный порог сходства для определения заголовка (0-1).
        
    Returns:
        True, если первая строка семантически похожа на известные заголовки, иначе False.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning(
            "sentence-transformers не установлен. "
            "Установите с помощью: pip install sentence-transformers"
        )
        return False
    
    logger.debug(f"Запуск семантического определения заголовка с моделью: {model_name}")
    
    # Если предоставлен путь, сначала читаем файл
    if isinstance(rows, (Path, str)):
        try:
            with open(rows, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')  # Используем разделитель ';'
                rows = list(reader)
            logger.debug(f"Прочитано {len(rows)} строк из файла")
        except Exception as e:
            logger.error(f"Ошибка чтения файла для семантического определения заголовка: {e}")
            return False
    
    if not isinstance(rows, list) or len(rows) < 1:
        logger.debug("Недостаточно строк для семантического анализа")
        return False
    
    # Получаем ячейки первой строки
    first_row = [cell.strip() for cell in rows[0] if cell.strip() != ""]
    
    if not first_row:
        logger.debug("Первая строка пустая")
        return False
    
    logger.debug(f"Анализ первой строки с {len(first_row)} ячейками")
    
    # Проверяем, что первая строка не состоит в основном из чисел
    numeric_cells = sum(
        1 for cell in first_row 
        if cell.replace('.', '').replace(',', '').replace('-', '').isdigit()
    )
    numeric_ratio = numeric_cells / len(first_row) if first_row else 0
    
    if numeric_ratio > 0.3:
        logger.debug(
            f"Первая строка содержит слишком много числовых значений ({numeric_ratio:.1%}), "
            f"вероятно, это строка данных, а не заголовок"
        )
        return False
    
    # Извлекаем reference headers из HEADER_ANCHORS
    reference_headers = []
    for _, header_variants in HEADER_ANCHORS:
        reference_headers.extend(header_variants)
    
    logger.debug(f"Используется {len(reference_headers)} эталонных заголовков из HEADER_ANCHORS")
    
    try:
        # Загружаем модель (кешируется после первой загрузки)
        logger.debug("Загрузка модели sentence-transformers...")
        model = SentenceTransformer(model_name)
        
        # Кодируем ячейки первой строки
        logger.debug(f"Кодирование {len(first_row)} ячеек из первой строки")
        first_row_embeddings = model.encode(first_row, convert_to_numpy=True)
        
        # Кодируем эталонные заголовки
        logger.debug(f"Кодирование {len(reference_headers)} эталонных заголовков")
        reference_embeddings = model.encode(reference_headers, convert_to_numpy=True)
        
        # Вычисляем сходство между первой строкой и эталонными заголовками
        # Для каждой ячейки первой строки находим максимальное сходство с любым эталонным заголовком
        max_similarities = []
        for cell_embedding in first_row_embeddings:
            # Косинусное сходство со всеми эталонными заголовками
            similarities = model.similarity(
                cell_embedding.reshape(1, -1),
                reference_embeddings
            )
            max_sim = float(similarities.max())
            max_similarities.append(max_sim)
            logger.debug(
                f"Ячейка '{first_row[len(max_similarities)-1]}' "
                f"максимальное сходство: {max_sim:.4f}"
            )
        
        # Вычисляем среднее сходство по всем ячейкам
        avg_similarity = float(np.mean(max_similarities))
        logger.info(
            f"Среднее семантическое сходство: {avg_similarity:.4f} "
            f"(порог: {similarity_threshold})"
        )
        
        # Также проверяем, сколько ячеек превышают порог
        cells_above_threshold = sum(
            1 for sim in max_similarities if sim >= similarity_threshold
        )
        ratio_above_threshold = cells_above_threshold / len(max_similarities)
        
        logger.info(
            f"Ячейки выше порога: {cells_above_threshold}/{len(max_similarities)} "
            f"({ratio_above_threshold:.2%})"
        )
        
        # Решение: высокое среднее И большинство ячеек выше порога (более строгое условие)
        is_header = (
            avg_similarity >= similarity_threshold and
            ratio_above_threshold >= 0.6
        )
        
        if is_header:
            logger.info(
                f"Заголовок обнаружен семантическим анализом "
                f"(среднее_сходство={avg_similarity:.4f}, "
                f"доля={ratio_above_threshold:.2%})"
            )
        else:
            logger.debug(
                f"Заголовок не обнаружен семантическим анализом "
                f"(среднее_сходство={avg_similarity:.4f}, "
                f"доля={ratio_above_threshold:.2%})"
            )
        
        return is_header
        
    except Exception as e:
        logger.error(f"Ошибка в семантическом определении заголовка: {e}")
        return False


def reconstruct_table_from_ocr(
    json_data: dict,
    image_width: int = 1200,
    min_gap_width: int = 10
) -> pd.DataFrame:
    """Восстанавливает структуру таблицы из результатов OCR на основе пространственного анализа.
    
    Эта функция восстанавливает табличную структуру из bounding box'ов OCR и текстов
    путем анализа пространственного распределения текстовых блоков для определения
    границ столбцов и позиций строк.
    
    Args:
        json_data: Словарь, содержащий результаты OCR с 'rec_boxes' и 'rec_texts'.
                  Ожидаемая структура: {'rec_boxes': [[x1,y1,x2,y2], ...], 'rec_texts': [...]}
        image_width: Ширина анализируемого изображения в пикселях для карты плотности.
        min_gap_width: Минимальная ширина пустой зоны для определения границы столбца.
        
    Returns:
        DataFrame с восстановленной структурой таблицы.
        
    Алгоритм:
        1. Построение карты плотности по ширине изображения
        2. Поиск непрерывных нулевых зон (пробелов) как границ столбцов
        3. Вычисление центров блоков (x_center, y_center)
        4. Распределение блоков по столбцам на основе x_center
        5. Сортировка блоков внутри столбцов по y_center
        6. Выравнивание строк между столбцами по максимальному количеству строк
    """
    try:
        logger.debug(f"Начало восстановления таблицы из данных OCR (ширина_изображения={image_width})")
        
        # Извлекаем boxes и texts из json_data
        boxes = json_data.get('rec_boxes', [])
        texts = json_data.get('rec_texts', [])
        
        if len(boxes) == 0 or len(texts) == 0:
            logger.warning("Не найдено OCR boxes или текстов в данных")
            return pd.DataFrame()
        
        logger.info(f"Обработка {len(boxes)} блоков OCR")
        
        # Step 1: Build density map
        density = np.zeros(image_width, dtype=int)
        for (x1, y1, x2, y2) in boxes:
            x1_int, x2_int = int(max(0, x1)), int(min(image_width - 1, x2))
            density[x1_int:x2_int] += 1
        
        logger.debug(f"Карта плотности построена с максимальной плотностью: {density.max()}")
        
        # Шаг 2: Поиск границ столбцов (непрерывные нулевые зоны)
        boundaries = []
        in_gap = False
        start = None
        
        for i, val in enumerate(density):
            if val == 0 and not in_gap:
                in_gap = True
                start = i
            elif val != 0 and in_gap:
                end = i
                in_gap = False
                if end - start > min_gap_width:
                    middle = int((start + end) / 2)
                    boundaries.append(middle)
        
        # Проверяем, если последний пробел доходит до края изображения
        if in_gap and (image_width - start) > min_gap_width:
            middle = int((start + image_width) / 2)
            boundaries.append(middle)
        
        logger.info(f"Найдено {len(boundaries)} границ столбцов: {boundaries}")
        
        # Шаг 3: Вычисление центров блоков
        blocks = []
        for (x1, y1, x2, y2), text in zip(boxes, texts):
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            blocks.append((x_center, y_center, text.strip()))
        
        # Шаг 4: Распределение блоков по столбцам
        num_columns = len(boundaries) + 1
        columns = [[] for _ in range(num_columns)]
        
        for x_center, y_center, text in blocks:
            col_index = sum(x_center > b for b in boundaries)
            columns[col_index].append((y_center, text))
        
        logger.debug(f"Блоки распределены по {num_columns} столбцам")
        
        # Шаг 5: Сортировка блоков внутри каждого столбца по y_center
        for col in columns:
            col.sort(key=lambda x: x[0])
        
        # Шаг 6: Построение таблицы с выравниванием строк
        max_rows = max(len(col) for col in columns) if columns else 0
        
        if max_rows == 0:
            logger.warning("Не найдено строк в таблице")
            return pd.DataFrame()
        
        table = []
        for i in range(max_rows):
            row = []
            for col in columns:
                if i < len(col):
                    row.append(col[i][1])  # Добавляем текст
                else:
                    row.append("")  # Пустая ячейка
            table.append(row)
        
        df = pd.DataFrame(table)
        logger.info(f"Таблица восстановлена с формой: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка восстановления таблицы из данных OCR: {e}", exc_info=True)
        return pd.DataFrame()


def write_to_json(
    file_path: Union[str, Path],
    data: Any,
    detect_headers: bool = False,
    temp_dir: Union[str, Path, None] = None
) -> bool:
    """Записывает данные в JSON файл с опциональным определением и генерацией заголовков.
    
    Args:
        file_path: Путь к выходному JSON файлу.
        data: Данные для записи в JSON. Может быть DataFrame, список списков или любые JSON-сериализуемые данные.
        detect_headers: Если True, пытается определить заголовки в данных DataFrame и генерирует их, если не найдены.
        temp_dir: Директория для временных файлов во время определения заголовков. Требуется, если detect_headers=True.
        
    Returns:
        True если успешно, False в противном случае.
    """
    try:
        file_path = Path(file_path)
        logger.debug(f"Запись JSON в: {file_path}")
        
        # Если данные - DataFrame и запрошено определение заголовков
        if detect_headers and isinstance(data, pd.DataFrame):
            logger.debug("Запрошено определение заголовков для DataFrame")
            
            if temp_dir is None:
                logger.warning("temp_dir не предоставлен, используется директория по умолчанию")
                temp_dir = file_path.parent
            
            temp_dir = Path(temp_dir)
            temp_csv_path = temp_dir / f"temp_{file_path.stem}.csv"
            
            try:
                # Сохраняем DataFrame во временный CSV без заголовков
                data.to_csv(temp_csv_path, index=False, sep=";", header=False)
                logger.debug(f"Создан временный CSV: {temp_csv_path}")
                
                # Проверяем, существует ли строка заголовков
                has_header = is_header_row_semantic(temp_csv_path)
                logger.info(f"Результат определения заголовков: {has_header}")
                
                if has_header:
                    # Читаем CSV с определенным заголовком
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=0, dtype=str)
                    logger.info(f"Используются определенные заголовки: {list(df_with_header.columns)}")
                else:
                    # Читаем CSV без заголовка и генерируем имена столбцов
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=None, dtype=str)
                    num_columns = len(df_with_header.columns)
                    generated_headers = [f"Заголовок {j+1}" for j in range(num_columns)]
                    df_with_header.columns = generated_headers
                    logger.info(f"Сгенерированы заголовки: {generated_headers}")
                
                # Преобразуем DataFrame в список словарей
                data = df_with_header.to_dict('records')
                logger.debug(f"DataFrame преобразован в {len(data)} записей")
                
            finally:
                # Временный CSV НЕ удаляется для отладки
                if temp_csv_path.exists():
                    logger.info(f"Временный CSV сохранен для отладки: {temp_csv_path}")
                    temp_csv_path.unlink()  # Закомментировано для отладки
        
        elif isinstance(data, pd.DataFrame):
            # Преобразуем DataFrame в записи без определения заголовков
            data = data.to_dict('records')
            logger.debug(f"DataFrame преобразован в {len(data)} записей")
        
        # Записываем данные в JSON файл
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON файл успешно записан: {file_path}")
        logger.debug(f"Размер файла: {file_path.stat().st_size} байт")
        return True
        
    except Exception as e:
        logger.error(f"Ошибка записи JSON файла {file_path}: {e}")
        return False


def read_from_json(file_path: Union[str, Path]) -> Any:
    """Читает данные из JSON файла.
    
    Args:
        file_path: Путь к JSON файлу.
        
    Returns:
        Загруженные данные или None при ошибке.
    """
    try:
        file_path = Path(file_path)
        logger.debug(f"Чтение JSON из: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON файл успешно прочитан: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Ошибка чтения JSON файла {file_path}: {e}")
        return None


def clean_dataframe(df: pd.DataFrame, use_languagetool: bool = False) -> pd.DataFrame:
    """Очищает DataFrame, удаляя пустые строки/столбцы и исправляя данные.
    
    Args:
        df: DataFrame для очистки.
        use_languagetool: Использовать ли языковой инструмент для исправления текста.
        
    Returns:
        Очищенный DataFrame.
    """
    logger.debug(f"Очистка DataFrame с формой: {df.shape}")
    
    try:
        original_shape = df.shape
        
        # Remove completely empty rows and columns (NaN values)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN with empty strings
        df = df.fillna('')
        
        # Strip whitespace from string columns and replace semicolons
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.replace(';', ' ', regex=False)
        
        # Remove columns where all values are empty strings (after stripping)
        non_empty_cols = (df != '').any(axis=0)
        df = df.loc[:, non_empty_cols]
        
        # # Remove rows where more than 1/3 of cells are empty
        # if len(df.columns) > 0:
        #     empty_cells_per_row = (df == '').sum(axis=1)
        #     total_cols = len(df.columns)
        #     empty_threshold = total_cols / 3
        #     rows_to_keep = empty_cells_per_row <= empty_threshold
        #     rows_removed = (~rows_to_keep).sum()
        #     if rows_removed > 0:
        #         logger.debug(f"Removing {rows_removed} rows with >1/3 empty cells")
        #     df = df.loc[rows_to_keep, :]
        
        # Remove rows where all values are empty strings
        non_empty_rows = (df != '').any(axis=1)
        df = df.loc[non_empty_rows, :]
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"DataFrame очищен: {original_shape} -> {df.shape}")
        
        if use_languagetool:
            logger.info("Запрошена коррекция языковым инструментом, но не реализована")
            # TODO: Реализовать коррекцию языковым инструментом при необходимости
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка очистки DataFrame: {e}")
        return df



def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Проверяет DataFrame и возвращает метрики качества.
    
    Args:
        df: DataFrame для проверки.
        
    Returns:
        Словарь с результатами проверки и метриками.
    """
    try:
        logger.debug(f"Проверка DataFrame с формой: {df.shape}")
        
        validation_result = {
            'is_valid': True,
            'shape': df.shape,
            'empty_cells': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'column_types': df.dtypes.to_dict(),
            'warnings': [],
            'errors': []
        }
        
        # Проверяем пустой DataFrame
        if df.empty:
            validation_result['errors'].append("DataFrame пустой")
            validation_result['is_valid'] = False
        
        # Проверяем слишком много пустых ячеек
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            empty_ratio = validation_result['empty_cells'] / total_cells
            if empty_ratio > 0.5:
                validation_result['warnings'].append(
                    f"Высокое соотношение пустых ячеек: {empty_ratio:.2%}"
                )
        
        # Проверяем дублирующиеся строки
        if validation_result['duplicate_rows'] > 0:
            validation_result['warnings'].append(
                f"Найдено {validation_result['duplicate_rows']} дублирующихся строк"
            )
        
        logger.info(f"Проверка DataFrame завершена. Валидный: {validation_result['is_valid']}")
        logger.debug(f"Предупреждения проверки: {len(validation_result['warnings'])}")
        logger.debug(f"Ошибки проверки: {len(validation_result['errors'])}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Ошибка проверки DataFrame: {e}")
        return {
            'is_valid': False,
            'errors': [f"Проверка не удалась: {str(e)}"]
        }