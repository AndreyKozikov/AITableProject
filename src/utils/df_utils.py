"""DataFrame Utilities Module.

Модуль утилит для работы с DataFrame.

This module provides utilities for DataFrame processing, header detection,
data cleaning, and file operations for the AITableProject.
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
    
    Использует sentence-transformers для кодирования первой строки и сравнения её
    с эталонным набором известных заголовков при помощи косинусного сходства.
    
    Args:
        rows: Список строк или путь к CSV файлу.
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
            "Установите командой: pip install sentence-transformers"
        )
        return False
    
    logger.debug(f"Начало семантического определения заголовка с моделью: {model_name}")
    
    # Если передан путь, сначала читаем файл
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
        logger.debug("Слишком мало строк для семантического анализа")
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
            f"вероятно это строка данных, а не заголовок"
        )
        return False
    
    # Извлекаем reference headers из HEADER_ANCHORS
    reference_headers = []
    for _, header_variants in HEADER_ANCHORS:
        reference_headers.extend(header_variants)
    
    logger.debug(f"Используем {len(reference_headers)} эталонных заголовков из HEADER_ANCHORS")
    
    try:
        # Загружаем модель (кэшируется после первой загрузки)
        logger.debug("Загрузка модели sentence-transformers...")
        model = SentenceTransformer(model_name)
        
        # Кодируем ячейки первой строки
        logger.debug(f"Кодирование {len(first_row)} ячеек первой строки")
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
                f"макс. сходство: {max_sim:.4f}"
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
            f"Ячеек выше порога: {cells_above_threshold}/{len(max_similarities)} "
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
                f"(среднее={avg_similarity:.4f}, "
                f"доля={ratio_above_threshold:.2%})"
            )
        else:
            logger.debug(
                f"Заголовок не обнаружен семантическим анализом "
                f"(среднее={avg_similarity:.4f}, "
                f"доля={ratio_above_threshold:.2%})"
            )
        
        return is_header
        
    except Exception as e:
        logger.error(f"Ошибка при семантическом определении заголовка: {e}")
        return False


def _calculate_iou(box1: list, box2: list) -> float:
    """Вычислить Intersection over Union (IoU) между двумя ограничивающими прямоугольниками.

    Args:
        box1: Первый bbox [x1, y1, x2, y2]
        box2: Второй bbox [x1, y1, x2, y2]

    Returns:
        Значение IoU (площадь пересечения / площадь объединения), диапазон [0, 1]
    """
    # Вычисляем площадь пересечения
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Проверяем, пересекаются ли боксы
    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    # Вычисляем площадь объединения
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Избегаем деления на ноль
    if union <= 0:
        return 0.0

    return intersection / union


def _merge_ocr_blocks_by_cells(json_data: dict, iou_threshold: float = 0.05) -> tuple:
    """Объединить OCR блоки на основе координат ячеек таблицы из PPStructure V3.

    Этот приватный вспомогательный метод агрегирует текстовые OCR блоки в соответствии
    с обнаруженными границами ячеек таблицы, используя двухэтапную стратегию:
    1. Быстрая фильтрация: проверка, попадает ли центр OCR блока внутрь ячейки
    2. IoU сопоставление: для несопоставленных блоков использует Intersection over Union

    Args:
        json_data: Словарь с результатами PPStructure V3, содержащий:
                  - 'table_res_list': Список результатов распознавания таблиц
                  - 'overall_ocr_res': Результаты OCR с 'rec_boxes' и 'rec_texts'
        iou_threshold: Минимальное значение IoU для назначения блока ячейке (по умолчанию: 0.05)

    Returns:
        Кортеж (merged_rec_boxes, merged_rec_texts), где каждый элемент соответствует
        одной ячейке таблицы, или (None, None), если объединение неприменимо.

    Алгоритм:
        1. Извлечение координат ячеек из table_res_list
        2. Извлечение OCR боксов и текстов из overall_ocr_res
        3. Этап 1: Сопоставление блоков по критерию центра в ячейке (быстро)
        4. Этап 2: Сопоставление оставшихся блоков по IoU (точно)
        5. Группировка блоков по строкам внутри каждой ячейки
        6. Умное объединение текста: пробел для той же строки, перенос для разных строк
        7. Использование координат ячейки как координат объединенного блока
    """
    try:
        # Проверяем наличие table_res_list
        table_res_list = json_data.get('table_res_list', [])
        if len(table_res_list) == 0:
            logger.debug("table_res_list не найден, пропускаем объединение OCR блоков")
            return None, None

        # Получаем первую таблицу (поддержка нескольких таблиц может быть добавлена позже)
        table_res = table_res_list[0]
        cell_box_list = table_res.get('cell_box_list', [])

        if len(cell_box_list) == 0:
            logger.debug("cell_box_list не найден в table_res, пропускаем объединение OCR блоков")
            return None, None

        # Извлекаем OCR боксы и тексты из overall_ocr_res
        overall_ocr_res = json_data.get('overall_ocr_res', {})
        rec_boxes = overall_ocr_res.get('rec_boxes', [])
        rec_texts = overall_ocr_res.get('rec_texts', [])

        if len(rec_boxes) == 0 or len(rec_texts) == 0:
            logger.debug("rec_boxes или rec_texts отсутствуют в overall_ocr_res")
            return None, None

        logger.info(f"Объединение {len(rec_boxes)} OCR блоков в {len(cell_box_list)} ячеек таблицы")

        # Счетчики статистики
        matched_by_center = 0
        matched_by_iou = 0
        unmatched_blocks = 0

        # Отслеживаем, какие OCR блоки уже назначены ячейкам
        assigned_blocks = set()

        # Словарь для хранения блоков каждой ячейки
        cell_blocks = {cell_idx: [] for cell_idx in range(len(cell_box_list))}

        # Этап 1: Сопоставление по критерию центра (быстрая фильтрация)
        for box_idx, (ocr_box, ocr_text) in enumerate(zip(rec_boxes, rec_texts)):
            if len(ocr_box) != 4:
                continue

            ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
            center_x = (ocr_x1 + ocr_x2) / 2
            center_y = (ocr_y1 + ocr_y2) / 2

            # Проверяем каждую ячейку
            for cell_idx, cell_box in enumerate(cell_box_list):
                cell_x1, cell_y1, cell_x2, cell_y2 = cell_box

                if (cell_x1 <= center_x <= cell_x2 and
                    cell_y1 <= center_y <= cell_y2):
                    cell_blocks[cell_idx].append({
                        'index': box_idx,
                        'box': [ocr_x1, ocr_y1, ocr_x2, ocr_y2],
                        'text': ocr_text,
                        'center_y': center_y,
                        'center_x': center_x,
                        'method': 'center'
                    })
                    assigned_blocks.add(box_idx)
                    matched_by_center += 1
                    break  # Блок назначен, переходим к следующему

        # Этап 2: Сопоставление оставшихся блоков по IoU
        for box_idx, (ocr_box, ocr_text) in enumerate(zip(rec_boxes, rec_texts)):
            if box_idx in assigned_blocks or len(ocr_box) != 4:
                continue

            ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
            center_x = (ocr_x1 + ocr_x2) / 2
            center_y = (ocr_y1 + ocr_y2) / 2

            # Вычисляем IoU со всеми ячейками
            best_iou = 0.0
            best_cell_idx = -1

            for cell_idx, cell_box in enumerate(cell_box_list):
                iou = _calculate_iou([ocr_x1, ocr_y1, ocr_x2, ocr_y2], cell_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cell_idx = cell_idx

            # Назначаем ячейке, если IoU превышает порог
            if best_iou >= iou_threshold and best_cell_idx >= 0:
                cell_blocks[best_cell_idx].append({
                    'index': box_idx,
                    'box': [ocr_x1, ocr_y1, ocr_x2, ocr_y2],
                    'text': ocr_text,
                    'center_y': center_y,
                    'center_x': center_x,
                    'method': 'iou',
                    'iou': best_iou
                })
                assigned_blocks.add(box_idx)
                matched_by_iou += 1
            else:
                unmatched_blocks += 1

        # Формируем объединенные результаты
        merged_boxes = []
        merged_texts = []
        total_lines_detected = 0
        cells_with_grouping = 0

        for cell_idx, cell_box in enumerate(cell_box_list):
            blocks = cell_blocks[cell_idx]

            if not blocks:
                # Пустая ячейка
                merged_texts.append('')
                merged_boxes.append(list(cell_box))
                continue

            # Вычисляем высоту ячейки для адаптивного порога
            cell_height = cell_box[3] - cell_box[1]

            # Определяем порог по Y для группировки строк
            # Используем минимум из: 15% высоты ячейки или средняя высота блока
            if len(blocks) > 0:
                avg_block_height = sum(b['box'][3] - b['box'][1] for b in blocks) / len(blocks)
                y_threshold = min(cell_height * 0.15, avg_block_height * 0.5, 15)
            else:
                y_threshold = 10

            # Группируем блоки в текстовые строки на основе вертикальной близости
            # Сначала сортируем все блоки по Y координате
            blocks_sorted_by_y = sorted(blocks, key=lambda b: b['center_y'])

            # Группируем в строки
            text_lines = []
            current_line = [blocks_sorted_by_y[0]]

            for block in blocks_sorted_by_y[1:]:
                # Проверяем, принадлежит ли этот блок текущей строке
                prev_y = current_line[-1]['center_y']
                current_y = block['center_y']

                if abs(current_y - prev_y) <= y_threshold:
                    # Та же строка
                    current_line.append(block)
                else:
                    # Новая строка - сохраняем текущую и начинаем новую
                    text_lines.append(current_line)
                    current_line = [block]

            # Не забываем последнюю строку
            if current_line:
                text_lines.append(current_line)

            # Сортируем блоки внутри каждой строки по горизонтали (слева направо)
            for line in text_lines:
                line.sort(key=lambda b: b['center_x'])

            # Сортируем строки по вертикали (сверху вниз)
            # Используем минимальный Y первого блока в каждой строке
            text_lines.sort(key=lambda line: line[0]['center_y'])

            # Объединяем текст из всех строк
            combined_parts = []

            for line_idx, line in enumerate(text_lines):
                # Добавляем разрыв строки между строками (кроме первой)
                if line_idx > 0:
                    # Проверяем, значителен ли вертикальный разрыв
                    prev_line_y = text_lines[line_idx - 1][0]['center_y']
                    current_line_y = line[0]['center_y']
                    y_gap = abs(current_line_y - prev_line_y)

                    if y_gap > y_threshold * 1.5:
                        combined_parts.append('\n')
                    else:
                        combined_parts.append(' ')

                # Объединяем блоки внутри строки
                for block_idx, block in enumerate(line):
                    if block_idx > 0:
                        combined_parts.append(' ')
                    combined_parts.append(block['text'])

            combined_text = ''.join(combined_parts)
            merged_texts.append(combined_text)
            merged_boxes.append(list(cell_box))

            # Статистика
            total_lines_detected += len(text_lines)
            if len(text_lines) > 1:
                cells_with_grouping += 1
                logger.debug(
                    f"Ячейка {cell_idx}: {len(blocks)} блоков сгруппировано в {len(text_lines)} строк "
                    f"(порог_y={y_threshold:.1f})"
                )

        # Логируем статистику
        logger.info(
            f"Объединение OCR блоков завершено: {len(rec_boxes)} блоков -> {len(merged_boxes)} ячеек"
        )
        logger.info(
            f"  Сопоставлено по центру: {matched_by_center}, "
            f"по IoU: {matched_by_iou}, "
            f"не сопоставлено: {unmatched_blocks}"
        )
        logger.info(
            f"  Группировка строк текста: {total_lines_detected} строк обнаружено в {cells_with_grouping} ячейках"
        )

        if matched_by_iou > 0:
            logger.debug(
                f"Порог IoU {iou_threshold} помог сопоставить {matched_by_iou} дополнительных блоков"
            )

        if cells_with_grouping > 0:
            logger.debug(
                f"Применена группировка по строкам к {cells_with_grouping} многострочным ячейкам"
            )

        return merged_boxes, merged_texts

    except Exception as e:
        logger.error(f"Ошибка в _merge_ocr_blocks_by_cells: {e}", exc_info=True)
        return None, None


def _detect_table_columns(
    rec_boxes: list,
    image_width: int = 1200,
    min_gap_width: int = 10
) -> List[int]:
    """Определить границы столбцов таблицы используя анализ карты плотности.
    
    Этот приватный вспомогательный метод анализирует горизонтальное распределение OCR боксов
    для выявления вертикальных промежутков, которые служат разделителями столбцов.
    
    Args:
        rec_boxes: Список ограничивающих прямоугольников [[x1, y1, x2, y2], ...]
        image_width: Ширина анализируемого изображения в пикселях для карты плотности.
        min_gap_width: Минимальная ширина пустой зоны для определения границы столбца.
        
    Returns:
        Список x-координат, представляющих границы столбцов (вертикальные разделители).
        Пустой список, если границы не найдены.
        
    Алгоритм:
        1. Построение горизонтальной карты плотности по ширине изображения
        2. Определение непрерывных нулевых зон (промежутков без текста)
        3. Фильтрация промежутков по пороговой ширине
        4. Возврат середины каждого валидного промежутка как границы
    """
    try:
        # Шаг 1: Построение карты плотности
        density = np.zeros(image_width, dtype=int)
        for box in rec_boxes:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                x1_int = int(max(0, x1))
                x2_int = int(min(image_width - 1, x2))
            density[x1_int:x2_int] += 1
        
        logger.debug(f"Карта плотности построена, макс. плотность: {density.max()}")
        
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
        
        # Проверяем, простирается ли последний промежуток до края изображения
        if in_gap and start is not None and (image_width - start) > min_gap_width:
            middle = int((start + image_width) / 2)
            boundaries.append(middle)
        
        logger.info(f"Обнаружено {len(boundaries)} границ столбцов на x-позициях: {boundaries}")

        return boundaries

    except Exception as e:
        logger.error(f"Ошибка определения столбцов таблицы: {e}", exc_info=True)
        return []


def reconstruct_table_from_ocr(
    json_data: dict,
    image_width: int = 1200,
    min_gap_width: int = 10
) -> pd.DataFrame:
    """Восстановить структуру таблицы из результатов OCR на основе пространственного анализа.

    Эта функция восстанавливает табличную структуру из OCR ограничивающих прямоугольников и текстов
    путем анализа пространственного распределения текстовых блоков для определения границ столбцов
    и позиций строк.

    Args:
        json_data: Словарь с результатами OCR, содержащий 'rec_boxes' и 'rec_texts'.
                  Ожидаемая структура: {'rec_boxes': [[x1,y1,x2,y2], ...], 'rec_texts': [...]}
                  Опционально содержит 'table_res_list' для объединения на основе ячеек.
        image_width: Ширина анализируемого изображения в пикселях для карты плотности.
        min_gap_width: Минимальная ширина пустой зоны для определения границы столбца.

    Returns:
        DataFrame с восстановленной структурой таблицы.
        
    Алгоритм:
        1. Определение границ столбцов из оригинальных OCR данных
        2. Объединение OCR блоков по ячейкам таблицы (если доступно)
        3. Вычисление центров блоков (x_center, y_center)
        4. Распределение блоков по столбцам на основе обнаруженных границ
        5. Сортировка блоков внутри столбцов по y_center
        6. Определение количества строк в каждом столбце
        7. Построение таблицы:
           - Если все столбцы имеют одинаковое количество строк: прямое объединение по индексу
           - Если количество строк различается: выравнивание с пустыми ячейками
    """
    try:
        logger.debug(f"Начало восстановления таблицы из OCR данных (ширина_изображения={image_width})")

        # Извлекаем оригинальные боксы и тексты для определения столбцов
        original_boxes = json_data.get('rec_boxes', [])
        original_texts = json_data.get('rec_texts', [])

        if len(original_boxes) == 0 or len(original_texts) == 0:
            logger.warning("OCR боксы или тексты не найдены в данных")
            return pd.DataFrame()

        # Шаг 1: Определение границ столбцов из оригинальных OCR данных
        boundaries = _detect_table_columns(
            original_boxes,
            image_width=image_width,
            min_gap_width=min_gap_width
        )

        # Шаг 2: Попытка объединить OCR блоки по ячейкам таблицы (если доступно)
        merged_boxes, merged_texts = _merge_ocr_blocks_by_cells(json_data)

        # Определяем, какие данные использовать для восстановления таблицы
        if merged_boxes is not None and merged_texts is not None:
            logger.info("Используются объединенные по ячейкам OCR данные для восстановления таблицы")
            boxes = merged_boxes
            texts = merged_texts
        else:
            logger.debug("Используются оригинальные OCR данные для восстановления таблицы")
            boxes = original_boxes
            texts = original_texts

        if len(boxes) == 0 or len(texts) == 0:
            logger.warning("Нет боксов или текстов после обработки")
            return pd.DataFrame()

        logger.info(f"Обработка {len(boxes)} блоков для восстановления таблицы")

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
        
        # Шаг 6: Определение количества строк в каждом столбце
        column_row_counts = [len(col) for col in columns]
        max_rows = max(column_row_counts) if column_row_counts else 0
        
        if max_rows == 0:
            logger.warning("В таблице не найдено строк")
            return pd.DataFrame()
        
        # Проверка на одинаковое количество строк во всех столбцах
        all_equal_rows = len(set(column_row_counts)) == 1

        if all_equal_rows:
            logger.info(
                f"Все {num_columns} столбца содержат одинаковое количество строк ({max_rows}), "
                f"используется прямое объединение по индексу"
            )

            # Прямое объединение строк по индексу
            table = []
            for i in range(max_rows):
                row = []
                for col in columns:
                    row.append(col[i][1])  # Добавляем текст (все столбцы имеют элемент с индексом i)
                table.append(row)
        else:
            logger.info(
                f"Столбцы имеют различное количество строк: {column_row_counts}, "
                f"используется выравнивание с пустыми ячейками"
            )

            # Объединение с учетом различного количества строк
            table = []
            for i in range(max_rows):
                row = []
                for col in columns:
                    if i < len(col):
                        row.append(col[i][1])  # Добавляем текст
                    else:
                        row.append("")  # Пустая ячейка для отсутствующих элементов
                table.append(row)
        
        df = pd.DataFrame(table)
        logger.info(f"Таблица восстановлена с формой: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка восстановления таблицы из OCR данных: {e}", exc_info=True)
        return pd.DataFrame()


def write_to_json(
    file_path: Union[str, Path],
    data: Any,
    detect_headers: bool = False,
    temp_dir: Union[str, Path, None] = None
) -> bool:
    """Записать данные в JSON файл с опциональным определением и генерацией заголовков.
    
    Args:
        file_path: Путь к выходному JSON файлу.
        data: Данные для записи в JSON. Может быть DataFrame, список списков или любые JSON-сериализуемые данные.
        detect_headers: Если True, пытается определить заголовки в DataFrame и генерирует их, если не найдены.
        temp_dir: Директория для временных файлов при определении заголовков. Требуется если detect_headers=True.
        
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
                logger.warning("temp_dir не предоставлена, используется директория по умолчанию")
                temp_dir = file_path.parent
            
            temp_dir = Path(temp_dir)
            temp_csv_path = temp_dir / f"temp_{file_path.stem}.csv"
            
            try:
                # Сохраняем DataFrame во временный CSV без заголовков
                data.to_csv(temp_csv_path, index=False, sep=";", header=False)
                logger.debug(f"Создан временный CSV: {temp_csv_path}")
                
                # Проверяем наличие строки заголовка
                has_header = is_header_row_semantic(temp_csv_path)
                logger.info(f"Результат определения заголовка: {has_header}")
                
                if has_header:
                    # Читаем CSV с обнаруженным заголовком
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=0, dtype=str)
                    logger.info(f"Используются обнаруженные заголовки: {list(df_with_header.columns)}")
                else:
                    # Читаем CSV без заголовка и генерируем названия столбцов
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=None, dtype=str)
                    num_columns = len(df_with_header.columns)
                    generated_headers = [f"Заголовок {j+1}" for j in range(num_columns)]
                    df_with_header.columns = generated_headers
                    logger.info(f"Сгенерированы заголовки: {generated_headers}")
                
                # Конвертируем DataFrame в список словарей
                data = df_with_header.to_dict('records')
                logger.debug(f"DataFrame конвертирован в {len(data)} записей")
                
            finally:
                # Временный CSV НЕ удаляется для отладки
                if temp_csv_path.exists():
                    logger.info(f"Временный CSV сохранен для отладки: {temp_csv_path}")
                    temp_csv_path.unlink()  # Закомментировано для отладки
        
        elif isinstance(data, pd.DataFrame):
            # Конвертируем DataFrame в записи без определения заголовков
            data = data.to_dict('records')
            logger.debug(f"DataFrame конвертирован в {len(data)} записей")
        
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
    """Прочитать данные из JSON файла.
    
    Args:
        file_path: Путь к JSON файлу.
        
    Returns:
        Загруженные данные или None в случае неудачи.
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
    """Очистить DataFrame, удалив пустые строки/столбцы и исправив данные.
    
    Args:
        df: DataFrame для очистки.
        use_languagetool: Использовать ли языковой инструмент для коррекции текста.
        
    Returns:
        Очищенный DataFrame.
    """
    logger.debug(f"Очистка DataFrame с формой: {df.shape}")
    
    try:
        original_shape = df.shape
        
        # Удаляем полностью пустые строки и столбцы (значения NaN)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Заполняем NaN пустыми строками
        df = df.fillna('')
        
        # Удаляем пробелы из строковых столбцов и заменяем точки с запятой
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.replace(';', ' ', regex=False)
        
        # Удаляем столбцы, где все значения - пустые строки (после удаления пробелов)
        non_empty_cols = (df != '').any(axis=0)
        df = df.loc[:, non_empty_cols]
        
        # # Удаляем строки, где больше 1/3 ячеек пусты
        # if len(df.columns) > 0:
        #     empty_cells_per_row = (df == '').sum(axis=1)
        #     total_cols = len(df.columns)
        #     empty_threshold = total_cols / 3
        #     rows_to_keep = empty_cells_per_row <= empty_threshold
        #     rows_removed = (~rows_to_keep).sum()
        #     if rows_removed > 0:
        #         logger.debug(f"Удаление {rows_removed} строк с >1/3 пустых ячеек")
        #     df = df.loc[rows_to_keep, :]
        
        # Удаляем строки, где все значения - пустые строки
        non_empty_rows = (df != '').any(axis=1)
        df = df.loc[non_empty_rows, :]
        
        # Сбрасываем индекс
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
    """Валидировать DataFrame и вернуть метрики качества.
    
    Args:
        df: DataFrame для валидации.
        
    Returns:
        Словарь с результатами валидации и метриками.
    """
    try:
        logger.debug(f"Валидация DataFrame с формой: {df.shape}")
        
        validation_result = {
            'is_valid': True,
            'shape': df.shape,
            'empty_cells': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'column_types': df.dtypes.to_dict(),
            'warnings': [],
            'errors': []
        }
        
        # Проверка на пустой DataFrame
        if df.empty:
            validation_result['errors'].append("DataFrame пустой")
            validation_result['is_valid'] = False
        
        # Проверка на слишком много пустых ячеек
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            empty_ratio = validation_result['empty_cells'] / total_cells
            if empty_ratio > 0.5:
                validation_result['warnings'].append(
                    f"Высокая доля пустых ячеек: {empty_ratio:.2%}"
                )
        
        # Проверка на дубликаты строк
        if validation_result['duplicate_rows'] > 0:
            validation_result['warnings'].append(
                f"Найдено {validation_result['duplicate_rows']} дублирующихся строк"
            )
        
        logger.info(f"Валидация DataFrame завершена. Валиден: {validation_result['is_valid']}")
        logger.debug(f"Предупреждений валидации: {len(validation_result['warnings'])}")
        logger.debug(f"Ошибок валидации: {len(validation_result['errors'])}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Ошибка валидации DataFrame: {e}")
        return {
            'is_valid': False,
            'errors': [f"Валидация не удалась: {str(e)}"]
        }