"""Image Parser Module.

Модуль парсинга изображений с OCR обработкой.

This module handles image parsing and OCR processing for extracting tabular data
from images using PaddleOCR and other AI models.
"""

from io import StringIO
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import pandas as pd
from paddleocr import PPStructureV3

from src.utils.config import (
    DET_LIMIT_SIDE_LEN,
    DET_LIMIT_TYPE,
    IMAGE_PREPROCESSING_MODE,
    PARSING_DIR,
    PPSTRUCTURE_CPU_THREADS,
    PPSTRUCTURE_DEVICE,
    PPSTRUCTURE_ENABLE_HPI,
    PPSTRUCTURE_ENABLE_MKLDNN,
    PPSTRUCTURE_LANG,
    PPSTRUCTURE_LAYOUT_MODEL_NAME,
    PPSTRUCTURE_LAYOUT_NMS,
    PPSTRUCTURE_LAYOUT_THRESHOLD,
    PPSTRUCTURE_OCR_VERSION,
    PPSTRUCTURE_PRECISION,
    PPSTRUCTURE_TEXT_DET_BOX_THRESH,
    PPSTRUCTURE_TEXT_DET_MODEL_NAME,
    PPSTRUCTURE_TEXT_DET_THRESH,
    PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO,
    PPSTRUCTURE_TEXT_REC_BATCH_SIZE,
    PPSTRUCTURE_TEXT_REC_MODEL_NAME,
    PPSTRUCTURE_TEXT_REC_SCORE_THRESH,
    PPSTRUCTURE_USE_TABLE_RECOGNITION,
    USE_PADDLEOCR_DOC_ORIENTATION,
    USE_PADDLEOCR_DOC_UNWARPING,
)
from src.utils.df_utils import write_to_json
from src.utils.logging_config import get_logger
from src.utils.preprocess_image import preprocess_image
from src.utils.registry import register_parser

# Получение настроенного логгера
logger = get_logger(__name__)


def _load_images(image_path: Path, preprocessing_mode: str) -> np.ndarray:
    """Загрузка изображения с выбранным типом предобработки.
    
    Args:
        image_path: Путь к файлу изображения.
        preprocessing_mode: Режим предобработки ("custom" или "paddleocr").
        
    Returns:
        np.ndarray: Загруженное изображение (предобработанное или исходное).
        
    Raises:
        Exception: Если загрузка или предобработка не удались.
    """
    try:
        logger.info(f"Загрузка изображения: {image_path}")
        logger.debug(f"Режим предобработки: {preprocessing_mode}")

        if preprocessing_mode == "custom":
            # Режим "custom": применяем собственную предобработку
            # PPStructureV3 будет инициализирован БЕЗ встроенной предобработки
            preprocessed_image = preprocess_image(image_path, save_to_disk=True)
            logger.info(f"Применена собственная предобработка. Размер: {preprocessed_image.shape}")
            return preprocessed_image
        
        elif preprocessing_mode == "paddleocr":
            # Режим "paddleocr": загружаем исходное изображение БЕЗ предобработки
            # PPStructureV3 будет инициализирован С встроенной предобработкой
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
            logger.info(f"Загружено исходное изображение для встроенной предобработки PaddleOCR. Размер: {image.shape}")
            return image
        
        else:
            logger.warning(f"Неизвестный режим предобработки: {preprocessing_mode}. Использую 'custom'")
            preprocessed_image = preprocess_image(image_path, save_to_disk=True)
            return preprocessed_image

    except Exception as e:
        logger.error(f"Не удалось загрузить изображение {image_path}: {e}", exc_info=True)
        raise


@register_parser(".jpg", ".jpeg", ".png")
def image_ocr(image_path: Path) -> List[Path]:
    """Выполнение OCR обработки изображения с извлечением таблиц.
    
    Args:
        image_path: Путь к файлу изображения.
        
    Returns:
        Список путей к созданным JSON файлам с извлеченными таблицами.
        
    Raises:
        Exception: Если OCR обработка не удалась.
    """
    logger.info(f"Запуск OCR обработки изображения: {image_path}")

    try:
        # Загрузка изображения с выбранным режимом предобработки
        image = _load_images(image_path, IMAGE_PREPROCESSING_MODE)
        
        # Инициализация PPStructureV3 в зависимости от режима предобработки
        logger.info("Инициализация моделей PaddleOCR...")
        
        if IMAGE_PREPROCESSING_MODE == "paddleocr":
            logger.info("Режим встроенной предобработки PaddleOCR")
            
            try:
                table_engine = PPStructureV3(
                    # Основные параметры (из конфига)
                    ocr_version=PPSTRUCTURE_OCR_VERSION,
                    lang=PPSTRUCTURE_LANG,
                    device=PPSTRUCTURE_DEVICE,
                    # Производительность (из конфига)
                    enable_mkldnn=PPSTRUCTURE_ENABLE_MKLDNN,
                    cpu_threads=PPSTRUCTURE_CPU_THREADS,
                    enable_hpi=PPSTRUCTURE_ENABLE_HPI,
                    precision=PPSTRUCTURE_PRECISION,
                    # Layout detection (из конфига)
                    layout_detection_model_name=PPSTRUCTURE_LAYOUT_MODEL_NAME,
                    layout_threshold=PPSTRUCTURE_LAYOUT_THRESHOLD,
                    layout_nms=PPSTRUCTURE_LAYOUT_NMS,
                    # Встроенная предобработка документа (ВКЛЮЧЕНА для режима paddleocr)
                    use_doc_orientation_classify=USE_PADDLEOCR_DOC_ORIENTATION,
                    use_doc_unwarping=USE_PADDLEOCR_DOC_UNWARPING,
                    # Text detection параметры (из конфига)
                    text_detection_model_name=PPSTRUCTURE_TEXT_DET_MODEL_NAME,
                    text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
                    text_det_limit_type=DET_LIMIT_TYPE,
                    text_det_thresh=PPSTRUCTURE_TEXT_DET_THRESH,
                    text_det_box_thresh=PPSTRUCTURE_TEXT_DET_BOX_THRESH,
                    text_det_unclip_ratio=PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO,
                    # Text recognition параметры (из конфига)
                    text_recognition_model_name=PPSTRUCTURE_TEXT_REC_MODEL_NAME,
                    text_recognition_batch_size=PPSTRUCTURE_TEXT_REC_BATCH_SIZE,
                    text_rec_score_thresh=PPSTRUCTURE_TEXT_REC_SCORE_THRESH,
                    use_textline_orientation=True,
                    # Table recognition (из конфига)
                    use_table_recognition=PPSTRUCTURE_USE_TABLE_RECOGNITION
                )
                logger.info("PPStructureV3 инициализирован с встроенной предобработкой PaddleOCR")
                
            except Exception as init_error:
                logger.error(
                    f"Ошибка инициализации PPStructureV3 (режим paddleocr): {init_error}", 
                    exc_info=True
                )
                raise
                
        else:
            logger.info("Режим собственной предобработки (встроенная предобработка PaddleOCR отключена)")
            
            try:
                table_engine = PPStructureV3(
                    # Основные параметры (из конфига)
                    ocr_version=PPSTRUCTURE_OCR_VERSION,
                    lang=PPSTRUCTURE_LANG,
                    device=PPSTRUCTURE_DEVICE,
                    # Производительность (из конфига)
                    enable_mkldnn=PPSTRUCTURE_ENABLE_MKLDNN,
                    cpu_threads=PPSTRUCTURE_CPU_THREADS,
                    enable_hpi=PPSTRUCTURE_ENABLE_HPI,
                    precision=PPSTRUCTURE_PRECISION,
                    # Layout detection (из конфига)
                    layout_detection_model_name=PPSTRUCTURE_LAYOUT_MODEL_NAME,
                    layout_threshold=PPSTRUCTURE_LAYOUT_THRESHOLD,
                    layout_nms=PPSTRUCTURE_LAYOUT_NMS,
                    # Встроенная предобработка ОТКЛЮЧЕНА (собственная уже применена)
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    # Text detection параметры (из конфига)
                    text_detection_model_name=PPSTRUCTURE_TEXT_DET_MODEL_NAME,
                    text_det_limit_side_len=DET_LIMIT_SIDE_LEN,
                    text_det_limit_type=DET_LIMIT_TYPE,
                    text_det_thresh=PPSTRUCTURE_TEXT_DET_THRESH,
                    text_det_box_thresh=PPSTRUCTURE_TEXT_DET_BOX_THRESH,
                    text_det_unclip_ratio=PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO,
                    # Text recognition параметры (из конфига)
                    text_recognition_model_name=PPSTRUCTURE_TEXT_REC_MODEL_NAME,
                    text_recognition_batch_size=PPSTRUCTURE_TEXT_REC_BATCH_SIZE,
                    text_rec_score_thresh=PPSTRUCTURE_TEXT_REC_SCORE_THRESH,
                    use_textline_orientation=False,
                    # Table recognition (из конфига)
                    use_table_recognition=PPSTRUCTURE_USE_TABLE_RECOGNITION
                )
                logger.info("PPStructureV3 инициализирован без встроенной предобработки (используется собственная)")
                
            except Exception as init_error:
                logger.error(
                    f"Ошибка инициализации PPStructureV3 (режим custom): {init_error}", 
                    exc_info=True
                )
                raise

        logger.info("Выполнение предсказания структуры таблицы...")
        results = table_engine.predict(image)

        files = []
        for i, r in enumerate(results):
            logger.info(f"Обработка результата {i}: найдено {len(r.get('parsing_res_list', []))} элементов")

            for item in r['parsing_res_list']:
                if item.label == 'table':
                    html = item.content
                    logger.debug(f"Найдено HTML содержимое таблицы: {len(html)} символов")

                    try:
                        dfs = pd.read_html(StringIO(html))
                        if dfs:
                            df = dfs[0]
                            logger.info(f"Извлечена таблица размером: {df.shape}")

                            # Сохранение как JSON файл с определением и генерацией заголовков
                            json_file_path = PARSING_DIR / f"table_{i}.json"
                            write_to_json(
                                json_file_path,
                                df,
                                detect_headers=True,
                                temp_dir=PARSING_DIR
                            )
                            files.append(json_file_path)
                            logger.info(f"Таблица сохранена как JSON: {json_file_path}")
                        else:
                            logger.warning(f"Не извлечено dataframes из HTML в результате {i}")
                    except Exception as e:
                        logger.error(f"Ошибка обработки HTML таблицы в результате {i}: {e}")
                        continue
                else:
                    logger.debug(f"Пропуск не-табличного элемента: {item.label}")

        logger.info(f"OCR обработка завершена. Создано {len(files)} файлов")
        return files

    except Exception as e:
        logger.error(f"Ошибка при OCR обработке {image_path}: {e}", exc_info=True)
        logger.debug(f"Тип ошибки: {type(e).__name__}")
        return []
