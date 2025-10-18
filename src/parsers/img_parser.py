"""Image Parser Module.

Модуль парсинга изображений с OCR обработкой.

This module handles image parsing and OCR processing for extracting tabular data
from images using PaddleOCR and other AI models.
"""

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
from src.utils.df_utils import write_to_json, clean_dataframe, reconstruct_table_from_ocr
from src.utils.image_preprocessor import ImagePreprocessor
from src.utils.logging_config import get_logger
from src.utils.registry import register_parser

# Получение настроенного логгера
logger = get_logger(__name__)

# OCR post-processing (опционально)
try:
    from src.utils.ocr_postprocessor import OCRPostProcessor, OCRConfig
    OCR_POSTPROCESSOR_AVAILABLE = True
except ImportError as e:
    OCR_POSTPROCESSOR_AVAILABLE = False
    logger.warning(f"OCR пост-процессор недоступен: {e}")


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

    image_preprocessor = ImagePreprocessor()


    logger.info(f"Запуск OCR обработки изображения: {image_path}")

    try:
        
        # Инициализация PPStructureV3 в зависимости от режима предобработки
        logger.info("Инициализация моделей PaddleOCR...")
        
        if IMAGE_PREPROCESSING_MODE == "paddleocr":
            logger.info("Режим встроенной предобработки PaddleOCR")
            
            try:
                image = image_preprocessor.load_image(image_path)
                image = image_preprocessor.resize_for_ocr(image)

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
                    #layout_detection_model_name=PPSTRUCTURE_LAYOUT_MODEL_NAME,
                    layout_threshold=PPSTRUCTURE_LAYOUT_THRESHOLD,
                    layout_nms=PPSTRUCTURE_LAYOUT_NMS,
                    # Встроенная предобработка документа (ВКЛЮЧЕНА для режима paddleocr)
                    #use_doc_orientation_classify=USE_PADDLEOCR_DOC_ORIENTATION,
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
                    #use_textline_orientation=True,
                    # Table recognition (из конфига)
                    use_table_recognition=PPSTRUCTURE_USE_TABLE_RECOGNITION,
                    use_formula_recognition=False
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
                image = image_preprocessor.run(image_path)
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
        
        # for res in results:
        #     res.save_to_img(save_path=PARSING_DIR)
        #     res.save_to_json(save_path=PARSING_DIR)


        # Get image dimensions for reconstruction
        image_width = image.shape[1] if hasattr(image, 'shape') else 1200
        logger.debug(f"Ширина изображения для реконструкции таблицы: {image_width}")

        files = []
        table_count = 0
        
        for i, r in enumerate(results):
            logger.info(f"Обработка результата {i}")
            
            # Check if this result contains table data
            parsing_res_list = r.get('parsing_res_list', [])
            has_table = any(
                getattr(item, 'label', None) == 'table' 
                for item in parsing_res_list
            )
            
            if not has_table:
                logger.debug(f"Результат {i} не содержит таблиц, пропуск")
                continue
            
            # Extract OCR data from overall_ocr_res
            overall_ocr_res = r.get('overall_ocr_res', {})
            
            if not overall_ocr_res:
                logger.warning(f"Результат {i} не содержит overall_ocr_res, пропуск")
                continue
            
            rec_boxes = overall_ocr_res.get('rec_boxes', [])
            rec_texts = overall_ocr_res.get('rec_texts', [])
            
            if len(rec_boxes) == 0 or len(rec_texts) == 0:
                logger.warning(f"Результат {i}: отсутствуют rec_boxes или rec_texts")
                continue
            
            logger.info(f"Найдено {len(rec_boxes)} OCR блоков для реконструкции таблицы")
            
            try:
                # Reconstruct table from OCR geometry
                # Pass full result structure for cell-based merging support
                ocr_data = {
                    'rec_boxes': rec_boxes,
                    'rec_texts': rec_texts,
                    'table_res_list': r.get('table_res_list', []),
                    'overall_ocr_res': overall_ocr_res
                }
                
                df = reconstruct_table_from_ocr(ocr_data, image_width=image_width)
                
                if df.empty:
                    logger.warning(f"Результат {i}: реконструкция таблицы вернула пустой DataFrame")
                    continue
                
                logger.info(f"Таблица реконструирована с размером: {df.shape}")
                
                # Clean DataFrame
                df_clean = clean_dataframe(df)
                
                # OCR post-processing (if available)
                if OCR_POSTPROCESSOR_AVAILABLE:
                    try:
                        ocr_config = OCRConfig(debug=False, log_corrections=False)
                        ocr_processor = OCRPostProcessor(config=ocr_config)
                        df_clean = ocr_processor.process_dataframe(df_clean)
                        logger.debug("OCR пост-обработка применена успешно")
                    except Exception as ocr_error:
                        logger.warning(f"OCR пост-обработка не удалась: {ocr_error}, продолжаем с очищенными данными")
                
                # Сохраняем как JSON с определением заголовков
                json_file_path = PARSING_DIR / f"{image_path.stem}_table_{table_count}.json"
                write_to_json(
                    json_file_path,
                    df_clean,
                    detect_headers=True,
                    temp_dir=PARSING_DIR
                )
                files.append(json_file_path)
                logger.info(f"Таблица {table_count} сохранена как JSON: {json_file_path}")
                table_count += 1
                
            except Exception as e:
                logger.error(f"Ошибка реконструкции таблицы из результата {i}: {e}", exc_info=True)
                continue
        
        if table_count == 0:
            logger.info("Таблицы не найдены на странице")

        logger.info(f"OCR обработка завершена. Создано {len(files)} файлов")
        return files

    except Exception as e:
        logger.error(f"Ошибка при OCR обработке {image_path}: {e}", exc_info=True)
        logger.debug(f"Тип ошибки: {type(e).__name__}")
        return []
