"""Image Parser Module.

Модуль парсинга изображений с OCR обработкой.

This module handles image parsing and OCR processing for extracting tabular data
from images using PaddleOCR and other AI models.
"""

import ast
import os
import re
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from paddleocr import PPStructureV3, PaddleOCR
from PIL import Image

from src.mapper.ask_qwen2 import ask_qwen2
from src.utils.config import PARSING_DIR
from src.utils.df_utils import write_to_json
from src.utils.logging_config import get_logger
from src.utils.preprocess_image import preprocess_image
from src.utils.registry import register_parser

# Получение настроенного логгера
logger = get_logger(__name__)


def _load_images(image_path: Union[str, Path]) -> np.ndarray:
    """Load image, apply preprocessing, and convert to numpy array.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Numpy array of the preprocessed image.
        
    Raises:
        Exception: If image loading or preprocessing fails.
    """
    try:
        logger.debug(f"Loading image: {image_path}")
        
        # Apply preprocessing without saving to disk
        preprocessed_image = preprocess_image(str(image_path), save_to_disk=False)
        logger.debug(f"Image preprocessed successfully. Shape: {preprocessed_image.shape}")
        
        return preprocessed_image
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        raise


@register_parser(".jpg", ".jpeg", ".png")
def image_ocr(image_path: Union[str, Path]) -> List[Path]:
    """Perform OCR processing on image with table extraction.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        List of paths to created JSON files with extracted tables.
        
    Raises:
        Exception: If OCR processing fails.
    """
    logger.info(f"Starting OCR processing for image: {image_path}")
    
    try:
        image = _load_images(image_path)
        logger.info("Initializing PaddleOCR models...")
        
        #ocr_model = PaddleOCR(use_angle_cls=True, lang="ru")
        table_engine = PPStructureV3(ocr_version="PP-OCRv5", lang="ru", device="cpu")
        
        logger.info("Running table structure prediction...")
        results = table_engine.predict(image)
        
        files = []
        for i, r in enumerate(results):
            logger.info(f"Processing result {i}: found {len(r.get('parsing_res_list', []))} elements")
            
            for item in r['parsing_res_list']:
                if item.label == 'table':
                    html = item.content
                    logger.debug(f"Found table HTML content: {len(html)} characters")
                    
                    try:
                        dfs = pd.read_html(StringIO(html))
                        if dfs:
                            df = dfs[0]
                            logger.info(f"Extracted table with shape: {df.shape}")
                            
                            # Save as JSON file with header detection and generation
                            json_file_path = PARSING_DIR / f"table_{i}.json"
                            write_to_json(
                                json_file_path,
                                df,
                                detect_headers=True,
                                temp_dir=PARSING_DIR
                            )
                            files.append(json_file_path)
                            logger.info(f"Table saved as JSON: {json_file_path}")
                        else:
                            logger.warning(f"No dataframes extracted from HTML in result {i}")
                    except Exception as e:
                        logger.error(f"Error processing HTML table in result {i}: {e}")
                        continue
                else:
                    logger.debug(f"Skipping non-table element: {item.label}")
        
        logger.info(f"OCR processing completed. Created {len(files)} files")
        return files
        
    except Exception as e:
        logger.error(f"Error in OCR processing for {image_path}: {e}")
        return []


def _extract_table(answer: str) -> Optional[List]:
    """Extract table data from AI response text.
    
    Args:
        answer: Text response from AI model.
        
    Returns:
        List representation of table data or None.
        
    Raises:
        Exception: If parsing fails.
    """
    logger.debug(f"Extracting table from answer with {len(answer)} characters")
    
    match = re.search(r"\[.*\]", answer, re.S)
    if match:
        try:
            table_data = ast.literal_eval(match.group(0))
            logger.info(f"Successfully extracted table with {len(table_data)} rows")
            return table_data
        except Exception as e:
            logger.warning(f"Failed to parse table data: {e}")
            return None
    
    logger.warning("No table pattern found in AI response")
    return None



def parse_images_ai(image_path: Union[str, Path]) -> Optional[str]:
    """Process image using AI model for table extraction.
    
    Args:
        image_path: Path to image file.
        
    Returns:
        Filename of processed JSON file or None.
        
    Raises:
        Exception: If AI processing fails.
    """
    logger.info(f"Starting AI-based image processing for: {image_path}")
    
    try:
        image = _load_images(image_path)
        image = preprocess_image(image_path, image)
        
        prompt = """
        Ты — распознаватель таблиц. 
        Нужно извлечь все строки и все столбцы таблицы.
        Формат ответа:
        - Строго список списков (Python-подобный массив).
        - Каждый вложенный список = одна строка таблицы.
        - Первый вложенный список = заголовки (если они есть).
        - Остальные вложенные списки = все строки таблицы по порядку.
        - В каждой строке должно быть одинаковое количество элементов (как в таблице).
        - Не добавляй пояснений, текста или комментариев вне списка.
        - Тип данных в одном столбце должен быть одинаковый для всех строк.
        
        Пример правильного ответа:
        [["Колонка1", "Колонка2"],
         ["Значение1", "Значение2"]]
        """
        
        logger.info("Sending image to AI model for processing...")
        answer = ask_qwen2(image_path=image, prompt=prompt)
        
        if answer:
            logger.info(f"AI response received: {len(answer)} characters")
            logger.debug(f"AI response preview: {answer[:200]}...")
        else:
            logger.warning("No response received from AI model")
            return None
        
        data = _extract_table(answer)
        if data:
            logger.info(f"Extracted data: {len(data)} rows")
        else:
            logger.warning("No table data extracted from AI response")
            return None
        
        file_name = f"{Path(image_path).stem}.json"
        out_path = PARSING_DIR / file_name
        write_to_json(out_path, data)
        
        logger.info(f"AI processing completed. Saved to: {out_path}")
        return file_name
        
    except Exception as e:
        logger.error(f"Error in AI image processing for {image_path}: {e}")
        return None