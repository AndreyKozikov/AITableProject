"""File Processing Utilities.

Утилиты для обработки файлов.

This module provides utilities for processing various file formats through
registered parsers and compiling results into Excel format.
"""

from pathlib import Path
from typing import Any, List, Optional, Union

import pandas as pd

from src.mapper.mapper import mapper_structured
from src.parsers.img_parser import image_ocr
from src.parsers.pdf_parser import parse_pdf
from src.parsers.txt_parser import load_txt_file
import src.parsers.docx_parser
import src.parsers.excel_parser
import src.parsers.openai_parser

from src.utils.config import OUT_DIR, PARSING_DIR
from src.utils.logging_config import get_logger
from src.utils.registry import PARSERS

# Получение настроенного логгера
logger = get_logger(__name__)


def _ensure_list(x: Any) -> List[Any]:
    """Convert value to list if it's not already a list.
    
    Args:
        x: Value to convert.
        
    Returns:
        List containing the value or empty list for None.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def parsing_files(files: List[Path]) -> List[Path]:
    """Process list of files through appropriate parsers.
    
    Args:
        files: List of Path objects to process.
        
    Returns:
        List of processed file paths.
    """
    logger.info(f"Starting to parse {len(files)} files")
    
    results = {}
    parsing_files_name = []

    for file in files:
        try:
            suffix = file.suffix.lower()
            file_name = file.name
            handler = PARSERS.get(suffix)
            
            logger.debug(f"Processing file: {file_name} with extension: {suffix}")
            
            if handler:
                logger.info(f"Using parser {handler.__name__} for file: {file_name}")
                files_list = _ensure_list(handler(file)) # Вызов парсера через зарегистрированные для расширений методы
                if files_list is not None:
                    parsing_files_name.extend(files_list)
                    results[file_name] = "Файл успешно обработан"
                    logger.info(f"Successfully processed file: {file_name}")
                else:
                    results[file_name] = "Ошибка при обработке файла"
                    logger.warning(f"Handler returned None for file: {file_name}")
            else:
                results[file_name] = "❌ Неподдерживаемый формат"
                logger.warning(f"Unsupported file format: {suffix} for file: {file_name}")
                
        except Exception as e:
            results[file_name] = f"Ошибка: {str(e)}"
            logger.error(f"Error processing file {file_name}: {e}")
            continue
    
    successful_files = sum(1 for result in results.values() if "успешно" in result)
    logger.info(f"Parsing completed. Successful: {successful_files}, Total: {len(files)}, Results: {len(parsing_files_name)}")
    
    return parsing_files_name


def save_to_xlsx(rows: List[dict]) -> Path:
    """Save data to Excel file.
    
    Args:
        rows: List of dictionaries where each dict represents a table row.
              Dictionary keys are column names, values are cell contents.
        
    Returns:
        Path to created Excel file.
    """
    logger.info(f"Saving {len(rows)} rows to Excel")
    
    if not rows:
        logger.warning("No data to save to Excel")
        return None
    
    # Create DataFrame from list of dictionaries
    df = pd.DataFrame(rows)
    
    file_path = OUT_DIR / 'result.xlsx'
    df.to_excel(file_path, index=False)
    
    logger.info(f"Data saved to Excel: {file_path}")
    logger.debug(f"Excel file size: {file_path.stat().st_size} bytes")
    
    return file_path


def process_files(files: List[Path], 
                 extended: bool = False, 
                 remote_model: bool = False) -> Optional[Path]:
    """Main file processing function.
    
    Args:
        files: List of files to process.
        extended: Extended processing flag.
        remote_model: Use remote AI model flag.
        
    Returns:
        Path to created result file or None.
    """
    logger.info(f"Starting processing of {len(files)} files")
    logger.info(f"Extended mode: {extended}, Remote model: {remote_model}")
    
    try:
        if remote_model:
            logger.info("Using remote OpenAI model for processing")
            handler = PARSERS.get("openai")
            if not handler:
                logger.error("OpenAI parser not found in registry")
                return None
            rows, header = handler([str(f) for f in files])
            logger.info("Remote model processing completed")
        else:
            logger.info("Using local processing pipeline")
            files_list_csv = parsing_files(files)
            if not files_list_csv:
                logger.warning("No processed files for mapping")
                return None
                
            rows = mapper_structured(
                        files=files_list_csv,
                        extended=extended,  # или True для расширенного режима
                        enable_thinking=False  # или True для режима рассуждений
                    )


            logger.info("Local processing completed")
        
        file_path = save_to_xlsx(rows)
        if file_path:
            logger.info(f"File processing completed successfully: {file_path}")
        else:
            logger.error("Failed to save results to Excel")
            
        return file_path
        
    except Exception as e:
        logger.error(f"Critical error in file processing: {e}")
        return None