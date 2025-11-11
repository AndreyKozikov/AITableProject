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
    """Преобразует значение в список, если оно еще не является списком.
    
    Args:
        x: Значение для преобразования.
        
    Returns:
        Список, содержащий значение, или пустой список для None.
    """
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def parsing_files(files: List[Path]) -> List[Path]:
    """Обрабатывает список файлов через соответствующие парсеры.
    
    Args:
        files: Список объектов Path для обработки.
        
    Returns:
        Список путей к обработанным файлам.
    """
    logger.info(f"Начинаем парсинг {len(files)} файлов")
    
    results = {}
    parsing_files_name = []

    for file in files:
        try:
            suffix = file.suffix.lower()
            file_name = file.name
            handler = PARSERS.get(suffix)
            
            logger.debug(f"Обработка файла: {file_name} с расширением: {suffix}")
            
            if handler:
                logger.info(f"Используем парсер {handler.__name__} для файла: {file_name}")
                files_list = _ensure_list(handler(file)) # Вызов парсера через зарегистрированные для расширений методы
                if files_list is not None:
                    parsing_files_name.extend(files_list)
                    results[file_name] = "Файл успешно обработан"
                    logger.info(f"Файл успешно обработан: {file_name}")
                else:
                    results[file_name] = "Ошибка при обработке файла"
                    logger.warning(f"Обработчик вернул None для файла: {file_name}")
            else:
                results[file_name] = "❌ Неподдерживаемый формат"
                logger.warning(f"Неподдерживаемый формат файла: {suffix} для файла: {file_name}")
                
        except Exception as e:
            results[file_name] = f"Ошибка: {str(e)}"
            logger.error(f"Ошибка обработки файла {file_name}: {e}")
            continue
    
    successful_files = sum(1 for result in results.values() if "успешно" in result)
    logger.info(f"Парсинг завершен. Успешно: {successful_files}, Всего: {len(files)}, Результатов: {len(parsing_files_name)}")
    
    return parsing_files_name


def save_to_xlsx(rows: List[dict]) -> Path:
    """Сохраняет данные в Excel файл.
    
    Args:
        rows: Список словарей, где каждый словарь представляет строку таблицы.
              Ключи словаря - имена столбцов, значения - содержимое ячеек.
        
    Returns:
        Путь к созданному Excel файлу.
    """
    logger.info(f"Сохраняем {len(rows)} строк в Excel")
    
    if not rows:
        logger.warning("Нет данных для сохранения в Excel")
        return None
    
    # Создаем DataFrame из списка словарей
    df = pd.DataFrame(rows)
    
    file_path = OUT_DIR / 'result.xlsx'
    df.to_excel(file_path, index=False)
    
    logger.info(f"Данные сохранены в Excel: {file_path}")
    logger.debug(f"Размер Excel файла: {file_path.stat().st_size} байт")
    
    return file_path


def process_files(files: List[Path], 
                 extended: bool = False, 
                 remote_model: bool = False,
                 use_cot: bool = False,
                 use_gguf: bool = False) -> Optional[Path]:
    """Главная функция обработки файлов.
    
    Args:
        files: Список файлов для обработки.
        extended: Флаг расширенной обработки.
        remote_model: Флаг использования удаленной AI модели.
        use_cot: Флаг использования модели с Chain-of-Thought reasoning.
        use_gguf: Флаг использования GGUF модели через llama-cpp-python.
        
    Returns:
        Путь к созданному файлу результата или None.
    """
    logger.info(f"Начинаем обработку {len(files)} файлов")
    logger.info(f"Расширенный режим: {extended}, Удаленная модель: {remote_model}, CoT: {use_cot}, GGUF: {use_gguf}")
    
    try:
        if remote_model:
            logger.info("Используем удаленную модель OpenAI для обработки")
            handler = PARSERS.get("openai")
            if not handler:
                logger.error("Парсер OpenAI не найден в реестре")
                return None
            rows, header = handler([str(f) for f in files])
            logger.info("Обработка удаленной моделью завершена")
        else:
            logger.info("Используем локальный конвейер обработки")
            files_list_csv = parsing_files(files)
            if not files_list_csv:
                logger.warning("Нет обработанных файлов для маппинга")
                return None
                
            rows = mapper_structured(
                        files=files_list_csv,
                        extended=extended,
                        enable_thinking=False,
                        use_cot=use_cot,
                        use_gguf=use_gguf
                    )


            logger.info("Локальная обработка завершена")
        
        file_path = save_to_xlsx(rows)
        if file_path:
            logger.info(f"Обработка файлов завершена успешно: {file_path}")
        else:
            logger.error("Не удалось сохранить результаты в Excel")
            
        return file_path
        
    except Exception as e:
        logger.error(f"Критическая ошибка при обработке файлов: {e}")
        return None