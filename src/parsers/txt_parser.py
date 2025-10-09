"""
Text File Parser Module
Модуль парсинга текстовых файлов

This module handles parsing of text files (TXT, CSV), extracting structured data.
Supports various text formats including CSV, TSV, and plain text with table-like structure.

Этот модуль обрабатывает парсинг текстовых файлов (TXT, CSV), извлекая структурированные данные.
Поддерживает различные текстовые форматы включая CSV, TSV и простой текст с табличной структурой.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import re
from io import StringIO

from src.utils.config import PARSING_DIR
from src.utils.registry import register_parser
from src.utils.df_utils import write_to_json

# Настройка логирования
logger = logging.getLogger(__name__)


class TextParser:
    """Класс для парсинга текстовых файлов"""
    
    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)
        self.content = None
        self.encoding = 'utf-8'
        
    def detect_encoding(self) -> str:
        """
        Определяет кодировку файла
        Returns: название кодировки
        """
        try:
            import chardet
            
            with open(self.file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"Определена кодировка: {encoding} (уверенность: {confidence:.2f})")
                
                # Если уверенность низкая, используем UTF-8 по умолчанию
                if confidence < 0.7:
                    logger.warning(f"Низкая уверенность в кодировке, используем UTF-8")
                    return 'utf-8'
                
                return encoding or 'utf-8'
                
        except ImportError:
            logger.warning("chardet не установлен, используем UTF-8 по умолчанию")
            return 'utf-8'
        except Exception as e:
            logger.error(f"Ошибка при определении кодировки: {e}")
            return 'utf-8'
    
    def load_content(self) -> bool:
        """
        Загружает содержимое файла
        Returns: True если успешно, False в противном случае
        """
        try:
            # Пытаемся определить кодировку
            self.encoding = self.detect_encoding()
            
            # Читаем файл
            with open(self.file_path, 'r', encoding=self.encoding, errors='replace') as f:
                self.content = f.read()
            
            if not self.content.strip():
                logger.warning(f"Файл {self.file_path} пуст")
                return False
            
            logger.info(f"Файл загружен успешно. Размер: {len(self.content)} символов")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {self.file_path}: {e}")
            return False
    
    def detect_format(self) -> str:
        """
        Определяет формат текстового файла
        Returns: тип формата ('csv', 'tsv', 'table', 'plain')
        """
        if not self.content:
            return 'plain'
        
        lines = self.content.strip().split('\n')
        if len(lines) < 2:
            return 'plain'
        
        # Проверяем на CSV (запятые)
        comma_count = sum(line.count(',') for line in lines[:5])
        if comma_count > len(lines[:5]):
            return 'csv'
        
        # Проверяем на TSV (табуляции)
        tab_count = sum(line.count('\t') for line in lines[:5])
        if tab_count > len(lines[:5]):
            return 'tsv'
        
        # Проверяем на разделители (точки с запятой, вертикальные черты)
        semicolon_count = sum(line.count(';') for line in lines[:5])
        if semicolon_count > len(lines[:5]):
            return 'semicolon'
        
        pipe_count = sum(line.count('|') for line in lines[:5])
        if pipe_count > len(lines[:5]):
            return 'pipe'
        
        # Проверяем на табличную структуру (несколько пробелов подряд)
        table_pattern = re.compile(r'\s{2,}')
        table_matches = sum(1 for line in lines[:5] if table_pattern.search(line))
        if table_matches > len(lines[:5]) * 0.6:
            return 'table'
        
        return 'plain'
    
    def parse_csv_format(self, separator: str = ';') -> Optional[pd.DataFrame]:
        """
        Парсит файл как CSV
        Args:
            separator: разделитель колонок
        Returns:
            DataFrame или None
        """
        try:
            df = pd.read_csv(
                StringIO(self.content),
                sep=separator,
                encoding=self.encoding,
                header=None,
                on_bad_lines='skip',
                dtype=str
            )
            
            logger.info(f"CSV парсинг успешен. Размер: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге CSV: {e}")
            return None
    
    def parse_table_format(self) -> Optional[pd.DataFrame]:
        """
        Парсит файл с табличной структурой (разделители - множественные пробелы)
        Returns:
            DataFrame или None
        """
        try:
            lines = self.content.strip().split('\n')
            parsed_rows = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Разделяем по множественным пробелам
                row = re.split(r'\s{2,}', line)
                row = [cell.strip() for cell in row if cell.strip()]
                
                if row:
                    parsed_rows.append(row)
            
            if not parsed_rows:
                return None
            
            # Определяем максимальное количество колонок
            max_cols = max(len(row) for row in parsed_rows)
            
            # Выравниваем все строки до одинакового количества колонок
            for row in parsed_rows:
                while len(row) < max_cols:
                    row.append('')
            
            # Создаем DataFrame
            df = pd.DataFrame(parsed_rows, dtype=str)
            
            logger.info(f"Табличный парсинг успешен. Размер: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге табличного формата: {e}")
            return None
    
    def parse_plain_text(self) -> Optional[pd.DataFrame]:
        """
        Парсит простой текст, пытаясь извлечь структурированные данные
        Returns:
            DataFrame или None
        """
        try:
            lines = [line.strip() for line in self.content.strip().split('\n') if line.strip()]
            
            if not lines:
                return None
            
            # Создаем простую структуру - каждая строка как отдельная запись
            df = pd.DataFrame({'Содержимое': lines}, dtype=str)
            
            logger.info(f"Текстовый парсинг успешен. Строк: {len(df)}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге простого текста: {e}")
            return None
    


def save_text_data(df: pd.DataFrame, base_filename: str, format_type: str) -> Optional[Path]:
    """
    Сохраняет данные из текстового файла в JSON формат
    Args:
        df: DataFrame с данными
        base_filename: базовое имя файла
        format_type: тип формата исходного файла
    Returns:
        путь к сохраненному файлу или None
    """
    try:
        filename = f"{base_filename}_{format_type}_data.json"
        file_path = PARSING_DIR / filename
        
        # Сохраняем в JSON с определением заголовков
        success = write_to_json(
            file_path,
            df,
            detect_headers=True,
            temp_dir=PARSING_DIR
        )
        
        if success:
            logger.info(f"Данные сохранены: {file_path}")
            return file_path
        else:
            logger.error(f"Ошибка при сохранении в JSON: {file_path}")
            return None
        
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {e}")
        return None


def load_txt_file(file_path: Union[str, Path]) -> List[Path]:
    """
    Основная функция парсинга текстовых файлов
    
    Args:
        file_path: путь к текстовому файлу
        
    Returns:
        список путей к обработанным файлам
    """
    return parse_text_file(file_path)


@register_parser(".txt", ".csv")
def parse_text_file(file_path: Union[str, Path]) -> List[Path]:
    """
    Основная функция парсинга текстовых файлов
    
    Args:
        file_path: путь к текстовому файлу
        
    Returns:
        список путей к обработанным файлам
    """
    file_path = Path(file_path)
    logger.info(f"Начинаем парсинг текстового файла: {file_path}")
    
    if not file_path.exists():
        logger.error(f"Файл не найден: {file_path}")
        return []
    
    try:
        # Создаем парсер
        parser = TextParser(file_path)
        
        # Загружаем содержимое
        if not parser.load_content():
            logger.error(f"Не удалось загрузить файл: {file_path}")
            return []
        
        # Определяем формат
        format_type = parser.detect_format()
        logger.info(f"Определен формат: {format_type}")
        
        # Парсим в зависимости от формата
        df = None
        
        if format_type == 'csv':
            df = parser.parse_csv_format(';')
        elif format_type == 'tsv':
            df = parser.parse_csv_format('\t')
        elif format_type == 'semicolon':
            df = parser.parse_csv_format(';')
        elif format_type == 'pipe':
            df = parser.parse_csv_format('|')
        elif format_type == 'table':
            df = parser.parse_table_format()
        else:  # plain
            df = parser.parse_plain_text()
        
        if df is None or len(df) == 0:
            logger.warning(f"Не удалось извлечь данные из файла: {file_path}")
            return []
        
        # Сохраняем данные
        base_filename = file_path.stem
        saved_file = save_text_data(df, base_filename, format_type)
        
        if saved_file:
            logger.info(f"Парсинг завершен успешно: {saved_file}")
            return [saved_file]
        else:
            logger.error("Не удалось сохранить обработанные данные")
            return []
        
    except Exception as e:
        logger.error(f"Критическая ошибка при парсинге {file_path}: {e}")
        return []
