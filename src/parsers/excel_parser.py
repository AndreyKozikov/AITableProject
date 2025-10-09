"""

Модуль парсинга Excel файлов (XLS, XLSX)


Этот модуль обрабатывает парсинг Excel файлов, извлекая таблицы и данные из электронных таблиц.
Поддерживает как старый (.xls), так и новый (.xlsx) формат Excel.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
import pandas as pd

try:
    import openpyxl
except ImportError:
    openpyxl = None

try:
    import xlrd
except ImportError:
    xlrd = None

from src.utils.config import PARSING_DIR, HEADER_ANCHORS
from src.utils.registry import register_parser
from src.utils.df_utils import write_to_json, clean_dataframe

# Настройка логирования
logger = logging.getLogger(__name__)


class ExcelParser:
    """Класс для парсинга Excel файлов"""

    def __init__(self, file_path: Union[str, Path]):
        self.file_path = Path(file_path)

    def _normalize_text(self, text: str) -> str:
        """
        Нормализует текст для сравнения (как в pdf_parser)
        Args:
            text: исходный текст
        Returns:
            нормализованный текст
        """
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().replace("ё", "е")
        text = " ".join(text.split())
        return text

    def _find_header_row(self, df: pd.DataFrame) -> Optional[int]:
        """
        Ищет строку с заголовками из HEADER_ANCHORS (алгоритм из pdf_parser)
        Args:
            df: DataFrame для поиска
        Returns:
            Индекс строки с заголовками или None если не найдено
        """
        try:
            best_idx = None
            best_score = 0

            # Проверяем первые 20 строк на наличие заголовков
            max_rows_to_check = min(20, len(df))

            for row_idx in range(max_rows_to_check):
                row = df.iloc[row_idx]

                # Собираем текст из всех ячеек строки
                row_text_parts = []
                for val in row:
                    if pd.notna(val) and str(val).strip():
                        row_text_parts.append(str(val))

                if not row_text_parts:
                    continue

                # Объединяем весь текст строки
                row_text = " ".join(row_text_parts)
                normalized_row = self._normalize_text(row_text)

                # Считаем совпадения с HEADER_ANCHORS (как в pdf_parser)
                score = 0
                for _, keys in HEADER_ANCHORS:
                    if any(k in normalized_row for k in keys):
                        score += 1

                # Обновляем лучший результат
                if score > best_score:
                    best_idx = row_idx
                    best_score = score
                    logger.debug(
                        f"Строка {row_idx}: score={score}, текст='{row_text[:50]}...'"
                    )

            # Если score >= 2, считаем что нашли заголовок (как в pdf_parser)
            if best_score >= 2:
                logger.info(
                    f"Найдена строка заголовков на позиции {best_idx} "
                    f"(score={best_score})"
                )
                return best_idx
            else:
                logger.debug(
                    f"Заголовки из HEADER_ANCHORS не найдены "
                    f"(лучший score={best_score})"
                )
                return None

        except Exception as e:
            logger.error(f"Ошибка при поиске заголовков: {e}")
            return None

    def _extract_table_from_header(self, df: pd.DataFrame, header_row_idx: int) -> pd.DataFrame:
        """
        Извлекает таблицу начиная со строки заголовков
        Args:
            df: Исходный DataFrame
            header_row_idx: Индекс строки с заголовками
        Returns:
            DataFrame с данными таблицы (включая строку заголовков)
        """
        try:
            # Берем таблицу начиная со строки заголовков (ВКЛЮЧАЯ её)
            table_df = df.iloc[header_row_idx:].copy()

            # Сбрасываем индекс
            table_df = table_df.reset_index(drop=True)

            logger.info(
                f"Извлечена таблица: {len(table_df)} строк (включая заголовок), "
                f"начиная со строки {header_row_idx}"
            )
            logger.debug(f"Первая строка (заголовок): {table_df.iloc[0].tolist()}")

            return table_df

        except Exception as e:
            logger.error(f"Ошибка при извлечении таблицы: {e}")
            return df

    def load_workbook(self) -> bool:
        """
        Загружает Excel файл
        Returns: True если успешно, False в противном случае
        """
        try:
            file_ext = self.file_path.suffix.lower()

            if file_ext == '.xlsx':
                # Для новых Excel файлов используем pandas с openpyxl
                if openpyxl is None:
                    logger.error("openpyxl не установлен. Установите: pip install openpyxl")
                    return False
                self.engine = 'openpyxl'

            elif file_ext == '.xls':
                # Для старых Excel файлов используем pandas с xlrd
                if xlrd is None:
                    logger.error("xlrd не установлен. Установите: pip install xlrd")
                    return False
                self.engine = 'xlrd'

            else:
                logger.error(f"Неподдерживаемый формат файла: {file_ext}")
                return False

            return True

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {self.file_path}: {e}")
            return False

    def get_sheet_names(self) -> List[str]:
        """
        Получает список имен листов в Excel файле
        Returns: список имен листов
        """
        try:
            # Используем pandas для получения списка листов
            excel_file = pd.ExcelFile(self.file_path, engine=self.engine)
            return excel_file.sheet_names

        except Exception as e:
            logger.error(f"Ошибка при получении списка листов: {e}")
            return []

    def parse_sheet(self, sheet_name: str = None, sheet_index: int = 0) -> Optional[pd.DataFrame]:
        """
        Парсит конкретный лист Excel файла
        Args:
            sheet_name: имя листа (если None, используется sheet_index)
            sheet_index: индекс листа (используется если sheet_name не указано)
        Returns:
            DataFrame с данными листа или None
        """
        try:
            # Читаем лист
            if sheet_name:
                df = pd.read_excel(
                    self.file_path,
                    sheet_name=sheet_name,
                    engine=self.engine,
                    header=None,  # Читаем без автоматического определения заголовков
                    dtype=str
                )
            else:
                df = pd.read_excel(
                    self.file_path,
                    sheet_name=sheet_index,
                    engine=self.engine,
                    header=None,
                    dtype=str
                )

            if df.empty:
                logger.warning(f"Лист {'с именем ' + sheet_name if sheet_name else f'с индексом {sheet_index}'} пуст")
                return None

            # Ищем строку с заголовками из HEADER_ANCHORS
            header_row_idx = self._find_header_row(df)

            if header_row_idx is not None:
                # Если заголовки найдены, извлекаем таблицу начиная с них
                df = self._extract_table_from_header(df, header_row_idx)
                logger.info(
                    f"Таблица извлечена со строки {header_row_idx}, "
                    f"игнорировано {header_row_idx} строк"
                )
            else:
                # Если заголовки не найдены, оставляем все данные как есть
                logger.info("Заголовки из HEADER_ANCHORS не найдены, используем все данные")

            return df

        except Exception as e:
            logger.error(f"Ошибка при парсинге листа {sheet_name or sheet_index}: {e}")
            return None

    def parse_all_sheets(self) -> List[pd.DataFrame]:
        """
        Парсит все листы Excel файла
        Returns:
            список DataFrame с данными всех листов
        """
        try:
            sheet_names = self.get_sheet_names()
            parsed_sheets = []

            for sheet_name in sheet_names:
                logger.info(f"Обрабатываем лист: {sheet_name}")

                df = self.parse_sheet(sheet_name=sheet_name)
                if df is not None and len(df) > 0:
                    # Добавляем информацию о листе
                    df.attrs['sheet_name'] = sheet_name
                    parsed_sheets.append(df)
                    logger.info(f"Лист '{sheet_name}' успешно обработан. Размер: {df.shape}")
                else:
                    logger.warning(f"Лист '{sheet_name}' пуст или не может быть обработан")

            return parsed_sheets

        except Exception as e:
            logger.error(f"Ошибка при парсинге всех листов: {e}")
            return []


def save_sheets_to_files(sheets: List[pd.DataFrame], base_filename: str) -> List[Path]:
    """
    Сохраняет листы Excel в отдельные JSON файлы
    Args:
        sheets: список DataFrame с листами
        base_filename: базовое имя файла
    Returns:
        список путей к сохраненным файлам
    """
    saved_files = []

    for i, df in enumerate(sheets):
        try:
            # Получаем имя листа из атрибутов или создаем стандартное
            sheet_name = getattr(df, 'attrs', {}).get('sheet_name', f'Sheet_{i + 1}')

            # Создаем безопасное имя файла
            safe_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_')).rstrip()

            if len(sheets) == 1:
                filename = f"{base_filename}_excel_data.json"
            else:
                filename = f"{base_filename}_{safe_sheet_name}.json"

            file_path = PARSING_DIR / filename

            # Очищаем DataFrame перед сохранением
            logger.debug(f"Cleaning DataFrame for sheet '{sheet_name}', original shape: {df.shape}")
            df_cleaned = clean_dataframe(df, use_languagetool=False)
            logger.debug(f"DataFrame cleaned, new shape: {df_cleaned.shape}")

            # Сохраняем в JSON с использованием write_to_json
            # detect_headers=True для автоматического определения заголовков
            # temp_dir=PARSING_DIR для временных файлов
            success = write_to_json(
                file_path,
                df_cleaned,
                detect_headers=True,
                temp_dir=PARSING_DIR
            )

            if success:
                saved_files.append(file_path)
                logger.info(f"Лист '{sheet_name}' сохранен в JSON: {file_path}")
            else:
                logger.error(f"Не удалось сохранить лист '{sheet_name}' в JSON")

        except Exception as e:
            logger.error(f"Ошибка при сохранении листа {i + 1}: {e}")
            continue

    return saved_files


@register_parser(".xlsx", ".xls")
def parse_excel(file_path: Union[str, Path]) -> List[Path]:
    """
    Основная функция парсинга Excel файлов
    
    Args:
        file_path: путь к Excel файлу
        
    Returns:
        список путей к обработанным файлам
    """
    file_path = Path(file_path)
    logger.info(f"Начинаем парсинг Excel файла: {file_path}")

    if not file_path.exists():
        logger.error(f"Файл не найден: {file_path}")
        return []

    try:
        # Создаем парсер
        parser = ExcelParser(file_path)

        # Загружаем файл
        if not parser.load_workbook():
            logger.error(f"Не удалось загрузить файл: {file_path}")
            return []

        base_filename = file_path.stem

        # Парсим все листы
        sheets = parser.parse_all_sheets()

        if not sheets:
            logger.warning(f"Не найдено данных в файле: {file_path}")
            return []

        logger.info(f"Найдено листов с данными: {len(sheets)}")

        # Сохраняем листы в файлы
        processed_files = save_sheets_to_files(sheets, base_filename)

        logger.info(f"Парсинг завершен. Обработано файлов: {len(processed_files)}")
        return processed_files

    except Exception as e:
        logger.error(f"Критическая ошибка при парсинге {file_path}: {e}")
        return []
