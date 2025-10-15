"""Parser Registry Module.

Модуль реестра парсеров.

Registry system for file format parsers allowing parsers to register
themselves for specific file extensions.
"""

from typing import Callable, Dict

from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

PARSERS: Dict[str, Callable] = {}


def register_parser(*suffixes: str) -> Callable:
    """Декоратор для регистрации парсера по расширению файла.
    
    Args:
        *suffixes: Расширения файлов (например, '.txt', '.pdf').
        
    Returns:
        Декоратор функции.
    """
    def wrapper(func: Callable) -> Callable:
        """Внутренняя функция декоратора.
        
        Args:
            func: Функция парсера для регистрации.
            
        Returns:
            Исходная функция без изменений.
        """
        for suffix in suffixes:
            normalized_suffix = suffix.lower().strip()
            if normalized_suffix:
                PARSERS[normalized_suffix] = func
                logger.debug(f"Зарегистрирован парсер {func.__name__} для {normalized_suffix}")
        return func
    return wrapper