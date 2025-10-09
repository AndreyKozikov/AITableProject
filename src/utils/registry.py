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
    """Decorator for registering parser by file extension.
    
    Args:
        *suffixes: File extensions (e.g. '.txt', '.pdf').
        
    Returns:
        Function decorator.
    """
    def wrapper(func: Callable) -> Callable:
        """Inner decorator function.
        
        Args:
            func: Parser function to register.
            
        Returns:
            Original function unchanged.
        """
        for suffix in suffixes:
            normalized_suffix = suffix.lower().strip()
            if normalized_suffix:
                PARSERS[normalized_suffix] = func
                logger.debug(f"Registered parser {func.__name__} for {normalized_suffix}")
        return func
    return wrapper