"""

Модуль централизованной конфигурации логирования.

Этот модуль обеспечивает единую конфигурацию логирования для всего проекта
AITableProject. Настраивает консольное и файловое логирование с ротацией,
кодировкой UTF-8 и настраиваемыми уровнями через переменные окружения.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from typing import Optional

# Создание директории для логов
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Флаг для предотвращения повторной настройки
_logging_configured = False


def configure_logging(level: Optional[int] = None) -> None:
    """Единая конфигурация логирования для всего проекта.
    
    Настраивает логирование с выводом в консоль и файлы с ротацией.
    Поддерживает настройку уровня логирования через переменную окружения.
    
    Args:
        level: Уровень логирования. Если None, используется INFO или значение
               из переменной окружения LOG_LEVEL.
    """
    global _logging_configured
    
    # Предотвращаем повторную настройку
    if _logging_configured:
        return
    
    # Определение уровня логирования
    if level is None:
        env_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        level = level_mapping.get(env_level, logging.INFO)
    
    # Создание форматтера
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Ротация по размеру для общего файла приложения
    app_log_path = os.path.join(LOG_DIR, "app.log")
    size_handler = logging.handlers.RotatingFileHandler(
        app_log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8"
    )
    size_handler.setFormatter(formatter)
    size_handler.setLevel(level)
    
    # Ежедневный файловый лог с хранением 7 дней
    daily_log_path = os.path.join(
        LOG_DIR, 
        datetime.now().strftime("%Y-%m-%d") + ".log"
    )
    time_handler = logging.handlers.TimedRotatingFileHandler(
        daily_log_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    time_handler.setFormatter(formatter)
    time_handler.setLevel(level)
    
    # Настройка корневого логгера
    root_logger = logging.getLogger()
    
    # Удаление существующих обработчиков для предотвращения дублирования
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Установка уровня и добавление обработчиков
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(size_handler)
    root_logger.addHandler(time_handler)
    
    # Отметка о том, что логирование настроено
    _logging_configured = True
    
    # Логирование успешной инициализации
    init_logger = logging.getLogger(__name__)
    init_logger.info(f"Logging configured successfully. Level: {logging.getLevelName(level)}")
    init_logger.info(f"Log directory: {LOG_DIR}")
    init_logger.info(f"App log: {app_log_path}")
    init_logger.info(f"Daily log: {daily_log_path}")


def get_logger(name: str) -> logging.Logger:
    """Получить настроенный логгер для модуля.
    
    Args:
        name: Имя модуля (обычно __name__).
        
    Returns:
        Настроенный объект логгера.
    """
    # Убеждаемся, что логирование настроено
    if not _logging_configured:
        configure_logging()
    
    return logging.getLogger(name)


def set_module_log_level(module_name: str, level: int) -> None:
    """Установить уровень логирования для конкретного модуля.
    
    Args:
        module_name: Имя модуля.
        level: Уровень логирования.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.info(f"Log level for {module_name} set to {logging.getLevelName(level)}")


def disable_module_logging(module_name: str) -> None:
    """Отключить логирование для конкретного модуля.
    
    Args:
        module_name: Имя модуля.
    """
    logger = logging.getLogger(module_name)
    logger.disabled = True
    

def get_log_directory() -> str:
    """Получить путь к директории логов.
    
    Returns:
        Путь к директории логов.
    """
    return LOG_DIR


def cleanup_old_logs(days_to_keep: int = 30) -> None:
    """Очистить старые лог-файлы.
    
    Args:
        days_to_keep: Количество дней для хранения логов.
    """
    import glob
    import time
    
    logger = get_logger(__name__)
    
    try:
        log_pattern = os.path.join(LOG_DIR, "*.log*")
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        
        for log_file in glob.glob(log_pattern):
            file_time = os.path.getmtime(log_file)
            if file_time < cutoff_time:
                os.remove(log_file)
                logger.info(f"Removed old log file: {log_file}")
                
    except Exception as e:
        logger.error(f"Error cleaning up old logs: {e}")


# Автоматическая настройка при импорте модуля, если переменная окружения установлена
if os.getenv('AUTO_CONFIGURE_LOGGING', '').lower() in ('true', '1', 'yes'):
    configure_logging()
