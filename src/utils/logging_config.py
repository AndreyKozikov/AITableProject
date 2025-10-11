"""
Модуль централизованной конфигурации логирования.

Этот модуль обеспечивает единую конфигурацию логирования для всего проекта
AITableProject. Настраивает консольное и файловое логирование с ротацией,
кодировкой UTF-8 и настраиваемыми уровнями через конфигурационный файл.
"""

import json
import logging
import logging.config
import logging.handlers
import os
from pathlib import Path
from typing import Optional

# Пути к директориям
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"

# Создание необходимых директорий
LOG_DIR.mkdir(exist_ok=True)
CONFIG_DIR.mkdir(exist_ok=True)

# Путь к файлу конфигурации логирования
LOGGING_CONFIG_FILE = CONFIG_DIR / "logging_config.json"

# Флаг для предотвращения повторной настройки
_logging_configured = False


def configure_logging(config_file: Optional[Path] = None) -> None:
    """Единая конфигурация логирования для всего проекта.
    
    Загружает конфигурацию логирования из JSON файла и применяет её ко всем модулям.
    Все уровни логирования для каждого модуля задаются в конфигурационном файле.
    
    Args:
        config_file: Путь к файлу конфигурации JSON. Если None, используется
                    config/logging_config.json из корня проекта.
    """
    global _logging_configured
    
    # Проверяем, настроено ли логирование
    root_logger = logging.getLogger()
    if _logging_configured:
        return
    
    # Определение пути к конфигурационному файлу
    if config_file is None:
        config_file = LOGGING_CONFIG_FILE
    
    try:
        # Проверка существования файла конфигурации
        if not config_file.exists():
            raise FileNotFoundError(
                f"Файл конфигурации логирования не найден: {config_file}\n"
                f"Создайте файл config/logging_config.json в корне проекта."
            )
        
        # Загрузка конфигурации из JSON файла
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Применение конфигурации
        logging.config.dictConfig(config)
        
        # Отметка о том, что логирование настроено
        _logging_configured = True
        
        # Логирование успешной инициализации
        init_logger = logging.getLogger(__name__)
        init_logger.info(f"Логирование успешно настроено из файла: {config_file}")
        init_logger.info(f"Директория логов: {LOG_DIR}")
        init_logger.info(f"Загружено логгеров: {len(config.get('loggers', {}))}")
        
    except json.JSONDecodeError as e:
        # Если файл конфигурации невалидный, используем базовую конфигурацию
        print(f"ОШИБКА: Невалидный JSON в файле конфигурации: {e}")
        _configure_basic_logging()
        
    except Exception as e:
        # Любая другая ошибка - используем базовую конфигурацию
        print(f"ОШИБКА при настройке логирования: {e}")
        _configure_basic_logging()


def _configure_basic_logging() -> None:
    """Базовая конфигурация логирования в случае ошибки загрузки конфига.
    
    Используется как fallback, если не удалось загрузить конфигурацию из файла.
    """
    global _logging_configured
    
    root_logger = logging.getLogger()
    
    # Удаление существующих обработчиков
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    
    # Базовая конфигурация
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    _logging_configured = True
    
    root_logger.warning("Используется базовая конфигурация логирования")


def get_logger(name: str) -> logging.Logger:
    """Получить настроенный логгер для модуля.
    
    Автоматически инициализирует систему логирования при первом вызове.
    Уровень логирования для каждого модуля определяется в config/logging_config.json.
    
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
    
    УСТАРЕЛО: Используйте config/logging_config.json для настройки уровней логирования.
    Эта функция оставлена для обратной совместимости.
    
    Args:
        module_name: Имя модуля.
        level: Уровень логирования.
    """
    logger = logging.getLogger(module_name)
    logger.setLevel(level)
    logger.warning(
        f"УСТАРЕЛО: Уровень логирования для {module_name} установлен на "
        f"{logging.getLevelName(level)} программно. "
        f"Рекомендуется использовать config/logging_config.json"
    )


def disable_module_logging(module_name: str) -> None:
    """Отключить логирование для конкретного модуля.
    
    УСТАРЕЛО: Используйте config/logging_config.json для настройки уровней логирования.
    Установите уровень CRITICAL или удалите логгер из конфигурации.
    
    Args:
        module_name: Имя модуля.
    """
    logger = logging.getLogger(module_name)
    logger.disabled = True
    print(f"ВНИМАНИЕ: Логирование для {module_name} отключено программно. "
          f"Рекомендуется использовать config/logging_config.json")
    

def get_log_directory() -> Path:
    """Получить путь к директории логов.
    
    Returns:
        Path объект директории логов.
    """
    return LOG_DIR


def cleanup_old_logs(days_to_keep: int = 30) -> None:
    """Очистить старые лог-файлы.
    
    Args:
        days_to_keep: Количество дней для хранения логов.
    """
    import time
    
    logger = get_logger(__name__)
    
    try:
        current_time = time.time()
        cutoff_time = current_time - (days_to_keep * 24 * 60 * 60)
        deleted_count = 0
        
        for log_file in LOG_DIR.glob("*.log*"):
            if log_file.is_file():
                file_time = log_file.stat().st_mtime
                if file_time < cutoff_time:
                    log_file.unlink()
                    deleted_count += 1
                    logger.info(f"Удален старый лог-файл: {log_file.name}")
        
        if deleted_count > 0:
            logger.info(f"Очистка завершена: удалено {deleted_count} файлов")
        else:
            logger.info("Старые лог-файлы не найдены")
                
    except Exception as e:
        logger.error(f"Ошибка при очистке старых логов: {e}")


# Автоматическая настройка при импорте модуля
# Логирование будет настроено при первом вызове get_logger() или configure_logging()
