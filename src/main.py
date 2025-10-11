"""
Главная точка входа в приложение.

Entry point for the AITableProject application that launches
the Streamlit web interface for document processing.
"""

import subprocess
import sys
from pathlib import Path

# Настройка централизованного логирования в самом начале
from src.utils.logging_config import configure_logging, get_logger
from src.utils.config import INBOX_DIR, PARSING_DIR

# Инициализация логирования из config/logging_config.json
configure_logging()
logger = get_logger(__name__)


def clean_directories() -> None:
    """Очистка всех файлов из директорий inbox и parsing.
    
    Удаляет все файлы из INBOX_DIR и PARSING_DIR для подготовки
    к новой сессии обработки.
    """
    try:
        # Очистка директории inbox
        if INBOX_DIR.exists():
            deleted_count = 0
            for file_path in INBOX_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Удален из inbox: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить {file_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Очищена директория inbox: удалено {deleted_count} файлов")
            else:
                logger.debug("Директория inbox пуста")
        else:
            logger.warning(f"Директория inbox не существует: {INBOX_DIR}")
        
        # Очистка директории parsing_files
        if PARSING_DIR.exists():
            deleted_count = 0
            for file_path in PARSING_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Удален из parsing_files: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Не удалось удалить {file_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Очищена директория parsing_files: удалено {deleted_count} файлов")
            else:
                logger.debug("Директория parsing_files пуста")
        else:
            logger.warning(f"Директория parsing не существует: {PARSING_DIR}")
            
    except Exception as e:
        logger.error(f"Ошибка при очистке директорий: {e}")


def main() -> None:
    """Главная функция приложения.
    
    Настраивает Python path и запускает Streamlit приложение.
    
    Raises:
        SystemExit: При критической ошибке запуска.
    """
    try:
        logger.info("Запуск приложения AITableProject...")
        
        # Очистка директорий перед запуском
        logger.info("Очистка директорий...")
        clean_directories()

        # Добавление пути src в Python path
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            logger.info(f"Добавлено в sys.path: {src_path}")
            
        # Запуск Streamlit приложения
        app_path = src_path / "app" / "enhanced_app.py"
        logger.info(f"Запуск Streamlit приложения: {app_path}")
        
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
        
    except FileNotFoundError:
        logger.error("Streamlit не найден. Установите: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Ошибка запуска Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()