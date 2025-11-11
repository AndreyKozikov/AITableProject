"""
Главная точка входа в PyQt5 приложение.

Entry point for the AITableProject PyQt5 application that launches
the GUI for document processing.
"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt

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
    
    Настраивает Python path и запускает PyQt5 приложение.
    
    Raises:
        SystemExit: При критической ошибке запуска.
    """
    try:
        logger.info("Запуск PyQt5 приложения AITableProject...")
        
        # Очистка директорий перед запуском
        logger.info("Очистка директорий...")
        clean_directories()

        # Добавление пути src в Python path
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            logger.info(f"Добавлено в sys.path: {src_path}")
        
        # Импортируем главное окно после добавления пути
        from src.app.qt5_app import MainWindow
        
        # Создаем приложение Qt
        app = QApplication(sys.argv)
        app.setApplicationName("AITableProject")
        app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        
        # Создаем и показываем главное окно
        window = MainWindow()
        window.show()
        
        logger.info("PyQt5 приложение запущено успешно")
        
        # Запускаем event loop
        sys.exit(app.exec_())
        
    except ImportError as e:
        logger.error(f"Ошибка импорта: {e}. Установите PyQt5: pip install PyQt5")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

