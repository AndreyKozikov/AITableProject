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

# Инициализация логирования
configure_logging()
logger = get_logger(__name__)


def clean_directories() -> None:
    """Clean all files from inbox and parsing directories.
    
    Removes all files from INBOX_DIR and PARSING_DIR to prepare
    for new processing session.
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
                        logger.debug(f"Deleted from inbox: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned inbox directory: {deleted_count} files deleted")
            else:
                logger.debug("Inbox directory is empty")
        else:
            logger.warning(f"Inbox directory does not exist: {INBOX_DIR}")
        
        # Очистка директории parsing_files
        if PARSING_DIR.exists():
            deleted_count = 0
            for file_path in PARSING_DIR.iterdir():
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted from parsing_files: {file_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path.name}: {e}")
            
            if deleted_count > 0:
                logger.info(f"Cleaned parsing_files directory: {deleted_count} files deleted")
            else:
                logger.debug("Parsing_files directory is empty")
        else:
            logger.warning(f"Parsing directory does not exist: {PARSING_DIR}")
            
    except Exception as e:
        logger.error(f"Error during directory cleanup: {e}")


def main() -> None:
    """Main application function.
    
    Sets up Python path and launches Streamlit application.
    
    Raises:
        SystemExit: On critical startup error.
    """
    try:
        logger.info("Starting AITableProject application...")
        
        # Clean directories before starting
        logger.info("Cleaning directories...")
        clean_directories()
        
        # Add src path to Python path
        src_path = Path(__file__).parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            logger.info(f"Added to sys.path: {src_path}")
            
        # Launch Streamlit application
        app_path = src_path / "app" / "enhanced_app.py"
        logger.info(f"Launching Streamlit app: {app_path}")
        
        subprocess.run(["streamlit", "run", str(app_path)], check=True)
        
    except FileNotFoundError:
        logger.error("Streamlit not found. Install: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()