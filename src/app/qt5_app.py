"""
PyQt5 версия AITableProject Application.

Реализует графический интерфейс на PyQt5 с профессиональным стилем
и расширенным пользовательским опытом.
"""

import sys
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QListWidget, QListWidgetItem, QComboBox, QRadioButton, QButtonGroup,
    QProgressBar, QFileDialog, QMessageBox, QFrame, QScrollArea, QSizePolicy,
    QTextEdit, QProgressDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMimeData
from PyQt5.QtGui import QFont, QIcon, QDragEnterEvent, QDropEvent, QPalette, QColor

# Импорты существующих утилит
from src.utils.config import INBOX_DIR, MODEL_ID, MODEL_CACHE_DIR
from src.utils.process_files import process_files
from src.utils.logging_config import get_logger
from src.utils.download_model import check_model_exists, download_model, get_model_size_info

logger = get_logger(__name__)


class FileProcessingThread(QThread):
    """Поток для обработки файлов в фоне."""
    
    progress = pyqtSignal(int, int, str)  # current_step, total_steps, status_message
    finished = pyqtSignal(object)  # result_path
    error = pyqtSignal(str)  # error_message
    
    def __init__(self, files: List[Path], extended: bool, remote_model: bool, use_cot: bool, use_gguf: bool = False):
        super().__init__()
        self.files = files
        self.extended = extended
        self.remote_model = remote_model
        self.use_cot = use_cot
        self.use_gguf = use_gguf
    
    def run(self):
        """Выполняет обработку файлов в фоновом потоке."""
        try:
            logger.info(f"Начинаем обработку {len(self.files)} файлов в фоновом потоке")
            self.progress.emit(1, 3, "Парсинг файлов...")
            
            result = process_files(
                self.files,
                extended=self.extended,
                remote_model=self.remote_model,
                use_cot=self.use_cot,
                use_gguf=self.use_gguf
            )
            
            self.progress.emit(3, 3, "Обработка завершена")
            self.finished.emit(result)
            
        except Exception as e:
            logger.error(f"Ошибка в потоке обработки: {e}")
            self.error.emit(str(e))


class ModelDownloadThread(QThread):
    """Поток для загрузки модели в фоне."""
    
    progress = pyqtSignal(str)  # status_message
    finished = pyqtSignal(bool, str)  # success, message
    
    def __init__(self, model_id: str, cache_dir: Path):
        super().__init__()
        self.model_id = model_id
        self.cache_dir = cache_dir
    
    def run(self):
        """Выполняет загрузку модели в фоновом потоке."""
        try:
            logger.info(f"Начинаем загрузку модели {self.model_id}")
            self.progress.emit(f"Загрузка модели {self.model_id}...")
            
            tokenizer, model = download_model(self.model_id, self.cache_dir)
            
            if tokenizer is not None and model is not None:
                size_info = get_model_size_info(self.cache_dir)
                message = f"Модель загружена успешно!\nРазмер: {size_info}"
                self.finished.emit(True, message)
                logger.info("Загрузка модели завершена успешно")
            else:
                self.finished.emit(False, "Не удалось загрузить модель. Проверьте логи.")
                logger.error("Загрузка модели не удалась")
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            self.finished.emit(False, f"Ошибка загрузки: {str(e)}")


class DropArea(QFrame):
    """Виджет зоны перетаскивания файлов."""
    
    files_dropped = pyqtSignal(list)  # List[str] - пути к файлам
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(2)
        
        # Настройка layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        
        # Иконка и текст
        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px;")
        
        text_label = QLabel("Перетащите файлы сюда\nили нажмите для выбора")
        text_label.setAlignment(Qt.AlignCenter)
        text_label.setStyleSheet("font-size: 14px; color: #64748b;")
        
        layout.addWidget(icon_label)
        layout.addWidget(text_label)
        
        # Стиль
        self.setStyleSheet("""
            DropArea {
                background-color: #f8fafc;
                border: 2px dashed #cbd5e1;
                border-radius: 8px;
                padding: 40px;
            }
            DropArea:hover {
                border-color: #3b82f6;
                background-color: #eff6ff;
            }
        """)
        
        self.setMinimumHeight(200)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Обрабатывает событие входа перетаскивания."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        """Обрабатывает событие сброса файлов."""
        files = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if Path(file_path).is_file():
                files.append(file_path)
        
        if files:
            self.files_dropped.emit(files)
    
    def mousePressEvent(self, event):
        """Открывает диалог выбора файлов при клике."""
        if event.button() == Qt.LeftButton:
            self.parent().parent().open_file_dialog()


class MainWindow(QMainWindow):
    """Главное окно приложения PyQt5."""
    
    def __init__(self):
        super().__init__()
        
        # Состояние приложения
        self.saved_files: List[Path] = []
        self.upload_map = {}  # Хранит сигнатуры файлов
        self.processing_start_time: Optional[datetime] = None
        self.processing_thread: Optional[FileProcessingThread] = None
        
        # Настройка окна
        self.setWindowTitle("AITableProject - Professional Document Processing")
        self.setMinimumSize(1200, 800)
        
        # Применяем стили
        self.setup_styles()
        
        # Создаем UI
        self.setup_ui()
        
        logger.info("Главное окно инициализировано")
    
    def setup_styles(self):
        """Применяет глобальные стили к приложению."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8fafc;
            }
            QLabel {
                color: #1e293b;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #cbd5e1;
                color: #94a3b8;
            }
            QComboBox, QRadioButton {
                font-size: 14px;
                color: #1e293b;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                background-color: white;
            }
            QComboBox:hover {
                border-color: #3b82f6;
            }
            QRadioButton {
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QProgressBar {
                border: 2px solid #e2e8f0;
                border-radius: 6px;
                text-align: center;
                background-color: #f1f5f9;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
            QListWidget {
                border: 2px solid #e2e8f0;
                border-radius: 8px;
                background-color: white;
                padding: 10px;
            }
            QListWidget::item {
                padding: 10px;
                border-bottom: 1px solid #e2e8f0;
            }
            QListWidget::item:hover {
                background-color: #f1f5f9;
            }
            QFrame.settings-card {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
                padding: 20px;
            }
        """)
    
    def setup_ui(self):
        """Создает пользовательский интерфейс."""
        # Центральный виджет
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Главный layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)
        
        # Заголовок
        header_layout = self.create_header()
        main_layout.addLayout(header_layout)
        
        # Контент (две колонки)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)
        
        # Левая колонка (загрузка файлов)
        left_column = self.create_left_column()
        content_layout.addLayout(left_column, 6)  # 60% ширины
        
        # Правая колонка (настройки)
        right_column = self.create_right_column()
        content_layout.addLayout(right_column, 4)  # 40% ширины
        
        main_layout.addLayout(content_layout)
    
    def create_header(self) -> QHBoxLayout:
        """Создает заголовок приложения."""
        header_layout = QHBoxLayout()
        
        # Левая часть - заголовок
        title_layout = QVBoxLayout()
        
        title = QLabel("📊 AITableProject")
        title.setStyleSheet("font-size: 32px; font-weight: bold; color: #1e293b;")
        
        subtitle = QLabel("Профессиональная обработка документов с помощью ИИ")
        subtitle.setStyleSheet("font-size: 16px; color: #64748b; margin-top: 5px;")
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        
        # Правая часть - кнопка загрузки модели
        self.download_model_button = QPushButton("⬇️ Загрузить модель")
        self.download_model_button.setMinimumHeight(40)
        self.download_model_button.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
            QPushButton:disabled {
                background-color: #cbd5e1;
                color: #94a3b8;
            }
        """)
        self.download_model_button.clicked.connect(self.start_model_download)
        
        header_layout.addWidget(self.download_model_button)
        
        return header_layout
    
    def create_left_column(self) -> QVBoxLayout:
        """Создает левую колонку с загрузкой файлов."""
        layout = QVBoxLayout()
        
        # Заголовок секции
        section_title = QLabel("📤 Загрузка документов")
        section_title.setStyleSheet("font-size: 18px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(section_title)
        
        # Зона перетаскивания
        self.drop_area = DropArea(self)
        self.drop_area.files_dropped.connect(self.handle_files_dropped)
        layout.addWidget(self.drop_area)
        
        # Список файлов
        files_label = QLabel("📋 Загруженные файлы")
        files_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-top: 20px; margin-bottom: 10px;")
        layout.addWidget(files_label)
        
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(300)
        layout.addWidget(self.file_list)
        
        return layout
    
    def create_right_column(self) -> QVBoxLayout:
        """Создает правую колонку с настройками."""
        layout = QVBoxLayout()
        
        # Карточка настроек
        settings_frame = QFrame()
        settings_frame.setObjectName("settings-card")
        settings_frame.setProperty("class", "settings-card")
        settings_frame.setStyleSheet("""
            #settings-card {
                background-color: white;
                border-radius: 12px;
                border: 1px solid #e2e8f0;
            }
        """)
        
        settings_layout = QVBoxLayout(settings_frame)
        settings_layout.setSpacing(20)
        
        # Заголовок
        title = QLabel("⚙️ Настройки обработки")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        settings_layout.addWidget(title)
        
        # Выбор модели
        model_label = QLabel("🤖 Модель ИИ")
        model_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        settings_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "Локальная модель Qwen 3",
            "Локальная модель Qwen 3 + CoT",
            "Локальная модель Qwen GGUF",
            "Облачная модель ChatGPT"
        ])
        self.model_combo.setToolTip("CoT (Chain-of-Thought) - модель с цепочками рассуждений для лучшей точности\nGGUF - квантованная модель для быстрого инференса")
        settings_layout.addWidget(self.model_combo)
        
        # Режим обработки
        mode_label = QLabel("📊 Режим обработки")
        mode_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        settings_layout.addWidget(mode_label)
        
        self.mode_group = QButtonGroup()
        self.smart_mode_radio = QRadioButton("Умное распределение")
        self.simple_mode_radio = QRadioButton("Упрощенное распределение")
        self.smart_mode_radio.setChecked(True)
        self.smart_mode_radio.setToolTip("Умное распределение обеспечивает более точную категоризацию данных")
        
        self.mode_group.addButton(self.smart_mode_radio)
        self.mode_group.addButton(self.simple_mode_radio)
        
        settings_layout.addWidget(self.smart_mode_radio)
        settings_layout.addWidget(self.simple_mode_radio)
        
        # Разделитель
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #e2e8f0;")
        settings_layout.addWidget(separator)
        
        # Прогресс бар
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setTextVisible(True)
        settings_layout.addWidget(self.progress_bar)
        
        # Статус
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #64748b; font-size: 13px;")
        self.status_label.setWordWrap(True)
        self.status_label.setVisible(False)
        settings_layout.addWidget(self.status_label)
        
        # Кнопка обработки
        self.process_button = QPushButton("🚀 Начать обработку")
        self.process_button.setEnabled(False)
        self.process_button.setMinimumHeight(45)
        self.process_button.clicked.connect(self.start_processing)
        settings_layout.addWidget(self.process_button)
        
        # Кнопка скачивания результата
        self.download_button = QPushButton("📥 Скачать результат Excel")
        self.download_button.setVisible(False)
        self.download_button.setMinimumHeight(45)
        self.download_button.clicked.connect(self.download_result)
        settings_layout.addWidget(self.download_button)
        
        # Информационная подсказка
        info_label = QLabel("👆 Загрузите файлы для начала обработки")
        info_label.setStyleSheet("color: #3b82f6; font-size: 13px; margin-top: 10px;")
        info_label.setWordWrap(True)
        settings_layout.addWidget(info_label)
        
        settings_layout.addStretch()
        
        layout.addWidget(settings_frame)
        
        return layout
    
    def open_file_dialog(self):
        """Открывает диалог выбора файлов."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите файлы для обработки",
            "",
            "Все поддерживаемые файлы (*.txt *.csv *.xlsx *.xls *.pdf *.doc *.docx *.jpg *.jpeg *.png);;Все файлы (*.*)"
        )
        
        if files:
            self.handle_files_dropped(files)
    
    def handle_files_dropped(self, files: List[str]):
        """Обрабатывает загруженные файлы."""
        logger.info(f"Получено {len(files)} файлов")
        
        next_idx = 1
        added_count = 0
        
        for file_path_str in files:
            file_path = Path(file_path_str)
            
            # Проверка расширения
            if file_path.suffix.lower() not in ['.txt', '.csv', '.xlsx', '.xls', '.pdf', 
                                                  '.doc', '.docx', '.jpg', '.jpeg', '.png']:
                logger.warning(f"Неподдерживаемый формат файла: {file_path.suffix}")
                continue
            
            # Генерируем сигнатуру
            sig = self._file_signature(file_path)
            
            if sig in self.upload_map:
                # Файл уже загружен
                saved_path = Path(self.upload_map[sig])
            else:
                # Сохраняем новый файл
                safe_name, used_idx = self._make_unique_name(file_path.name, next_idx)
                saved_path = INBOX_DIR / safe_name
                
                try:
                    # Копируем файл
                    with open(file_path, 'rb') as src, open(saved_path, 'wb') as dst:
                        dst.write(src.read())
                    
                    self.upload_map[sig] = str(saved_path)
                    next_idx = used_idx + 1
                    added_count += 1
                    
                except Exception as e:
                    logger.error(f"Ошибка копирования файла {file_path.name}: {e}")
                    continue
            
            # Добавляем в список, если ещё нет
            if saved_path not in self.saved_files:
                self.saved_files.append(saved_path)
                self.add_file_to_list(saved_path)
        
        if added_count > 0:
            logger.info(f"Добавлено {added_count} новых файлов")
            self.process_button.setEnabled(True)
    
    def add_file_to_list(self, file_path: Path):
        """Добавляет файл в список отображения."""
        file_size = file_path.stat().st_size / 1024  # KB
        icon = self._get_file_icon(file_path)
        
        item = QListWidgetItem(f"{icon} {file_path.name} ({file_size:.1f} KB)")
        item.setData(Qt.UserRole, str(file_path))
        
        self.file_list.addItem(item)
    
    def _get_file_icon(self, file_path: Path) -> str:
        """Получает emoji иконку для типа файла."""
        ext = file_path.suffix.lower()
        icon_map = {
            '.pdf': "📄",
            '.xlsx': "📊",
            '.xls': "📊",
            '.doc': "📘",
            '.docx': "📘",
            '.jpg': "🖼️",
            '.jpeg': "🖼️",
            '.png': "🖼️",
            '.txt': "📝",
            '.csv': "📝",
        }
        return icon_map.get(ext, "📎")
    
    def _transliterate_ru_to_latin(self, text: str) -> str:
        """Транслитерирует русский текст в латинские символы."""
        mapping = {
            'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E', 
            'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 
            'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U', 
            'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sch', 
            'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
            'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 
            'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 
            'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 
            'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch', 
            'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        }
        return ''.join(mapping.get(ch, ch) for ch in text)
    
    def _sanitize_stem(self, stem: str) -> str:
        """Очищает основу имени файла."""
        translit = self._transliterate_ru_to_latin(stem)
        translit = translit.strip()
        translit = re.sub(r"[^A-Za-z0-9._-]+", "_", translit)
        translit = re.sub(r"_+", "_", translit)
        translit = translit.strip("._-")
        translit = translit.lower()
        return translit or "file"
    
    def _make_unique_name(self, original_name: str, start_index: int) -> tuple:
        """Генерирует уникальное имя файла."""
        p = Path(original_name)
        base = self._sanitize_stem(p.stem)
        ext = p.suffix.lower()
        idx = max(1, start_index)
        while True:
            candidate = f"{base}_{idx:03d}{ext}"
            if not (INBOX_DIR / candidate).exists():
                return candidate, idx
            idx += 1
    
    def _file_signature(self, file_path: Path) -> str:
        """Генерирует уникальную подпись для файла."""
        with open(file_path, 'rb') as f:
            content = f.read()
            md5 = hashlib.md5(content).hexdigest()
        
        size = file_path.stat().st_size
        return f"{file_path.name}:{size}:{md5}"
    
    def start_processing(self):
        """Начинает обработку файлов."""
        if not self.saved_files:
            QMessageBox.warning(self, "Предупреждение", "Пожалуйста, загрузите файлы для обработки")
            return
        
        # Определяем параметры обработки
        model_choice = self.model_combo.currentText()
        remote_model = model_choice == "Облачная модель ChatGPT"
        use_cot = model_choice == "Локальная модель Qwen 3 + CoT"
        use_gguf = model_choice == "Локальная модель Qwen GGUF"
        extended = self.smart_mode_radio.isChecked()
        
        logger.info(f"Начинаем обработку: extended={extended}, remote={remote_model}, cot={use_cot}, gguf={use_gguf}")
        
        # Отключаем кнопку и показываем прогресс
        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_label.setVisible(True)
        self.status_label.setText("Запуск обработки...")
        
        # Сохраняем время старта
        self.processing_start_time = datetime.now()
        
        # Создаем и запускаем поток обработки
        self.processing_thread = FileProcessingThread(
            self.saved_files,
            extended,
            remote_model,
            use_cot,
            use_gguf
        )
        
        self.processing_thread.progress.connect(self.update_progress)
        self.processing_thread.finished.connect(self.processing_finished)
        self.processing_thread.error.connect(self.processing_error)
        
        self.processing_thread.start()
    
    def update_progress(self, current: int, total: int, message: str):
        """Обновляет прогресс обработки."""
        percentage = int((current / total) * 100)
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        logger.info(f"Прогресс: {current}/{total} - {message}")
    
    def processing_finished(self, result_path: Optional[Path]):
        """Обрабатывает завершение обработки."""
        end_time = datetime.now()
        duration = end_time - self.processing_start_time
        duration_str = str(duration).split('.')[0]
        
        self.progress_bar.setVisible(False)
        
        if result_path and Path(result_path).exists():
            self.status_label.setText(f"✅ Обработка завершена за {duration_str}")
            self.status_label.setStyleSheet("color: #10b981; font-size: 13px;")
            
            # Показываем кнопку скачивания
            self.download_button.setVisible(True)
            self.result_path = result_path
            
            QMessageBox.information(
                self, 
                "Успех", 
                f"Обработка завершена успешно за {duration_str}!\n\nРезультат сохранен в:\n{result_path}"
            )
            
            # Сброс состояния
            self.saved_files = []
            self.upload_map = {}
            self.file_list.clear()
            
        else:
            self.status_label.setText("❌ Ошибка: результат не создан")
            self.status_label.setStyleSheet("color: #ef4444; font-size: 13px;")
            self.process_button.setEnabled(True)
            
            QMessageBox.critical(self, "Ошибка", "Не удалось создать файл результата")
        
        logger.info(f"Обработка завершена. Результат: {result_path}")
    
    def processing_error(self, error_message: str):
        """Обрабатывает ошибку обработки."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"❌ Ошибка: {error_message}")
        self.status_label.setStyleSheet("color: #ef4444; font-size: 13px;")
        self.process_button.setEnabled(True)
        
        QMessageBox.critical(self, "Ошибка обработки", f"Произошла ошибка:\n\n{error_message}")
        logger.error(f"Ошибка обработки: {error_message}")
    
    def download_result(self):
        """Открывает диалог сохранения результата."""
        if not hasattr(self, 'result_path') or not Path(self.result_path).exists():
            QMessageBox.warning(self, "Предупреждение", "Файл результата не найден")
            return
        
        default_name = Path(self.result_path).name
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить результат",
            default_name,
            "Excel файлы (*.xlsx);;Все файлы (*.*)"
        )
        
        if save_path:
            try:
                # Копируем файл
                with open(self.result_path, 'rb') as src, open(save_path, 'wb') as dst:
                    dst.write(src.read())
                
                QMessageBox.information(self, "Успех", f"Файл сохранен:\n{save_path}")
                logger.info(f"Результат сохранен в: {save_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить файл:\n{e}")
                logger.error(f"Ошибка сохранения файла: {e}")
    
    def start_model_download(self):
        """Начинает загрузку модели."""
        # Проверяем, не загружена ли модель уже
        if check_model_exists(MODEL_CACHE_DIR):
            size_info = get_model_size_info(MODEL_CACHE_DIR)
            reply = QMessageBox.question(
                self,
                "Модель уже загружена",
                f"Модель {MODEL_ID} уже загружена.\n"
                f"Размер: {size_info}\n\n"
                f"Загрузить заново?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.No:
                return
        
        # Подтверждение загрузки
        reply = QMessageBox.question(
            self,
            "Загрузка модели",
            f"Начать загрузку модели {MODEL_ID}?\n\n"
            f"Это может занять несколько минут и требует интернет-соединения.\n"
            f"Размер загрузки: ~3-5 GB",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        
        if reply == QMessageBox.No:
            return
        
        # Отключаем кнопку и создаем диалог прогресса
        self.download_model_button.setEnabled(False)
        
        self.model_progress_dialog = QProgressDialog(
            "Загрузка модели...",
            "Отмена",
            0, 0,
            self
        )
        self.model_progress_dialog.setWindowTitle("Загрузка модели")
        self.model_progress_dialog.setWindowModality(Qt.WindowModal)
        self.model_progress_dialog.setMinimumDuration(0)
        self.model_progress_dialog.setCancelButton(None)  # Нельзя отменить
        self.model_progress_dialog.show()
        
        # Создаем и запускаем поток загрузки
        self.model_download_thread = ModelDownloadThread(MODEL_ID, MODEL_CACHE_DIR)
        self.model_download_thread.progress.connect(self.update_model_download_progress)
        self.model_download_thread.finished.connect(self.model_download_finished)
        self.model_download_thread.start()
        
        logger.info(f"Начата загрузка модели {MODEL_ID}")
    
    def update_model_download_progress(self, message: str):
        """Обновляет прогресс загрузки модели."""
        self.model_progress_dialog.setLabelText(message)
        logger.info(f"Прогресс загрузки: {message}")
    
    def model_download_finished(self, success: bool, message: str):
        """Обрабатывает завершение загрузки модели."""
        self.model_progress_dialog.close()
        self.download_model_button.setEnabled(True)
        
        if success:
            QMessageBox.information(
                self,
                "Загрузка завершена",
                message
            )
            logger.info("Загрузка модели завершена успешно")
        else:
            QMessageBox.critical(
                self,
                "Ошибка загрузки",
                message
            )
            logger.error(f"Ошибка загрузки модели: {message}")


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

