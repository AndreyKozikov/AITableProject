"""Configuration Module.

Модуль конфигурации проекта.

Contains all configuration settings, directory paths, and constants
for the AITableProject application.
"""

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

# ==================== БАЗОВЫЕ НАСТРОЙКИ ====================
# Используются во всех модулях проекта

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROOT = Path(__file__).resolve().parents[1]
INBOX_DIR = ROOT.parent / "inbox"
PARSING_DIR = ROOT.parent / "parsing_files"
OUT_DIR = ROOT.parent / "out"
OUT_DIR_LEARNING_DATA = ROOT / "learning" / "datasets"
MODEL_DIR = ROOT.parent / "model_table"

# ==================== ПАРСИНГ ТАБЛИЦ ====================
# Используется в: excel_parser.py, docx_parser.py, pdf_parser.py, df_utils.py

HEADER_ANCHORS: List[Tuple[str, List[str]]] = [
    ("poz", ["поз", "позиция", "артикул", "№", "п/п", "№ п/п"]),  # Поз
    ("name", ["наименование", "товар", "товары", "обозначение", "инструмент"]),  # Наименование
    ("pic", ["рис"]),  # Рис
    ("qty", ["кол", "кол-во", "количество", "кол."]),  # Кол-во
    ("price", ["цена"]),  # Цена (подойдёт и "Цена за шт.")
    ("curr", ["валюта"]),  # Валюта
    ("disc", ["скидка"]),  # Скидка
    ("other", ["ндс", "сумма", "ставка", "бренд", "производитель", "модель", "фирма", "единица измерения", "потребность"])
]

# ==================== МАППИНГ ДАННЫХ ====================
# Используется в: mapper.py, ask_qwen3_so.py

# Модель Qwen для структурированного вывода
# ID модели на HuggingFace Hub (для загрузки)
MODEL_ID = "Qwen/Qwen3-1.7B"

# Локальная директория для хранения модели
MODEL_CACHE_DIR = ROOT.parent / "models" / "qwen"

# Использовать локальную модель (если True, загружает из MODEL_CACHE_DIR)
# Если False, загружает напрямую из HuggingFace Hub
USE_LOCAL_MODEL = True

PROMPT_TEMPLATE = """
Ты — ассистент по обработке табличных данных.
Формат выходных данных: **только таблица в Markdown**, без пояснений и заголовков.

Правила:
- Используй ровно эти столбцы: {header}.
- Каждое входное значение помещается только в один столбец.
- Если подходит под 'Количество', 'Наименование' или 'Единица измерения' — ставь туда.
- Если данных нет для столбца — оставляй пустую ячейку.
- Не добавляй новых строк или столбцов.
- Артикул не является наименованием.

Входные данные представлены в формате json:
{tables_text}
Выведи результат в виде Markdown-таблицы.
"""

PROMPT_TEMPLATE_SO = """
Ты — ассистент по структурированной обработке табличных данных.  
Извлекай данные из текста и представляй их строго в формате JSON.  

Правила:  
- Используй ровно эти ключи: {header}.  
- Структура ответа должна полностью соответствовать JSON Schema (см. ниже).  
- Не добавляй новых ключей или уровней вложенности.  
- Если данных нет для ключа — вставляй пустую строку.  
- Не используй Markdown, блоки ```json или пояснительный текст.  
- Вывод — только корректный JSON, без заголовков и комментариев.  

<output-format>  
Твой вывод обязан строго соответствовать следующей JSON-схеме:  
{schema}  
</output-format>  

Входные данные:  
{tables_text}  
"""

# # ==================== ПРЕДОБРАБОТКА ИЗОБРАЖЕНИЙ ====================
# # Используется в: preprocess_image.py
# # Параметры сгруппированы в порядке применения операций
#
# # ---------- Общие параметры масштабирования (_resize_for_ocr) ----------
# # PaddleOCR параметры разрешения для table detection
# # Рекомендуемое значение для распознавания таблиц: 736
# # Для текстового OCR по умолчанию используется 960
DET_LIMIT_SIDE_LEN = 736
#
# # Тип ограничения стороны изображения: 'min' или 'max'
# # 'min' - минимальная сторона изображения не будет меньше DET_LIMIT_SIDE_LEN
# # 'max' - максимальная сторона изображения не будет больше DET_LIMIT_SIDE_LEN
DET_LIMIT_TYPE = 'min'

# ---------- Режим предобработки изображений ----------
# 
# Доступные режимы:
# 
# "custom" - Собственная предобработка из preprocess_image.py
#   Применяется: preprocess_image() перед передачей в PaddleOCR
#   PPStructureV3 инициализируется без встроенной предобработки
#   Этапы обработки:
#     ✓ Преобразование в оттенки серого
#     ✓ Коррекция перспективы (perspective correction)
#     ✓ Выравнивание наклона (deskew)
#     ✓ Масштабирование до оптимального размера (resize для OCR)
#     ✓ Шумоподавление (denoise)
#     ✓ Повышение резкости (sharpen)
#     ✓ Улучшение контраста (CLAHE)
#     ✓ Опционально: бинаризация (binarization)
#   Рекомендуется для: сканированных документов, фото документов
#
# "paddleocr" - Встроенная предобработка PaddleOCR
#   Применяется: исходное изображение загружается и передается в PaddleOCR
#   PPStructureV3 инициализируется с модулями предобработки:
#     ✓ use_doc_orientation_classify (классификация ориентации 0°, 90°, 180°, 270°)
#     ✓ use_doc_unwarping (исправление геометрических искажений UVDoc)
#   ВНИМАНИЕ: doc_unwarping требует: pip install shapely pyclipper
#   Рекомендуется для: документов с неправильной ориентацией, сильными искажениями
#
IMAGE_PREPROCESSING_MODE = "paddleocr"
#IMAGE_PREPROCESSING_MODE = "custom"

# ---------- Параметры режима "paddleocr" ----------
# (игнорируются при режиме "custom")

# Использовать классификацию ориентации документа (0°, 90°, 180°, 270°)
# True - автоматически определять и исправлять ориентацию
# False - предполагать, что документ уже правильно ориентирован
USE_PADDLEOCR_DOC_ORIENTATION = False

# Использовать исправление геометрических искажений документа (UVDoc модель)
# True - исправлять искажения, деформации, кривизну документа
# False - не исправлять геометрические искажения
# ВНИМАНИЕ: требует установки дополнительных зависимостей
#   pip install shapely pyclipper
USE_PADDLEOCR_DOC_UNWARPING = True

# ---------- Основные параметры PPStructureV3 ----------
PPSTRUCTURE_OCR_VERSION = "PP-OCRv5"  # Версия OCR моделей (последняя версия)
PPSTRUCTURE_LANG = "ru"  # Язык: автоматически использует eslav_PP-OCRv5_mobile_rec
                         # Поддерживает: русский + английский + цифры
PPSTRUCTURE_DEVICE = "cpu"  # Устройство: "cpu", "gpu:0", "gpu:1", и т.д.

# ---------- Производительность ----------
PPSTRUCTURE_ENABLE_MKLDNN = True  # MKL-DNN ускорение для CPU (Intel оптимизация)
PPSTRUCTURE_CPU_THREADS = 8  # Количество потоков CPU для инференса
PPSTRUCTURE_ENABLE_HPI = False  # High-Performance Inference (требует установки плагина)
PPSTRUCTURE_PRECISION = "fp32"  # Точность вычислений: "fp32", "fp16", "int8"

# ---------- Layout Detection ----------
PPSTRUCTURE_LAYOUT_MODEL_NAME = "RT-DETR-H_layout_3cls"  # Модель для обнаружения макета
                                                         # RT-DETR-H - высокая точность
                                                         # 3cls: table, image, stamp
PPSTRUCTURE_LAYOUT_THRESHOLD = 0.5  # Порог уверенности для layout detection (0.0-1.0)
PPSTRUCTURE_LAYOUT_NMS = True  # Non-Maximum Suppression для удаления дублирующих boxes

# ---------- Text Detection (обнаружение текста) ----------
PPSTRUCTURE_TEXT_DET_MODEL_NAME = "PP-OCRv5_server_det"  # Модель детекции текста
                                                         # server - высокая точность
                                                         # mobile - быстрее для CPU
PPSTRUCTURE_TEXT_DET_THRESH = 0.2  # Порог для пикселей текста в probability map
PPSTRUCTURE_TEXT_DET_BOX_THRESH = 0.2  # Порог для текстовых областей (bounding boxes)
PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO = 1.0  # Коэффициент расширения bounding boxes
                                         # 1.0-1.5: для таблиц (больше разделения слов)
                                         # 1.5-2.0: для обычного текста (меньше разделения)
                                         # Влияет на распознавание ПРОБЕЛОВ между словами!

# ---------- Text Recognition (распознавание текста) ----------
PPSTRUCTURE_TEXT_REC_MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"  # Модель распознавания текста
                                                               # eslav - East Slavic (русский + английский)
                                                               # КРИТИЧНО для правильных пробелов!
PPSTRUCTURE_TEXT_REC_BATCH_SIZE = 1  # Размер батча для распознавания (1 для CPU)
PPSTRUCTURE_TEXT_REC_SCORE_THRESH = 0.4  # Минимальный порог уверенности для результатов
                                         # 0.0 - без фильтрации
                                         # 0.5-0.7 - фильтр ненадежных результатов

# ---------- Table Recognition ----------
PPSTRUCTURE_USE_TABLE_RECOGNITION = True  # Включить модуль распознавания таблиц

# ==================== ИНИЦИАЛИЗАЦИЯ ДИРЕКТОРИЙ ====================
for d in (INBOX_DIR, PARSING_DIR, OUT_DIR, OUT_DIR_LEARNING_DATA, MODEL_CACHE_DIR):
    d.mkdir(exist_ok=True, parents=True)