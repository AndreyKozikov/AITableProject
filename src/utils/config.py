"""Configuration Module.

Модуль конфигурации проекта.

Contains all configuration settings, directory paths, and constants
for the AITableProject application.
"""

import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROOT = Path(__file__).resolve().parents[1]
INBOX_DIR = ROOT.parent / "inbox"
PARSING_DIR = ROOT.parent / "parsing_files"
OUT_DIR = ROOT.parent / "out"
MODEL_DIR = ROOT.parent / "model_table"

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

for d in (INBOX_DIR, PARSING_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)