from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ROOT = Path(__file__).resolve().parents[1]
INBOX_DIR = ROOT.parent / "inbox"
PARSING_DIR = ROOT.parent / "parsing_files"
OUT_DIR = ROOT.parent / "out"
MODEL_DIR = ROOT.parent / "model_table"
HEADER_ANCHORS = [
    ("poz", ["поз", "позиция", "артикул"]),  # Поз
    ("name", ["наименование", "товар", "товары"]),  # Наименование
    ("pic", ["рис"]),  # Рис
    ("qty", ["кол", "кол-во", "количество"]),  # Кол-во
    ("price", ["цена"]),  # Цена (подойдёт и "Цена за шт.")
    ("curr", ["валюта"]),  # Валюта
    ("disc", ["скидка"]),  # Скидка
    ("other", ["ндс", "сумма", "ставка"])
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

for d in (INBOX_DIR, PARSING_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)

