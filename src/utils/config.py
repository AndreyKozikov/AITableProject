from pathlib import Path

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

for d in (INBOX_DIR, PARSING_DIR, OUT_DIR):
    d.mkdir(exist_ok=True)