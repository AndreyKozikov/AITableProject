import pdfplumber
import re
import pandas as pd
from src.utils.config import INBOX_DIR

# ---------- 1. Чтение PDF и извлечение текста ----------
pdf_path = INBOX_DIR / "Каталог D'Andrea оснастка.pdf"

all_text = []
with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages, start=1):
        text = page.extract_text()
        if text:  # если на странице есть текст
            all_text.append({"page": i, "text": text})

# превратим в DataFrame для удобства
df_text = pd.DataFrame(all_text)
print(df_text.head())

# ---------- 2. Очистка и выделение ключевых данных ----------
# Пример: ищем строки с REF. CODE и диапазонами Ø
pattern_ref = re.compile(r"(REF\.?\s*CODE.*)", re.IGNORECASE)
pattern_diam = re.compile(r"Ø\s*\d+(\.\d+)?\s*~\s*\d+(\.\d+)?")

extracted = []

for row in df_text.itertuples():
    lines = row.text.split("\n")
    for line in lines:
        # проверка на REF. CODE
        if pattern_ref.search(line) or pattern_diam.search(line):
            extracted.append({
                "page": row.page,
                "line": line.strip()
            })

df_extracted = pd.DataFrame(extracted)
print(df_extracted.head(20))