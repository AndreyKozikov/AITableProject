from paddleocr import PPStructureV3
from pathlib import Path
import pandas as pd
from src.mapper.model_answer import ask_qwen
from src.utils.config import INBOX_DIR, PARSING_DIR, OUT_DIR, MODEL_DIR

def load_tables_from_csv(file):
    return pd.read_csv(file)


def mapper(extended=False):
    if extended:
        pass
    else:
        file_path = MODEL_DIR / "simplified.csv"
        header = ", ".join(pd.read_csv(file_path).columns)
        files = PARSING_DIR.glob("*.csv")
        question = (
            f"Преобразовать её в таблицу с заданными столбцами.")
        for file in files:
            df = load_tables_from_csv(file)
            for i in range(0, len(df), 4):
                batch = df.iloc[i:i + 4]
                tables_text = batch.to_csv(index=False, header=False)
                result = ask_qwen(image_path=None, question=question, tables_text=tables_text, header=header)
                print(result)



if __name__ == "__main__":
    print(mapper(extended=False))
