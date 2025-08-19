import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import pandas as pd

# параметры модели
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
MAX_CTX = 4096   # максимально поддерживаемый контекст

# инициализация
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to("cpu")

def estimate_tokens(image_path: str, csv_text: str, question: str = "Анализируй таблицу") -> dict:
    """
    Возвращает число токенов для картинки+текста и запас для CSV
    """

    # формируем сообщение
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": Image.open(image_path)},
            {"type": "text", "text": csv_text + "\n\n" + question},
        ]
    }]

    # превращаем в токены
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = processor.image_processor(images=Image.open(image_path), return_tensors="pt")
    inputs = processor(text=[text], images=[image_inputs["pixel_values"][0]], padding=True, return_tensors="pt")

    total_tokens = inputs.input_ids.shape[1]
    free_tokens = MAX_CTX - total_tokens
    return {
        "total_tokens": int(total_tokens),
        "free_tokens": int(free_tokens),
        "fits": free_tokens > 0
    }


def max_rows_for_csv(image_path: str, df: pd.DataFrame, question: str = "Анализируй таблицу", avg_tokens_per_row: int = 25) -> int:
    """
    Прикидывает сколько строк из DataFrame можно безопасно добавить в CSV,
    учитывая размер картинки.
    """
    # проверим нулевой текст
    dummy = estimate_tokens(image_path, "", question)
    reserve = dummy["free_tokens"]

    max_rows = reserve // avg_tokens_per_row
    return max_rows


# === пример использования ===
if __name__ == "__main__":
    img = r"D:\Andrew\GeekBrains\Python\AITableProject\src\data\scan_1.jpg"
    df = pd.DataFrame({
        "Обозначение": ["А1", "А2", "А3"],
        "Наименование": ["Болт", "Гайка", "Шайба"],
        "Ед. изм.": ["шт.", "шт.", "шт."],
        "Кол-во": [10, 20, 30]
    })

    rows = max_rows_for_csv(img, df)
    print(f"Можно безопасно добавить ~{rows} строк CSV для этой картинки")
