import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from PIL import Image

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"  # пример пути к LLaMA 2 7B chat в Hugging Face
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model = None
tokenizer = None

def load_model():
    global model, tokenizer
    if model is None or tokenizer is None:
        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
        model = LlamaForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            device_map="auto" if DEVICE == "cuda" else None
        )
        model.to(DEVICE)

def tokens_count_llama(text):
    global tokenizer
    if tokenizer is None:
        load_model()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)

def ask_llama2(question: str, tables_text: str, header: str, max_new_tokens: int = 2048) -> str:
    """
    Обработка запроса с использованием LLaMA 2.
    Поддержка картинок здесь отсутствует, т.к. LLaMA 2 — текстовая модель.
    При необходимости изображения нужно обрабатывать отдельно и вставлять в подсказку текстом.
    """

    load_model()

    # Формируем контекст с инструкцией
    prompt = f"""
Ты — ассистент по обработке табличных данных.
Формат выходных данных: **только таблица в Markdown**, без пояснений и заголовков.
Правила:
- Используй ровно эти столбцы: {header}.
- Каждое входное значение помещается только в один столбец.
- Если подходит под 'Количество', 'Наименование' или 'Единица измерения' - ставь туда.
- Возьми входные табличные данные и распредели их содержимое строго в эти столбцы.
- Всё остальное пиши в 'Техническое задание'. Если элементов несколько — объединяй через символ '_'.
- Если данных нет для столбца — оставляй пустую ячейку/
- Не добавляй новых строк или столбцов

Входные данные (разделитель столбцов = `;`):
{tables_text}

Выведи результат в виде Markdown-таблицы без заголовка.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    # Декодируем ответ, убирая входной запрос
    generated_text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    return generated_text.strip()