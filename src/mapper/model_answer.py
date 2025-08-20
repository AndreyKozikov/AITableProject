import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model = None
processor = None


def load_model():
    global model, processor
    if model is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=TORCH_DTYPE).to("cpu")
        model = model.to(DEVICE)

    if processor is None:
        processor = AutoProcessor.from_pretrained(MODEL_ID)


def tokens_count(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def ask_qwen(image_path: str, question: str, tables_text: str, header: str, max_new_tokens: int = 32768, ) -> str:
    load_model()

    if image_path is not None:
        img = Image.open(image_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text":
                    "Вот таблица, которую необходимо преобразовать:\n"
                    f"{tables_text}\n\n"
                    "Задача: \n"
                    f"{question}\n\n"
                    "Важно:\n"
                    "1. Используй только формат CSV с разделителем ','.\n"
                    "2. Первая строка — это заголовки (столбцы).\n"
                    "3. Если данных нет, ставь '-'.\n"
                    "4. Не добавляй лишний текст или комментарии, возвращай только CSV.\n"
                 }
            ],
        }]
    else:
        messages = [
            {
                "role": "system",
                "content":
                    [{
                        "type": "text", "text":
                            f"Ты — ассистент по обработке табличных данных. У тебя есть шаблон таблицы с фиксированными столбцами: '{header}'."
                            "Задача: "
                            "- Возьми входные табличные данные и распредели их содержимое строго в эти столбцы."
                            "- Данные из одной входной ячейки должны помещаться только в одну подходящую по смыслу ячейку выходной таблицы. Не разбивай данные из одной ячейки на несколько. "
                            "- Если данные подходят по смыслу к столбцам 'Количество', 'Наименование' или 'Единица измерения', размести их там."
                            "- Все остальные данные (например, описания, характеристики, требования) записывай в столбец 'Техническое задание'. "
                            "- Если данных для 'Технического задания' несколько, объедини их в одну ячейку с помощью точек или запятых. "
                            "- Если в входных данных несколько строк, обработай каждую строку отдельно и выведи таблицу с несколькими строками. "
                            "- Не добавляй новые столбцы или строки. Не меняй порядок столбцов. Если данных для какого-то столбца нет, оставь ячейку пустой. "
                            "- Выводи результат только в формате таблицы Markdown. Ничего больше не пиши."
                    }],
            },
            {
                "role": "user",
                "content":
                    [{
                        "type": "text", "text":
                            "Вот таблица, которую необходимо преобразовать:\n"
                            f"{tables_text}\n\n"
                            "Задача: \n"
                            f"{question}\n\n"
                    }]
            }
        ]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return answer
