import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
#MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
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


def ask_qwen2(image_path: str = None, prompt: str = None, max_new_tokens: int = 32768, ) -> str:
    load_model()

    if image_path is not None:
        img = Image.open(image_path)

        messages = [
            {"role": "system", "content": "Ты — распознаватель таблиц. Отвечай только списком списков, без пояснений."},
            {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ],
        }]
    else:
        messages = [
            {"role": "system",
             "content": "Ты — распознаватель таблиц. Отвечай только списком списков (Python-массив), без текста и пояснений."},
            {"role": "user", "content": [{"type": "text", "text": prompt}]}]

    chat_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    # EOS токен
    eos_token_id = processor.tokenizer.eos_token_id

    inputs = processor(
        text=[chat_text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(**inputs,
                                    max_new_tokens=max_new_tokens,
                                    do_sample=False,  # отключаем случайность
                                    eos_token_id=eos_token_id,  # 👈 останавливаемся строго
                                    )

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]

    raw_answer = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # --- Фильтрация: оставляем только массив ---
    start = raw_answer.find("[[")
    end = raw_answer.rfind("]]")
    if start != -1 and end != -1:
        answer = raw_answer[start:end + 2]  # включаем закрывающие ]]
    else:
        answer = raw_answer  # fallback

    return answer
