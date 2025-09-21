import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

model = None
tokenizer = None


def load_model():
    global model, tokenizer
    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=TORCH_DTYPE,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )


def tokens_count3(text):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)


def ask_qwen3(prompt: str = None, max_new_tokens: int = 2048) -> str:
    load_model()
    messages = [{"role": "system",
             "content": "Ты — распознаватель таблиц. Отвечай только списком списков (Python-массив), без текста и пояснений. Распределяй данные только в заданные столбцы, новых не добавляй."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    eos_token_id = tokenizer.eos_token_id
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # отключаем случайность
        eos_token_id=eos_token_id,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    return answer
