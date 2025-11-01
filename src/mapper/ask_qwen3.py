"""Qwen3 Language Model Integration.

Интеграция модели Qwen3.

This module provides integration with Qwen3 language model for text processing
and table recognition tasks.
"""

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

MODEL_ID = "Qwen/Qwen3-1.7B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Глобальные переменные для модели
model = None
tokenizer = None


def load_model() -> None:
    """Загружает модель Qwen3 и токенизатор.
    
    Инициализирует глобальные переменные модели и токенизатора, если они еще не
    были загружены. Использует подходящее устройство (CUDA/CPU) и тип данных.
    
    Raises:
        Exception: Если загрузка модели не удалась.
    """
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            logger.info(f"Загрузка модели Qwen3: {MODEL_ID}")
            logger.info(f"Используемое устройство: {DEVICE}, тип данных: {TORCH_DTYPE}")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            logger.info("Модель Qwen3 и токенизатор загружены успешно")
            
    except Exception as e:
        logger.error(f"Не удалось загрузить модель Qwen3: {e}")
        raise



def ask_qwen3(prompt: Optional[str] = None, max_new_tokens: int = 2048) -> str:
    """Запрашивает модель Qwen3 с текстовым промптом.
    
    Args:
        prompt: Текстовый промпт для модели.
        max_new_tokens: Максимальное количество токенов для генерации.
        
    Returns:
        Сгенерированный ответ от модели.
        
    Raises:
        Exception: Если инференс модели не удался.
    """
    if not prompt:
        logger.warning("Пустой промпт предоставлен в Qwen3")
        return ""
    
    logger.info(f"Начало инференса Qwen3. Тип входа: single. Длина: {len(prompt)}")
    logger.info("ВХОДНЫЕ ДАННЫЕ ДЛЯ МОДЕЛИ (single):")
    logger.info(f"{prompt}")
    
    try:
        load_model()
        
        messages = [
            {
                "role": "system",
                "content": "Ты — распознаватель таблиц. Отвечай только списком списков (Python-массив), без текста и пояснений. Распределяй данные только в заданные столбцы, новых не добавляй."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        logger.info("Количество сообщений, передаваемых в шаблон: %d", len(messages))
        logger.info("Длина user-сообщения: %d", len(prompt))
        
        logger.info("Подготовка входных данных модели...")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        eos_token_id = tokenizer.eos_token_id
        
        logger.info(f"Запуск инференса модели с max_new_tokens: {max_new_tokens}")
        logger.debug(f"Количество входных токенов: {model_inputs.input_ids.shape[1]}")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # отключаем случайность
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.9
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        logger.info(f"Инференс модели завершен. Сгенерировано {len(output_text)} символов")
        logger.debug(f"Превью сгенерированного текста: {output_text[:200]}...")
        
        return output_text
        
    except Exception as e:
        logger.error(f"Ошибка в инференсе Qwen3: {e}")
        raise