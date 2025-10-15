"""LLaMA2 Language Model Integration.

Интеграция модели LLaMA2.

This module provides integration with LLaMA2 language model for text processing
and table recognition tasks.
"""

from typing import Optional

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

MODEL_ID = "meta-llama/Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Глобальные переменные для модели
model = None
tokenizer = None


def load_model() -> None:
    """Загружает модель LLaMA2 и токенизатор.
    
    Инициализирует глобальные переменные модели и токенизатора, если они еще не
    были загружены. Использует подходящее устройство (CUDA/CPU) и тип данных.
    
    Raises:
        Exception: Если загрузка модели не удалась.
    """
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            logger.info(f"Загрузка модели LLaMA2: {MODEL_ID}")
            logger.info(f"Используемое устройство: {DEVICE}, тип данных: {TORCH_DTYPE}")
            
            tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
            model = LlamaForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if DEVICE == "cuda" else None
            )
            model.to(DEVICE)
            
            logger.info("Модель LLaMA2 и токенизатор загружены успешно")
            
    except Exception as e:
        logger.error(f"Не удалось загрузить модель LLaMA2: {e}")
        raise


def tokens_count_llama(text: str) -> int:
    """Подсчитывает токены в тексте используя токенизатор LLaMA2.
    
    Args:
        text: Текст для подсчета токенов.
        
    Returns:
        Количество токенов в тексте.
        
    Raises:
        Exception: Если загрузка токенизатора не удалась.
    """
    global tokenizer
    
    try:
        if tokenizer is None:
            load_model()
        
        logger.debug(f"Подсчет токенов для текста с {len(text)} символами")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        logger.debug(f"Количество токенов: {token_count}")
        return token_count
    except Exception as e:
        logger.error(f"Ошибка при подсчете токенов: {e}")
        raise


def ask_llama2(question: str, 
               tables_text: str, 
               header: str, 
               max_new_tokens: int = 2048) -> str:
    """Запрашивает модель LLaMA2 для обработки таблиц.
    
    Обрабатывает данные таблиц используя LLaMA2. Примечание: LLaMA2 - это только текстовая модель,
    поэтому изображения нужно обрабатывать отдельно и вставлять как текст.
    
    Args:
        question: Вопрос или инструкция для модели.
        tables_text: Данные таблицы в текстовом формате.
        header: Заголовки столбцов для таблицы.
        max_new_tokens: Максимальное количество токенов для генерации.
        
    Returns:
        Сгенерированный ответ от модели.
        
    Raises:
        Exception: Если инференс модели не удался.
    """
    logger.info(f"Начало инференса LLaMA2")
    logger.info(f"Длина вопроса: {len(question)}, "
                f"Длина текста таблиц: {len(tables_text)}, "
                f"Заголовок: {header}")
    
    try:
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

Входные данные:
{tables_text}

Вопрос: {question}

Ответ (только таблица в Markdown):
"""
        
        logger.debug(f"Полная длина промпта: {len(prompt)} символов")
        logger.debug(f"Превью промпта: {prompt[:300]}...")
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = inputs.to(DEVICE)
        
        logger.info(f"Запуск инференса модели с max_new_tokens: {max_new_tokens}")
        logger.debug(f"Количество входных токенов: {inputs.input_ids.shape[1]}")
        
        # Генерация ответа
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Декодирование ответа
        generated_text = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):], 
            skip_special_tokens=True
        )
        
        logger.info(f"Инференс модели завершен. Сгенерировано {len(generated_text)} символов")
        logger.debug(f"Превью сгенерированного текста: {generated_text[:200]}...")
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Ошибка в инференсе LLaMA2: {e}")
        raise