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
    """Load LLaMA2 model and tokenizer.
    
    Initializes the global model and tokenizer variables if they haven't been
    loaded yet. Uses appropriate device (CUDA/CPU) and data type.
    
    Raises:
        Exception: If model loading fails.
    """
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            logger.info(f"Loading LLaMA2 model: {MODEL_ID}")
            logger.info(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
            
            tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)
            model = LlamaForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if DEVICE == "cuda" else None
            )
            model.to(DEVICE)
            
            logger.info("LLaMA2 model and tokenizer loaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to load LLaMA2 model: {e}")
        raise


def tokens_count_llama(text: str) -> int:
    """Count tokens in text using LLaMA2 tokenizer.
    
    Args:
        text: Text to count tokens for.
        
    Returns:
        Number of tokens in the text.
        
    Raises:
        Exception: If tokenizer loading fails.
    """
    global tokenizer
    
    try:
        if tokenizer is None:
            load_model()
        
        logger.debug(f"Counting tokens for text with {len(text)} characters")
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        logger.debug(f"Token count: {token_count}")
        return token_count
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise


def ask_llama2(question: str, 
               tables_text: str, 
               header: str, 
               max_new_tokens: int = 2048) -> str:
    """Query LLaMA2 model for table processing.
    
    Process table data using LLaMA2. Note: LLaMA2 is a text-only model,
    so images need to be processed separately and inserted as text.
    
    Args:
        question: Question or instruction for the model.
        tables_text: Table data in text format.
        header: Column headers for the table.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        Generated response from the model.
        
    Raises:
        Exception: If model inference fails.
    """
    logger.info(f"Starting LLaMA2 inference")
    logger.info(f"Question length: {len(question)}, "
                f"Tables text length: {len(tables_text)}, "
                f"Header: {header}")
    
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
        
        logger.debug(f"Full prompt length: {len(prompt)} characters")
        logger.debug(f"Prompt preview: {prompt[:300]}...")
        
        # Токенизация
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = inputs.to(DEVICE)
        
        logger.info(f"Running model inference with max_new_tokens: {max_new_tokens}")
        logger.debug(f"Input token count: {inputs.input_ids.shape[1]}")
        
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
        
        logger.info(f"Model inference completed. Generated {len(generated_text)} characters")
        logger.debug(f"Generated text preview: {generated_text[:200]}...")
        
        return generated_text.strip()
        
    except Exception as e:
        logger.error(f"Error in LLaMA2 inference: {e}")
        raise