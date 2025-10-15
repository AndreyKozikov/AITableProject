"""Qwen2 Vision-Language Model Integration.

Интеграция модели Qwen2 Vision-Language.

This module provides integration with Qwen2-VL model for vision-language tasks,
specifically for table recognition and extraction from images.
"""

from typing import Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Глобальные переменные для модели
model = None
processor = None


def load_model() -> None:
    """Загружает модель Qwen2-VL и процессор.
    
    Инициализирует глобальные переменные модели и процессора, если они еще не
    были загружены. Использует подходящее устройство (CUDA/CPU) и тип данных.
    
    Raises:
        Exception: Если загрузка модели не удалась.
    """
    global model, processor
    
    try:
        if model is None:
            logger.info(f"Загрузка модели Qwen2-VL: {MODEL_ID}")
            logger.info(f"Используемое устройство: {DEVICE}, тип данных: {TORCH_DTYPE}")
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=TORCH_DTYPE
            ).to("cpu")
            model = model.to(DEVICE)
            logger.info("Модель загружена успешно")

        if processor is None:
            logger.info("Загрузка процессора...")
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            logger.info("Процессор загружен успешно")
            
    except Exception as e:
        logger.error(f"Не удалось загрузить модель Qwen2-VL: {e}")
        raise


def tokens_count(text: str) -> int:
    """Подсчитывает токены в тексте используя токенизатор Qwen2.
    
    Args:
        text: Текст для подсчета токенов.
        
    Returns:
        Количество токенов в тексте.
        
    Raises:
        Exception: Если загрузка токенизатора не удалась.
    """
    try:
        logger.debug(f"Подсчет токенов для текста с {len(text)} символами")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        logger.debug(f"Количество токенов: {token_count}")
        return token_count
    except Exception as e:
        logger.error(f"Ошибка при подсчете токенов: {e}")
        raise


def ask_qwen2(image_path: Optional[str] = None, 
              prompt: Optional[str] = None, 
              max_new_tokens: int = 32768) -> str:
    """Запрашивает модель Qwen2-VL с изображением и/или текстовым промптом.
    
    Args:
        image_path: Путь к файлу изображения (опционально).
        prompt: Текстовый промпт для модели.
        max_new_tokens: Максимальное количество токенов для генерации.
        
    Returns:
        Сгенерированный ответ от модели.
        
    Raises:
        Exception: Если инференс модели не удался.
    """
    logger.info(f"Начало инференса Qwen2 с изображением: {image_path is not None}, "
                f"длина промпта: {len(prompt) if prompt else 0}")
    
    try:
        load_model()
        
        if image_path is not None:
            logger.info(f"Загрузка изображения: {image_path}")
            img = Image.open(image_path)
            logger.debug(f"Изображение загружено, размер: {img.size}, режим: {img.mode}")

            messages = [
                {
                    "role": "system", 
                    "content": "Ты — распознаватель таблиц. Отвечай только списком списков, без пояснений."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": prompt}
                    ],
                }
            ]
        else:
            logger.info("Обработка запроса только с текстом")
            messages = [
                {
                    "role": "system",
                    "content": "Ты — распознаватель таблиц. Отвечай только списком списков (Python-массив), без текста и пояснений."
                },
                {
                    "role": "user", 
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

        logger.info("Подготовка входных данных модели...")
        text = processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(DEVICE)
        
        logger.info(f"Запуск инференса модели с max_new_tokens: {max_new_tokens}")
        
        # Генерируем вывод
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Отключаем случайность для воспроизводимости
            temperature=0.1,
            top_p=0.9
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"Инференс модели завершен. Сгенерировано {len(output_text)} символов")
        logger.debug(f"Превью сгенерированного текста: {output_text[:200]}...")
        
        return output_text
        
    except Exception as e:
        logger.error(f"Ошибка в инференсе Qwen2: {e}")
        raise