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
    """Load Qwen2-VL model and processor.
    
    Initializes the global model and processor variables if they haven't been
    loaded yet. Uses appropriate device (CUDA/CPU) and data type.
    
    Raises:
        Exception: If model loading fails.
    """
    global model, processor
    
    try:
        if model is None:
            logger.info(f"Loading Qwen2-VL model: {MODEL_ID}")
            logger.info(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
            
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                MODEL_ID, 
                torch_dtype=TORCH_DTYPE
            ).to("cpu")
            model = model.to(DEVICE)
            logger.info("Model loaded successfully")

        if processor is None:
            logger.info("Loading processor...")
            processor = AutoProcessor.from_pretrained(MODEL_ID)
            logger.info("Processor loaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to load Qwen2-VL model: {e}")
        raise


def tokens_count(text: str) -> int:
    """Count tokens in text using Qwen2 tokenizer.
    
    Args:
        text: Text to count tokens for.
        
    Returns:
        Number of tokens in the text.
        
    Raises:
        Exception: If tokenizer loading fails.
    """
    try:
        logger.debug(f"Counting tokens for text with {len(text)} characters")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_count = len(tokens)
        logger.debug(f"Token count: {token_count}")
        return token_count
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise


def ask_qwen2(image_path: Optional[str] = None, 
              prompt: Optional[str] = None, 
              max_new_tokens: int = 32768) -> str:
    """Query Qwen2-VL model with image and/or text prompt.
    
    Args:
        image_path: Path to image file (optional).
        prompt: Text prompt for the model.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        Generated response from the model.
        
    Raises:
        Exception: If model inference fails.
    """
    logger.info(f"Starting Qwen2 inference with image: {image_path is not None}, "
                f"prompt length: {len(prompt) if prompt else 0}")
    
    try:
        load_model()
        
        if image_path is not None:
            logger.info(f"Loading image: {image_path}")
            img = Image.open(image_path)
            logger.debug(f"Image loaded, size: {img.size}, mode: {img.mode}")

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
            logger.info("Processing text-only request")
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

        logger.info("Preparing model input...")
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
        
        logger.info(f"Running model inference with max_new_tokens: {max_new_tokens}")
        
        # Generate output
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
        
        logger.info(f"Model inference completed. Generated {len(output_text)} characters")
        logger.debug(f"Generated text preview: {output_text[:200]}...")
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error in Qwen2 inference: {e}")
        raise