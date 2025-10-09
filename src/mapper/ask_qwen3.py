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
    """Load Qwen3 model and tokenizer.
    
    Initializes the global model and tokenizer variables if they haven't been
    loaded yet. Uses appropriate device (CUDA/CPU) and data type.
    
    Raises:
        Exception: If model loading fails.
    """
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            logger.info(f"Loading Qwen3 model: {MODEL_ID}")
            logger.info(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            logger.info("Qwen3 model and tokenizer loaded successfully")
            
    except Exception as e:
        logger.error(f"Failed to load Qwen3 model: {e}")
        raise



def ask_qwen3(prompt: Optional[str] = None, max_new_tokens: int = 2048) -> str:
    """Query Qwen3 model with text prompt.
    
    Args:
        prompt: Text prompt for the model.
        max_new_tokens: Maximum number of tokens to generate.
        
    Returns:
        Generated response from the model.
        
    Raises:
        Exception: If model inference fails.
    """
    if not prompt:
        logger.warning("Empty prompt provided to Qwen3")
        return ""
    
    logger.info(f"Starting Qwen3 inference with prompt length: {len(prompt)}")
    logger.debug(f"Prompt preview: {prompt[:200]}...")
    
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
        
        logger.info("Preparing model input...")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # Switches between thinking and non-thinking modes
        )
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        eos_token_id = tokenizer.eos_token_id
        
        logger.info(f"Running model inference with max_new_tokens: {max_new_tokens}")
        logger.debug(f"Input token count: {model_inputs.input_ids.shape[1]}")
        
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
        
        logger.info(f"Model inference completed. Generated {len(output_text)} characters")
        logger.debug(f"Generated text preview: {output_text[:200]}...")
        
        return output_text
        
    except Exception as e:
        logger.error(f"Error in Qwen3 inference: {e}")
        raise