"""Qwen3 Structured Output Integration.

Интеграция модели Qwen3 со структурированным выводом.

This module provides integration with Qwen3 language model for structured
data output using JSON schema approach for table recognition tasks.
Based on: https://www.dataleadsfuture.com/build-autogen-agents-with-qwen3-structured-output-thinking-mode/

Ключевые принципы из статьи:
1. JSON schema встраивается в system_prompt для контроля формата вывода
2. Используется Pydantic BaseModel для валидации структурированных данных
3. Параметр enable_thinking управляет режимом Chain of Thought
4. Валидация через model_validate_json() обеспечивает типобезопасность
5. Схема загружается из CSV файлов, обеспечивая гибкость при изменениях

Модель загружается из models/qwen/, LoRA адаптеры применяются автоматически если существуют.
"""

from pathlib import Path
from typing import Optional, List, Type
import re

import pandas as pd
import torch
from pydantic import BaseModel, Field, create_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.utils.config import (
    MODEL_DIR, 
    PROMPT_TEMPLATE_SO, 
    MODEL_CACHE_DIR,
    LORA_ADAPTER_PATH
)
from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Глобальные переменные для модели
model = None
tokenizer = None

# Кэш для динамически созданных моделей
_model_cache = {}


def _normalize_field_name(field_name: str) -> str:
    """Normalize Russian field name to English identifier.
    
    Конвертирует русское название поля в валидный Python идентификатор.
    Используется транслитерация для сохранения читаемости.
    
    Args:
        field_name: Original field name (может быть на русском).
        
    Returns:
        Normalized English field name suitable for Python identifier.
    """
    # Словарь транслитерации для полей
    transliteration = {
        'Обозначение': 'designation',
        'Наименование': 'name',
        'Производитель': 'manufacturer',
        'единица измерения': 'unit',
        'Единица измерения': 'unit',
        'Количество': 'quantity',
        'Техническое задание': 'technical_specification'
    }
    
    # Проверяем прямое соответствие
    if field_name in transliteration:
        return transliteration[field_name]
    
    # Если нет прямого соответствия, делаем базовую нормализацию
    normalized = field_name.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)  # Удаляем спецсимволы
    normalized = re.sub(r'\s+', '_', normalized)  # Пробелы в подчеркивания
    
    return normalized


def _load_csv_schema(csv_path: Path) -> List[str]:
    """Load column names from CSV schema file.
    
    Загружает названия колонок из CSV файла схемы.
    
    Args:
        csv_path: Path to CSV schema file.
        
    Returns:
        List of column names from CSV header.
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist.
    """
    try:
        df = pd.read_csv(csv_path, sep=',', nrows=0)
        columns = list(df.columns)
        return columns
    except Exception as e:
        logger.error(f"Ошибка загрузки CSV схемы из {csv_path}: {e}")
        raise


def _create_table_row_model(
    columns: List[str],
    model_name: str = "DynamicTableRow"
) -> Type[BaseModel]:
    """Create Pydantic model dynamically from column names.
    
    Динамически создает Pydantic модель из списка колонок CSV.
    Реализация на основе статьи - использование схемы напрямую из CSV.
    
    Args:
        columns: List of column names.
        model_name: Name for the created model.
        
    Returns:
        Dynamically created Pydantic BaseModel class.
    """
    # Создаем поля для модели
    fields = {}
    
    for column in columns:
        # Нормализуем имя поля в английский идентификатор
        field_name = _normalize_field_name(column)
        
        # Создаем Field с описанием и дефолтным значением
        field_def = (
            str,
            Field(
                default="",
                description=f"Field for {column}",
                alias=column  # Используем алиас для оригинального имени
            )
        )
        
        fields[field_name] = field_def
    
    # Создаем модель динамически используя create_model
    dynamic_model = create_model(model_name, **fields)
    
    return dynamic_model


def _create_structured_output_model(
    row_model: Type[BaseModel],
    model_name: str = "TableStructuredOutput"
) -> Type[BaseModel]:
    """Create container model for list of rows.
    
    Создает модель-контейнер для списка строк таблицы.
    
    Args:
        row_model: Pydantic model for single row.
        model_name: Name for container model.
        
    Returns:
        Container model with 'rows' field.
    """
    container_model = create_model(
        model_name,
        rows=(List[row_model], Field(description="List of table rows"))
    )
    
    return container_model


def get_table_models(extended: bool = False) -> tuple[Type[BaseModel], Type[BaseModel]]:
    """Get or create Pydantic models for table schema.
    
    Получает или создает Pydantic модели на основе CSV схемы.
    Использует кэширование для повторных вызовов.
    
    Args:
        extended: Use extended schema if True, simplified if False.
        
    Returns:
        Tuple of (RowModel, ContainerModel).
    """
    cache_key = 'extended' if extended else 'simplified'
    
    # Проверяем кэш
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    # Определяем путь к CSV
    if extended:
        csv_path = MODEL_DIR / "extended.csv"
        row_model_name = "TableRowExtended"
        container_model_name = "TableStructuredOutputExtended"
    else:
        csv_path = MODEL_DIR / "simplified.csv"
        row_model_name = "TableRowSimplified"
        container_model_name = "TableStructuredOutput"
    
    # Загружаем колонки из CSV
    columns = _load_csv_schema(csv_path)
    
    # Создаем модели динамически
    row_model = _create_table_row_model(columns, row_model_name)
    container_model = _create_structured_output_model(row_model, container_model_name)
    
    # Кэшируем
    _model_cache[cache_key] = (row_model, container_model)
    
    return row_model, container_model


def load_model() -> None:
    """Load Qwen model and tokenizer from local models directory.
    
    Loads model from MODEL_CACHE_DIR and automatically applies LoRA adapters
    if they exist in LORA_ADAPTER_PATH.
    
    Raises:
        Exception: If model loading fails.
    """
    global model, tokenizer
    
    try:
        if model is None or tokenizer is None:
            # Загружаем токенизатор
            tokenizer = AutoTokenizer.from_pretrained(
                str(MODEL_CACHE_DIR),
                trust_remote_code=True
            )
            
            # Загружаем базовую модель
            base_model = AutoModelForCausalLM.from_pretrained(
                str(MODEL_CACHE_DIR),
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            # Проверяем наличие LoRA адаптеров и загружаем их автоматически
            if LORA_ADAPTER_PATH.exists():
                model = PeftModel.from_pretrained(
                    base_model,
                    str(LORA_ADAPTER_PATH),
                    is_trainable=False
                )
                logger.info("✓ Модель загружена с LoRA адаптерами")
            else:
                model = base_model
                logger.info("✓ Модель загружена без LoRA адаптеров (базовая модель)")
            
    except Exception as e:
        logger.error(f"Ошибка загрузки модели Qwen: {e}")
        raise


def ask_qwen3_structured(
    prompt: Optional[str] = None,
    extended: bool = False,
    max_new_tokens: int = 2048,
    enable_thinking: bool = False
) -> BaseModel:
    """Query Qwen3 model with structured output.
    
    Реализация на основе статьи:
    1. JSON schema встраивается в system_prompt через PROMPT_TEMPLATE_SO
    2. enable_thinking управляет режимом CoT через apply_chat_template
    3. Валидация через model_validate_json() (Pydantic)
    4. Возврат строго типизированного объекта, никаких сырых выводов
    5. Схема загружается динамически из CSV файлов
    
    Args:
        prompt: Text prompt for the model.
        extended: Use extended table schema (6 columns) if True, simplified (4 columns) if False.
        max_new_tokens: Maximum number of tokens to generate.
        enable_thinking: Enable thinking mode (Chain of Thought reasoning).
        
    Returns:
        Structured output model with validated data.
        
    Raises:
        Exception: If model inference or validation fails.
    """
    if not prompt:
        # Возвращаем пустую структуру
        _, container_model = get_table_models(extended=extended)
        return container_model(rows=[])
    
    try:
        load_model()
        
        # Получаем динамически созданные модели из CSV схемы
        row_model, container_model = get_table_models(extended=extended)
        
        # Получаем JSON схему для промпта
        json_schema = container_model.model_json_schema()
        

        # Получаем список колонок для header
        if extended:
            csv_path = MODEL_DIR / "extended.csv"
        else:
            csv_path = MODEL_DIR / "simplified.csv"
        
        columns = _load_csv_schema(csv_path)
        header_str = ", ".join(columns)
        
        # Определяем режим работы
        mode = "extended" if extended else "simplified"
        
        # Формируем system prompt используя PROMPT_TEMPLATE_SO из конфига
        system_content = PROMPT_TEMPLATE_SO.format(
            header=header_str,
            schema=json_schema,
            tables_text="{tables_text}"  # Placeholder, будет заменен в user message
        )
        
        # User message содержит только данные
        system_content_clean = system_content.replace("{tables_text}", "")  # Убираем placeholder из system
        
        # Добавляем режим работы в system prompt (как при обучении)
        system_message_content = f"Режим работы: {mode}. {system_content_clean}"
        
        messages = [
            {
                "role": "system",
                "content": system_message_content
            },
            {
                "role": "user", 
                "content": prompt
            }
        ]
        
        # Применяем chat template с параметром enable_thinking
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        
        # Логируем финальный промпт
        logger.info(f"\n{'='*60}")
        logger.info("ФИНАЛЬНЫЙ ПРОМПТ:")
        logger.info(f"{'='*60}")
        logger.info(f"{text}")
        logger.info(f"{'='*60}\n")
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        eos_token_id = tokenizer.eos_token_id
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.9
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        # Логируем ответ модели
        logger.info(f"\n{'='*60}")
        logger.info("ОТВЕТ МОДЕЛИ:")
        logger.info(f"{'='*60}")
        logger.info(f"{output_text}")
        logger.info(f"{'='*60}\n")
        
        # Очистка вывода от возможного Markdown синтаксиса
        output_text_clean = output_text.strip()
        
        # Если вывод обернут в ```json или ``` блок
        if output_text_clean.startswith("```"):
            lines = output_text_clean.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.startswith("```"):
                    if in_json:
                        break
                    else:
                        in_json = True
                        continue
                if in_json:
                    json_lines.append(line)
            output_text_clean = "\n".join(json_lines)
        
        
        # Валидация через model_validate_json
        structured_result = container_model.model_validate_json(output_text_clean)
        
        # Логируем финальный обработанный результат
        logger.info(f"\n{'='*60}")
        logger.info("ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        logger.info(f"{'='*60}")
        logger.info(f"Количество строк: {len(structured_result.rows)}")
        for i, row in enumerate(structured_result.rows, 1):
            logger.info(f"Строка {i}: {row.model_dump(by_alias=True)}")
        logger.info(f"{'='*60}\n")
        
        return structured_result
        
    except Exception as e:
        logger.error(f"Ошибка обработки: {e}")
        if 'output_text' in locals():
            logger.error(f"Проблемный вывод: {output_text}")
        raise


def extract_rows_as_dicts(
    structured_output: BaseModel,
    use_aliases: bool = True
) -> List[dict]:
    """Extract rows from structured output as list of dictionaries.
    
    Вспомогательная функция для извлечения данных из Pydantic модели
    в формат списка словарей для дальнейшей обработки.
    
    Args:
        structured_output: Validated structured output from ask_qwen3_structured.
        use_aliases: Use field aliases (original Russian names) if True.
        
    Returns:
        List of dictionaries with row data.
    """
    rows_as_dicts = []
    
    for row in structured_output.rows:
        # Используем model_dump() для конвертации Pydantic модели в dict
        # by_alias=True возвращает оригинальные русские названия полей
        row_dict = row.model_dump(by_alias=use_aliases)
        rows_as_dicts.append(row_dict)
    
    return rows_as_dicts
