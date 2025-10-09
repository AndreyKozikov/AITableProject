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
"""

from pathlib import Path
from typing import Optional, List, Type
import re

import pandas as pd
import torch
from pydantic import BaseModel, Field, create_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils.config import MODEL_DIR, PROMPT_TEMPLATE_SO
from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)

MODEL_ID = "Qwen/Qwen3-1.7B"
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
        logger.debug(f"Loaded {len(columns)} columns from {csv_path}: {columns}")
        return columns
    except Exception as e:
        logger.error(f"Failed to load CSV schema from {csv_path}: {e}")
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
    
    logger.debug(f"Created dynamic model '{model_name}' with fields: {list(fields.keys())}")
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
    
    logger.debug(f"Created container model '{model_name}'")
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
        logger.debug(f"Using cached models for {cache_key} mode")
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
    
    logger.info(f"Loading schema from {csv_path}")
    
    # Загружаем колонки из CSV
    columns = _load_csv_schema(csv_path)
    
    # Создаем модели динамически
    row_model = _create_table_row_model(columns, row_model_name)
    container_model = _create_structured_output_model(row_model, container_model_name)
    
    # Кэшируем
    _model_cache[cache_key] = (row_model, container_model)
    
    logger.info(f"Created and cached models for {cache_key} mode with {len(columns)} columns")
    return row_model, container_model


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
            logger.info(f"Loading Qwen3 model for structured output: {MODEL_ID}")
            logger.info(f"Using device: {DEVICE}, dtype: {TORCH_DTYPE}")
            
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=TORCH_DTYPE,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            logger.info("Qwen3 model and tokenizer loaded successfully for structured output")
            
    except Exception as e:
        logger.error(f"Failed to load Qwen3 model: {e}")
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
        logger.warning("Empty prompt provided to Qwen3 structured output")
        # Возвращаем пустую структуру
        _, container_model = get_table_models(extended=extended)
        return container_model(rows=[])
    
    logger.info(f"Starting Qwen3 structured inference with prompt length: {len(prompt)}")
    logger.info(f"Mode: {'extended' if extended else 'simplified'}, Thinking: {enable_thinking}")
    logger.debug(f"Prompt preview: {prompt[:200]}...")
    
    try:
        load_model()
        
        # Получаем динамически созданные модели из CSV схемы
        row_model, container_model = get_table_models(extended=extended)
        
        # Получаем JSON схему для промпта
        json_schema = container_model.model_json_schema()
        
        # Логируем JSON схему
        logger.info(f"\n{'='*60}")
        logger.info("JSON Schema for structured output:")
        logger.info(f"{'='*60}")
        logger.info(f"{json_schema}")
        logger.info(f"{'='*60}\n")
        
        # Получаем список колонок для header
        if extended:
            csv_path = MODEL_DIR / "extended.csv"
        else:
            csv_path = MODEL_DIR / "simplified.csv"
        
        columns = _load_csv_schema(csv_path)
        header_str = ", ".join(columns)
        
        # Формируем system prompt используя PROMPT_TEMPLATE_SO из конфига
        system_content = PROMPT_TEMPLATE_SO.format(
            header=header_str,
            schema=json_schema,
            tables_text="{tables_text}"  # Placeholder, будет заменен в user message
        )
        
        # User message содержит только данные
        system_message_content = system_content.replace("{tables_text}", "")  # Убираем placeholder из system
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
        
        logger.info("Preparing model input with JSON schema from CSV...")
        logger.info(f"Schema columns: {header_str}")
        logger.info(f"Total messages count: {len(messages)}")
        
        # Логируем полностью все сообщения
        for idx, message in enumerate(messages, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Message {idx}/{len(messages)} - Role: {message['role']}")
            logger.info(f"{'='*60}")
            logger.info(f"Content length: {len(message['content'])} characters")
            logger.info(f"Full content:\n{message['content']}")
            logger.info(f"{'='*60}\n")
        
        # Применяем chat template с параметром enable_thinking (из статьи)
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking  # Switches between thinking and non-thinking modes
        )
        
        # Логируем финальный текст после применения chat template
        logger.info(f"\n{'='*60}")
        logger.info("Final text after applying chat template:")
        logger.info(f"{'='*60}")
        logger.info(f"Text length: {len(text)} characters")
        logger.info(f"Full text:\n{text}")
        logger.info(f"{'='*60}\n")
        
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        eos_token_id = tokenizer.eos_token_id
        
        logger.info(f"Running model inference with max_new_tokens: {max_new_tokens}")
        logger.info(f"Input token count: {model_inputs.input_ids.shape[1]}")
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # отключаем случайность для стабильности (из статьи)
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            temperature=0.1,
            top_p=0.9
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
        
        logger.info(f"Model inference completed. Generated {len(output_text)} characters")
        
        # Логируем полный ответ модели
        logger.info(f"\n{'='*60}")
        logger.info("RAW Model Response:")
        logger.info(f"{'='*60}")
        logger.info(f"Response length: {len(output_text)} characters")
        logger.info(f"Full response:\n{output_text}")
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
        
        # Логируем очищенный JSON перед валидацией
        logger.info(f"\n{'='*60}")
        logger.info("Cleaned JSON for validation:")
        logger.info(f"{'='*60}")
        logger.info(f"Cleaned JSON length: {len(output_text_clean)} characters")
        logger.info(f"Cleaned JSON:\n{output_text_clean}")
        logger.info(f"{'='*60}\n")
        
        # Валидация через model_validate_json (метод из статьи)
        logger.info("Validating output with Pydantic model_validate_json...")
        structured_result = container_model.model_validate_json(output_text_clean)
        
        logger.info(f"Successfully validated structured output with {len(structured_result.rows)} rows")
        
        # Логируем валидированный результат
        logger.info(f"\n{'='*60}")
        logger.info("Validated Pydantic Result:")
        logger.info(f"{'='*60}")
        logger.info(f"Number of rows: {len(structured_result.rows)}")
        logger.info(f"Result object: {structured_result}")
        if structured_result.rows:
            logger.info(f"First row sample: {structured_result.rows[0]}")
        logger.info(f"{'='*60}\n")
        
        return structured_result
        
    except Exception as e:
        logger.error(f"Error in Qwen3 structured inference: {e}")
        logger.error(f"Failed output text: {output_text if 'output_text' in locals() else 'N/A'}")
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
    
    logger.debug(f"Extracted {len(rows_as_dicts)} rows as dictionaries")
    return rows_as_dicts
