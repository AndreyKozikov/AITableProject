"""
Модуль для создания обучающего датасета с Chain-of-Thought reasoning.

Расширяет существующий функционал добавлением цепочек рассуждений
и интеграции с графом знаний.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from .knowledge_graph import knowledge_graph
from .cot_reasoning import cot_generator
from .create_training_dataset import (
    _load_csv_schema, _create_pydantic_model, _create_container_model,
    get_json_schema_for_mode
)

# Константа для ограничения количества обучающих данных
MAX_TRAINING_EXAMPLES = 50


def calculate_examples_per_file(excel_files: List[Path], max_examples: int) -> Dict[str, int]:
    """
    Рассчитать количество примеров для каждого файла.
    
    Args:
        excel_files: Список путей к Excel файлам
        max_examples: Максимальное количество примеров
        
    Returns:
        Словарь {имя_файла: количество_примеров}
    """
    examples_per_file = {}
    total_available = 0
    
    # Сначала подсчитываем доступные данные в каждом файле
    file_capacities = {}
    for excel_file in excel_files:
        try:
            input_df = pd.read_excel(excel_file, sheet_name="INPUT")
            capacity = len(input_df)
            file_capacities[excel_file.name] = capacity
            total_available += capacity
        except Exception as e:
            print(f"Ошибка чтения {excel_file.name}: {e}")
            file_capacities[excel_file.name] = 0
    
    print(f"Общее количество доступных записей: {total_available}")
    print(f"Максимальное количество для обучения: {max_examples}")
    
    if total_available <= max_examples:
        # Если данных меньше чем нужно, берем все
        for filename, capacity in file_capacities.items():
            examples_per_file[filename] = capacity
    else:
        # Распределяем пропорционально
        base_per_file = max_examples // len(excel_files)
        remainder = max_examples % len(excel_files)
        
        for i, (filename, capacity) in enumerate(file_capacities.items()):
            # Базовое количество + остаток для первых файлов
            target_examples = base_per_file + (1 if i < remainder else 0)
            
            # Не превышаем доступную емкость файла
            actual_examples = min(target_examples, capacity)
            examples_per_file[filename] = actual_examples
    
    return examples_per_file


def select_random_rows(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    """
    Выбрать случайные строки из DataFrame.
    
    Args:
        df: Исходный DataFrame
        num_rows: Количество строк для выбора
        
    Returns:
        DataFrame с выбранными строками
    """
    if len(df) <= num_rows:
        return df
    
    # Выбираем случайные индексы
    import random
    random_indices = random.sample(range(len(df)), num_rows)
    return df.iloc[random_indices].reset_index(drop=True)


def create_cot_training_examples(
    input_df: pd.DataFrame, 
    simplified_df: pd.DataFrame, 
    extended_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """Создать примеры для обучения с Chain-of-Thought reasoning."""
    
    examples = []
    
    # Получаем схемы для обоих режимов
    simplified_schema, simplified_columns_str = get_json_schema_for_mode("simplified")
    extended_schema, extended_columns_str = get_json_schema_for_mode("extended")
    
    # Получаем списки колонок из CSV файлов
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.utils.config import MODEL_DIR
    
    simplified_csv_path = MODEL_DIR / "simplified.csv"
    extended_csv_path = MODEL_DIR / "extended.csv"
    
    simplified_columns = _load_csv_schema(simplified_csv_path)
    extended_columns = _load_csv_schema(extended_csv_path)
    
    # Создаем модели Pydantic
    simplified_model = _create_pydantic_model(simplified_columns, "TableRowSimplified")
    extended_model = _create_pydantic_model(extended_columns, "TableRowExtended")
    
    simplified_container = _create_container_model(simplified_model)
    extended_container = _create_container_model(extended_model)
    
    # Обрабатываем каждую строку
    # Все DataFrame'ы уже отфильтрованы и имеют одинаковую длину
    min_length = min(len(input_df), len(simplified_df), len(extended_df))
    
    for position in range(min_length):
        row = input_df.iloc[position]
        
        # Формируем входной текст
        input_text = _format_input_text(row)
        
        # Создаем реалистичные рассуждения на основе данных из листов
        simplified_reasoning = _create_realistic_reasoning("simplified", simplified_df.iloc[position])
        extended_reasoning = _create_realistic_reasoning("extended", extended_df.iloc[position])
        
        # Создаем примеры для simplified режима
        simplified_example = _create_cot_example(
            "simplified", 
            input_text, 
            simplified_df.iloc[position], 
            simplified_schema,
            simplified_reasoning
        )
        examples.append(simplified_example)
        
        # Создаем примеры для extended режима
        extended_example = _create_cot_example(
            "extended",
            input_text,
            extended_df.iloc[position],
            extended_schema, 
            extended_reasoning
        )
        examples.append(extended_example)
    
    return examples


def _format_input_text(row: pd.Series) -> str:
    """Форматировать входной текст из строки DataFrame в структурированном виде."""
    parts = []
    
    # Добавляем заголовок записи
    parts.append("Запись 1:")
    
    # Показываем все поля из строки, даже если они пустые
    for column_name, value in row.items():
        if pd.notna(value) and str(value).strip():
            parts.append(f"  {column_name}: {value}")
        else:
            parts.append(f"  {column_name}: ")
    
    return "\n".join(parts)


def _create_realistic_reasoning(mode: str, target_row: pd.Series) -> str:
    """Создать реалистичные рассуждения на основе реальных данных из листов."""
    
    # Извлекаем данные из target_row
    if mode == "extended":
        denotation = str(target_row.get("Обозначение", ""))
        tool_name = str(target_row.get("Наименование", ""))
        manufacturer = str(target_row.get("Производитель", ""))
        technical_task = str(target_row.get("Техническое задание", ""))
        quantity = str(target_row.get("Количество", ""))
        unit = str(target_row.get("Единица измерения", ""))
    else:  # simplified
        tool_name = str(target_row.get("Наименование", ""))
        technical_task = str(target_row.get("Техническое задание", ""))
        quantity = str(target_row.get("Количество", ""))
        unit = str(target_row.get("Единица измерения", ""))
        manufacturer = ""
        if technical_task and "Производитель:" in technical_task:
            # Извлекаем производителя из технического задания
            import re
            mfg_match = re.search(r'Производитель:\s*([^;]+)', technical_task)
            manufacturer = mfg_match.group(1).strip() if mfg_match else ""
    
    # Формируем рассуждения
    reasoning_steps = []
    
    # Шаг 1: Анализ структуры
    reasoning_steps.append("1. Анализирую структуру входных данных. Вижу структурированную запись с несколькими полями, разделенными переносом строки.")
    
    # Шаг 2: Определение наименования
    reasoning_steps.append(f"2. Определяю наименование инструмента: {tool_name}")
    
    # Шаг 3: Извлечение обозначения (только для extended)
    if mode == "extended":
        if denotation and denotation.strip() and denotation != "nan":
            reasoning_steps.append(f"3. Извлекаю обозначение инструмента: {denotation}")
        else:
            reasoning_steps.append("3. Извлекаю обозначение инструмента: Обозначение не найдено")
    
    # Шаг 4: Определение производителя (только для extended)
    if mode == "extended":
        if manufacturer and manufacturer.strip() and manufacturer != "nan":
            reasoning_steps.append(f"4. Определяю производителя: {manufacturer}")
        else:
            reasoning_steps.append("4. Определяю производителя: Производитель не указан явно")
    
    # Шаг 5: Извлечение количества
    if quantity and quantity.strip() and quantity != "nan":
        reasoning_steps.append(f"5. Извлекаю количество: {quantity}")
    else:
        reasoning_steps.append("5. Извлекаю количество: Количество не найдено")
    
    # Шаг 6: Извлечение единицы измерения
    if unit and unit.strip() and unit != "nan":
        reasoning_steps.append(f"6. Извлекаю единицу измерения: {unit}")
    else:
        reasoning_steps.append("6. Извлекаю единицу измерения: Единица измерения не найдена")
    
    # Шаг 7: Извлечение технического задания
    if technical_task and technical_task.strip() and technical_task != "nan":
        reasoning_steps.append(f"7. Извлекаю техническое задание: {technical_task}")
    else:
        reasoning_steps.append("7. Извлекаю техническое задание: Техническое задание не найдено")
    
    # Шаг 8: Валидация результата
    reasoning_steps.append("8. Валидирую результат разбора. Все необходимые поля заполнены корректно.")
    
    return "\n".join(reasoning_steps)


def _create_catalog_realistic_reasoning(mode: str, target_row: pd.Series) -> str:
    """Создать реалистичные рассуждения для каталогов на основе реальных данных."""
    
    # Извлекаем данные из target_row каталога
    if mode == "extended":
        denotation = str(target_row.get("Обозначение", ""))
        tool_name = str(target_row.get("Наименование", ""))
        manufacturer = str(target_row.get("Производитель", ""))
        technical_task = str(target_row.get("Техническое задание", ""))
        quantity = str(target_row.get("Количество", ""))
        unit = str(target_row.get("Единица измерения", ""))
    else:  # simplified
        tool_name = str(target_row.get("Наименование", ""))
        technical_task = str(target_row.get("Техническое задание", ""))
        quantity = str(target_row.get("Количество", ""))
        unit = str(target_row.get("Единица измерения", ""))
        manufacturer = ""
        if technical_task and "Производитель:" in technical_task:
            # Извлекаем производителя из технического задания
            import re
            mfg_match = re.search(r'Производитель:\s*([^;]+)', technical_task)
            manufacturer = mfg_match.group(1).strip() if mfg_match else ""
    
    # Формируем рассуждения
    reasoning_steps = []
    
    # Шаг 1: Анализ структуры
    reasoning_steps.append("1. Анализирую структуру входных данных. Вижу структурированную запись с несколькими полями, разделенными переносом строки.")
    
    # Шаг 2: Определение наименования
    reasoning_steps.append(f"2. Определяю наименование инструмента: {tool_name}")
    
    # Шаг 3: Извлечение обозначения (только для extended)
    if mode == "extended":
        if denotation and denotation.strip() and denotation != "nan":
            reasoning_steps.append(f"3. Извлекаю обозначение инструмента: {denotation}")
        else:
            reasoning_steps.append("3. Извлекаю обозначение инструмента: Обозначение не найдено")
    
    # Шаг 4: Определение производителя (только для extended)
    if mode == "extended":
        if manufacturer and manufacturer.strip() and manufacturer != "nan":
            reasoning_steps.append(f"4. Определяю производителя: {manufacturer}")
        else:
            reasoning_steps.append("4. Определяю производителя: Производитель не указан явно")
    
    # Шаг 5: Извлечение количества
    if quantity and quantity.strip() and quantity != "nan":
        reasoning_steps.append(f"5. Извлекаю количество: {quantity}")
    else:
        reasoning_steps.append("5. Извлекаю количество: Количество не найдено")
    
    # Шаг 6: Извлечение единицы измерения
    if unit and unit.strip() and unit != "nan":
        reasoning_steps.append(f"6. Извлекаю единицу измерения: {unit}")
    else:
        reasoning_steps.append("6. Извлекаю единицу измерения: Единица измерения не найдена")
    
    # Шаг 7: Извлечение технического задания
    if technical_task and technical_task.strip() and technical_task != "nan":
        reasoning_steps.append(f"7. Извлекаю техническое задание: {technical_task}")
    else:
        reasoning_steps.append("7. Извлекаю техническое задание: Техническое задание не найдено")
    
    # Шаг 8: Валидация результата
    reasoning_steps.append("8. Валидирую результат разбора. Все необходимые поля заполнены корректно.")
    
    return "\n".join(reasoning_steps)


def _create_cot_example(
    mode: str, 
    input_text: str, 
    target_row: pd.Series, 
    schema: str,
    reasoning_text: str
) -> Dict[str, Any]:
    """Создать пример с Chain-of-Thought reasoning."""
    
    # Создаем system prompt БЕЗ reasoning (как в инференсе)
    system_prompt = f"""Ты — ассистент по структурированной обработке промышленных инструментов.

Правила анализа:
1. Сначала объясни шаги разбора (chain-of-thought)
2. Используй знания о производителях и типах инструментов
3. Выдели обозначение, наименование, производителя и параметры
4. Представь результат в JSON формате

Правила вывода:
- Используй ровно эти ключи: {', '.join(_get_columns_for_mode(mode))}
- Структура ответа должна полностью соответствовать JSON Schema
- Не добавляй новых ключей или уровней вложенности
- Если данных нет для ключа — вставляй пустую строку
- Не используй Markdown, блоки ```json или пояснительный текст
- Вывод — только корректный JSON, без заголовков и комментариев

<output-format>
{schema}
</output-format>

Входные данные:"""

    # Создаем user prompt с демонстрацией reasoning
    user_prompt = f"""Проанализируй данные:
{input_text}

Сначала объясни шаги разбора (chain-of-thought), затем выведи структурированный ответ в JSON.

Рассуждения:
{reasoning_text}

Структурированный ответ:"""

    # Создаем assistant ответ ТОЛЬКО с JSON (без reasoning)
    assistant_data = _create_assistant_data(mode, target_row)
    
    return {
        "mode": mode,
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_data
    }


def _get_json_schema(mode: str) -> str:
    """Получить JSON Schema для указанного режима."""
    if mode == "simplified":
        return '''{
  "$defs": {
    "TableRowSimplified": {
      "additionalProperties": false,
      "properties": {
        "Наименование": {
          "default": "",
          "description": "Field for Наименование",
          "title": "Наименование",
          "type": "string"
        },
        "Единица измерения": {
          "default": "",
          "description": "Field for Единица измерения",
          "title": "Единица измерения",
          "type": "string"
        },
        "Количество": {
          "default": "",
          "description": "Field for Количество",
          "title": "Количество",
          "type": "string"
        },
        "Техническое задание": {
          "default": "",
          "description": "Field for Техническое задание",
          "title": "Техническое задание",
          "type": "string"
        }
      },
      "title": "TableRowSimplified",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "rows": {
      "description": "List of table rows",
      "items": {
        "$ref": "#/$defs/TableRowSimplified"
      },
      "title": "Rows",
      "type": "array"
    }
  },
  "required": [
    "rows"
  ],
  "title": "TableStructuredOutput",
  "type": "object"
}'''
    else:  # extended
        return '''{
  "$defs": {
    "TableRowExtended": {
      "additionalProperties": false,
      "properties": {
        "Обозначение": {
          "default": "",
          "description": "Field for Обозначение",
          "title": "Обозначение",
          "type": "string"
        },
        "Наименование": {
          "default": "",
          "description": "Field for Наименование",
          "title": "Наименование",
          "type": "string"
        },
        "Производитель": {
          "default": "",
          "description": "Field for Производитель",
          "title": "Производитель",
          "type": "string"
        },
        "Единица измерения": {
          "default": "",
          "description": "Field for Единица измерения",
          "title": "Единица измерения",
          "type": "string"
        },
        "Количество": {
          "default": "",
          "description": "Field for Количество",
          "title": "Количество",
          "type": "string"
        },
        "Техническое задание": {
          "default": "",
          "description": "Field for Техническое задание",
          "title": "Техническое задание",
          "type": "string"
        }
      },
      "title": "TableRowExtended",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "rows": {
      "description": "List of table rows",
      "items": {
        "$ref": "#/$defs/TableRowExtended"
      },
      "title": "Rows",
      "type": "array"
    }
  },
  "required": [
    "rows"
  ],
  "title": "TableStructuredOutput",
  "type": "object"
}'''


def _get_columns_for_mode(mode: str) -> List[str]:
    """Получить колонки для указанного режима."""
    if mode == "simplified":
        return ["Наименование", "Единица измерения", "Количество", "Техническое задание"]
    else:  # extended
        return ["Обозначение", "Наименование", "Производитель", "Единица измерения", "Количество", "Техническое задание"]


def _create_assistant_data(mode: str, target_row: pd.Series) -> Dict[str, Any]:
    """Создать данные assistant из целевой строки."""
    
    if mode == "simplified":
        # Для режима simplified используем данные напрямую из столбцов листа SIMPLIFIED
        return {
            "rows": [{
                "Наименование": str(target_row.get("Наименование", "")),
                "Единица измерения": str(target_row.get("Единица измерения", "")) or "шт.",
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": str(target_row.get("Техническое задание", ""))
            }]
        }
    else:  # extended
        # Для режима extended используем данные напрямую из столбцов листа EXTENDED
        return {
            "rows": [{
                "Обозначение": str(target_row.get("Обозначение", "")),
                "Наименование": str(target_row.get("Наименование", "")),
                "Производитель": str(target_row.get("Производитель", "")),
                "Единица измерения": str(target_row.get("Единица измерения", "")) or "шт.",
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": str(target_row.get("Техническое задание", ""))
            }]
        }


def create_catalog_training_examples(catalog_file: Path, num_examples: int) -> List[Dict[str, Any]]:
    """Создать примеры для обучения из файла каталога."""
    examples = []
    
    try:
        df = pd.read_excel(catalog_file, sheet_name='Sheet1')
        
        # Выбираем случайные строки
        selected_rows = select_random_rows(df, num_examples)
        
        for idx, row in selected_rows.iterrows():
            # Формируем входной текст из каталога
            input_text = _format_catalog_input_text(row)
            
            # Создаем реалистичные рассуждения на основе данных из каталога
            simplified_reasoning = _create_catalog_realistic_reasoning("simplified", row)
            extended_reasoning = _create_catalog_realistic_reasoning("extended", row)
            
            # Создаем примеры для обоих режимов
            simplified_example = _create_catalog_cot_example(
                "simplified", 
                input_text, 
                row,
                simplified_reasoning
            )
            examples.append(simplified_example)
            
            extended_example = _create_catalog_cot_example(
                "extended",
                input_text,
                row,
                extended_reasoning
            )
            examples.append(extended_example)
            
    except Exception as e:
        print(f"Ошибка обработки каталога {catalog_file.name}: {e}")
    
    return examples


def create_log_training_examples(log_file: Path, num_examples: int) -> List[Dict[str, Any]]:
    """Создать примеры для обучения из файла лога."""
    examples = []
    
    try:
        # Читаем CSV файл лога
        df = pd.read_csv(log_file)
        
        # Выбираем случайные строки
        selected_rows = select_random_rows(df, num_examples)
        
        for idx, row in selected_rows.iterrows():
            # Формируем входной текст из лога
            input_text = _format_log_input_text(row)
            
            # Генерируем reasoning для этого текста
            reasoning_steps = cot_generator.generate_reasoning(input_text)
            reasoning_text = cot_generator.format_reasoning_for_prompt(reasoning_steps)
            
            # Создаем примеры для обоих режимов
            simplified_example = _create_log_cot_example(
                "simplified", 
                input_text, 
                row,
                reasoning_text
            )
            examples.append(simplified_example)
            
            extended_example = _create_log_cot_example(
                "extended",
                input_text,
                row,
                reasoning_text
            )
            examples.append(extended_example)
            
    except Exception as e:
        print(f"Ошибка обработки лога {log_file.name}: {e}")
    
    return examples


def _format_catalog_input_text(row: pd.Series) -> str:
    """Форматировать входной текст из строки каталога в структурированном виде."""
    parts = []
    
    # Добавляем заголовок записи
    parts.append("Запись 1:")
    
    # Показываем все поля из строки, даже если они пустые
    for column_name, value in row.items():
        if pd.notna(value) and str(value).strip():
            parts.append(f"  {column_name}: {value}")
        else:
            parts.append(f"  {column_name}: ")
    
    return "\n".join(parts)


def _format_log_input_text(row: pd.Series) -> str:
    """Форматировать входной текст из строки лога в структурированном виде."""
    parts = []
    
    # Формируем структурированный текст как в реальном логе
    if 'Unnamed: 0' in row and pd.notna(row['Unnamed: 0']):
        parts.append(f"Запись {row['Unnamed: 0']}:")
    else:
        parts.append("Запись 1:")
    
    # Показываем все поля из строки, даже если они пустые
    for column_name, value in row.items():
        # Пропускаем поле 'Unnamed: 0' так как оно уже использовано в заголовке
        if column_name == 'Unnamed: 0':
            continue
            
        if pd.notna(value) and str(value).strip():
            parts.append(f"  {column_name}: {value}")
        else:
            parts.append(f"  {column_name}: ")
    
    return "\n".join(parts)


def _create_catalog_cot_example(
    mode: str, 
    input_text: str, 
    target_row: pd.Series,
    reasoning_text: str
) -> Dict[str, Any]:
    """Создать пример с Chain-of-Thought reasoning из каталога."""
    
    # Создаем system prompt БЕЗ reasoning (как в инференсе)
    system_prompt = f"""Ты — ассистент по структурированной обработке промышленных инструментов.

Правила анализа:
1. Сначала объясни шаги разбора (chain-of-thought)
2. Используй знания о производителях и типах инструментов
3. Выдели обозначение, наименование, производителя и параметры
4. Представь результат в JSON формате

Правила вывода:
- Используй ровно эти ключи: {', '.join(_get_columns_for_mode(mode))}
- Структура ответа должна полностью соответствовать JSON Schema
- Не добавляй новых ключей или уровней вложенности
- Если данных нет для ключа — вставляй пустую строку
- Не используй Markdown, блоки ```json или пояснительный текст
- Вывод — только корректный JSON, без заголовков и комментариев

<output-format>
{_get_json_schema(mode)}
</output-format>

Входные данные:"""

    # Создаем user prompt с демонстрацией reasoning
    user_prompt = f"""Проанализируй данные:
{input_text}

Сначала объясни шаги разбора (chain-of-thought), затем выведи структурированный ответ в JSON.

Рассуждения:
{reasoning_text}

Структурированный ответ:"""

    # Создаем assistant ответ ТОЛЬКО с JSON (без reasoning)
    assistant_data = _create_catalog_assistant_data(mode, target_row)
    
    return {
        "mode": mode,
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_data
    }


def _create_log_cot_example(
    mode: str, 
    input_text: str, 
    target_row: pd.Series,
    reasoning_text: str
) -> Dict[str, Any]:
    """Создать пример с Chain-of-Thought reasoning из лога."""
    
    # Создаем system prompt БЕЗ reasoning (как в инференсе)
    system_prompt = f"""Ты — ассистент по структурированной обработке промышленных инструментов.

Правила анализа:
1. Сначала объясни шаги разбора (chain-of-thought)
2. Используй знания о производителях и типах инструментов
3. Выдели обозначение, наименование, производителя и параметры
4. Представь результат в JSON формате

Правила вывода:
- Используй ровно эти ключи: {', '.join(_get_columns_for_mode(mode))}
- Структура ответа должна полностью соответствовать JSON Schema
- Не добавляй новых ключей или уровней вложенности
- Если данных нет для ключа — вставляй пустую строку
- Не используй Markdown, блоки ```json или пояснительный текст
- Вывод — только корректный JSON, без заголовков и комментариев

<output-format>
{_get_json_schema(mode)}
</output-format>

Входные данные:"""

    # Создаем user prompt с демонстрацией reasoning
    user_prompt = f"""Проанализируй данные:
{input_text}

Сначала объясни шаги разбора (chain-of-thought), затем выведи структурированный ответ в JSON.

Рассуждения:
{reasoning_text}

Структурированный ответ:"""

    # Создаем assistant ответ ТОЛЬКО с JSON (без reasoning)
    assistant_data = _create_log_assistant_data(mode, target_row)
    
    return {
        "mode": mode,
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_data
    }


def _create_catalog_assistant_data(mode: str, target_row: pd.Series) -> Dict[str, Any]:
    """Создать данные assistant из строки каталога."""
    
    # Извлекаем полное наименование из поля "Наименование"
    full_name = str(target_row.get("Наименование", ""))
    
    # Ищем технические обозначения в наименовании
    import re
    denotation_patterns = [
        r'[A-Z]{2,}\d+[A-Z]*\d*[A-Z]*',  # SDJCR 1212 K11-S
        r'[A-Z]\d+[A-Z]*\d*[A-Z]*',      # DCMT 11T304
        r'[A-Z]{2,}\d+',                 # DH423051
        r'[A-Z]+\d+[A-Z]*\d*',           # ER16 426
        r'HSK-[A-Z]\d+',                 # HSK-A40
        r'BT\d+',                        # BT40
        r'ER\d+',                        # ER16
        r'CNMG\d{6}',                    # CNMG090304
        r'A\d{6}',                       # A011011
        r'[A-Z]+\d+-\d+',                # ER32-80
    ]
    
    found_denotations = []
    for pattern in denotation_patterns:
        matches = re.findall(pattern, full_name)
        found_denotations.extend(matches)
    
    if found_denotations:
        # Объединяем все найденные обозначения
        all_denotations = []
        for denot in found_denotations:
            if denot not in all_denotations:
                all_denotations.append(denot)
        
        denotation = " ".join(all_denotations)
        
        # Извлекаем наименование - все до первого технического обозначения
        words = full_name.split()
        tool_name_words = []
        
        for word in words:
            # Если встретили техническое обозначение (буквы+цифры), останавливаемся
            if re.match(r'^[A-Z]+\d+', word) or word.isdigit():
                break
            tool_name_words.append(word)
        
        tool_name = " ".join(tool_name_words).strip()
    else:
        denotation = ""
        tool_name = full_name
    
    # Получаем код каталога
    denotation_col = 'Код детали (REF. CODE)' if 'Код детали (REF. CODE)' in target_row.index else 'Код детали'
    catalog_code = str(target_row.get(denotation_col, ""))
    
    # Определяем единицу измерения - если не указана, используем "шт." по умолчанию
    unit = str(target_row.get("Единица измерения", ""))
    if not unit or unit.strip() == "":
        unit = "шт."
    
    if mode == "simplified":
        return {
            "rows": [{
                "Наименование": tool_name,
                "Единица измерения": unit,
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": f"Код: {catalog_code}" if catalog_code else ""
            }]
        }
    else:  # extended
        return {
            "rows": [{
                "Обозначение": denotation,
                "Наименование": tool_name,
                "Производитель": str(target_row.get("Производитель", "")),
                "Единица измерения": unit,
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": f"Код: {catalog_code}" if catalog_code else ""
            }]
        }


def _create_log_assistant_data(mode: str, target_row: pd.Series) -> Dict[str, Any]:
    """Создать данные assistant из строки лога."""
    
    # Извлекаем полное наименование из поля "Товары работы услуги"
    full_name = str(target_row.get("Товары работы услуги", ""))
    
    # Ищем технические обозначения в наименовании
    import re
    denotation_patterns = [
        r'[A-Z]{2,}\d+[A-Z]*\d*[A-Z]*',  # SDJCR 1212 K11-S
        r'[A-Z]\d+[A-Z]*\d*[A-Z]*',      # DCMT 11T304
        r'[A-Z]{2,}\d+',                 # DH423051
        r'[A-Z]+\d+[A-Z]*\d*',           # ER16 426
        r'HSK-[A-Z]\d+',                 # HSK-A40
        r'BT\d+',                        # BT40
        r'ER\d+',                        # ER16
        r'CNMG\d{6}',                    # CNMG090304
        r'A\d{6}',                       # A011011
        r'[A-Z]+\d+-\d+',                # ER32-80
        r'[A-Z]+\d+[A-Z]*\d*[A-Z]*',     # MAS403BT
        r'[A-Z]{1,3}\d+',                # H100, AD, G63
    ]
    
    found_denotations = []
    for pattern in denotation_patterns:
        matches = re.findall(pattern, full_name)
        found_denotations.extend(matches)
    
    if found_denotations:
        # Объединяем все найденные обозначения
        all_denotations = []
        for denot in found_denotations:
            if denot not in all_denotations:
                all_denotations.append(denot)
        
        denotation = " ".join(all_denotations)
        
        # Извлекаем наименование - все до первого технического обозначения
        words = full_name.split()
        tool_name_words = []
        
        for word in words:
            # Если встретили техническое обозначение (буквы+цифры), останавливаемся
            if re.match(r'^[A-Z]+\d+', word) or word.isdigit():
                break
            tool_name_words.append(word)
        
        tool_name = " ".join(tool_name_words).strip()
    else:
        denotation = ""
        tool_name = full_name
    
    # Ищем производителя в наименовании (слова с заглавными буквами)
    words = tool_name.split()
    possible_manufacturers = []
    for word in words:
        if word[0].isupper() and len(word) > 2 and not word.isdigit():
            # Исключаем технические обозначения
            if not re.match(r'^[A-Z]+\d+', word) and word not in ['Цанговый', 'Патрон', 'мм']:
                possible_manufacturers.append(word)
    
    manufacturer = possible_manufacturers[0] if possible_manufacturers else ""
    
    # Извлекаем параметры (размеры как единое целое)
    size_match = re.search(r'ø\s*(\d+(?:,\d+)?)\s*x\s*(\d+(?:,\d+)?)', full_name)
    if size_match:
        parameters = f"Размеры: ø{size_match.group(1)} x {size_match.group(2)}"
    else:
        # Если не найдены размеры, ищем только диаметр
        diameter_match = re.search(r'ø\s*(\d+(?:,\d+)?)\s*мм', full_name)
        parameters = f"Диаметр: ø{diameter_match.group(1)} мм" if diameter_match else ""
    
    # Определяем единицу измерения - если не указана, используем "шт." по умолчанию
    unit = str(target_row.get("Unnamed: 4", ""))
    if not unit or unit.strip() == "":
        unit = "шт."
    
    if mode == "simplified":
        return {
            "rows": [{
                "Наименование": tool_name,
                "Единица измерения": unit,
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": parameters
            }]
        }
    else:  # extended
        return {
            "rows": [{
                "Обозначение": denotation,
                "Наименование": tool_name,
                "Производитель": manufacturer,
                "Единица измерения": unit,
                "Количество": str(target_row.get("Количество", "")),
                "Техническое задание": parameters
            }]
        }


def create_enhanced_training_dataset():
    """Создать расширенный датасет с CoT reasoning."""
    
    # Пути к файлам
    script_dir = Path(__file__).parent
    datasets_dir = script_dir / "datasets"
    catalogs_dir = datasets_dir / "catalogs"
    output_jsonl_path = script_dir / "qwen3_cot_structured_train.jsonl"
    
    print("=" * 70)
    print("СОЗДАНИЕ ДАТАСЕТА С CHAIN-OF-THOUGHT REASONING")
    print("=" * 70)
    print(f"Максимальное количество примеров: {MAX_TRAINING_EXAMPLES}")
    
    # Инициализация графа знаний
    print("Инициализация графа знаний...")
    kg_file = script_dir / "knowledge_graph.json"
    if kg_file.exists():
        knowledge_graph.load_from_file(kg_file)
        print(f"✓ Граф знаний загружен из {kg_file}")
    else:
        knowledge_graph.save_to_file(kg_file)
        print(f"✓ Граф знаний сохранен в {kg_file}")
    
    # Поиск Excel файлов в основной директории
    excel_files = list(datasets_dir.glob("*.xlsx"))
    catalog_files = list(catalogs_dir.glob("*.xlsx")) if catalogs_dir.exists() else []
    
    if not excel_files and not catalog_files:
        print(f"❌ Excel файлы не найдены в {datasets_dir} и {catalogs_dir}")
        return
    
    print(f"Найдено {len(excel_files)} Excel файлов в основной директории")
    print(f"Найдено {len(catalog_files)} Excel файлов в каталогах")
    
    # Рассчитываем количество примеров для каждого файла
    all_files = excel_files + catalog_files
    examples_per_file = calculate_examples_per_file(all_files, MAX_TRAINING_EXAMPLES)
    
    print("\nРаспределение примеров по файлам:")
    for filename, count in examples_per_file.items():
        print(f"  {filename}: {count} примеров")
    
    all_examples = []
    
    # Обработка основных Excel файлов
    for excel_file in excel_files:
        filename = excel_file.name
        num_examples = examples_per_file.get(filename, 0)
        
        if num_examples == 0:
            print(f"\nПропуск файла: {filename} (0 примеров)")
            continue
            
        print(f"\nОбработка файла: {filename} ({num_examples} примеров)")
        
        try:
            # Читаем Excel файл с тремя листами
            input_df = pd.read_excel(excel_file, sheet_name="INPUT")
            simplified_df = pd.read_excel(excel_file, sheet_name="SIMPLIFIED")
            extended_df = pd.read_excel(excel_file, sheet_name="EXTENDED")
            
            print(f"  INPUT: {len(input_df)} строк")
            print(f"  SIMPLIFIED: {len(simplified_df)} строк")
            print(f"  EXTENDED: {len(extended_df)} строк")
            
            # Выбираем одинаковые индексы из всех листов для сохранения соответствия строк
            if len(input_df) <= num_examples:
                selected_input = input_df
                selected_simplified = simplified_df
                selected_extended = extended_df
            else:
                # Выбираем случайные индексы один раз
                import random
                random_indices = random.sample(range(len(input_df)), num_examples)
                
                # Используем одинаковые индексы для всех листов
                selected_input = input_df.iloc[random_indices].reset_index(drop=True)
                selected_simplified = simplified_df.iloc[random_indices].reset_index(drop=True)
                selected_extended = extended_df.iloc[random_indices].reset_index(drop=True)
            
            print(f"  Выбрано: {len(selected_input)} строк")
            
            # Создаем примеры с CoT reasoning
            examples = create_cot_training_examples(selected_input, selected_simplified, selected_extended)
            all_examples.extend(examples)
            
            print(f"  ✓ Создано {len(examples)} примеров с CoT reasoning")
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки {excel_file.name}: {e}")
            continue
    
    # Обработка файлов каталогов
    for catalog_file in catalog_files:
        filename = catalog_file.name
        num_examples = examples_per_file.get(filename, 0)
        
        if num_examples == 0:
            print(f"\nПропуск каталога: {filename} (0 примеров)")
            continue
            
        print(f"\nОбработка каталога: {filename} ({num_examples} примеров)")
        
        try:
            # Создаем примеры из каталога
            examples = create_catalog_training_examples(catalog_file, num_examples)
            all_examples.extend(examples)
            
            print(f"  ✓ Создано {len(examples)} примеров с CoT reasoning из каталога")
            
        except Exception as e:
            print(f"  ❌ Ошибка обработки каталога {catalog_file.name}: {e}")
            continue
    
    # Сохраняем результат
    print(f"\nСохранение {len(all_examples)} примеров в {output_jsonl_path}")
    
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print(f"✓ Датасет сохранен: {output_jsonl_path}")
    
    # Статистика
    simplified_count = len([ex for ex in all_examples if ex['mode'] == 'simplified'])
    extended_count = len([ex for ex in all_examples if ex['mode'] == 'extended'])
    
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 70)
    print(f"Всего примеров: {len(all_examples)}")
    print(f"Simplified режим: {simplified_count}")
    print(f"Extended режим: {extended_count}")
    print(f"Максимальное количество: {MAX_TRAINING_EXAMPLES}")
    print(f"Файл: {output_jsonl_path}")
    print("=" * 70)


if __name__ == "__main__":
    create_enhanced_training_dataset()
