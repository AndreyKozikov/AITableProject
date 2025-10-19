"""Модуль для создания обучающего датасета в формате JSONL.

Этот модуль создает датасет для обучения модели на основе Excel файлов
с тремя листами: INPUT, SIMPLIFIED и EXTENDED.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, create_model

from src.utils.config import MODEL_DIR, OUT_DIR_LEARNING_DATA, PROMPT_TEMPLATE_SO

# Путь к выходному JSONL файлу (в той же директории, что и текущий модуль)
OUTPUT_JSONL_PATH = Path(__file__).parent / "qwen3_structured_train.jsonl"


def _load_csv_schema(csv_path: Path) -> List[str]:
    """Загрузить названия колонок из CSV файла схемы.
    
    Args:
        csv_path: Путь к CSV файлу схемы.
        
    Returns:
        Список названий колонок из заголовка CSV.
    """
    try:
        df = pd.read_csv(csv_path, sep=',', nrows=0)
        columns = list(df.columns)
        print(f"Загружено {len(columns)} колонок из {csv_path}: {columns}")
        return columns
    except Exception as e:
        print(f"Ошибка загрузки CSV схемы из {csv_path}: {e}")
        raise


def _create_pydantic_model(columns: List[str], model_name: str) -> type[BaseModel]:
    """Создать Pydantic модель динамически из списка колонок.
    
    Args:
        columns: Список названий колонок.
        model_name: Имя создаваемой модели.
        
    Returns:
        Динамически созданный класс Pydantic BaseModel.
    """
    fields = {}
    
    for column in columns:
        field_def = (
            str,
            Field(
                default="",
                description=f"Field for {column}",
                title=column
            )
        )
        fields[column] = field_def
    
    dynamic_model = create_model(
        model_name,
        **fields,
        __config__=ConfigDict(extra='forbid')
    )
    
    print(f"Создана модель '{model_name}' с полями: {list(fields.keys())}")
    return dynamic_model


def _create_container_model(row_model: type[BaseModel]) -> type[BaseModel]:
    """Создать контейнерную модель для массива строк таблицы.
    
    Args:
        row_model: Модель для одной строки таблицы.
        
    Returns:
        Модель-контейнер с полем rows.
    """
    container_model = create_model(
        'TableStructuredOutput',
        rows=(
            List[row_model],
            Field(
                description="List of table rows",
                title="Rows"
            )
        ),
        __config__=ConfigDict(extra='forbid')
    )
    
    return container_model


def get_json_schema_for_mode(mode: str) -> Tuple[str, str]:
    """Получить JSON схему и заголовки для указанного режима.
    
    Args:
        mode: Режим работы ('simplified' или 'extended').
        
    Returns:
        Кортеж (schema_json_str, header_str).
    """
    if mode == "simplified":
        csv_path = MODEL_DIR / "simplified.csv"
        row_model_name = "TableRowSimplified"
    elif mode == "extended":
        csv_path = MODEL_DIR / "extended.csv"
        row_model_name = "TableRowExtended"
    else:
        raise ValueError(f"Неизвестный режим: {mode}")
    
    # Загружаем схему из CSV
    columns = _load_csv_schema(csv_path)
    
    # Создаем Pydantic модели
    row_model = _create_pydantic_model(columns, row_model_name)
    container_model = _create_container_model(row_model)
    
    # Получаем JSON схему
    schema_dict = container_model.model_json_schema()
    schema_json_str = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    
    # Формируем строку заголовков
    header_str = ", ".join(columns)
    
    return schema_json_str, header_str


def read_excel_sheet_as_dataframe(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Прочитать лист Excel как DataFrame.
    
    Args:
        excel_path: Путь к Excel файлу.
        sheet_name: Имя листа для чтения.
        
    Returns:
        DataFrame с данными листа.
    """
    try:
        # Читаем лист, сохраняя пустые значения как пустые строки
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False
        )
        
        # Удаляем столбцы с названиями, содержащими "Unnamed"
        unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
        if unnamed_cols:
            df = df.drop(columns=unnamed_cols)
            print(f"  Удалены столбцы из {sheet_name}: {unnamed_cols}")
        
        # Заменяем NaN на пустые строки
        df = df.fillna("")
        
        print(f"Прочитано {len(df)} строк из листа {sheet_name}")
        return df
        
    except Exception as e:
        print(f"Ошибка чтения листа {sheet_name} из {excel_path}: {e}")
        return pd.DataFrame()


def format_row_as_text(row: pd.Series, row_num: int) -> str:
    """Форматировать одну строку DataFrame как текстовый блок.
    
    Args:
        row: Pandas Series (строка DataFrame).
        row_num: Номер записи для отображения.
        
    Returns:
        Текстовое представление строки.
    """
    lines = [f"Запись {row_num}:"]
    
    for col_name, value in row.items():
        value_str = str(value).strip()
        if value_str:
            lines.append(f"  {col_name}: {value_str}")
        else:
            lines.append(f"  {col_name}:")
    
    return "\n".join(lines)


def row_to_dict(row: pd.Series) -> Dict[str, str]:
    """Конвертировать строку DataFrame в словарь.
    
    Args:
        row: Pandas Series (строка DataFrame).
        
    Returns:
        Словарь с данными строки.
    """
    return row.to_dict()


def create_jsonl_record(
    mode: str,
    input_text: str,
    target_row: Dict[str, str]
) -> Dict:
    """Создать одну запись для JSONL файла.
    
    Args:
        mode: Режим ('simplified' или 'extended').
        input_text: Текст одной строки из листа INPUT.
        target_row: Данные одной строки из листа SIMPLIFIED или EXTENDED.
        
    Returns:
        Словарь с полями mode, system, user, assistant.
    """
    # Получаем JSON схему и заголовки для режима
    schema_json_str, header_str = get_json_schema_for_mode(mode)
    
    # Формируем system prompt как при инференсе
    # Заменяем {tables_text} на пустую строку, оставляя "Входные данные:\n"
    system_prompt = PROMPT_TEMPLATE_SO.format(
        header=header_str,
        schema=schema_json_str,
        tables_text=""
    ).strip()
    
    # User prompt - это текст одной строки из INPUT
    user_prompt = input_text
    
    # Assistant response - объект с данными одной строки (в массиве rows)
    assistant_data = {"rows": [target_row]}
    
    record = {
        "mode": mode,
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_data
    }
    
    return record


def process_excel_file(excel_path: Path) -> List[Dict]:
    """Обработать один Excel файл построчно и создать два примера для каждой строки.
    
    Для каждой строки i создаются две записи:
    - mode="simplified" с данными из SIMPLIFIED[i]
    - mode="extended" с данными из EXTENDED[i]
    
    Args:
        excel_path: Путь к Excel файлу.
        
    Returns:
        Список записей для JSONL файла (количество строк × 2).
    """
    print(f"Обработка файла: {excel_path.name}")
    
    records = []
    
    try:
        # Читаем все три листа как DataFrame
        df_input = read_excel_sheet_as_dataframe(excel_path, "INPUT")
        df_simplified = read_excel_sheet_as_dataframe(excel_path, "SIMPLIFIED")
        df_extended = read_excel_sheet_as_dataframe(excel_path, "EXTENDED")
        
        # Количество строк (предполагаем одинаковое на всех листах)
        num_rows = len(df_input)
        print(f"  Обработка {num_rows} строк...")
        
        # Обрабатываем каждую строку
        for i in range(num_rows):
            row_num = i + 1
            
            # Формируем user-промпт из строки INPUT
            input_row = df_input.iloc[i]
            input_text = format_row_as_text(input_row, row_num)
            
            # Получаем данные строки из SIMPLIFIED
            simplified_row = df_simplified.iloc[i]
            simplified_dict = row_to_dict(simplified_row)
            
            # Создаем запись для simplified
            record_simplified = create_jsonl_record("simplified", input_text, simplified_dict)
            records.append(record_simplified)
            
            # Получаем данные строки из EXTENDED
            extended_row = df_extended.iloc[i]
            extended_dict = row_to_dict(extended_row)
            
            # Создаем запись для extended
            record_extended = create_jsonl_record("extended", input_text, extended_dict)
            records.append(record_extended)
        
        print(f"  Создано {len(records)} записей ({num_rows} строк × 2 режима)")
        return records
        
    except Exception as e:
        print(f"Ошибка обработки файла {excel_path.name}: {e}")
        return []


def create_training_dataset() -> bool:
    """Создать обучающий датасет JSONL из всех Excel файлов.
    
    Returns:
        True если датасет успешно создан, False в противном случае.
    """
    try:
        print("=" * 70)
        print("Начало создания обучающего датасета JSONL")
        print("=" * 70)
        
        # Получаем список Excel файлов
        excel_files = sorted(OUT_DIR_LEARNING_DATA.glob("*.xlsx"))
        
        if not excel_files:
            print(f"Excel файлы не найдены в {OUT_DIR_LEARNING_DATA}")
            return False
        
        print(f"Найдено {len(excel_files)} Excel файлов")
        
        # Обрабатываем все файлы
        all_records = []
        
        for excel_file in excel_files:
            records = process_excel_file(excel_file)
            all_records.extend(records)
        
        if not all_records:
            print("Не создано ни одной записи")
            return False
        
        # Сохраняем в JSONL файл
        print(f"Сохранение {len(all_records)} записей в {OUTPUT_JSONL_PATH}")
        
        with open(OUTPUT_JSONL_PATH, 'w', encoding='utf-8') as f:
            for record in all_records:
                # Сохраняем каждую запись как одну строку JSON
                # assistant - это dict, будет сериализован как объект JSON
                json_line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                f.write(json_line + '\n')
        
        print("=" * 70)
        print(f"Датасет успешно создан: {OUTPUT_JSONL_PATH}")
        print(f"Всего записей: {len(all_records)}")
        print(f"Файлов обработано: {len(excel_files)}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"Критическая ошибка при создании датасета: {e}")
        return False


if __name__ == "__main__":
    # Запуск создания датасета при прямом выполнении скрипта
    create_training_dataset()

