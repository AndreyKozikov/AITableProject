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


def read_excel_sheet_as_text(excel_path: Path, sheet_name: str) -> str:
    """Прочитать лист INPUT и сформировать текстовый блок.
    
    Args:
        excel_path: Путь к Excel файлу.
        sheet_name: Имя листа для чтения.
        
    Returns:
        Текстовое представление листа в формате "Запись N: ...".
    """
    try:
        # Читаем лист, сохраняя пустые значения как пустые строки
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False
        )
        
        # Заменяем NaN на пустые строки
        df = df.fillna("")
        
        text_blocks = []
        record_num = 1
        
        for idx, row in df.iterrows():
            # Проверяем, не является ли строка полностью пустой
            if row.astype(str).str.strip().eq("").all():
                continue
            
            # Формируем текст записи
            lines = [f"Запись {record_num}:"]
            
            for col_name, value in row.items():
                value_str = str(value).strip()
                if value_str:
                    lines.append(f"  {col_name}: {value_str}")
                else:
                    lines.append(f"  {col_name}:")
            
            text_blocks.append("\n".join(lines))
            record_num += 1
        
        result = "\n\n".join(text_blocks)
        print(f"Сформирован текст из {len(text_blocks)} записей листа {sheet_name}")
        return result
        
    except Exception as e:
        print(f"Ошибка чтения листа {sheet_name} из {excel_path}: {e}")
        return ""


def read_excel_sheet_as_json(excel_path: Path, sheet_name: str) -> List[Dict[str, str]]:
    """Прочитать лист SIMPLIFIED/EXTENDED и сформировать список словарей.
    
    Args:
        excel_path: Путь к Excel файлу.
        sheet_name: Имя листа для чтения.
        
    Returns:
        Список словарей с данными из листа.
    """
    try:
        # Читаем лист, сохраняя пустые значения как пустые строки
        df = pd.read_excel(
            excel_path,
            sheet_name=sheet_name,
            dtype=str,
            keep_default_na=False
        )
        
        # Заменяем NaN на пустые строки
        df = df.fillna("")
        
        # Удаляем полностью пустые строки
        df = df[~df.astype(str).apply(lambda x: x.str.strip().eq("").all(), axis=1)]
        
        # Конвертируем в список словарей
        rows = df.to_dict('records')
        
        print(f"Прочитано {len(rows)} строк из листа {sheet_name}")
        return rows
        
    except Exception as e:
        print(f"Ошибка чтения листа {sheet_name} из {excel_path}: {e}")
        return []


def create_jsonl_record(
    mode: str,
    input_text: str,
    target_rows: List[Dict[str, str]]
) -> Dict:
    """Создать одну запись для JSONL файла.
    
    Args:
        mode: Режим ('simplified' или 'extended').
        input_text: Текст из листа INPUT.
        target_rows: Данные из листа SIMPLIFIED или EXTENDED.
        
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
    
    # User prompt - это текст из INPUT
    user_prompt = input_text
    
    # Assistant response - объект с данными (не строка!)
    assistant_data = {"rows": target_rows}
    
    record = {
        "mode": mode,
        "system": system_prompt,
        "user": user_prompt,
        "assistant": assistant_data
    }
    
    return record


def process_excel_file(excel_path: Path) -> List[Dict]:
    """Обработать один Excel файл и создать две записи (simplified + extended).
    
    Args:
        excel_path: Путь к Excel файлу.
        
    Returns:
        Список записей для JSONL файла.
    """
    print(f"Обработка файла: {excel_path.name}")
    
    records = []
    
    try:
        # Читаем лист INPUT как текст
        input_text = read_excel_sheet_as_text(excel_path, "INPUT")
        
        if not input_text:
            print(f"Пустой INPUT в файле {excel_path.name}, пропускаем")
            return []
        
        # Обрабатываем SIMPLIFIED
        simplified_rows = read_excel_sheet_as_json(excel_path, "SIMPLIFIED")
        if simplified_rows:
            record_simplified = create_jsonl_record("simplified", input_text, simplified_rows)
            records.append(record_simplified)
            print(f"  Создана запись для SIMPLIFIED ({len(simplified_rows)} строк)")
        else:
            print(f"  Пустой SIMPLIFIED в файле {excel_path.name}")
        
        # Обрабатываем EXTENDED
        extended_rows = read_excel_sheet_as_json(excel_path, "EXTENDED")
        if extended_rows:
            record_extended = create_jsonl_record("extended", input_text, extended_rows)
            records.append(record_extended)
            print(f"  Создана запись для EXTENDED ({len(extended_rows)} строк)")
        else:
            print(f"  Пустой EXTENDED в файле {excel_path.name}")
        
        print(f"Создано {len(records)} записей из файла {excel_path.name}")
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

