"""
Скрипт для fine-tuning модели Qwen3 с использованием LoRA адаптеров.

Скрипт выполняет дообучение модели Qwen3 на структурированных данных
для задачи извлечения табличных данных с использованием техники LoRA
(Low-Rank Adaptation) для эффективного обучения.
"""

import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

from src.utils.config import MODEL_ID, MODEL_CACHE_DIR, USE_LOCAL_MODEL

# Определяем путь к модели в зависимости от настроек
MODEL_PATH = str(MODEL_CACHE_DIR) if USE_LOCAL_MODEL else MODEL_ID


def get_device():
    """
    Определение доступного устройства: TPU → GPU → CPU.
    
    Returns:
        Кортеж (device, device_type) где:
        - device: torch.device объект
        - device_type: строка "tpu", "gpu" или "cpu"
    """
    # Проверка TPU
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        num_cores = os.environ.get("TPU_NUM_CORES", "8")
        print(f"Используем TPU (ядра: {num_cores})")
        return device, "tpu"
    except ImportError:
        pass
    except Exception as e:
        print(f"TPU найден, но ошибка при инициализации: {e}")
    
    # Проверка GPU
    if torch.cuda.is_available():
        print("GPU доступен, используем CUDA")
        return torch.device("cuda"), "gpu"
    
    # Используем CPU
    print("Используем CPU")
    return torch.device("cpu"), "cpu"


def get_torch_dtype(device_type):
    """
    Определяет оптимальный тип данных torch для устройства.
    
    Args:
        device_type: Тип устройства ("tpu", "gpu" или "cpu")
        
    Returns:
        torch.dtype для модели
    """
    if device_type == "tpu":
        return torch.bfloat16  # TPU работает только с bfloat16
    elif device_type == "gpu":
        return torch.float16  # На GPU можно использовать float16
    else:
        return torch.float32  # На CPU используем float32


def load_model_and_tokenizer(model_path, device, torch_dtype):
    """
    Загружает модель и токенизатор.
    
    Args:
        model_path: Путь к модели (локальный или HuggingFace ID)
        device: Устройство для загрузки модели
        torch_dtype: Тип данных для модели
        
    Returns:
        Кортеж (tokenizer, model)
    """
    source = "local cache" if USE_LOCAL_MODEL else "HuggingFace Hub"
    print(f"Загрузка модели из {source}: {model_path}")
    print(f"Устройство: {device}, dtype: {torch_dtype}")
    
    try:
        # Загружаем токенизатор
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Загружаем модель для causal language modeling
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        model = model.to(device)
        
        print(f"Модель и токенизатор загружены успешно из {source}")
        return tokenizer, model
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        if USE_LOCAL_MODEL:
            print(f"Модель не найдена в локальном кэше: {MODEL_CACHE_DIR}")
            print(f"Запустите 'python src/utils/download_model.py' для загрузки модели")
        raise


def setup_lora(model):
    """
    Настраивает LoRA адаптеры для модели.
    
    Args:
        model: Предобученная модель
        
    Returns:
        Модель с LoRA адаптерами
    """
    print("Настройка LoRA адаптеров...")
    
    # Конфигурация LoRA
    peft_config = LoraConfig(
        r=16,  # Ранг матриц (чем больше, тем больше параметров, но выше точность)
        lora_alpha=32,  # Коэффициент усиления LoRA весов
        target_modules=["q_proj", "v_proj"],  # Целевые слои для Qwen3
        lora_dropout=0.05,  # Dropout для регуляризации
        bias="none",  # Не обучаем bias
        task_type="CAUSAL_LM"  # Тип задачи - авторегрессионная языковая модель
    )
    
    # Применяем LoRA к модели
    model = get_peft_model(model, peft_config)
    
    # Выводим информацию об обучаемых параметрах
    model.print_trainable_parameters()
    
    print("LoRA адаптеры настроены")
    return model


def load_training_dataset(data_path):
    """
    Загружает датасет для обучения из JSONL файла.
    
    Args:
        data_path: Путь к JSONL файлу с данными
        
    Returns:
        Dataset объект из библиотеки datasets
    """
    print(f"Загрузка датасета из: {data_path}")
    
    # Загружаем JSONL файл
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    print(f"Загружено примеров: {len(dataset)}")
    return dataset


def tokenize_function(example, tokenizer):
    """
    Токенизирует один пример из датасета.
    
    Формирует промпт из system и user сообщений, добавляет ответ assistant.
    Маскирует промпт в labels (значения -100), чтобы модель обучалась
    только на генерации ответа.
    
    Args:
        example: Словарь с полями 'system', 'user', 'assistant'
        tokenizer: Токенизатор модели
        
    Returns:
        Словарь с токенизированными данными
    """
    # Формируем полный промпт
    prompt = f"System: {example['system']}\n\nUser: {example['user']}\n\nAssistant: "
    output = example['assistant']
    
    # Полный текст = промпт + ответ
    full_text = prompt + output
    
    # Токенизируем весь текст
    tokenized = tokenizer(
        full_text,
        max_length=2048,  # Максимальная длина для Qwen3
        truncation=True,
        padding="max_length"
    )
    
    # Создаем labels - копия input_ids
    labels = tokenized["input_ids"].copy()
    
    # Маскируем промпт в labels (не учитываем в loss)
    prompt_tokenized = tokenizer(prompt, add_special_tokens=False)
    prompt_len = len(prompt_tokenized["input_ids"])
    
    # Заменяем токены промпта на -100 (игнорируются в loss)
    labels[:prompt_len] = [-100] * prompt_len
    
    tokenized["labels"] = labels
    
    return tokenized


def prepare_dataset(dataset, tokenizer):
    """
    Подготавливает датасет к обучению - токенизирует все примеры.
    
    Args:
        dataset: Исходный датасет
        tokenizer: Токенизатор модели
        
    Returns:
        Токенизированный датасет
    """
    print("Токенизация датасета...")
    
    # Применяем функцию токенизации ко всем примерам
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=["mode", "system", "user", "assistant"]  # Удаляем исходные колонки
    )
    
    print("Датасет токенизирован")
    return tokenized_dataset


def setup_training_arguments(device_type, output_dir):
    """
    Настраивает параметры обучения.
    
    Args:
        device_type: Тип устройства ("tpu", "gpu" или "cpu")
        output_dir: Директория для сохранения чекпоинтов
        
    Returns:
        TrainingArguments объект
    """
    print("Настройка параметров обучения...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1 if device_type != "gpu" else 2,  # Размер батча
        gradient_accumulation_steps=8,  # Накопление градиентов для эффективности
        learning_rate=2e-5,  # Скорость обучения
        num_train_epochs=3,  # Количество эпох
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,  # Логирование каждые 10 шагов
        save_strategy="epoch",  # Сохранять после каждой эпохи
        save_total_limit=2,  # Хранить только 2 последних чекпоинта
        report_to="none",  # Не отправлять в wandb/tensorboard
        remove_unused_columns=False,
        fp16=True if device_type == "gpu" else False,  # Mixed precision для GPU
        bf16=True if device_type == "tpu" else False,  # bfloat16 для TPU
        warmup_steps=100,  # Постепенное увеличение learning rate
        weight_decay=0.01,  # L2 регуляризация
        dataloader_num_workers=0,  # Количество воркеров для загрузки данных
    )
    
    print(f"Параметры обучения настроены:")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    
    return training_args


def train_model(model, tokenizer, tokenized_dataset, training_args):
    """
    Запускает процесс обучения модели.
    
    Args:
        model: Модель с LoRA адаптерами
        tokenizer: Токенизатор
        tokenized_dataset: Токенизированный датасет
        training_args: Параметры обучения
        
    Returns:
        Обученная модель
    """
    print("Инициализация Trainer...")
    
    # Data collator для динамического паддинга
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Не используем Masked Language Modeling
    )
    
    # Создаем Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 70)
    
    # Запускаем обучение
    trainer.train()
    
    print("=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    return model


def save_lora_adapters(model, tokenizer, output_path):
    """
    Сохраняет LoRA адаптеры и токенизатор.
    
    Args:
        model: Обученная модель с LoRA адаптерами
        tokenizer: Токенизатор
        output_path: Путь для сохранения адаптеров
    """
    print(f"Сохранение LoRA адаптеров в: {output_path}")
    
    # Создаем директорию если не существует
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем только LoRA адаптеры (не всю модель)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("LoRA адаптеры и токенизатор сохранены")


def main():
    """
    Главная функция скрипта.
    """
    print("=" * 70)
    print("FINE-TUNING МОДЕЛИ QWEN С LORA")
    print("=" * 70)
    
    # Определяем пути
    script_dir = Path(__file__).parent
    data_path = script_dir / "qwen3_structured_train.jsonl"
    
    # Сохраняем результаты обучения в директорию с моделью
    model_base_dir = MODEL_CACHE_DIR
    output_dir = model_base_dir / "checkpoints"
    adapters_dir = model_base_dir / "lora_adapters"
    
    source = "локальный кэш" if USE_LOCAL_MODEL else "HuggingFace Hub"
    print(f"Источник модели: {source}")
    print(f"Путь к модели: {MODEL_PATH}")
    print(f"Данные: {data_path}")
    print(f"Чекпоинты: {output_dir}")
    print(f"Адаптеры: {adapters_dir}")
    
    # Проверяем существование файла данных
    if not data_path.exists():
        print(f"ОШИБКА: Файл данных не найден: {data_path}")
        return
    
    # Шаг 1: Определяем устройство
    print("\n" + "=" * 70)
    print("ШАГ 1: Определение устройства")
    print("=" * 70)
    device, device_type = get_device()
    torch_dtype = get_torch_dtype(device_type)
    
    # Шаг 2: Загружаем модель и токенизатор
    print("\n" + "=" * 70)
    print("ШАГ 2: Загрузка модели и токенизатора")
    print("=" * 70)
    tokenizer, model = load_model_and_tokenizer(MODEL_PATH, device, torch_dtype)
    
    # Шаг 3: Настраиваем LoRA
    print("\n" + "=" * 70)
    print("ШАГ 3: Настройка LoRA адаптеров")
    print("=" * 70)
    model = setup_lora(model)
    
    # Шаг 4: Загружаем данные
    print("\n" + "=" * 70)
    print("ШАГ 4: Загрузка датасета")
    print("=" * 70)
    dataset = load_training_dataset(data_path)
    
    # Шаг 5: Токенизируем данные
    print("\n" + "=" * 70)
    print("ШАГ 5: Токенизация датасета")
    print("=" * 70)
    tokenized_dataset = prepare_dataset(dataset, tokenizer)
    
    # Шаг 6: Настраиваем параметры обучения
    print("\n" + "=" * 70)
    print("ШАГ 6: Настройка параметров обучения")
    print("=" * 70)
    training_args = setup_training_arguments(device_type, output_dir)
    
    # Шаг 7: Обучаем модель
    print("\n" + "=" * 70)
    print("ШАГ 7: Обучение модели")
    print("=" * 70)
    model = train_model(model, tokenizer, tokenized_dataset, training_args)
    
    # Шаг 8: Сохраняем адаптеры
    print("\n" + "=" * 70)
    print("ШАГ 8: Сохранение LoRA адаптеров")
    print("=" * 70)
    save_lora_adapters(model, tokenizer, adapters_dir)
    
    print("\n" + "=" * 70)
    print("ПРОЦЕСС FINE-TUNING ЗАВЕРШЕН УСПЕШНО!")
    print("=" * 70)
    print(f"\nLoRA адаптеры сохранены в: {adapters_dir}")
    print("Для использования обученной модели загрузите базовую модель")
    print("и примените сохраненные адаптеры с помощью PeftModel.from_pretrained()")


if __name__ == "__main__":
    main()

