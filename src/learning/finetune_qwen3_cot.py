"""
Скрипт для fine-tuning модели Qwen3 с поддержкой Chain-of-Thought reasoning.

Расширяет базовый функционал finetune_qwen3.py добавлением поддержки
цепочек рассуждений и интеграции с графом знаний.
"""

import os
import json
from pathlib import Path
from datetime import datetime

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

from src.utils.config import MODEL_CACHE_DIR
from .knowledge_graph import knowledge_graph
from .cot_reasoning import cot_generator


def get_device():
    """Определение доступного устройства: TPU → GPU → CPU."""
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
    """Определяет оптимальный тип данных torch для устройства."""
    if device_type == "tpu":
        return torch.bfloat16
    elif device_type == "gpu":
        return torch.float16
    else:
        return torch.float32


def load_model_and_tokenizer(device, torch_dtype):
    """Загружает модель и токенизатор из локальной директории."""
    print(f"Загрузка модели из {MODEL_CACHE_DIR}")
    print(f"Устройство: {device}, dtype: {torch_dtype}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(MODEL_CACHE_DIR),
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            str(MODEL_CACHE_DIR),
            torch_dtype=torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        model = model.to(device)
        
        print("Модель и токенизатор загружены успешно")
        return tokenizer, model
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        print(f"Путь к модели: {MODEL_CACHE_DIR}")
        raise


def setup_lora(model):
    """Настраивает LoRA адаптеры для модели."""
    print("Настройка LoRA адаптеров...")
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print("LoRA адаптеры настроены")
    return model


def load_training_dataset(data_path):
    """Загружает датасет для обучения из JSONL файла."""
    print(f"Загрузка датасета из: {data_path}")
    
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    print(f"Загружено примеров: {len(dataset)}")
    return dataset


def tokenize_cot_function(example, idx, tokenizer):
    """
    Токенизирует пример с Chain-of-Thought reasoning.
    
    Расширяет базовую функцию токенизации поддержкой цепочек рассуждений.
    """
    mode = example.get("mode", "unknown")
    
    # Преобразуем assistant из dict в JSON строку, если это dict
    assistant_text = example['assistant']
    if isinstance(assistant_text, dict):
        assistant_text = json.dumps(assistant_text, ensure_ascii=False, separators=(',', ':'))
    
    # Формируем system prompt с указанием режима работы
    system_prompt = f"Режим работы: {mode}. {example['system']}"
    
    # Шаг 1: Формируем prompt_text (system + user) с generation prompt
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['user']}
    ]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Шаг 2: Формируем full_text (system + user + assistant) без generation prompt
    full_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": example['user']},
        {"role": "assistant", "content": assistant_text}
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    # Логирование каждого 100-го промпта
    prompt_num = idx + 1
    if prompt_num == 1 or prompt_num % 100 == 0:
        print(f"\n{'='*70}")
        print(f"CoT ПРОМПТ #{prompt_num} (режим: {mode})")
        print(f"{'='*70}")
        print(f"Prompt часть:\n{prompt_text}")
        print(f"\nFull text:\n{full_text}")
        print(f"{'='*70}\n")
    
    # Шаг 3: Токенизируем весь текст БЕЗ truncation и БЕЗ padding
    tokenized = tokenizer(
        full_text,
        truncation=False,
        padding=False
    )
    
    # Шаг 4: Создаем labels - копия input_ids
    labels = tokenized["input_ids"].copy()
    
    # Шаг 5: Маскируем промпт в labels (system + user)
    prompt_tokenized = tokenizer(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_tokenized["input_ids"])
    
    # Заменяем токены prompt на -100 (игнорируются в loss)
    labels[:prompt_len] = [-100] * prompt_len
    
    tokenized["labels"] = labels
    
    return tokenized


def prepare_dataset(dataset, tokenizer):
    """Подготавливает датасет к обучению - токенизирует все примеры."""
    print("Токенизация датасета с CoT reasoning...")
    
    tokenized_dataset = dataset.map(
        lambda x, idx: tokenize_cot_function(x, idx, tokenizer),
        with_indices=True,
        remove_columns=["mode", "system", "user", "assistant"]
    )
    
    print("Датасет токенизирован")
    return tokenized_dataset


def setup_training_arguments(device_type, output_dir):
    """Настраивает параметры обучения."""
    print("Настройка параметров обучения...")
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        save_strategy="steps",
        save_steps=5,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=True if device_type == "gpu" else False,
        bf16=True if device_type == "tpu" else False,
        warmup_steps=100,
        weight_decay=0.01,
        dataloader_num_workers=0,
    )
    
    print(f"Параметры обучения настроены:")
    print(f"  - Batch size: 1 (один пример за раз)")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Чекпоинты: каждые {training_args.save_steps} шагов")
    print(f"  - Хранится чекпоинтов: {training_args.save_total_limit}")
    
    return training_args


def train_model(model, tokenizer, tokenized_dataset, training_args, resume_from_checkpoint=None):
    """Запускает процесс обучения модели."""
    print("Инициализация Trainer...")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    print("=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ С CHAIN-OF-THOUGHT REASONING")
    print("=" * 70)
    
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    print("=" * 70)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)
    
    return model


def save_lora_adapters(model, tokenizer, output_path):
    """Сохраняет LoRA адаптеры и токенизатор."""
    print(f"Сохранение LoRA адаптеров в: {output_path}")
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print("LoRA адаптеры и токенизатор сохранены")


def main():
    """Главная функция скрипта."""
    start_time = datetime.now()
    
    print("=" * 70)
    print("FINE-TUNING МОДЕЛИ QWEN С LORA + CHAIN-OF-THOUGHT")
    print("=" * 70)
    print(f"Время начала: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Определяем пути
    script_dir = Path(__file__).parent
    data_path = script_dir / "qwen3_cot_structured_train.jsonl"
    
    # Сохраняем результаты обучения в директорию с моделью
    model_base_dir = MODEL_CACHE_DIR
    output_dir = model_base_dir / "checkpoints_cot"
    adapters_dir = model_base_dir / "lora_adapters_cot"
    
    print(f"Путь к модели: {MODEL_CACHE_DIR}")
    print(f"Данные: {data_path}")
    print(f"Чекпоинты: {output_dir}")
    print(f"Адаптеры: {adapters_dir}")
    
    # Проверяем существование файла данных
    if not data_path.exists():
        print(f"ОШИБКА: Файл данных не найден: {data_path}")
        print("Сначала запустите create_cot_training_dataset.py")
        return
    
    # Инициализация графа знаний
    print("\n" + "=" * 70)
    print("ИНИЦИАЛИЗАЦИЯ ГРАФА ЗНАНИЙ")
    print("=" * 70)
    kg_file = script_dir / "knowledge_graph.json"
    if kg_file.exists():
        knowledge_graph.load_from_file(kg_file)
        print(f"✓ Граф знаний загружен из {kg_file}")
    else:
        knowledge_graph.save_to_file(kg_file)
        print(f"✓ Граф знаний сохранен в {kg_file}")
    
    # Выводим статистику графа знаний
    print(f"✓ Сущностей в графе: {len(knowledge_graph.entities)}")
    print(f"✓ Связей в графе: {len(knowledge_graph.relations)}")
    print(f"✓ Производителей из каталогов: {len(knowledge_graph.get_catalog_manufacturers())}")
    print(f"✓ Паттернов типов инструментов: {len(knowledge_graph.get_tool_type_patterns())}")
    
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
    tokenizer, model = load_model_and_tokenizer(device, torch_dtype)
    
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
    print("ШАГ 5: Токенизация датасета с CoT")
    print("=" * 70)
    tokenized_dataset = prepare_dataset(dataset, tokenizer)
    
    # Шаг 6: Настраиваем параметры обучения
    print("\n" + "=" * 70)
    print("ШАГ 6: Настройка параметров обучения")
    print("=" * 70)
    training_args = setup_training_arguments(device_type, output_dir)
    
    # Шаг 7: Обучаем модель
    print("\n" + "=" * 70)
    print("ШАГ 7: Обучение модели с CoT reasoning")
    print("=" * 70)
    model = train_model(model, tokenizer, tokenized_dataset, training_args)
    
    # Шаг 8: Сохраняем адаптеры
    print("\n" + "=" * 70)
    print("ШАГ 8: Сохранение LoRA адаптеров")
    print("=" * 70)
    save_lora_adapters(model, tokenizer, adapters_dir)
    
    # Вычисляем время обучения
    end_time = datetime.now()
    training_duration = end_time - start_time
    hours, remainder = divmod(training_duration.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Создаём файл-маркер о завершении обучения
    completion_marker = output_dir / "TRAINING_COMPLETED_COT.txt"
    with open(completion_marker, 'w', encoding='utf-8') as f:
        f.write(f"Обучение с CoT завершено: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Начало обучения: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Длительность: {int(hours)}ч {int(minutes)}м {int(seconds)}с\n")
        f.write(f"\nLoRA адаптеры: {adapters_dir}\n")
        f.write(f"Чекпоинты: {output_dir}\n")
        f.write(f"Граф знаний: {kg_file}\n")
    
    # Выводим заметное сообщение о завершении
    print("\n" + "🎉" * 35)
    print("\n" + " " * 15 + "ОБУЧЕНИЕ С CoT ЗАВЕРШЕНО!")
    print(" " * 10 + "TRAINING WITH CHAIN-OF-THOUGHT COMPLETED!")
    print("\n" + "🎉" * 35)
    print("\n" + "=" * 70)
    print("ИНФОРМАЦИЯ О ЗАВЕРШЕНИИ")
    print("=" * 70)
    print(f"Время начала:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Время окончания:  {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Длительность:     {int(hours)}ч {int(minutes)}м {int(seconds)}с")
    print("=" * 70)
    print("\n📁 СОХРАНЁННЫЕ ФАЙЛЫ:")
    print(f"   • LoRA адаптеры: {adapters_dir}")
    print(f"   • Чекпоинты: {output_dir}")
    print(f"   • Граф знаний: {kg_file}")
    print(f"   • Маркер завершения: {completion_marker}")
    print("\n📖 ИСПОЛЬЗОВАНИЕ:")
    print("   Для использования обученной модели загрузите базовую модель")
    print("   и примените адаптеры через PeftModel.from_pretrained()")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
