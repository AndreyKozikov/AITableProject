"""
Скрипт для загрузки модели Qwen с HuggingFace Hub.

Скрипт проверяет наличие модели в локальном кэше и загружает её,
если она отсутствует. Использует настройки из config.py для определения
модели и директории для загрузки.
"""

import os
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from src.utils.config import MODEL_ID, MODEL_CACHE_DIR


def check_model_exists(cache_dir):
    """
    Проверяет наличие модели в локальной директории.
    
    Модель считается загруженной, если в директории существуют
    необходимые файлы: config.json и хотя бы один файл весов.
    
    Args:
        cache_dir: Путь к директории с моделью
        
    Returns:
        True если модель уже загружена, False иначе
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print(f"Директория модели не существует: {cache_path}")
        return False
    
    # Проверяем наличие ключевых файлов модели
    config_file = cache_path / "config.json"
    
    # Ищем файлы весов (могут быть в разных форматах)
    weight_patterns = [
        "*.bin",  # PyTorch binary format
        "*.safetensors",  # SafeTensors format
        "pytorch_model*.bin",  # Multi-part models
        "model*.safetensors"  # Multi-part safetensors
    ]
    
    has_weights = False
    for pattern in weight_patterns:
        if list(cache_path.glob(pattern)):
            has_weights = True
            break
    
    # Проверяем наличие tokenizer файлов
    tokenizer_file = cache_path / "tokenizer_config.json"
    
    if config_file.exists() and has_weights and tokenizer_file.exists():
        print(f"✓ Модель найдена в локальной директории: {cache_path}")
        return True
    else:
        print(f"Модель не найдена или неполная в: {cache_path}")
        if not config_file.exists():
            print("  - Отсутствует config.json")
        if not has_weights:
            print("  - Отсутствуют файлы весов")
        if not tokenizer_file.exists():
            print("  - Отсутствует tokenizer_config.json")
        return False


def download_model(model_id, cache_dir):
    """
    Загружает модель и токенизатор с HuggingFace Hub.
    
    Args:
        model_id: ID модели на HuggingFace Hub (например, "Qwen/Qwen2.5-1.5B")
        cache_dir: Директория для сохранения модели
        
    Returns:
        Кортеж (tokenizer, model) или (None, None) в случае ошибки
    """
    print("=" * 70)
    print(f"ЗАГРУЗКА МОДЕЛИ: {model_id}")
    print("=" * 70)
    print(f"Целевая директория: {cache_dir}")
    
    # Создаем директорию если не существует
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Загружаем токенизатор
        print("\nШаг 1/2: Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        print("✓ Токенизатор загружен успешно")
        
        # Загружаем модель
        print("\nШаг 2/2: Загрузка модели (это может занять несколько минут)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        print("✓ Модель загружена успешно")
        
        # Сохраняем в удобном формате
        print("\nСохранение модели в локальную директорию...")
        model.save_pretrained(cache_dir)
        tokenizer.save_pretrained(cache_dir)
        print(f"✓ Модель сохранена в: {cache_dir}")
        
        print("\n" + "=" * 70)
        print("ЗАГРУЗКА ЗАВЕРШЕНА УСПЕШНО")
        print("=" * 70)
        
        return tokenizer, model
        
    except Exception as e:
        print(f"\n{'='*70}")
        print("ОШИБКА ПРИ ЗАГРУЗКЕ МОДЕЛИ")
        print(f"{'='*70}")
        print(f"Ошибка: {e}")
        print(f"\nВозможные причины:")
        print(f"1. Отсутствует интернет-соединение")
        print(f"2. Неверный ID модели: {model_id}")
        print(f"3. Недостаточно места на диске")
        print(f"4. Требуется авторизация на HuggingFace (для закрытых моделей)")
        print(f"\nПроверьте доступность модели: https://huggingface.co/{model_id}")
        return None, None


def get_model_size_info(cache_dir):
    """
    Вычисляет размер загруженной модели.
    
    Args:
        cache_dir: Путь к директории с моделью
        
    Returns:
        Строка с информацией о размере или None
    """
    cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return None
    
    total_size = 0
    file_count = 0
    
    for file in cache_path.rglob("*"):
        if file.is_file():
            total_size += file.stat().st_size
            file_count += 1
    
    # Конвертируем в читаемый формат
    if total_size < 1024:
        size_str = f"{total_size} B"
    elif total_size < 1024 ** 2:
        size_str = f"{total_size / 1024:.2f} KB"
    elif total_size < 1024 ** 3:
        size_str = f"{total_size / (1024 ** 2):.2f} MB"
    else:
        size_str = f"{total_size / (1024 ** 3):.2f} GB"
    
    return f"{size_str} ({file_count} файлов)"


def main():
    """
    Главная функция скрипта.
    """
    print("=" * 70)
    print("СКРИПТ ЗАГРУЗКИ МОДЕЛИ QWEN")
    print("=" * 70)
    print(f"ID модели: {MODEL_ID}")
    print(f"Директория: {MODEL_CACHE_DIR}")
    print()
    
    # Проверяем наличие модели
    print("Проверка наличия модели в локальном кэше...")
    
    if check_model_exists(MODEL_CACHE_DIR):
        print("\n✓ Модель уже загружена!")
        
        # Показываем информацию о размере
        size_info = get_model_size_info(MODEL_CACHE_DIR)
        if size_info:
            print(f"Размер модели: {size_info}")
        
        print("\nДля повторной загрузки удалите директорию:")
        print(f"  {MODEL_CACHE_DIR}")
        return
    
    # Загружаем модель
    print("\nМодель не найдена. Начинаем загрузку...")
    print("(Это может занять несколько минут в зависимости от скорости интернета)")
    print()
    
    tokenizer, model = download_model(MODEL_ID, MODEL_CACHE_DIR)
    
    if tokenizer is not None and model is not None:
        # Показываем информацию о размере
        size_info = get_model_size_info(MODEL_CACHE_DIR)
        if size_info:
            print(f"\nРазмер загруженной модели: {size_info}")
        
        print("\nМодель готова к использованию!")
        print(f"Обновите USE_LOCAL_MODEL = True в config.py для использования локальной модели")
    else:
        print("\nЗагрузка модели не удалась. Проверьте логи выше для деталей.")


if __name__ == "__main__":
    main()

