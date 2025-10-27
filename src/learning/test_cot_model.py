"""
Скрипт для тестирования модели с Chain-of-Thought reasoning.

Позволяет проверить качество работы обученной модели на тестовых данных.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

import sys
from pathlib import Path

# Добавляем путь к src в sys.path для импорта
sys.path.append(str(Path(__file__).parent.parent))

from mapper.ask_qwen3_cot import Qwen3CoTModel


def load_test_data(test_file: Path) -> List[Dict[str, Any]]:
    """Загрузить тестовые данные из файла."""
    test_data = []
    
    if test_file.suffix == '.json':
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    elif test_file.suffix == '.xlsx':
        df = pd.read_excel(test_file)
        for _, row in df.iterrows():
            test_data.append({
                "input": str(row.get("input", "")),
                "expected": {
                    "simplified": {
                        "Наименование": str(row.get("simplified_name", "")),
                        "Единица измерения": str(row.get("simplified_unit", "шт.")),
                        "Количество": str(row.get("simplified_quantity", "")),
                        "Техническое задание": str(row.get("simplified_spec", ""))
                    },
                    "extended": {
                        "Обозначение": str(row.get("extended_denotation", "")),
                        "Наименование": str(row.get("extended_name", "")),
                        "Производитель": str(row.get("extended_manufacturer", "")),
                        "Единица измерения": str(row.get("extended_unit", "шт.")),
                        "Количество": str(row.get("extended_quantity", "")),
                        "Техническое задание": str(row.get("extended_spec", ""))
                    }
                }
            })
    
    return test_data


def calculate_accuracy(predicted: Dict, expected: Dict) -> float:
    """Вычислить точность предсказания."""
    if not predicted or not expected:
        return 0.0
    
    total_fields = len(expected)
    correct_fields = 0
    
    for key, expected_value in expected.items():
        predicted_value = predicted.get(key, "")
        
        # Нормализуем значения для сравнения
        expected_norm = str(expected_value).strip().lower()
        predicted_norm = str(predicted_value).strip().lower()
        
        if expected_norm == predicted_norm:
            correct_fields += 1
        elif expected_norm in predicted_norm or predicted_norm in expected_norm:
            # Частичное совпадение
            correct_fields += 0.5
    
    return correct_fields / total_fields if total_fields > 0 else 0.0


def test_model_on_data(model: Qwen3CoTModel, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Тестировать модель на данных."""
    results = {
        "total_tests": len(test_data),
        "simplified_results": [],
        "extended_results": [],
        "overall_accuracy": 0.0
    }
    
    print(f"Тестирование модели на {len(test_data)} примерах...")
    
    for i, test_case in enumerate(test_data):
        print(f"\n--- Тест {i+1}/{len(test_data)} ---")
        print(f"Входной текст: {test_case['input']}")
        
        # Тестируем simplified режим
        if "simplified" in test_case.get("expected", {}):
            print("\n🔍 Тестирование simplified режима...")
            simplified_result = model.ask_qwen3_cot(test_case["input"], mode="simplified")
            
            if simplified_result["success"]:
                predicted = simplified_result["result"].get("rows", [{}])[0] if simplified_result["result"].get("rows") else {}
                expected = test_case["expected"]["simplified"]
                accuracy = calculate_accuracy(predicted, expected)
                
                results["simplified_results"].append({
                    "test_id": i+1,
                    "input": test_case["input"],
                    "predicted": predicted,
                    "expected": expected,
                    "accuracy": accuracy,
                    "reasoning": simplified_result.get("reasoning", "")
                })
                
                print(f"✓ Точность: {accuracy:.2f}")
                print(f"Предсказано: {predicted}")
                print(f"Ожидалось: {expected}")
            else:
                print(f"❌ Ошибка: {simplified_result['error']}")
                results["simplified_results"].append({
                    "test_id": i+1,
                    "input": test_case["input"],
                    "error": simplified_result["error"],
                    "accuracy": 0.0
                })
        
        # Тестируем extended режим
        if "extended" in test_case.get("expected", {}):
            print("\n🔍 Тестирование extended режима...")
            extended_result = model.ask_qwen3_cot(test_case["input"], mode="extended")
            
            if extended_result["success"]:
                predicted = extended_result["result"].get("rows", [{}])[0] if extended_result["result"].get("rows") else {}
                expected = test_case["expected"]["extended"]
                accuracy = calculate_accuracy(predicted, expected)
                
                results["extended_results"].append({
                    "test_id": i+1,
                    "input": test_case["input"],
                    "predicted": predicted,
                    "expected": expected,
                    "accuracy": accuracy,
                    "reasoning": extended_result.get("reasoning", "")
                })
                
                print(f"✓ Точность: {accuracy:.2f}")
                print(f"Предсказано: {predicted}")
                print(f"Ожидалось: {expected}")
            else:
                print(f"❌ Ошибка: {extended_result['error']}")
                results["extended_results"].append({
                    "test_id": i+1,
                    "input": test_case["input"],
                    "error": extended_result["error"],
                    "accuracy": 0.0
                })
    
    # Вычисляем общую точность
    all_accuracies = []
    for result in results["simplified_results"]:
        all_accuracies.append(result["accuracy"])
    for result in results["extended_results"]:
        all_accuracies.append(result["accuracy"])
    
    if all_accuracies:
        results["overall_accuracy"] = sum(all_accuracies) / len(all_accuracies)
    
    return results


def save_test_results(results: Dict[str, Any], output_file: Path):
    """Сохранить результаты тестирования."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n📁 Результаты сохранены в: {output_file}")


def print_test_summary(results: Dict[str, Any]):
    """Вывести сводку результатов тестирования."""
    print("\n" + "=" * 70)
    print("СВОДКА РЕЗУЛЬТАТОВ ТЕСТИРОВАНИЯ")
    print("=" * 70)
    
    print(f"Всего тестов: {results['total_tests']}")
    print(f"Общая точность: {results['overall_accuracy']:.2f}")
    
    # Simplified режим
    if results["simplified_results"]:
        simplified_acc = [r["accuracy"] for r in results["simplified_results"]]
        print(f"\nSimplified режим:")
        print(f"  Тестов: {len(simplified_acc)}")
        print(f"  Средняя точность: {sum(simplified_acc)/len(simplified_acc):.2f}")
        print(f"  Лучший результат: {max(simplified_acc):.2f}")
        print(f"  Худший результат: {min(simplified_acc):.2f}")
    
    # Extended режим
    if results["extended_results"]:
        extended_acc = [r["accuracy"] for r in results["extended_results"]]
        print(f"\nExtended режим:")
        print(f"  Тестов: {len(extended_acc)}")
        print(f"  Средняя точность: {sum(extended_acc)/len(extended_acc):.2f}")
        print(f"  Лучший результат: {max(extended_acc):.2f}")
        print(f"  Худший результат: {min(extended_acc):.2f}")
    
    # Топ-3 лучших результата
    all_results = results["simplified_results"] + results["extended_results"]
    top_results = sorted(all_results, key=lambda x: x.get("accuracy", 0), reverse=True)[:3]
    
    print(f"\n🏆 Топ-3 лучших результата:")
    for i, result in enumerate(top_results, 1):
        print(f"  {i}. Точность: {result.get('accuracy', 0):.2f}")
        print(f"     Вход: {result['input']}")
        if 'predicted' in result:
            print(f"     Предсказано: {result['predicted']}")
    
    print("=" * 70)


def main():
    """Главная функция тестирования."""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ МОДЕЛИ С CHAIN-OF-THOUGHT REASONING")
    print("=" * 70)
    
    # Определяем пути
    script_dir = Path(__file__).parent
    test_data_file = script_dir / "test_data.json"  # Создайте этот файл с тестовыми данными
    output_file = script_dir / "test_results.json"
    
    # Проверяем наличие тестовых данных
    if not test_data_file.exists():
        print(f"❌ Файл тестовых данных не найден: {test_data_file}")
        print("Создайте файл test_data.json с тестовыми данными")
        return
    
    # Загружаем тестовые данные
    print(f"Загрузка тестовых данных из: {test_data_file}")
    test_data = load_test_data(test_data_file)
    print(f"✓ Загружено {len(test_data)} тестовых примеров")
    
    # Инициализируем модель
    print("\nИнициализация модели...")
    try:
        model = Qwen3CoTModel()
        print("✓ Модель инициализирована")
    except Exception as e:
        print(f"❌ Ошибка инициализации модели: {e}")
        return
    
    # Запускаем тестирование
    print("\nЗапуск тестирования...")
    results = test_model_on_data(model, test_data)
    
    # Сохраняем результаты
    save_test_results(results, output_file)
    
    # Выводим сводку
    print_test_summary(results)


if __name__ == "__main__":
    main()
