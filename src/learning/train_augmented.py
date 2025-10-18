import random
import json

def augment_example(example, n_aug=3):
    """Создаём n_aug аугментированных примеров из одного"""
    input_data = example["input"].copy()
    output_data = example["output"].copy()
    augmented = []

    for _ in range(n_aug):
        new_input = input_data.copy()

        # 1. Перемешаем случайно часть элементов (кроме наименования)
        if len(new_input) > 3:
            middle = new_input[1:-1]
            random.shuffle(middle)
            new_input = [new_input[0]] + middle + [new_input[-1]]

        # 2. Вариации единиц измерения
        units_variants = ["шт.", "штука", "pcs", "ед."]
        if random.random() < 0.3:
            output_data["Единица измерения"] = random.choice(units_variants)

        # 3. Иногда убираем одно случайное поле во входе
        if random.random() < 0.2 and len(new_input) > 2:
            idx = random.randint(0, len(new_input) - 1)
            new_input[idx] = None

        augmented.append({
            "input": new_input,
            "output": output_data.copy()
        })

    return augmented


# Пример использования
file_path = "datasets/data/train_augmented.jsonl"
dataset = []
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        dataset.append(json.loads(line))

augmented_dataset = []
for ex in dataset:
    augmented_dataset.append(ex)  # оригинал
    augmented_dataset.extend(augment_example(ex, n_aug=3))  # +3 новых

print(f"Было: {len(dataset)}, стало: {len(augmented_dataset)}")

# Сохраняем
with open("./data/train_augmented_1.jsonl", "w", encoding="utf-8") as f:
    for row in augmented_dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")