# Руководство по настройке параметров PPStructureV3

## Расположение параметров

Все параметры PPStructureV3 для распознавания таблиц находятся в файле:
```
src/utils/config.py
```

Раздел: **ПАРАМЕТРЫ PPStructureV3 ДЛЯ TABLE RECOGNITION** (строки 156-203)

---

## Группы параметров

### 1. Основные параметры

```python
PPSTRUCTURE_OCR_VERSION = "PP-OCRv5"  # Версия моделей
PPSTRUCTURE_LANG = "ru"               # Язык (eslav модель)
PPSTRUCTURE_DEVICE = "cpu"            # Устройство
```

### 2. Производительность

```python
PPSTRUCTURE_ENABLE_MKLDNN = True   # MKL-DNN ускорение
PPSTRUCTURE_CPU_THREADS = 8        # Потоки CPU
PPSTRUCTURE_ENABLE_HPI = False     # HPI плагин
PPSTRUCTURE_PRECISION = "fp32"     # Точность
```

### 3. Layout Detection

```python
PPSTRUCTURE_LAYOUT_MODEL_NAME = "RT-DETR-H_layout_3cls"
PPSTRUCTURE_LAYOUT_THRESHOLD = 0.7
PPSTRUCTURE_LAYOUT_NMS = True
```

### 4. Text Detection (влияет на пробелы!)

```python
PPSTRUCTURE_TEXT_DET_MODEL_NAME = "PP-OCRv5_server_det"
PPSTRUCTURE_TEXT_DET_THRESH = 0.3
PPSTRUCTURE_TEXT_DET_BOX_THRESH = 0.6
PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO = 1.2  # ⭐ Влияет на пробелы!
```

### 5. Text Recognition (влияет на пробелы!)

```python
PPSTRUCTURE_TEXT_REC_MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"  # ⭐ Критично!
PPSTRUCTURE_TEXT_REC_BATCH_SIZE = 1
PPSTRUCTURE_TEXT_REC_SCORE_THRESH = 0.6
```

### 6. Table Recognition

```python
PPSTRUCTURE_USE_TABLE_RECOGNITION = True
```

---

## Быстрые настройки

### Проблема: Слова склеиваются (нет пробелов)

```python
PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO = 1.0  # Уменьшить с 1.2
PPSTRUCTURE_TEXT_DET_BOX_THRESH = 0.5    # Снизить с 0.6
```

### Проблема: Лишние пробелы (слова разрываются)

```python
PPSTRUCTURE_TEXT_DET_UNCLIP_RATIO = 2.0  # Увеличить с 1.2
PPSTRUCTURE_TEXT_DET_BOX_THRESH = 0.65   # Повысить с 0.6
```

### Проблема: Английские слова без пробелов

⚠️ ПРОВЕРЬТЕ:
```python
PPSTRUCTURE_LANG = "ru"  # ✅ Должно быть "ru" (eslav модель)
PPSTRUCTURE_TEXT_REC_MODEL_NAME = "eslav_PP-OCRv5_mobile_rec"  # ✅ Должна быть eslav
```

❌ НЕ используйте:
```python
PPSTRUCTURE_TEXT_REC_MODEL_NAME = "cyrillic_PP-OCRv3_mobile_rec"  # НЕТ английского!
```

### Ускорение на CPU

```python
PPSTRUCTURE_CPU_THREADS = 16  # Увеличить (по кол-ву ядер)
PPSTRUCTURE_ENABLE_MKLDNN = True  # Обязательно
```

### Максимальная точность

```python
PPSTRUCTURE_LAYOUT_MODEL_NAME = "RT-DETR-H_layout_3cls"  # Уже высокая точность
PPSTRUCTURE_TEXT_DET_MODEL_NAME = "PP-OCRv5_server_det"  # Server модель
PPSTRUCTURE_TEXT_REC_SCORE_THRESH = 0.7  # Выше порог
```

### Максимальная скорость

```python
PPSTRUCTURE_LAYOUT_MODEL_NAME = "PicoDet_layout_1x_table"  # Легковесная
PPSTRUCTURE_TEXT_DET_MODEL_NAME = "PP-OCRv5_mobile_det"    # Mobile модель
PPSTRUCTURE_TEXT_REC_SCORE_THRESH = 0.5  # Ниже порог
```

---

## Альтернативные модели

### Layout Detection:

| Модель | Точность | Скорость | Когда использовать |
|--------|----------|----------|-------------------|
| `RT-DETR-H_layout_3cls` | ⭐⭐⭐ | ⚡ | Максимальная точность |
| `PicoDet-L_layout_3cls` | ⭐⭐ | ⚡⚡ | Баланс |
| `PicoDet-S_layout_3cls` | ⭐ | ⚡⚡⚡ | Скорость |
| `PicoDet_layout_1x_table` | ⭐⭐ | ⚡⚡⚡ | Только таблицы |

### Text Detection:

| Модель | Точность | Скорость CPU | Когда использовать |
|--------|----------|--------------|-------------------|
| `PP-OCRv5_server_det` | ⭐⭐⭐ | ⚡ (медленно) | Высокая точность |
| `PP-OCRv5_mobile_det` | ⭐⭐ | ⚡⚡⚡ (быстро) | CPU оптимизация |

### Text Recognition:

| Модель | Языки | Точность | Когда использовать |
|--------|-------|----------|-------------------|
| `eslav_PP-OCRv5_mobile_rec` | RU+EN+цифры | 81.6% | ⭐ Русский+Английский |
| `en_PP-OCRv5_mobile_rec` | EN+цифры | 85.25% | Только английский |
| `latin_PP-OCRv5_mobile_rec` | Латиница+цифры | 84.7% | Европейские языки |
| `cyrillic_PP-OCRv3_mobile_rec` | Только RU+цифры | 70% | ❌ НЕ использовать |

---

## Применение изменений

После изменения параметров в `config.py`:

1. Сохраните файл
2. Перезапустите приложение
3. Изменения применятся автоматически

Логирование покажет используемые параметры при инициализации.

---

## Параметры специфичные для режима

Эти параметры НЕ в константах (различаются по режимам):

### Режим "paddleocr":
```python
use_doc_orientation_classify=USE_PADDLEOCR_DOC_ORIENTATION  # Из конфига
use_doc_unwarping=USE_PADDLEOCR_DOC_UNWARPING              # Из конфига
use_textline_orientation=True                               # Фиксировано
```

### Режим "custom":
```python
use_doc_orientation_classify=False  # Фиксировано
use_doc_unwarping=False             # Фиксировано
use_textline_orientation=False      # Фиксировано
```

---

## См. также

- `docs/PPStructureV3_Parameters_Guide.md` - полная документация всех параметров
- `config/logging_config.json` - настройка логирования

