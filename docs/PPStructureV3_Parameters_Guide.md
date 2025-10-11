# Руководство по параметрам PPStructureV3

## Содержание
1. [Общие параметры](#общие-параметры)
2. [Параметры Layout Detection (обнаружение макета)](#параметры-layout-detection)
3. [Параметры Chart Recognition (распознавание диаграмм)](#параметры-chart-recognition)
4. [Параметры Region Detection (обнаружение регионов)](#параметры-region-detection)
5. [Параметры Document Preprocessing (предобработка документов)](#параметры-document-preprocessing)
6. [Параметры Text Detection (обнаружение текста)](#параметры-text-detection)
7. [Параметры Text Recognition (распознавание текста)](#параметры-text-recognition)
8. [Параметры Table Recognition (распознавание таблиц)](#параметры-table-recognition)
9. [Параметры Seal Recognition (распознавание печатей)](#параметры-seal-recognition)
10. [Параметры Formula Recognition (распознавание формул)](#параметры-formula-recognition)
11. [Параметры производительности](#параметры-производительности)
12. [Примеры использования](#примеры-использования)

---

## Общие параметры

### `lang`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Язык для текстового распознавания
- **Значения**: 
  - `"ch"` - китайский (по умолчанию)
  - `"en"` - английский
  - `"ru"` - русский
  - и др. (80+ языков)
- **Пример**:
```python
pipeline = PPStructureV3(lang="ru")
```

### `ocr_version`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Версия OCR моделей
- **Значения**: `"PP-OCRv4"`, `"PP-OCRv5"` и др.
- **Пример**:
```python
pipeline = PPStructureV3(ocr_version="PP-OCRv5")
```

### `device`
- **Тип**: `str | None`
- **По умолчанию**: `None` (автоматически выбирается GPU:0 или CPU)
- **Описание**: Устройство для инференса
- **Значения**: 
  - `"cpu"` - процессор
  - `"gpu:0"`, `"gpu:1"` - GPU с номером
  - `"npu:0"` - NPU (Neural Processing Unit)
  - `"xpu:0"` - XPU
  - `"mlu:0"` - MLU (Machine Learning Unit)
  - `"dcu:0"` - DCU
- **Пример**:
```python
pipeline = PPStructureV3(device="cpu")
```

---

## Параметры Layout Detection

Layout Detection определяет структуру документа (заголовки, таблицы, изображения, текст и т.д.)

### `layout_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None` (используется модель по умолчанию)
- **Описание**: Имя модели для обнаружения макета
- **Доступные модели**:
  - `"PicoDet-S_layout_3cls"` - легковесная, 3 класса (table, image, stamp)
  - `"PicoDet-L_layout_3cls"` - баланс точности и скорости
  - `"RT-DETR-H_layout_3cls"` - высокая точность
  - `"PicoDet-S_layout_17cls"` - 17 классов документов
  - `"PicoDet_layout_1x_table"` - специализированная для таблиц
  - `"PP-DocLayout-S"` - эффективная
  - `"PP-DocLayout-M"` - средняя
  - `"PP-DocLayout_plus-L"` - высокая точность

### `layout_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к директории с кастомной моделью layout detection
- **Пример**:
```python
pipeline = PPStructureV3(
    layout_detection_model_name="PicoDet-S_layout_3cls",
    layout_detection_model_dir="./models/custom_layout_det"
)
```

### `layout_threshold`
- **Тип**: `float | dict | None`
- **По умолчанию**: `None`
- **Описание**: Порог уверенности для обнаружения элементов макета
- **Диапазон**: 0.0 - 1.0 (обычно 0.5-0.7)
- **Пример**:
```python
# Единый порог для всех классов
pipeline = PPStructureV3(layout_threshold=0.6)

# Разные пороги для разных классов
pipeline = PPStructureV3(layout_threshold={
    "table": 0.7,
    "figure": 0.5,
    "title": 0.6
})
```

### `layout_nms`
- **Тип**: `bool | None`
- **По умолчанию**: `None` (True)
- **Описание**: Применять ли Non-Maximum Suppression для удаления перекрывающихся bounding boxes
- **Пример**:
```python
pipeline = PPStructureV3(layout_nms=True)
```

### `layout_unclip_ratio`
- **Тип**: `float | tuple[float, float] | dict | None`
- **По умолчанию**: `None`
- **Описание**: Коэффициент расширения bounding boxes для layout detection
- **Диапазон**: обычно 1.0 - 1.5
- **Пример**:
```python
# Единый коэффициент
pipeline = PPStructureV3(layout_unclip_ratio=1.2)

# Разные коэффициенты для ширины и высоты
pipeline = PPStructureV3(layout_unclip_ratio=(1.1, 1.3))

# Разные коэффициенты для разных классов
pipeline = PPStructureV3(layout_unclip_ratio={
    "table": 1.3,
    "figure": 1.1
})
```

### `layout_merge_bboxes_mode`
- **Тип**: `str | dict | None`
- **По умолчанию**: `None`
- **Описание**: Режим слияния перекрывающихся bounding boxes
- **Значения**: 
  - `"small"` - удалить маленькие перекрывающиеся боксы
  - `"large"` - удалить большие перекрывающиеся боксы
  - `None` - не сливать
- **Пример**:
```python
pipeline = PPStructureV3(layout_merge_bboxes_mode="small")
```

---

## Параметры Chart Recognition

### `chart_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания диаграмм

### `chart_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания диаграмм

### `chart_recognition_batch_size`
- **Тип**: `int | None`
- **По умолчанию**: `None` (1)
- **Описание**: Размер батча для обработки диаграмм
- **Пример**:
```python
pipeline = PPStructureV3(
    use_chart_recognition=True,
    chart_recognition_batch_size=4
)
```

---

## Параметры Region Detection

### `region_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для обнаружения регионов документа

### `region_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели обнаружения регионов

---

## Параметры Document Preprocessing

### `doc_orientation_classify_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для классификации ориентации документа
- **Модели**: `"PP-LCNet_x1_0_doc_ori"` (классифицирует 0°, 90°, 180°, 270°)

### `doc_orientation_classify_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели классификации ориентации

### `doc_unwarping_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для исправления геометрических искажений
- **Модели**: `"UVDoc"` (исправление деформаций документа)

### `doc_unwarping_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели исправления искажений

### `use_doc_orientation_classify`
- **Тип**: `bool | None`
- **По умолчанию**: `False`
- **Описание**: Использовать ли классификацию ориентации документа
- **Пример**:
```python
pipeline = PPStructureV3(use_doc_orientation_classify=True)
```

### `use_doc_unwarping`
- **Тип**: `bool | None`
- **По умолчанию**: `False`
- **Описание**: Использовать ли исправление геометрических искажений
- **Пример**:
```python
pipeline = PPStructureV3(use_doc_unwarping=True)
```

---

## Параметры Text Detection

Text Detection обнаруживает области текста на изображении.

### `text_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для обнаружения текста
- **Модели**:
  - `"PP-OCRv4_server_det"` - серверная версия, высокая точность
  - `"PP-OCRv4_mobile_det"` - мобильная версия, быстрая
  - `"PP-OCRv5_server_det"` - последняя серверная версия
  - `"PP-OCRv5_mobile_det"` - последняя мобильная версия

### `text_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели обнаружения текста

### `text_det_limit_side_len`
- **Тип**: `int | None`
- **По умолчанию**: `None` (960 для общего OCR)
- **Описание**: Ограничение длины стороны изображения для text detection
- **Рекомендации**: 
  - 960 - для общего OCR
  - 736 - для table detection
- **Пример**:
```python
pipeline = PPStructureV3(text_det_limit_side_len=960)
```

### `text_det_limit_type`
- **Тип**: `str | None`
- **По умолчанию**: `None` (`"max"` для общего OCR)
- **Описание**: Тип ограничения стороны изображения
- **Значения**:
  - `"min"` - минимальная сторона не меньше `text_det_limit_side_len`
  - `"max"` - максимальная сторона не больше `text_det_limit_side_len`
- **Пример**:
```python
pipeline = PPStructureV3(
    text_det_limit_side_len=960,
    text_det_limit_type="max"
)
```

### `text_det_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.3)
- **Описание**: Порог для определения пикселей текста в probability map
- **Диапазон**: 0.0 - 1.0
- **Пояснение**: Пиксели со score > `text_det_thresh` считаются текстовыми
- **Пример**:
```python
pipeline = PPStructureV3(text_det_thresh=0.3)
```

### `text_det_box_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.6)
- **Описание**: Порог для определения текстовых регионов
- **Диапазон**: 0.0 - 1.0
- **Пояснение**: Bounding box считается текстовым, если средний score всех пикселей > `text_det_box_thresh`
- **Пример**:
```python
pipeline = PPStructureV3(text_det_box_thresh=0.6)
```

### `text_det_unclip_ratio`
- **Тип**: `float | None`
- **По умолчанию**: `None` (1.5)
- **Описание**: Коэффициент расширения текстовых областей (Vatti clipping algorithm)
- **Диапазон**: обычно 1.0 - 2.0
- **Пояснение**: Увеличивает размер bounding box для захвата всего текста
- **Пример**:
```python
pipeline = PPStructureV3(text_det_unclip_ratio=1.5)
```

---

## Параметры Text Recognition

Text Recognition распознает текст в обнаруженных областях.

### `text_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания текста
- **Модели**:
  - `"PP-OCRv4_server_rec"` - серверная версия
  - `"PP-OCRv4_mobile_rec"` - мобильная версия
  - `"PP-OCRv5_server_rec"` - последняя серверная
  - `"PP-OCRv5_mobile_rec"` - последняя мобильная
  - `"en_PP-OCRv4_mobile_rec"` - для английского текста
- **Пример**:
```python
# Для английского текста
pipeline = PPStructureV3(
    lang="en",
    text_recognition_model_name="en_PP-OCRv4_mobile_rec"
)
```

### `text_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания текста

### `text_recognition_batch_size`
- **Тип**: `int | None`
- **По умолчанию**: `None` (1)
- **Описание**: Размер батча для распознавания текста
- **Рекомендации**: увеличить для ускорения на GPU
- **Пример**:
```python
pipeline = PPStructureV3(text_recognition_batch_size=8)
```

### `text_rec_score_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.0 - без фильтрации)
- **Описание**: Минимальный порог уверенности для сохранения результатов распознавания
- **Диапазон**: 0.0 - 1.0
- **Пример**:
```python
pipeline = PPStructureV3(text_rec_score_thresh=0.5)
```

---

## Параметры Textline Orientation

### `textline_orientation_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для определения ориентации текстовых линий

### `textline_orientation_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели определения ориентации

### `textline_orientation_batch_size`
- **Тип**: `int | None`
- **По умолчанию**: `None` (1)
- **Описание**: Размер батча для определения ориентации текстовых линий

### `use_textline_orientation`
- **Тип**: `bool | None`
- **По умолчанию**: `None` (False)
- **Описание**: Использовать ли модуль определения ориентации текстовых линий
- **Пример**:
```python
pipeline = PPStructureV3(use_textline_orientation=True)
```

---

## Параметры Table Recognition

### `table_classification_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для классификации таблиц (wired/wireless)

### `table_classification_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели классификации таблиц

### `wired_table_structure_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания структуры таблиц с линиями
- **Модели**: `"SLANet"`, `"SLANet_plus"`

### `wired_table_structure_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания таблиц с линиями

### `wireless_table_structure_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания структуры таблиц без линий

### `wireless_table_structure_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания таблиц без линий

### `wired_table_cells_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для обнаружения ячеек в таблицах с линиями

### `wired_table_cells_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели обнаружения ячеек (wired tables)

### `wireless_table_cells_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для обнаружения ячеек в таблицах без линий

### `wireless_table_cells_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели обнаружения ячеек (wireless tables)

### `table_orientation_classify_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для классификации ориентации таблиц

### `table_orientation_classify_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели классификации ориентации таблиц

### `use_table_recognition`
- **Тип**: `bool | None`
- **По умолчанию**: `None` (True в PP-StructureV3)
- **Описание**: Использовать ли модуль распознавания таблиц
- **Пример**:
```python
pipeline = PPStructureV3(use_table_recognition=True)
```

---

## Параметры Seal Recognition

Seal Recognition используется для распознавания текста на печатях/штампах.

### `seal_text_detection_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для обнаружения текста на печатях
- **Модели**:
  - `"PP-OCRv4_server_seal_det"` - серверная версия, высокая точность
  - `"PP-OCRv4_mobile_seal_det"` - мобильная версия

### `seal_text_detection_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели обнаружения текста на печатях

### `seal_det_limit_side_len`
- **Тип**: `int | None`
- **По умолчанию**: `None` (736)
- **Описание**: Ограничение длины стороны изображения для seal detection
- **Пример**:
```python
pipeline = PPStructureV3(seal_det_limit_side_len=736)
```

### `seal_det_limit_type`
- **Тип**: `str | None`
- **По умолчанию**: `None` (`"min"`)
- **Описание**: Тип ограничения стороны изображения для печатей
- **Значения**: `"min"`, `"max"`
- **Пример**:
```python
pipeline = PPStructureV3(seal_det_limit_type="min")
```

### `seal_det_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.2)
- **Описание**: Порог для определения пикселей текста на печатях
- **Диапазон**: 0.0 - 1.0
- **Пример**:
```python
pipeline = PPStructureV3(seal_det_thresh=0.2)
```

### `seal_det_box_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.6)
- **Описание**: Порог для определения текстовых регионов на печатях
- **Диапазон**: 0.0 - 1.0
- **Пример**:
```python
pipeline = PPStructureV3(seal_det_box_thresh=0.6)
```

### `seal_det_unclip_ratio`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.5 для печатей)
- **Описание**: Коэффициент расширения текстовых областей на печатях
- **Диапазон**: 0.3 - 1.0 (меньше, чем для обычного текста)
- **Пример**:
```python
pipeline = PPStructureV3(seal_det_unclip_ratio=0.5)
```

### `seal_text_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания текста на печатях

### `seal_text_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания текста на печатях

### `seal_text_recognition_batch_size`
- **Тип**: `int | None`
- **По умолчанию**: `None` (1)
- **Описание**: Размер батча для распознавания текста на печатях
- **Пример**:
```python
pipeline = PPStructureV3(seal_text_recognition_batch_size=4)
```

### `seal_rec_score_thresh`
- **Тип**: `float | None`
- **По умолчанию**: `None` (0.0)
- **Описание**: Минимальный порог уверенности для результатов распознавания печатей
- **Диапазон**: 0.0 - 1.0
- **Пример**:
```python
pipeline = PPStructureV3(seal_rec_score_thresh=0.5)
```

### `use_seal_recognition`
- **Тип**: `bool | None`
- **По умолчанию**: `None` (False)
- **Описание**: Использовать ли модуль распознавания печатей
- **Пример**:
```python
pipeline = PPStructureV3(use_seal_recognition=True)
```

---

## Параметры Formula Recognition

### `formula_recognition_model_name`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Имя модели для распознавания математических формул

### `formula_recognition_model_dir`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к модели распознавания формул

### `formula_recognition_batch_size`
- **Тип**: `int | None`
- **По умолчанию**: `None` (1)
- **Описание**: Размер батча для распознавания формул
- **Пример**:
```python
pipeline = PPStructureV3(
    use_formula_recognition=True,
    formula_recognition_batch_size=2
)
```

### `use_formula_recognition`
- **Тип**: `bool | None`
- **По умолчанию**: `None` (True в PP-StructureV3)
- **Описание**: Использовать ли модуль распознавания формул
- **Пример**:
```python
pipeline = PPStructureV3(use_formula_recognition=True)
```

---

## Параметры производительности

### `enable_hpi`
- **Тип**: `bool`
- **По умолчанию**: `False`
- **Описание**: Включить High Performance Inference (HPI)
- **Пояснение**: Ускоряет инференс за счет оптимизаций
- **Пример**:
```python
pipeline = PPStructureV3(enable_hpi=True)
```

### `use_tensorrt`
- **Тип**: `bool`
- **По умолчанию**: `False`
- **Описание**: Использовать ли TensorRT для ускорения на GPU
- **Требования**: NVIDIA GPU, установленный TensorRT
- **Пример**:
```python
pipeline = PPStructureV3(
    device="gpu:0",
    use_tensorrt=True
)
```

### `precision`
- **Тип**: `str`
- **По умолчанию**: `"fp32"`
- **Описание**: Точность вычислений
- **Значения**:
  - `"fp32"` - 32-битная точность (максимальная точность)
  - `"fp16"` - 16-битная точность (быстрее, меньше памяти)
  - `"int8"` - 8-битная квантизация (самая быстрая)
- **Пример**:
```python
pipeline = PPStructureV3(
    device="gpu:0",
    precision="fp16"
)
```

### `enable_mkldnn`
- **Тип**: `bool`
- **По умолчанию**: `True`
- **Описание**: Включить ускорение MKL-DNN для CPU
- **Пояснение**: Intel Math Kernel Library - Deep Neural Networks
- **Пример**:
```python
pipeline = PPStructureV3(
    device="cpu",
    enable_mkldnn=True
)
```

### `mkldnn_cache_capacity`
- **Тип**: `int`
- **По умолчанию**: `10`
- **Описание**: Размер кэша MKL-DNN
- **Диапазон**: 1 - 100
- **Пример**:
```python
pipeline = PPStructureV3(
    enable_mkldnn=True,
    mkldnn_cache_capacity=20
)
```

### `cpu_threads`
- **Тип**: `int`
- **По умолчанию**: `8`
- **Описание**: Количество потоков CPU для инференса
- **Рекомендации**: установить равным количеству физических ядер
- **Пример**:
```python
pipeline = PPStructureV3(cpu_threads=16)
```

### `paddlex_config`
- **Тип**: `str | None`
- **По умолчанию**: `None`
- **Описание**: Путь к файлу конфигурации YAML
- **Пояснение**: Позволяет загрузить все параметры из файла
- **Пример**:
```python
pipeline = PPStructureV3(paddlex_config="PP-StructureV3.yaml")
```

---

## Параметры модулей (флаги)

### `use_region_detection`
- **Тип**: `bool | None`
- **По умолчанию**: `None`
- **Описание**: Использовать ли обнаружение регионов документа

### `use_chart_recognition`
- **Тип**: `bool`
- **По умолчанию**: `False`
- **Описание**: Использовать ли модуль распознавания диаграмм

---

## Примеры использования

### Базовая инициализация для русского языка
```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    lang="ru",
    ocr_version="PP-OCRv5",
    device="cpu"
)

# Обработка изображения
results = pipeline.predict("document.jpg")
```

### Оптимизированная конфигурация для распознавания таблиц
```python
pipeline = PPStructureV3(
    lang="ru",
    ocr_version="PP-OCRv5",
    device="gpu:0",
    # Layout detection для таблиц
    layout_detection_model_name="PicoDet_layout_1x_table",
    layout_threshold=0.7,
    layout_nms=True,
    # Text detection
    text_det_limit_side_len=736,  # Рекомендуется для таблиц
    text_det_limit_type="min",
    text_det_thresh=0.3,
    text_det_box_thresh=0.6,
    text_det_unclip_ratio=1.5,
    # Text recognition
    text_recognition_batch_size=8,
    text_rec_score_thresh=0.5,
    # Table recognition
    use_table_recognition=True,
    # Performance
    precision="fp16",
    enable_mkldnn=True,
    cpu_threads=8
)
```

### Полная конфигурация со всеми модулями
```python
pipeline = PPStructureV3(
    # Основные параметры
    lang="ru",
    ocr_version="PP-OCRv5",
    device="gpu:0",
    
    # Document preprocessing
    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    
    # Layout detection
    layout_threshold=0.6,
    layout_nms=True,
    layout_unclip_ratio=1.2,
    
    # Text detection & recognition
    text_det_limit_side_len=960,
    text_det_limit_type="max",
    text_recognition_batch_size=8,
    use_textline_orientation=True,
    
    # Специализированные модули
    use_seal_recognition=True,
    use_table_recognition=True,
    use_formula_recognition=True,
    use_chart_recognition=False,
    
    # Performance
    use_tensorrt=True,
    precision="fp16",
    enable_mkldnn=True,
    mkldnn_cache_capacity=20,
    cpu_threads=16
)
```

### Конфигурация для CPU с оптимизацией производительности
```python
pipeline = PPStructureV3(
    lang="ru",
    device="cpu",
    # Используем легковесные модели
    layout_detection_model_name="PicoDet-S_layout_3cls",
    # Оптимизация для CPU
    enable_mkldnn=True,
    mkldnn_cache_capacity=20,
    cpu_threads=16,
    # Батч для ускорения
    text_recognition_batch_size=4,
    # Отключаем ненужные модули
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_seal_recognition=False,
    use_formula_recognition=False,
    use_chart_recognition=False
)
```

### Использование кастомных fine-tuned моделей
```python
pipeline = PPStructureV3(
    # Пути к fine-tuned моделям
    layout_detection_model_dir="./models/custom_layout_det",
    text_detection_model_dir="./models/custom_text_det",
    text_recognition_model_dir="./models/custom_text_rec",
    wired_table_structure_recognition_model_dir="./models/custom_table",
    
    # Остальные параметры
    lang="ru",
    device="gpu:0"
)
```

### Экспорт и загрузка конфигурации из YAML
```python
# Экспорт текущей конфигурации
pipeline = PPStructureV3()
pipeline.export_paddlex_config_to_yaml("my_config.yaml")

# Редактируем my_config.yaml...

# Загрузка конфигурации
pipeline = PPStructureV3(paddlex_config="my_config.yaml")
```

---

## Рекомендации по параметрам

### Для распознавания таблиц:
- `text_det_limit_side_len=736`
- `text_det_limit_type="min"`
- `layout_detection_model_name="PicoDet_layout_1x_table"`
- `use_table_recognition=True`

### Для документов с печатями:
- `use_seal_recognition=True`
- `seal_det_limit_side_len=736`
- `seal_det_thresh=0.2`
- `seal_det_box_thresh=0.6`

### Для искаженных документов:
- `use_doc_orientation_classify=True`
- `use_doc_unwarping=True`

### Для максимальной производительности на GPU:
- `device="gpu:0"`
- `use_tensorrt=True`
- `precision="fp16"`
- Увеличить `*_batch_size` параметры

### Для CPU:
- `device="cpu"`
- `enable_mkldnn=True`
- `cpu_threads=<число физических ядер>`
- Использовать `_mobile` модели вместо `_server`

---

## Ссылки
- [Официальная документация PaddleOCR](https://github.com/paddlepaddle/paddleocr)
- [PP-StructureV3 Documentation](https://github.com/paddlepaddle/paddleocr/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.md)
- [Model Zoo](https://github.com/paddlepaddle/paddleocr/blob/main/docs/version3.x/models_list.md)

