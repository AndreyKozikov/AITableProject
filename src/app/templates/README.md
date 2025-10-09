# HTML Templates Directory

Этот каталог содержит HTML-шаблоны для приложения AITableProject.

## Структура

### Основные шаблоны

- **`header.html`** - Заголовок приложения
- **`upload_card.html`** - Карточка загрузки файлов с drag & drop зоной
- **`settings_card.html`** - Карточка настроек обработки
- **`file_list.html`** - Контейнер списка загруженных файлов
- **`file_item.html`** - Отдельный элемент файла в списке
- **`progress_tracker.html`** - Компонент отслеживания прогресса обработки
- **`results_display.html`** - Отображение результатов обработки

### Модуль загрузчика

- **`template_loader.py`** - Утилита для загрузки и рендеринга шаблонов

## Использование

### Базовая загрузка шаблона

```python
from src.app.templates.template_loader import load_template

# Загрузить шаблон без подстановки переменных
html = load_template("header")
st.markdown(html, unsafe_allow_html=True)
```

### Рендеринг с переменными

```python
from src.app.templates.template_loader import render_template

# Рендерить шаблон с подстановкой переменных
html = render_template(
    "file_item",
    icon="📄",
    file_name="document.pdf",
    file_size="125.5"
)
```

### Использование класса TemplateLoader

```python
from src.app.templates.template_loader import TemplateLoader
from pathlib import Path

# Создать экземпляр загрузчика
loader = TemplateLoader(templates_dir=Path("path/to/templates"))

# Загрузить шаблон
template = loader.load_template("header")

# Рендерить с переменными
rendered = loader.render_template("file_item", icon="📄", file_name="test.pdf")

# Очистить кэш
loader.clear_cache()
```

## Синтаксис шаблонов

Шаблоны используют стандартный синтаксис форматирования строк Python:

```html
<div class="file-name">{file_name}</div>
<div class="file-size">{file_size} KB</div>
```

При рендеринге переменные заменяются соответствующими значениями:

```python
render_template("file_item", file_name="document.pdf", file_size="125.5")
```

## Преимущества

1. **Разделение ответственности** - HTML-код отделен от бизнес-логики Python
2. **Переиспользование** - Шаблоны можно использовать многократно
3. **Легкость поддержки** - Изменения в UI не требуют правки кода Python
4. **Кэширование** - Шаблоны кэшируются для повышения производительности
5. **Чистый код** - Основной файл приложения содержит только логику

## Стилизация

Для стилизации используется:
- **CSS-переменные** для единообразия дизайна
- **Внешний CSS-файл** (`static/styles.css`) для глобальных стилей
- **Встроенные стили** в шаблонах для компонент-специфичных стилей

## Примечания

- Все шаблоны используют кодировку UTF-8
- Шаблоны поддерживают emoji и кириллицу
- Для iframe-компонентов используются полные HTML-документы с `<head>` и `<body>`
- Для встроенных компонентов используются HTML-фрагменты

