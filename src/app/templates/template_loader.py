"""
Модуль загрузчика шаблонов
Обрабатывает загрузку и рендеринг HTML шаблонов для приложения Streamlit.
"""

from pathlib import Path
from typing import Dict, Any


class TemplateLoader:
    """Загружает и рендерит HTML шаблоны из директории шаблонов."""
    
    def __init__(self, templates_dir: Path = None):
        """
        Инициализирует загрузчик шаблонов.
        
        Args:
            templates_dir: Путь к директории шаблонов. 
                          Если None, используется директория этого файла.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent
        self.templates_dir = templates_dir
        self._cache = {}
    
    def load_template(self, template_name: str) -> str:
        """
        Загружает файл шаблона из директории шаблонов.
        
        Args:
            template_name: Имя файла шаблона (без расширения .html)
        
        Returns:
            Содержимое шаблона в виде строки
        """
        # Проверяем кэш сначала
        if template_name in self._cache:
            return self._cache[template_name]
        
        # Загружаем из файла
        template_path = self.templates_dir / f"{template_name}.html"
        
        if not template_path.exists():
            raise FileNotFoundError(
                f"Шаблон '{template_name}' не найден по пути {template_path}"
            )
        
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Кэшируем шаблон
        self._cache[template_name] = content
        
        return content
    
    def render_template(self, template_name: str, **kwargs: Any) -> str:
        """
        Загружает и рендерит шаблон с предоставленными переменными.
        
        Args:
            template_name: Имя файла шаблона (без расширения .html)
            **kwargs: Переменные для подстановки в шаблон
        
        Returns:
            Отрендеренный шаблон в виде строки
        """
        template = self.load_template(template_name)
        return template.format(**kwargs)
    
    def clear_cache(self):
        """Очищает кэш шаблонов."""
        self._cache.clear()


# Глобальный экземпляр загрузчика шаблонов
_template_loader = TemplateLoader()


def get_template_loader() -> TemplateLoader:
    """Получает глобальный экземпляр загрузчика шаблонов."""
    return _template_loader


def render_template(template_name: str, **kwargs: Any) -> str:
    """
    Удобная функция для рендеринга шаблона используя глобальный загрузчик.
    
    Args:
        template_name: Имя файла шаблона (без расширения .html)
        **kwargs: Переменные для подстановки в шаблон
    
    Returns:
        Отрендеренный шаблон в виде строки
    """
    return _template_loader.render_template(template_name, **kwargs)


def load_template(template_name: str) -> str:
    """
    Удобная функция для загрузки шаблона используя глобальный загрузчик.
    
    Args:
        template_name: Имя файла шаблона (без расширения .html)
    
    Returns:
        Содержимое шаблона в виде строки
    """
    return _template_loader.load_template(template_name)

