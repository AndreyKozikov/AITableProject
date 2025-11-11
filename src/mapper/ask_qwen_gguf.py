"""
Модуль для работы с Qwen GGUF моделью через llama-cpp-python.

Загружает квантованную GGUF модель для быстрого инференса с поддержкой
Chain-of-Thought reasoning.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from llama_cpp import Llama

from src.learning.knowledge_graph import knowledge_graph

# Настройка логирования
logger = logging.getLogger(__name__)

# Путь к GGUF модели
GGUF_MODEL_PATH = Path("models/qwen4b/qwen-4b-q4_K_M.gguf")

# Путь для логирования ответов модели
GGUF_LOG_DIR = Path("logs/gguf_responses")
GGUF_LOG_DIR.mkdir(parents=True, exist_ok=True)


class QwenGGUFModel:
    """Модель Qwen GGUF с поддержкой Chain-of-Thought reasoning."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация GGUF модели.
        
        Args:
            model_path: Путь к GGUF модели (по умолчанию из константы)
        """
        self.model_path = model_path or str(GGUF_MODEL_PATH)
        self.llm = None
        
        # Инициализация графа знаний
        self.knowledge_graph = knowledge_graph
        
        self._load_model()
    
    def _load_model(self):
        """Загрузка GGUF модели через llama-cpp-python."""
        logger.info(f"Загрузка GGUF модели из: {self.model_path}")
        
        try:
            # Проверяем существование файла
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
            
            logger.info(f"Размер файла: {Path(self.model_path).stat().st_size / (1024**3):.2f} GB")
            
            # Пробуем загрузить модель только на CPU без GPU
            # Это более безопасно для первого запуска
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Размер контекста (уменьшен для стабильности)
                n_threads=4,  # Количество потоков CPU (уменьшено)
                n_gpu_layers=0,  # Отключаем GPU для стабильности
                verbose=True,  # Включаем verbose для диагностики
                use_mlock=False,  # Отключаем mlock
                use_mmap=True,  # Используем memory mapping
                n_batch=512  # Размер батча
            )
            logger.info("✓ GGUF модель загружена успешно")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки GGUF модели: {e}")
            logger.error("Проверьте установку llama-cpp-python:")
            logger.error("pip uninstall llama-cpp-python -y")
            logger.error("pip install llama-cpp-python --force-reinstall --no-cache-dir")
            raise
    
    def format_cot_prompt(self, input_text: str, mode: str = "extended") -> str:
        """
        Форматирование промпта с инструкциями для CoT reasoning.
        
        Args:
            input_text: Входной текст для обработки
            mode: Режим обработки ('extended' или 'simplified')
            
        Returns:
            Отформатированный промпт
        """
        if mode == "extended":
            columns = "Артикул, Наименование, Количество, Единица измерения, Цена, Примечание"
        else:
            columns = "Артикул, Наименование, Количество, Единица измерения"
        
        prompt = f"""Ты — эксперт по обработке данных об инструментах и оборудовании.

Задача: Проанализируй данные и заполни таблицу со следующими колонками:
{columns}

Правила:
1. Используй свой опыт работы с инструментами
2. Внимательно анализируй каждое поле
3. Артикул не является наименованием
4. Заполняй все доступные колонки
5. Если данных нет — оставляй пустым

Входные данные:
{input_text}

Выведи результат ТОЛЬКО в формате JSON массива объектов без дополнительных пояснений."""

        return prompt
    
    def _extract_knowledge(self, text: str) -> Dict[str, List[str]]:
        """
        Извлечение знаний из графа знаний.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Словарь с релевантными знаниями
        """
        # Получаем контекст из графа знаний
        context = self.knowledge_graph.get_context(text)
        
        knowledge = {
            "инструменты": context.get("tools", [])[:3],
            "единицы_измерения": context.get("units", [])[:3],
            "материалы": context.get("materials", [])[:3]
        }
        
        return knowledge
    
    def _save_response_log(self, prompt: str, response: str, thinking: str = ""):
        """
        Сохранение лога запроса и ответа.
        
        Args:
            prompt: Промпт
            response: Ответ модели
            thinking: Рассуждения модели (опционально)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = GGUF_LOG_DIR / f"response_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "prompt": prompt,
            "thinking": thinking,
            "response": response,
            "model_path": self.model_path
        }
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Лог сохранён: {log_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить лог: {e}")
    
    def generate(self, prompt: str, max_tokens: int = 4096, temperature: float = 0.7) -> str:
        """
        Генерация текста через GGUF модель.
        
        Args:
            prompt: Промпт для модели
            max_tokens: Максимальное количество токенов
            temperature: Температура генерации
            
        Returns:
            Сгенерированный текст
        """
        logger.info("Начало генерации через GGUF модель")
        logger.debug(f"Длина промпта: {len(prompt)} символов")
        
        try:
            # Генерация через llama-cpp-python
            output = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["###", "\n\n\n"],
                echo=False
            )
            
            generated_text = output['choices'][0]['text'].strip()
            
            logger.info(f"Генерация завершена. Длина ответа: {len(generated_text)} символов")
            
            # Сохраняем лог
            self._save_response_log(prompt, generated_text)
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Ошибка генерации: {e}")
            raise
    
    def process_with_cot(
        self,
        input_data: Union[str, List[Dict[str, Any]]],
        mode: str = "extended"
    ) -> str:
        """
        Обработка данных с Chain-of-Thought reasoning.
        
        Args:
            input_data: Входные данные (строка или список словарей)
            mode: Режим обработки
            
        Returns:
            JSON строка с результатами
        """
        logger.info(f"Начало обработки в режиме CoT (mode={mode})")
        
        # Конвертируем данные в строку если нужно
        if isinstance(input_data, list):
            input_text = json.dumps(input_data, ensure_ascii=False, indent=2)
        else:
            input_text = input_data
        
        # Извлекаем знания
        knowledge = self._extract_knowledge(input_text)
        logger.debug(f"Извлечено знаний: {len(knowledge)} категорий")
        
        # Форматируем промпт
        prompt = self.format_cot_prompt(input_text, mode)
        
        # Генерируем ответ
        response = self.generate(prompt, max_tokens=4096, temperature=0.3)
        
        return response


# Глобальный экземпляр модели (ленивая инициализация)
_model_instance = None


def get_model() -> QwenGGUFModel:
    """
    Получить глобальный экземпляр модели (singleton).
    
    Returns:
        Экземпляр QwenGGUFModel
    """
    global _model_instance
    if _model_instance is None:
        logger.info("Инициализация глобального экземпляра GGUF модели")
        _model_instance = QwenGGUFModel()
    return _model_instance


def ask_qwen_gguf(
    prompt: Optional[Union[str, List[str]]] = None,
    extended: bool = False,
    max_new_tokens: int = 4096
) -> str:
    """
    Запрос к Qwen GGUF модели.
    
    Args:
        prompt: Промпт или список промптов
        extended: Использовать расширенный режим
        max_new_tokens: Максимальное количество токенов
        
    Returns:
        Ответ модели
    """
    logger.info(f"ask_qwen_gguf вызван (extended={extended})")
    
    model = get_model()
    mode = "extended" if extended else "simplified"
    
    # Обрабатываем промпт
    if isinstance(prompt, list):
        input_data = prompt
    else:
        input_data = prompt
    
    response = model.process_with_cot(input_data, mode=mode)
    
    return response


def extract_cot_rows_as_dicts(response_text: str) -> List[Dict[str, str]]:
    """
    Извлечение строк из JSON ответа модели.
    
    Args:
        response_text: Текст ответа от модели
        
    Returns:
        Список словарей с данными
    """
    logger.info("Извлечение данных из ответа GGUF модели")
    
    try:
        # Ищем JSON в ответе
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            if isinstance(data, list):
                logger.info(f"Извлечено {len(data)} записей")
                return data
        
        # Если не нашли JSON массив, пробуем парсить как объект
        try:
            data = json.loads(response_text)
            if isinstance(data, dict):
                return [data]
            elif isinstance(data, list):
                return data
        except:
            pass
        
        logger.warning("Не удалось извлечь JSON из ответа")
        return []
        
    except json.JSONDecodeError as e:
        logger.error(f"Ошибка парсинга JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"Ошибка извлечения данных: {e}")
        return []

