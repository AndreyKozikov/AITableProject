"""
Модуль для работы с OpenAI GPT через API с поддержкой Chain-of-Thought reasoning.

Интегрирует цепочки рассуждений и граф знаний в процесс обработки
промышленных инструментов через OpenAI API.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.config import OPENAI_API_KEY
from src.learning.knowledge_graph import knowledge_graph

# Настройка логирования
logger = logging.getLogger(__name__)

# Путь для логирования ответов OpenAI CoT модели
OPENAI_COT_LOG_DIR = Path("logs/openai_cot_responses")
OPENAI_COT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def is_retryable_exception(e: Exception) -> bool:
    """Проверка, можно ли повторить запрос при ошибке."""
    text = str(e)
    # Не повторяем при rate_limit_exceeded
    if "rate_limit_exceeded" in text:
        logger.warning(f"Rate limit exceeded, not retrying: {text}")
        return False
    logger.debug(f"Exception is retryable: {text}")
    return True


class OpenAICoTModel:
    """Модель OpenAI GPT с поддержкой Chain-of-Thought reasoning."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Инициализация модели OpenAI с CoT reasoning.
        
        Args:
            api_key: API ключ OpenAI (по умолчанию из config)
            model: Модель OpenAI для использования
        """
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не установлен. Установите его в .env файле.")
        
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        
        # Инициализация графа знаний
        self.knowledge_graph = knowledge_graph
        
        logger.info(f"OpenAI CoT модель инициализирована: {model}")
    
    def format_cot_prompt(self, input_text: str, mode: str = "extended") -> tuple:
        """
        Форматировать промпт с Chain-of-Thought reasoning.
        
        Args:
            input_text: Входной текст
            mode: Режим обработки ("simplified" или "extended")
            
        Returns:
            Кортеж (system_prompt, user_prompt)
        """
        # Получаем информацию из графа знаний для контекстной подачи
        knowledge_context = self._get_knowledge_context(input_text, mode)
        
        # Создаем system prompt с полной информацией из графа знаний
        system_prompt = f"""Режим работы: {mode}. Ты — ассистент по структурированной обработке промышленных инструментов.

БАЗА ЗНАНИЙ:
{knowledge_context}

КРИТИЧЕСКИ ВАЖНО:
- Отвечай ТОЛЬКО JSON структурой
- НЕ ВКЛЮЧАЙ никаких рассуждений, объяснений или промежуточных шагов
- НЕ ПИШИ текст перед или после JSON
- НЕ ИСПОЛЬЗУЙ Markdown блоки ```json
- НЕ ДОБАВЛЯЙ комментарии или пояснения

Правила анализа:
1. Используй информацию из базы знаний для точного определения производителей и типов инструментов
2. Выдели обозначение, наименование, производителя и параметры
3. Применяй правила определения производителя по обозначению из базы знаний
4. Представь результат в JSON формате

Правила вывода:
- Используй ровно эти ключи: {', '.join(self._get_columns(mode))}
- Структура ответа должна полностью соответствовать JSON Schema
- Не добавляй новых ключей или уровней вложенности
- Если данных нет для ключа — вставляй пустую строку
- Если единица измерения не указана в исходных данных — используй "шт." по умолчанию
- Не используй Markdown, блоки ```json или пояснительный текст
- Вывод — только корректный JSON, без заголовков и комментариев
- НЕ ВКЛЮЧАЙ рассуждения, объяснения или промежуточные шаги в ответ
- Отвечай ТОЛЬКО JSON структурой

<output-format>
{self._get_json_schema(mode)}
</output-format>"""

        # Создаем user prompt ТОЛЬКО с входной строкой
        user_prompt = f"Проанализируй данные:\n{input_text}"

        return system_prompt, user_prompt
    
    def _get_knowledge_context(self, input_text: str, mode: str) -> str:
        """
        Получить контекстную информацию из графа знаний для подачи в модель.
        
        Args:
            input_text: Входной текст для анализа
            mode: Режим обработки
            
        Returns:
            Форматированная строка с информацией из графа знаний
        """
        try:
            # Получаем базовую информацию из графа знаний
            manufacturers = self.knowledge_graph.get_catalog_manufacturers()
            tool_types = list(self.knowledge_graph.get_tool_type_patterns().keys())
            
            # Находим релевантные сущности в тексте
            relevant_entities = self._find_relevant_entities(input_text)
            
            # Получаем правила определения производителей
            manufacturer_rules = self._get_manufacturer_rules()
            
            # Формируем контекст
            context_parts = []
            
            # Базовая информация о производителях
            if manufacturers:
                context_parts.append(f"ПРОИЗВОДИТЕЛИ: {', '.join(sorted(manufacturers))}")
            
            # Типы инструментов
            if tool_types:
                context_parts.append(f"ТИПЫ ИНСТРУМЕНТОВ: {', '.join(sorted(tool_types))}")
            
            # Правила определения производителей
            if manufacturer_rules:
                context_parts.append(f"ПРАВИЛА ОПРЕДЕЛЕНИЯ ПРОИЗВОДИТЕЛЕЙ:\n{manufacturer_rules}")
            
            # Релевантные сущности
            if relevant_entities:
                context_parts.append(f"РЕЛЕВАНТНЫЕ СУЩНОСТИ В ТЕКСТЕ: {', '.join(relevant_entities)}")
            
            return "\n".join(context_parts) if context_parts else "База знаний недоступна"
            
        except Exception as e:
            logger.warning(f"Ошибка получения контекста из графа знаний: {e}")
            return "База знаний недоступна"
    
    def _find_relevant_entities(self, input_text: str) -> List[str]:
        """
        Найти релевантные сущности в тексте.
        
        Args:
            input_text: Входной текст
            
        Returns:
            Список найденных сущностей
        """
        try:
            relevant_entities = []
            text_lower = input_text.lower()
            
            # Ищем производителей по обозначениям из параметров
            parameter_patterns = self.knowledge_graph.get_parameter_patterns()
            for pattern, description in parameter_patterns.items():
                if pattern.lower() in text_lower:
                    relevant_entities.append(f"{pattern}→{description}")
            
            # Ищем типы инструментов
            tool_patterns = self.knowledge_graph.get_tool_type_patterns()
            for tool_type, patterns in tool_patterns.items():
                for pattern in patterns:
                    if pattern.lower() in text_lower:
                        relevant_entities.append(f"{pattern}→{tool_type}")
                        break
            
            return relevant_entities[:10]  # Ограничиваем количество
            
        except Exception as e:
            logger.warning(f"Ошибка поиска релевантных сущностей: {e}")
            return []
    
    def _get_manufacturer_rules(self) -> str:
        """
        Получить правила определения производителей из графа знаний.
        
        Returns:
            Форматированная строка с правилами
        """
        try:
            rules = []
            parameter_patterns = self.knowledge_graph.get_parameter_patterns()
            
            # Добавляем основные правила определения производителей
            basic_rules = [
                "SDJCR/CNMG → Sandvik",
                "YG/DH → YG-1", 
                "KGM → KELITE",
                "DCMT → ZCC",
                "ER → EROGLU",
                "HSK/BT → DAndrea",
                "A0/A1 → JieHe"
            ]
            
            for rule in basic_rules:
                rules.append(f"- {rule}")
            
            # Добавляем дополнительные паттерны из графа знаний
            for pattern, description in parameter_patterns.items():
                if len(pattern) <= 10:  # Короткие паттерны для производителей
                    rules.append(f"- {pattern} → {description}")
            
            return "\n".join(rules) if rules else "Правила недоступны"
            
        except Exception as e:
            logger.warning(f"Ошибка получения правил производителей: {e}")
            return "Правила недоступны"
    
    def _get_columns(self, mode: str) -> List[str]:
        """Получить список колонок для указанного режима."""
        if mode == "simplified":
            return ["Наименование", "Единица измерения", "Количество", "Техническое задание"]
        else:
            return ["Обозначение", "Наименование", "Производитель", "Единица измерения", "Количество", "Техническое задание"]
    
    def _get_json_schema(self, mode: str) -> str:
        """Получить JSON Schema для указанного режима."""
        if mode == "simplified":
            return '''{
  "$defs": {
    "TableRowSimplified": {
      "additionalProperties": false,
      "properties": {
        "Наименование": {
          "default": "",
          "description": "Field for Наименование",
          "title": "Наименование",
          "type": "string"
        },
        "Единица измерения": {
          "default": "",
          "description": "Field for Единица измерения",
          "title": "Единица измерения",
          "type": "string"
        },
        "Количество": {
          "default": "",
          "description": "Field for Количество",
          "title": "Количество",
          "type": "string"
        },
        "Техническое задание": {
          "default": "",
          "description": "Field for Техническое задание",
          "title": "Техническое задание",
          "type": "string"
        }
      },
      "title": "TableRowSimplified",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "rows": {
      "description": "List of table rows",
      "items": {
        "$ref": "#/$defs/TableRowSimplified"
      },
      "title": "Rows",
      "type": "array"
    }
  },
  "required": [
    "rows"
  ],
  "title": "TableStructuredOutput",
  "type": "object"
}'''
        else:  # extended
            return '''{
  "$defs": {
    "TableRowExtended": {
      "additionalProperties": false,
      "properties": {
        "Обозначение": {
          "default": "",
          "description": "Field for Обозначение",
          "title": "Обозначение",
          "type": "string"
        },
        "Наименование": {
          "default": "",
          "description": "Field for Наименование",
          "title": "Наименование",
          "type": "string"
        },
        "Производитель": {
          "default": "",
          "description": "Field for Производитель",
          "title": "Производитель",
          "type": "string"
        },
        "Единица измерения": {
          "default": "",
          "description": "Field for Единица измерения",
          "title": "Единица измерения",
          "type": "string"
        },
        "Количество": {
          "default": "",
          "description": "Field for Количество",
          "title": "Количество",
          "type": "string"
        },
        "Техническое задание": {
          "default": "",
          "description": "Field for Техническое задание",
          "title": "Техническое задание",
          "type": "string"
        }
      },
      "title": "TableRowExtended",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "properties": {
    "rows": {
      "description": "List of table rows",
      "items": {
        "$ref": "#/$defs/TableRowExtended"
      },
      "title": "Rows",
      "type": "array"
    }
  },
  "required": [
    "rows"
  ],
  "title": "TableStructuredOutput",
  "type": "object"
}'''
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        retry=retry_if_exception(is_retryable_exception)
    )
    def ask_openai_cot(self, input_text: Union[str, List[str]], mode: str = "extended") -> Dict[str, Any]:
        """
        Обработать текст с использованием Chain-of-Thought reasoning через OpenAI API.
        
        Args:
            input_text: Входной текст для обработки (строка или список строк)
            mode: Режим обработки ("simplified" или "extended")
            
        Returns:
            Словарь с результатом обработки в формате: {"success": bool, "result": dict, "raw_response": str}
        """
        def _infer_single(single_text: str, index: Optional[int] = None, total: Optional[int] = None) -> Dict[str, Any]:
            # Форматируем промпт с CoT
            system_prompt, user_prompt = self.format_cot_prompt(single_text, mode)
            
            # Логируем входные данные
            if index is not None and total is not None:
                logger.info(f"ВХОД В OpenAI [chunk {index+1}/{total}]:\n{single_text}")
            else:
                logger.info(f"ВХОД В OpenAI [single]:\n{single_text}")

            try:
                # Вызов OpenAI API с structured output
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=4096
                )
                
                # Извлекаем ответ
                content = response.choices[0].message.content
                
                # Логирование полного ответа модели
                self._log_response(single_text, content, mode)
                
                # Парсим JSON ответ
                try:
                    parsed_json = json.loads(content)
                    logger.info(f"OpenAI model parsed JSON successfully")
                    return {"success": True, "result": parsed_json, "raw_response": content}
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    logger.error(f"Raw response: {content}")
                    return {"success": False, "error": f"Invalid JSON response: {str(e)}", "raw_response": content}
                    
            except Exception as e:
                error_msg = f"Ошибка вызова OpenAI API: {str(e)}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg, "raw_response": ""}

        try:
            # Обработка батча: список строк или одна строка
            if isinstance(input_text, list):
                combined_rows: List[Dict[str, Any]] = []
                raw_responses: List[str] = []
                total = len(input_text)
                logger.info(f"РЕЖИМ СПИСКА: получено {total} чанков, обрабатываем все")
                for idx, chunk in enumerate(input_text):
                    result = _infer_single(chunk, index=idx, total=total)
                    raw_responses.append(result.get("raw_response", ""))
                    if result.get("success") and isinstance(result.get("result"), dict):
                        rows = result["result"].get("rows", [])
                        if isinstance(rows, list):
                            combined_rows.extend(rows)
                return {"success": True, "result": {"rows": combined_rows}, "raw_response": "\n".join(raw_responses)}

            # Одиночный вызов
            return _infer_single(input_text)
            
        except Exception as e:
            error_msg = f"Ошибка обработки: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg, "raw_response": ""}
    
    def _log_response(self, input_text: str, response: str, mode: str):
        """
        Логировать ответ модели в отдельный файл.
        
        Args:
            input_text: Входной текст
            response: Ответ модели
            mode: Режим обработки
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = OPENAI_COT_LOG_DIR / f"openai_cot_response_{timestamp}.txt"
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: {mode}\n")
                f.write("=" * 80 + "\n\n")
                f.write("INPUT TEXT:\n")
                f.write("-" * 80 + "\n")
                f.write(input_text + "\n\n")
                f.write("=" * 80 + "\n\n")
                f.write("MODEL RESPONSE:\n")
                f.write("-" * 80 + "\n")
                f.write(response + "\n")
                f.write("=" * 80 + "\n")
            
            logger.info(f"OpenAI CoT response logged to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to log OpenAI CoT response: {e}")


# Глобальный экземпляр модели
openai_cot_model = None


def get_openai_cot_model() -> OpenAICoTModel:
    """Получить глобальный экземпляр модели OpenAI с CoT."""
    global openai_cot_model
    if openai_cot_model is None:
        openai_cot_model = OpenAICoTModel()
    return openai_cot_model


def ask_openai_cot(input_text: str, mode: str = "extended") -> Dict[str, Any]:
    """
    Удобная функция для вызова модели OpenAI с CoT reasoning.
    
    Args:
        input_text: Входной текст для обработки
        mode: Режим обработки ("simplified" или "extended")
        
    Returns:
        Результат обработки в формате: {"success": bool, "result": dict, "raw_response": str}
    """
    model = get_openai_cot_model()
    return model.ask_openai_cot(input_text, mode)


def extract_cot_rows_as_dicts(cot_result: Dict[str, Any], use_aliases: bool = True) -> List[Dict[str, Any]]:
    """
    Извлечь строки из результата CoT модели как список словарей.
    
    Args:
        cot_result: Результат от ask_openai_cot
        use_aliases: Использовать алиасы полей (игнорируется для CoT)
        
    Returns:
        Список словарей с данными строк
    """
    if not cot_result.get("success", False):
        return []
    
    result_data = cot_result.get("result", {})
    if not isinstance(result_data, dict) or "rows" not in result_data:
        return []
    
    return result_data["rows"]

