"""
Модуль для работы с Qwen3 с поддержкой Chain-of-Thought reasoning.

Интегрирует цепочки рассуждений и граф знаний в процесс обработки
промышленных инструментов.
"""

import json
import torch
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.utils.config import MODEL_CACHE_DIR
from src.learning.knowledge_graph import knowledge_graph

# Настройка логирования
logger = logging.getLogger(__name__)

# Путь для логирования ответов CoT модели
COT_LOG_DIR = Path("logs/cot_responses")
COT_LOG_DIR.mkdir(parents=True, exist_ok=True)


class Qwen3CoTModel:
    """Модель Qwen3 с поддержкой Chain-of-Thought reasoning."""
    
    def __init__(self, model_path: Optional[str] = None, adapters_path: Optional[str] = None):
        """
        Инициализация модели с CoT reasoning.
        
        Args:
            model_path: Путь к базовой модели (по умолчанию из config)
            adapters_path: Путь к LoRA адаптерам (по умолчанию из config)
        """
        self.model_path = model_path or str(MODEL_CACHE_DIR)
        self.adapters_path = adapters_path or str(MODEL_CACHE_DIR / "lora_adapters_cot")
        
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()
        
        # Инициализация графа знаний
        self.knowledge_graph = knowledge_graph
        
        self._load_model()
    
    def _get_device(self):
        """Определение устройства для модели."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_model(self):
        """Загрузка модели и токенизатора."""
        print(f"Загрузка модели из: {self.model_path}")
        print(f"Устройство: {self.device}")
        
        try:
            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Загружаем базовую модель
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                device_map="auto" if self.device.type == "cuda" else None,
                trust_remote_code=True
            )
            
            # Загружаем LoRA адаптеры, если они существуют
            adapters_path = Path(self.adapters_path)
            if adapters_path.exists():
                print(f"Загрузка LoRA адаптеров из: {adapters_path}")
                self.model = PeftModel.from_pretrained(self.model, str(adapters_path))
                print("✓ LoRA адаптеры загружены")
            else:
                print(f"⚠ LoRA адаптеры не найдены в {adapters_path}")
                print("Используется базовая модель")
            
            self.model = self.model.to(self.device)
            print("✓ Модель загружена успешно")
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            raise

    
    def format_cot_prompt(self, input_text: str, mode: str = "extended") -> str:
        """
        Форматировать промпт с Chain-of-Thought reasoning.
        
        Args:
            input_text: Входной текст
            mode: Режим обработки ("simplified" или "extended")
            
        Returns:
            Отформатированный промпт
        """
        # Создаем system prompt
        system_prompt = f"""Режим работы: {mode}. Ты — ассистент по структурированной обработке промышленных инструментов.

КРИТИЧЕСКИ ВАЖНО:
- Отвечай ТОЛЬКО JSON структурой
- НЕ ВКЛЮЧАЙ никаких рассуждений, объяснений или промежуточных шагов
- НЕ ПИШИ текст перед или после JSON
- НЕ ИСПОЛЬЗУЙ Markdown блоки ```json
- НЕ ДОБАВЛЯЙ комментарии или пояснения

Правила анализа:
1. Выдели обозначение, наименование, производителя и параметры
2. Определи производителя по обозначению (SDJCR/CNMG→Sandvik, YG/DH→YG-1, KGM→KELITE, DCMT→ZCC, ER→EROGLU, HSK/BT→DAndrea, A0/A1→JieHe)
3. Представь результат в JSON формате

Правила вывода:
- Используй ровно эти ключи: {', '.join(self._get_columns(mode))}
- Структура ответа должна полностью соответствовать JSON Schema
- Не добавляй новых ключей или уровней вложенности
- Если данных нет для ключа — вставляй пустую строку
- Не используй Markdown, блоки ```json или пояснительный текст
- Вывод — только корректный JSON, без заголовков и комментариев
- НЕ ВКЛЮЧАЙ рассуждения, объяснения или промежуточные шаги в ответ
- Отвечай ТОЛЬКО JSON структурой

<output-format>
{self._get_json_schema(mode)}
</output-format>"""

        # Создаем user prompt ТОЛЬКО с входной строкой (как в продакшене)
        user_prompt = f"Проанализируй данные:\n{input_text}"

        return system_prompt, user_prompt
    
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
    
    def ask_qwen3_cot(self, input_text: str, mode: str = "extended", max_length: int = 4096) -> Dict[str, Any]:
        """
        Обработать текст с использованием Chain-of-Thought reasoning.
        
        Args:
            input_text: Входной текст для обработки
            mode: Режим обработки ("simplified" или "extended")
            max_length: Максимальная длина ответа
            
        Returns:
            Словарь с результатом обработки
        """
        try:
            # Форматируем промпт с CoT
            system_prompt, user_prompt = self.format_cot_prompt(input_text, mode)
            
            # Формируем сообщения для чата
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Применяем chat template
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Токенизируем
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            ).to(self.device)
            
            # Генерируем ответ
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    top_k=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.0
                )
            
            # Декодируем ответ
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Извлекаем только ответ ассистента
            if "assistant" in response.lower():
                assistant_start = response.lower().find("assistant")
                response = response[assistant_start + 9:].strip()
            
            # Логирование полного ответа модели
            self._log_response(input_text, response, mode)
            
            # Парсим JSON ответ
            try:
                parsed_json = json.loads(response)
                logger.info(f"CoT model parsed JSON successfully")
                
                return {
                    "success": True,
                    "result": parsed_json,
                    "raw_response": response
                }
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.error(f"Raw response: {response}")
                
                return {
                    "success": False,
                    "error": f"Invalid JSON response: {str(e)}",
                    "raw_response": response
                }
                
        except Exception as e:
            error_msg = f"Ошибка обработки: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "raw_response": ""
            }
    
    def _log_response(self, input_text: str, response: str, mode: str):
        """
        Логировать ответ модели в отдельный файл.
        
        Args:
            input_text: Входной текст
            response: Ответ модели
            mode: Режим обработки
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        log_file = COT_LOG_DIR / f"cot_response_{timestamp}.txt"
        
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
            
            logger.info(f"CoT response logged to: {log_file}")
        except Exception as e:
            logger.error(f"Failed to log CoT response: {e}")
    
    def get_knowledge_info(self, entity_name: str) -> Dict[str, Any]:
        """
        Получить информацию о сущности из графа знаний.
        
        Args:
            entity_name: Название сущности
            
        Returns:
            Информация о сущности
        """
        return self.knowledge_graph.get_entity_info(entity_name)
    
    def find_manufacturer_by_denotation(self, denotation: str) -> str:
        """
        Найти производителя по обозначению.
        
        Args:
            denotation: Обозначение инструмента
            
        Returns:
            Название производителя
        """
        return self.knowledge_graph.find_manufacturer_by_denotation(denotation)


# Глобальный экземпляр модели
qwen3_cot_model = None


def get_qwen3_cot_model() -> Qwen3CoTModel:
    """Получить глобальный экземпляр модели Qwen3 с CoT."""
    global qwen3_cot_model
    if qwen3_cot_model is None:
        qwen3_cot_model = Qwen3CoTModel()
    return qwen3_cot_model


def ask_qwen3_cot(input_text: str, mode: str = "extended") -> Dict[str, Any]:
    """
    Удобная функция для вызова модели Qwen3 с CoT reasoning.
    
    Args:
        input_text: Входной текст для обработки
        mode: Режим обработки ("simplified" или "extended")
        
    Returns:
        Результат обработки
    """
    model = get_qwen3_cot_model()
    return model.ask_qwen3_cot(input_text, mode)


def extract_cot_rows_as_dicts(cot_result: Dict[str, Any], use_aliases: bool = True) -> List[Dict[str, Any]]:
    """
    Извлечь строки из результата CoT модели как список словарей.
    
    Args:
        cot_result: Результат от ask_qwen3_cot
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
