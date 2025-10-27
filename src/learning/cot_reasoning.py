"""
Модуль для генерации Chain-of-Thought reasoning для промышленных инструментов.

Содержит логику для создания цепочек рассуждений при разборе строк описания инструментов.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .knowledge_graph import knowledge_graph


@dataclass
class ReasoningStep:
    """Шаг в цепочке рассуждений."""
    step_number: int
    description: str
    confidence: float
    evidence: List[str]


class CoTReasoningGenerator:
    """Генератор цепочек рассуждений для разбора промышленных инструментов."""
    
    def __init__(self):
        self.knowledge_graph = knowledge_graph
        self.parameter_patterns = self.knowledge_graph.get_parameter_patterns()
    
    def generate_reasoning(self, input_text: str) -> List[ReasoningStep]:
        """Генерировать цепочку рассуждений для входного текста."""
        reasoning_steps = []
        
        # Шаг 1: Анализ общей структуры строки
        reasoning_steps.append(self._analyze_structure(input_text))
        
        # Шаг 2: Определение типа инструмента
        reasoning_steps.append(self._identify_tool_type(input_text))
        
        # Шаг 3: Извлечение обозначения
        reasoning_steps.append(self._extract_denotation(input_text))
        
        # Шаг 4: Определение производителя
        reasoning_steps.append(self._identify_manufacturer(input_text))
        
        # Шаг 5: Извлечение параметров
        reasoning_steps.append(self._extract_parameters(input_text))
        
        # Шаг 6: Извлечение технического задания
        reasoning_steps.append(self._extract_technical_task(input_text))
        
        # Шаг 7: Валидация результата
        reasoning_steps.append(self._validate_result(input_text, reasoning_steps))
        
        return reasoning_steps
    
    def _analyze_structure(self, text: str) -> ReasoningStep:
        """Анализ общей структуры строки."""
        # Подсчет компонентов
        words = text.split()
        has_numbers = bool(re.search(r'\d+', text))
        has_letters = bool(re.search(r'[A-Za-z]', text))
        has_symbols = bool(re.search(r'[^\w\s]', text))
        
        structure_analysis = []
        structure_analysis.append(f"Строка содержит {len(words)} слов")
        structure_analysis.append(f"Есть числа: {'да' if has_numbers else 'нет'}")
        structure_analysis.append(f"Есть буквы: {'да' if has_letters else 'нет'}")
        structure_analysis.append(f"Есть символы: {'да' if has_symbols else 'нет'}")
        
        confidence = 0.9 if len(words) > 2 else 0.6
        
        return ReasoningStep(
            step_number=1,
            description=f"Анализирую структуру строки: '{text[:50]}...'",
            confidence=confidence,
            evidence=structure_analysis
        )
    
    def _identify_tool_type(self, text: str) -> ReasoningStep:
        """Определение наименования инструмента."""
        # Ищем полное наименование инструмента
        
        # Сначала ищем в поле "Наименование и обозначение"
        if "Наименование и обозначение:" in text:
            # Извлекаем часть после "Наименование и обозначение:"
            parts = text.split("Наименование и обозначение:")
            if len(parts) > 1:
                name_part = parts[1].strip()
                # Извлекаем наименование до первого технического обозначения
                words = name_part.split()
                tool_name_words = []
                
                for word in words:
                    # Если встретили техническое обозначение (буквы+цифры), останавливаемся
                    if re.match(r'^[A-Z]+\d+', word) or word.isdigit():
                        break
                    tool_name_words.append(word)
                
                tool_name = " ".join(tool_name_words).strip()
                evidence = [f"Найдено наименование: {tool_name}"]
                confidence = 0.9
            else:
                tool_name = text
                evidence = [f"Найдено наименование: {tool_name}"]
                confidence = 0.7
        elif "Наименование:" in text:
            # Извлекаем часть после "Наименование:"
            parts = text.split("Наименование:")
            if len(parts) > 1:
                name_part = parts[1].split("|")[0].strip()
            else:
                name_part = text
            
            # Ищем базовые типы инструментов для определения категории
            tool_types = ["Державка", "Пластина", "Сверло", "Фреза", "Цанга", "Винт", "Патрон"]
            found_type = None
            
            for tool_type in tool_types:
                if tool_type.lower() in name_part.lower():
                    found_type = tool_type
                    break
            
            if found_type:
                evidence = [f"Найдено наименование: {name_part} (тип: {found_type})"]
                confidence = 0.9
            else:
                evidence = [f"Найдено наименование: {name_part} (тип не определен)"]
                confidence = 0.7
        else:
            # Если нет поля "Наименование:", ищем в самой строке
            tool_types = ["Державка", "Пластина", "Сверло", "Фреза", "Цанга", "Винт", "Патрон"]
            found_type = None
            name_part = text
            
            for tool_type in tool_types:
                if tool_type.lower() in text.lower():
                    found_type = tool_type
                    # Извлекаем полное наименование
                    words = text.split()
                    for i, word in enumerate(words):
                        if tool_type.lower() in word.lower():
                            # Берем несколько слов вокруг найденного типа
                            start = max(0, i-2)
                            end = min(len(words), i+3)
                            name_part = " ".join(words[start:end])
                            break
                    break
            
            if found_type:
                evidence = [f"Найдено наименование: {name_part} (тип: {found_type})"]
                confidence = 0.8
            else:
                evidence = [f"Наименование не определено явно"]
                confidence = 0.3
        
        return ReasoningStep(
            step_number=2,
            description="Определяю наименование инструмента",
            confidence=confidence,
            evidence=evidence
        )
    
    def _extract_denotation(self, text: str) -> ReasoningStep:
        """Извлечение обозначения инструмента."""
        # Ищем технические обозначения в строке
        # Обозначения обычно содержат буквы и цифры
        
        # Если есть поле "Товары работы услуги:", извлекаем из него
        if "Товары работы услуги:" in text:
            parts = text.split("Товары работы услуги:")
            if len(parts) > 1:
                description = parts[1].split("\n")[0].strip()
            else:
                description = text
            
            # Паттерны для обозначений
            denotation_patterns = [
                r'[A-Z]{2,}\d+[A-Z]*\d*[A-Z]*',  # SDJCR 1212 K11-S
                r'[A-Z]\d+[A-Z]*\d*[A-Z]*',      # DCMT 11T304
                r'[A-Z]{2,}\d+',                 # DH423051
                r'[A-Z]+\d+[A-Z]*\d*',           # ER16 426
                r'HSK-[A-Z]\d+',                 # HSK-A40
                r'BT\d+',                        # BT40
                r'ER\d+',                        # ER16
                r'CNMG\d{6}',                    # CNMG090304
                r'A\d{6}',                       # A011011
                r'[A-Z]+\d+-\d+',                # ER32-80
                r'[A-Z]+\d+[A-Z]*\d*[A-Z]*',     # MAS403BT
                r'[A-Z]{1,3}\d+',                # H100, AD, G63
                r'\d{9,12}'                      # 416321504020
            ]
            
            found_denotations = []
            for pattern in denotation_patterns:
                matches = re.findall(pattern, description)
                found_denotations.extend(matches)
            
            if found_denotations:
                # Объединяем все найденные обозначения
                all_denotations = []
                for denot in found_denotations:
                    if denot not in all_denotations:
                        all_denotations.append(denot)
                
                main_denotation = " ".join(all_denotations)
                evidence = [f"Найдено обозначение: {main_denotation}"]
                confidence = 0.9
            else:
                evidence = ["Обозначение не найдено в описании товара"]
                confidence = 0.2
        else:
            # Обычный поиск в тексте
            denotation_patterns = [
                r'[A-Z]{2,}\d+[A-Z]*\d*[A-Z]*',  # SDJCR 1212 K11-S
                r'[A-Z]\d+[A-Z]*\d*[A-Z]*',      # DCMT 11T304
                r'[A-Z]{2,}\d+',                 # DH423051
                r'[A-Z]+\d+[A-Z]*\d*',           # ER16 426
                r'HSK-[A-Z]\d+',                 # HSK-A40
                r'BT\d+',                        # BT40
                r'ER\d+',                        # ER16
                r'CNMG\d{6}',                    # CNMG090304
                r'A\d{6}',                       # A011011
                r'[A-Z]+\d+-\d+',                # ER32-80
                r'\d{9,12}'                      # 416321504020
            ]
            
            found_denotations = []
            for pattern in denotation_patterns:
                matches = re.findall(pattern, text)
                found_denotations.extend(matches)
            
            if found_denotations:
                # Берем самое длинное обозначение как основное
                main_denotation = max(found_denotations, key=len)
                evidence = [f"Найдено обозначение: {main_denotation}"]
                confidence = 0.9
            else:
                evidence = ["Обозначение не найдено"]
                confidence = 0.2
        
        return ReasoningStep(
            step_number=3,
            description="Извлекаю обозначение инструмента",
            confidence=confidence,
            evidence=evidence
        )
    
    def _identify_manufacturer(self, text: str) -> ReasoningStep:
        """Определение производителя."""
        # Ищем явное указание производителя в разных форматах
        
        # Сначала ищем в поле "Производитель"
        if "Производитель:" in text:
            # Извлекаем часть после "Производитель:"
            parts = text.split("Производитель:")
            if len(parts) > 1:
                mfg_part = parts[1].split("|")[0].strip()
            else:
                mfg_part = ""
            if mfg_part:
                evidence = [f"Найден производитель: {mfg_part}"]
                confidence = 0.95
            else:
                evidence = ["Поле 'Производитель' пустое"]
                confidence = 0.2
        elif "Фирма:" in text:
            # Извлекаем производителя из поля "Фирма"
            firm_match = re.search(r'Фирма:\s*([A-Za-z0-9-]+)', text)
            if firm_match:
                manufacturer = firm_match.group(1)
                evidence = [f"Найден производитель: {manufacturer} (из поля Фирма)"]
                confidence = 0.95
            else:
                evidence = ["Поле 'Фирма' не содержит производителя"]
                confidence = 0.2
        elif "Бренд:" in text:
            # Извлекаем производителя из поля "Бренд"
            brand_match = re.search(r'Бренд:\s*([A-Za-z0-9]+)', text)
            if brand_match:
                manufacturer = brand_match.group(1)
                evidence = [f"Найден производитель: {manufacturer} (из поля Бренд)"]
                confidence = 0.95
            else:
                evidence = ["Поле 'Бренд' не содержит производителя"]
                confidence = 0.2
        else:
            # Если нет поля "Производитель:" или "Бренд:", ищем в описании товара
            if "Товары работы услуги:" in text:
                # Извлекаем описание товара
                parts = text.split("Товары работы услуги:")
                if len(parts) > 1:
                    description = parts[1].split("\n")[0].strip()
                else:
                    description = text
                
                # Ищем известных производителей в описании
                manufacturers = self.knowledge_graph.get_related_entities("производитель")
                found_manufacturers = []
                
                for mfg in manufacturers:
                    if mfg.lower() in description.lower():
                        found_manufacturers.append(mfg)
                
                if found_manufacturers:
                    manufacturer = found_manufacturers[0]
                    evidence = [f"Найден производитель: {manufacturer} (в описании товара)"]
                    confidence = 0.9
                else:
                    # Ищем другие возможные названия производителей (слова с заглавными буквами)
                    words = description.split()
                    possible_manufacturers = []
                    for word in words:
                        if word[0].isupper() and len(word) > 2 and not word.isdigit():
                            # Исключаем технические обозначения
                            if not re.match(r'^[A-Z]+\d+', word) and word not in ['Цанговый', 'Патрон', 'мм']:
                                possible_manufacturers.append(word)
                    
                    if possible_manufacturers:
                        manufacturer = possible_manufacturers[0]
                        evidence = [f"Найден возможный производитель: {manufacturer} (в описании товара)"]
                        confidence = 0.7
                    else:
                        evidence = ["Производитель не найден в описании товара"]
                        confidence = 0.9
            else:
                # Если нет структурированных полей, ищем в тексте
                manufacturers = self.knowledge_graph.get_related_entities("производитель")
                found_manufacturers = []
                
                for mfg in manufacturers:
                    if mfg.lower() in text.lower():
                        found_manufacturers.append(mfg)
                
                if found_manufacturers:
                    manufacturer = found_manufacturers[0]
                    evidence = [f"Найден производитель: {manufacturer}"]
                    confidence = 0.9
                else:
                    evidence = ["Производитель не указан явно"]
                    confidence = 0.9  # Высокая уверенность в отсутствии производителя
        
        return ReasoningStep(
            step_number=4,
            description="Определяю производителя",
            confidence=confidence,
            evidence=evidence
        )
    
    def _extract_parameters(self, text: str) -> ReasoningStep:
        """Извлечение параметров инструмента."""
        # Ищем только явные параметры с единицами измерения
        # Не интерпретируем числа без контекста как параметры
        
        # Если есть поле "Товары работы услуги:", ищем в нем
        if "Товары работы услуги:" in text:
            parts = text.split("Товары работы услуги:")
            if len(parts) > 1:
                description = parts[1].split("\n")[0].strip()
            else:
                description = text
            search_text = description
        else:
            search_text = text
        
        found_parameters = {}
        
        # Расширенные паттерны для параметров
        extended_patterns = {
            "размеры": r'ø\s*(\d+(?:,\d+)?)\s*x\s*(\d+(?:,\d+)?)',  # Ø 40 x 10
            "диаметр": r'ø\s*(\d+(?:,\d+)?)\s*мм',
            "Lобщ": r"Lобщ\.?\s*=\s*(\d+(?:,\d+)?)\s*мм",
            "Lраб": r"Lраб\s*=\s*(\d+(?:,\d+)?)\s*мм",
            "угол": r"(\d+)\s*°",
            "количество": r"Кол\.?\s*:\s*(\d+)",
            "HSK": r"HSK-([A-Z]\d+)",
            "BT": r"BT(\d+)",
            "ER": r"ER(\d+)",
            "CNMG": r"CNMG(\d{6})",
            "размер_хвостовика": r"(\d+)\s*мм"
        }
        
        for param_name, pattern in extended_patterns.items():
            matches = re.findall(pattern, search_text, re.IGNORECASE)
            if matches:
                found_parameters[param_name] = matches
        
        if found_parameters:
            evidence = []
            for param, values in found_parameters.items():
                if param == "размеры":
                    # Сохраняем размеры как единое целое
                    evidence.append(f"Размеры: ø{values[0][0]} x {values[0][1]}")
                elif param == "диаметр":
                    evidence.append(f"Диаметр: ø{', '.join(values)} мм")
                else:
                    evidence.append(f"{param}: {', '.join(values)}")
            confidence = 0.8
        else:
            # Проверяем, есть ли числа без явных единиц измерения
            has_numbers = bool(re.search(r'\d+\.?\d*', search_text))
            if has_numbers:
                evidence = ["Найдены числа, но без явных единиц измерения (мм, °, и т.д.)"]
                confidence = 0.4
            else:
                evidence = ["Параметры не найдены"]
                confidence = 0.4
        
        return ReasoningStep(
            step_number=5,
            description="Извлекаю технические параметры с единицами измерения",
            confidence=confidence,
            evidence=evidence
        )
    
    def _extract_technical_task(self, text: str) -> ReasoningStep:
        """Извлечение технического задания (код заказа и другие параметры)."""
        technical_info = []
        
        # Ищем код заказа
        order_code_match = re.search(r'Код заказа:\s*(\d+)', text)
        if order_code_match:
            technical_info.append(f"Код заказа: {order_code_match.group(1)}")
        
        # Ищем параметры диаметра
        diameter_match = re.search(r'ø\s*(\d+(?:,\d+)?)\s*мм', text)
        if diameter_match:
            technical_info.append(f"Диаметр: ø{diameter_match.group(1)} мм")
        
        # Ищем технические требования в скобках
        bracket_match = re.search(r'\(([^)]+)\)', text)
        if bracket_match:
            bracket_content = bracket_match.group(1).strip()
            technical_info.append(bracket_content)
        
        # Ищем другие технические параметры
        if "Срок поставки:" in text:
            delivery_match = re.search(r'Срок поставки:\s*([^К]+)', text)
            if delivery_match:
                technical_info.append(f"Срок поставки: {delivery_match.group(1).strip()}")
        
        if technical_info:
            evidence = technical_info
            confidence = 0.9
        else:
            evidence = ["Техническое задание не найдено"]
            confidence = 0.3
        
        return ReasoningStep(
            step_number=7,
            description="Извлекаю техническое задание",
            confidence=confidence,
            evidence=evidence
        )
    
    def _validate_result(self, text: str, previous_steps: List[ReasoningStep]) -> ReasoningStep:
        """Валидация результата разбора."""
        # Проверяем согласованность результатов
        validation_checks = []
        
        # Проверка 1: Есть ли основные извлеченные элементы
        has_denotation = any("обозначение" in step.evidence[0].lower() and "найдено" in step.evidence[0].lower() 
                            for step in previous_steps if step.step_number == 3)
        has_name = any("наименование" in step.evidence[0].lower() and "найдено" in step.evidence[0].lower() 
                      for step in previous_steps if step.step_number == 2)
        has_manufacturer = any("производитель" in step.evidence[0].lower() and "найден" in step.evidence[0].lower() 
                              for step in previous_steps if step.step_number == 4)
        has_quantity = "Количество:" in text or "Кол.:" in text or re.search(r'\d+\s*шт', text)
        has_unit = "Единица:" in text or "Единица измерения:" in text or re.search(r'\d+\s*шт', text)
        
        # Если есть количество, но нет единицы измерения, устанавливаем "шт." по умолчанию
        if has_quantity and not has_unit:
            validation_checks.append("✓ Количество найдено, единица измерения установлена как 'шт.' по умолчанию")
            has_unit = True  # Считаем, что единица есть
        elif has_quantity and has_unit:
            validation_checks.append("✓ Количество и единица измерения найдены")
        elif not has_quantity:
            validation_checks.append("⚠ Количество не найдено")
        
        if has_denotation and has_name and has_quantity and has_unit:
            if has_manufacturer:
                validation_checks.append("✓ Извлечена вся основная информация (наименование, обозначение, производитель, количество, единица)")
            else:
                validation_checks.append("✓ Извлечена основная информация (наименование, обозначение, количество, единица) - производитель не указан")
        elif has_denotation and has_name:
            validation_checks.append("✓ Извлечена основная информация (наименование, обозначение)")
        else:
            validation_checks.append("✗ Не вся основная информация извлечена")
        
        # Проверка 2: Согласованность наименования и обозначения
        name_step = next((step for step in previous_steps if step.step_number == 2), None)
        denotation_step = next((step for step in previous_steps if step.step_number == 3), None)
        
        if name_step and denotation_step:
            if name_step.confidence > 0.7 and denotation_step.confidence > 0.7:
                validation_checks.append("✓ Наименование и обозначение согласованы")
            else:
                validation_checks.append("⚠ Низкая уверенность в наименовании или обозначении")
        
        # Общая оценка
        overall_confidence = sum(step.confidence for step in previous_steps) / len(previous_steps)
        
        return ReasoningStep(
            step_number=7,
            description="Валидирую результат разбора",
            confidence=overall_confidence,
            evidence=validation_checks
        )
    
    def _calculate_type_confidence(self, text: str, tool_type: str) -> float:
        """Вычисление уверенности в определении типа инструмента."""
        base_confidence = 0.5
        
        # Бонус за позицию в начале строки
        if text.lower().startswith(tool_type.lower()):
            base_confidence += 0.3
        
        # Бонус за длину совпадения
        if len(tool_type) > 5:
            base_confidence += 0.1
        
        # Бонус за контекстные слова
        context_words = ["инструмент", "пластина", "державка", "сверло", "фреза"]
        if any(word in text.lower() for word in context_words):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _extract_denotation_from_text(self, text: str) -> Optional[str]:
        """Извлечение обозначения из текста."""
        denotation_patterns = [
            r'[A-Z]{2,}\d+[A-Z]*\d*[A-Z]*',
            r'[A-Z]\d+[A-Z]*\d*[A-Z]*',
            r'[A-Z]{2,}\d+',
            r'[A-Z]+\d+[A-Z]*\d*'
        ]
        
        for pattern in denotation_patterns:
            matches = re.findall(pattern, text)
            if matches:
                return max(matches, key=len)
        
        return None
    
    def format_reasoning_for_prompt(self, reasoning_steps: List[ReasoningStep]) -> str:
        """Форматирование цепочки рассуждений для промпта."""
        formatted_steps = []
        
        for step in reasoning_steps:
            step_text = f"{step.step_number}. {step.description}"
            if step.evidence:
                step_text += f" ({', '.join(step.evidence)})"
            formatted_steps.append(step_text)
        
        return "\n".join(formatted_steps)
    
    def get_reasoning_summary(self, reasoning_steps: List[ReasoningStep]) -> Dict:
        """Получить сводку по цепочке рассуждений."""
        return {
            "total_steps": len(reasoning_steps),
            "average_confidence": sum(step.confidence for step in reasoning_steps) / len(reasoning_steps),
            "high_confidence_steps": len([step for step in reasoning_steps if step.confidence > 0.7]),
            "low_confidence_steps": len([step for step in reasoning_steps if step.confidence < 0.5]),
            "reasoning_text": self.format_reasoning_for_prompt(reasoning_steps)
        }


# Глобальный экземпляр генератора
cot_generator = CoTReasoningGenerator()
