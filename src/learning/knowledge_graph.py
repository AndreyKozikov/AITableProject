"""
Модуль для работы с графом знаний промышленных инструментов.

Содержит структурированные знания о производителях, типах инструментов,
обозначениях и параметрах для улучшения точности разбора.
"""

from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class Entity:
    """Сущность в графе знаний."""
    name: str
    type: str
    attributes: Dict[str, str] = None


@dataclass
class Relation:
    """Связь между сущностями в графе знаний."""
    from_entity: str
    to_entity: str
    relation_type: str
    confidence: float = 1.0


class IndustrialKnowledgeGraph:
    """Граф знаний для промышленных инструментов."""
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Инициализация базовых знаний о промышленных инструментах."""
        
        # Производители (расширено данными из каталогов)
        manufacturers = [
            "Sandvik", "YG-1", "KELITE", "ZCC", "EROGLU", "DAndrea", 
            "JieHe", "ZeTOOL", "KGM", "CNMG", "DCMT"
        ]
        
        for mfg in manufacturers:
            self.add_entity(mfg, "производитель", {
                "страна": self._get_manufacturer_country(mfg),
                "специализация": self._get_manufacturer_specialization(mfg)
            })
        
        # Типы инструментов (расширено данными из каталогов)
        tool_types = [
            "Державка", "Пластина", "Сверло", "Фреза", "Цанга", 
            "Винт", "Сверло центровочное", "Твердосплавная концевая фреза",
            "Державка для сверла", "Державка для фрезы", "Державка для пластины",
            "Цанга для сверла", "Цанга для фрезы", "Цанга для инструмента"
        ]
        
        for tool_type in tool_types:
            self.add_entity(tool_type, "тип_инструмента", {
                "категория": self._get_tool_category(tool_type),
                "назначение": self._get_tool_purpose(tool_type)
            })
        
        # Параметры
        parameters = [
            "Lобщ", "Lраб", "Ø", "угол", "материал", "покрытие",
            "тип_крепления", "размер_хвостовика"
        ]
        
        for param in parameters:
            self.add_entity(param, "параметр", {
                "единица_измерения": self._get_parameter_unit(param),
                "тип_значения": self._get_parameter_type(param)
            })
        
        # Единицы измерения
        units = ["мм", "градус", "шт", "мкм", "дюйм"]
        for unit in units:
            self.add_entity(unit, "единица_измерения")
        
        # Создание связей
        self._create_relations()
    
    def add_entity(self, name: str, entity_type: str, attributes: Dict[str, str] = None):
        """Добавить сущность в граф."""
        self.entities[name] = Entity(name, entity_type, attributes or {})
    
    def add_relation(self, from_entity: str, to_entity: str, relation_type: str, confidence: float = 1.0):
        """Добавить связь между сущностями."""
        self.relations.append(Relation(from_entity, to_entity, relation_type, confidence))
    
    def get_related_entities(self, entity_name: str, relation_type: str = None) -> List[str]:
        """Получить связанные сущности."""
        related = []
        for rel in self.relations:
            if rel.from_entity == entity_name and (relation_type is None or rel.relation_type == relation_type):
                related.append(rel.to_entity)
            elif rel.to_entity == entity_name and (relation_type is None or rel.relation_type == relation_type):
                related.append(rel.from_entity)
        return related
    
    def get_entity_info(self, entity_name: str) -> Dict:
        """Получить информацию о сущности."""
        if entity_name not in self.entities:
            return {}
        
        entity = self.entities[entity_name]
        return {
            "name": entity.name,
            "type": entity.type,
            "attributes": entity.attributes,
            "relations": self.get_related_entities(entity_name)
        }
    
    def find_manufacturer_by_denotation(self, denotation: str) -> str:
        """Найти производителя по обозначению."""
        # Простые правила для определения производителя по обозначению
        denotation_upper = denotation.upper()
        
        if "SDJCR" in denotation_upper or "CNMG" in denotation_upper:
            return "Sandvik"
        elif "YG" in denotation_upper or "DH" in denotation_upper or "DJ" in denotation_upper:
            return "YG-1"
        elif "KGM" in denotation_upper:
            return "KELITE"
        elif "DCMT" in denotation_upper:
            return "ZCC"
        elif "ER" in denotation_upper:
            return "EROGLU"
        elif "HSK" in denotation_upper or "BT" in denotation_upper:
            return "DAndrea"
        elif "A0" in denotation_upper or "A1" in denotation_upper:
            return "JieHe"
        elif "CNMG" in denotation_upper and "MB" in denotation_upper:
            return "KELITE"
        elif "BT40" in denotation_upper or "ER16" in denotation_upper:
            return "ZeTOOL"
        
        return "Неизвестно"
    
    def get_parameter_patterns(self) -> Dict[str, str]:
        """Получить паттерны для извлечения параметров."""
        return {
            "Lобщ": r"Lобщ\.?\s*=\s*(\d+(?:,\d+)?)\s*мм",
            "Lраб": r"Lраб\s*=\s*(\d+(?:,\d+)?)\s*мм",
            "диаметр": r"Ø\s*(\d+(?:,\d+)?)\s*мм",
            "угол": r"(\d+)\s*°",
            "количество": r"Кол\.?\s*:\s*(\d+)",
            "HSK": r"HSK-([A-Z]\d+)",
            "BT": r"BT(\d+)",
            "ER": r"ER(\d+)",
            "CNMG": r"CNMG(\d{6})",
            "размер_хвостовика": r"(\d+)\s*мм"
        }
    
    def get_catalog_manufacturers(self) -> List[str]:
        """Получить список производителей из каталогов."""
        return ["DAndrea", "JieHe", "KELITE", "ZeTOOL"]
    
    def get_tool_type_patterns(self) -> Dict[str, List[str]]:
        """Получить паттерны для определения типов инструментов."""
        return {
            "Державка": ["державка", "holder", "HSK", "BT"],
            "Пластина": ["пластина", "insert", "CNMG", "DCMT"],
            "Сверло": ["сверло", "drill", "центровочное"],
            "Фреза": ["фреза", "mill", "концевая", "end mill"],
            "Цанга": ["цанга", "collet", "ER", "зажимная"],
            "Винт": ["винт", "screw", "болт"]
        }
    
    def identify_tool_type_from_description(self, description: str) -> str:
        """Определить тип инструмента по описанию."""
        description_lower = description.lower()
        patterns = self.get_tool_type_patterns()
        
        for tool_type, keywords in patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                return tool_type
        
        return "Неизвестно"
    
    def _create_relations(self):
        """Создать связи между сущностями."""
        # Связи производителей с типами инструментов
        self.add_relation("Sandvik", "Державка", "производит")
        self.add_relation("Sandvik", "Пластина", "производит")
        self.add_relation("YG-1", "Сверло", "производит")
        self.add_relation("YG-1", "Фреза", "производит")
        self.add_relation("KELITE", "Фреза", "производит")
        self.add_relation("KELITE", "Пластина", "производит")
        self.add_relation("ZCC", "Пластина", "производит")
        self.add_relation("EROGLU", "Цанга", "производит")
        self.add_relation("DAndrea", "Державка", "производит")
        self.add_relation("DAndrea", "Державка для сверла", "производит")
        self.add_relation("DAndrea", "Державка для фрезы", "производит")
        self.add_relation("JieHe", "Державка", "производит")
        self.add_relation("JieHe", "Державка для сверла", "производит")
        self.add_relation("JieHe", "Державка для фрезы", "производит")
        self.add_relation("ZeTOOL", "Цанга", "производит")
        self.add_relation("ZeTOOL", "Цанга для сверла", "производит")
        self.add_relation("ZeTOOL", "Цанга для фрезы", "производит")
        
        # Связи параметров с единицами измерения
        self.add_relation("Lобщ", "мм", "единица_измерения")
        self.add_relation("Lраб", "мм", "единица_измерения")
        self.add_relation("Ø", "мм", "единица_измерения")
        self.add_relation("угол", "градус", "единица_измерения")
        
        # Связи типов инструментов с категориями
        self.add_relation("Державка", "режущий_инструмент", "категория")
        self.add_relation("Пластина", "режущий_инструмент", "категория")
        self.add_relation("Сверло", "сверлильный_инструмент", "категория")
        self.add_relation("Фреза", "фрезерный_инструмент", "категория")
        self.add_relation("Цанга", "зажимной_инструмент", "категория")
    
    def _get_manufacturer_country(self, manufacturer: str) -> str:
        """Получить страну производителя."""
        country_map = {
            "Sandvik": "Швеция",
            "YG-1": "Южная Корея", 
            "KELITE": "Китай",
            "ZCC": "Китай",
            "EROGLU": "Турция",
            "DAndrea": "Италия",
            "JieHe": "Китай",
            "ZeTOOL": "Китай"
        }
        return country_map.get(manufacturer, "Неизвестно")
    
    def _get_manufacturer_specialization(self, manufacturer: str) -> str:
        """Получить специализацию производителя."""
        specialization_map = {
            "Sandvik": "Металлообработка, горное дело",
            "YG-1": "Режущие инструменты",
            "KELITE": "Твердосплавные инструменты",
            "ZCC": "Пластины и державки",
            "EROGLU": "Зажимные системы",
            "DAndrea": "Державки и зажимные системы",
            "JieHe": "Державки и инструментальная оснастка",
            "ZeTOOL": "Цанги и зажимные системы"
        }
        return specialization_map.get(manufacturer, "Общее машиностроение")
    
    def _get_tool_category(self, tool_type: str) -> str:
        """Получить категорию инструмента."""
        category_map = {
            "Державка": "режущий_инструмент",
            "Пластина": "режущий_инструмент", 
            "Сверло": "сверлильный_инструмент",
            "Фреза": "фрезерный_инструмент",
            "Цанга": "зажимной_инструмент",
            "Винт": "крепежный_элемент"
        }
        return category_map.get(tool_type, "общий_инструмент")
    
    def _get_tool_purpose(self, tool_type: str) -> str:
        """Получить назначение инструмента."""
        purpose_map = {
            "Державка": "Крепление режущих пластин",
            "Пластина": "Резание металла",
            "Сверло": "Сверление отверстий",
            "Фреза": "Фрезерование поверхностей",
            "Цанга": "Зажим заготовок"
        }
        return purpose_map.get(tool_type, "Обработка материалов")
    
    def _get_parameter_unit(self, parameter: str) -> str:
        """Получить единицу измерения параметра."""
        unit_map = {
            "Lобщ": "мм",
            "Lраб": "мм", 
            "Ø": "мм",
            "угол": "градус",
            "материал": "текст"
        }
        return unit_map.get(parameter, "безразмерная")
    
    def _get_parameter_type(self, parameter: str) -> str:
        """Получить тип значения параметра."""
        type_map = {
            "Lобщ": "число",
            "Lраб": "число",
            "Ø": "число", 
            "угол": "число",
            "материал": "строка"
        }
        return type_map.get(parameter, "строка")
    
    def save_to_file(self, file_path: Path):
        """Сохранить граф знаний в файл."""
        data = {
            "entities": {name: {
                "name": entity.name,
                "type": entity.type,
                "attributes": entity.attributes
            } for name, entity in self.entities.items()},
            "relations": [{
                "from_entity": rel.from_entity,
                "to_entity": rel.to_entity,
                "relation_type": rel.relation_type,
                "confidence": rel.confidence
            } for rel in self.relations]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, file_path: Path):
        """Загрузить граф знаний из файла."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Загружаем сущности
        for name, entity_data in data["entities"].items():
            self.add_entity(
                entity_data["name"],
                entity_data["type"], 
                entity_data["attributes"]
            )
        
        # Загружаем связи
        for rel_data in data["relations"]:
            self.add_relation(
                rel_data["from_entity"],
                rel_data["to_entity"],
                rel_data["relation_type"],
                rel_data["confidence"]
            )


# Глобальный экземпляр графа знаний
knowledge_graph = IndustrialKnowledgeGraph()
