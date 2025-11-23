"""
Модуль для работы с графом знаний промышленных инструментов.

Содержит структурированные знания о производителях, типах инструментов,
обозначениях и параметрах для улучшения точности разбора.

Граф знаний загружается из файла knowledge_graph.json при инициализации.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


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
    
    def __init__(self, json_file_path: Optional[Path] = None):
        """
        Инициализация графа знаний.
        
        Args:
            json_file_path: Путь к JSON файлу с графом знаний. 
                           Если None, используется файл knowledge_graph.json 
                           в той же директории, что и модуль.
        """
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        
        # Определяем путь к JSON файлу
        if json_file_path is None:
            json_file_path = Path(__file__).parent / "knowledge_graph.json"
        
        self.json_file_path = json_file_path
        self._load_from_json()
    
    def _load_from_json(self):
        """Загрузка графа знаний из JSON файла."""
        try:
            if not self.json_file_path.exists():
                logger.error(f"Файл графа знаний не найден: {self.json_file_path}")
                logger.warning("Инициализация пустого графа знаний")
                return
            
            logger.info(f"Загрузка графа знаний из файла: {self.json_file_path}")
            
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Загружаем сущности
            entities_data = data.get("entities", {})
            for name, entity_data in entities_data.items():
                self.add_entity(
                    entity_data.get("name", name),
                    entity_data.get("type", "неизвестно"),
                    entity_data.get("attributes", {})
                )
            
            logger.info(f"Загружено сущностей: {len(self.entities)}")
            
            # Загружаем связи
            relations_data = data.get("relations", [])
            for rel_data in relations_data:
                self.add_relation(
                    rel_data.get("from_entity", ""),
                    rel_data.get("to_entity", ""),
                    rel_data.get("relation_type", ""),
                    rel_data.get("confidence", 1.0)
                )
            
            logger.info(f"Загружено связей: {len(self.relations)}")
            
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON файла: {e}")
            logger.warning("Инициализация пустого графа знаний")
        except Exception as e:
            logger.error(f"Ошибка загрузки графа знаний: {e}")
            logger.warning("Инициализация пустого графа знаний")
    
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
        # Базовые паттерны для известных параметров
        base_patterns = {
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
        
        # Можно расширить паттерны на основе загруженных параметров из графа
        # Пока возвращаем базовые паттерны
        return base_patterns
    
    def get_catalog_manufacturers(self) -> List[str]:
        """Получить список производителей из каталогов."""
        # Извлекаем производителей из загруженных сущностей
        manufacturers = []
        for name, entity in self.entities.items():
            if entity.type == "производитель":
                manufacturers.append(name)
        return sorted(manufacturers) if manufacturers else ["DAndrea", "JieHe", "KELITE", "ZeTOOL"]
    
    def get_tool_type_patterns(self) -> Dict[str, List[str]]:
        """Получить паттерны для определения типов инструментов."""
        # Извлекаем типы инструментов из загруженных сущностей
        patterns = {}
        for name, entity in self.entities.items():
            if entity.type == "тип_инструмента":
                # Базовые паттерны на основе названия
                name_lower = name.lower()
                patterns[name] = [name_lower]
                
                # Добавляем английские эквиваленты для известных типов
                if "державка" in name_lower:
                    patterns[name].extend(["holder", "HSK", "BT"])
                elif "пластина" in name_lower:
                    patterns[name].extend(["insert", "CNMG", "DCMT"])
                elif "сверло" in name_lower:
                    patterns[name].extend(["drill", "центровочное"])
                elif "фреза" in name_lower:
                    patterns[name].extend(["mill", "концевая", "end mill"])
                elif "цанга" in name_lower:
                    patterns[name].extend(["collet", "ER", "зажимная"])
                elif "винт" in name_lower:
                    patterns[name].extend(["screw", "болт"])
        
        # Если паттерны не найдены, возвращаем базовые
        if not patterns:
            return {
                "Державка": ["державка", "holder", "HSK", "BT"],
                "Пластина": ["пластина", "insert", "CNMG", "DCMT"],
                "Сверло": ["сверло", "drill", "центровочное"],
                "Фреза": ["фреза", "mill", "концевая", "end mill"],
                "Цанга": ["цанга", "collet", "ER", "зажимная"],
                "Винт": ["винт", "screw", "болт"]
            }
        
        return patterns
    
    def identify_tool_type_from_description(self, description: str) -> str:
        """Определить тип инструмента по описанию."""
        description_lower = description.lower()
        patterns = self.get_tool_type_patterns()
        
        for tool_type, keywords in patterns.items():
            if any(keyword in description_lower for keyword in keywords):
                return tool_type
        
        return "Неизвестно"
    
    
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
        """Загрузить граф знаний из файла (перезагрузка данных)."""
        # Очищаем текущие данные
        self.entities.clear()
        self.relations.clear()
        
        # Устанавливаем новый путь и загружаем
        self.json_file_path = file_path
        self._load_from_json()


# Глобальный экземпляр графа знаний
knowledge_graph = IndustrialKnowledgeGraph()
