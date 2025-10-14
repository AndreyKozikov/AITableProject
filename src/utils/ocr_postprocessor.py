"""
Модуль постобработки OCR-результатов из CSV-таблиц (Модернизированная версия).

Выполняет очистку и коррекцию текста с искажениями:
- смешение кириллицы и латиницы
- неверный регистр
- артефакты распознавания
- неправильные окончания

Использует специализированные библиотеки для улучшения качества:
- ftfy: исправление кодировочных артефактов
- rapidfuzz: быстрая нечеткая строковая подстановка
- pymorphy3: морфологический анализ
- razdel: токенизация с сохранением пробелов (fallback на regex)
- confusable_homoglyphs: замена омографов
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Callable, Any
import unicodedata

# Обязательные библиотеки
import regex  # Вместо стандартного re
import ftfy
import rapidfuzz
from rapidfuzz import fuzz, process
import pymorphy3
import pandas as pd

# Опциональные библиотеки
try:
    import razdel
    RAZDEL_AVAILABLE = True
except ImportError:
    RAZDEL_AVAILABLE = False
    logging.warning("razdel not available, using regex fallback for tokenization")

try:
    from confusable_homoglyphs import confusables
    CONFUSABLES_AVAILABLE = True
except ImportError:
    CONFUSABLES_AVAILABLE = False
    logging.warning("confusable-homoglyphs not available, using basic conversion")

# Настройка логирования
logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Конфигурация препроцессора OCR"""
    
    # Чувствительность исправлений
    max_distance_long: int = 2  # Для слов длиной > 5
    max_distance_short: int = 3  # Для слов длиной ≤ 5
    short_word_threshold: int = 5
    
    # Токенизация
    use_razdel: bool = RAZDEL_AVAILABLE
    
    # Защита технических токенов
    protect_technical: bool = True
    protect_brands: bool = True
    
    # Debug режим
    debug: bool = False
    log_corrections: bool = False
    
    # Технические паттерны (используем regex, не re)
    technical_patterns: List[str] = field(default_factory=lambda: [
        r'[A-ZА-Я]{1,3}\d{1,4}[A-ZА-Яа-я\d\'\.\-]*',
        r'[A-ZА-Яа-я]+\d+[A-ZА-Яа-я\d\-\.]*',
        r'[A-Z]{2,}(?:\s+\d+)?',  # Аббревиатуры типа TRM, BT
        r'[A-ZА-Я]{2,}[A-ZА-Я]{2,}',  # CCMT, ССМТ и т.д.
        r'[A-ZА-Я]{2}[A-ZА-Я]{2}\s*\d+',  # CCMT 120404
    ])
    
    # Известные бренды (эталонные названия)
    known_brands: List[str] = field(default_factory=lambda: [
        "D'Andrea", "Sandvik", "Iscar", "Kennametal", "Seco", "Walter",
        "Mitsubishi", "Sumitomo", "Kyocera", "Tungaloy", "Korloy"
    ])
    
    # Минимальный score для распознавания бренда
    brand_fuzzy_threshold: int = 75
    
    # Доменные термины (технические слова)
    domain_terms: List[str] = field(default_factory=lambda: [
        "резец", "державка", "пластина", "цанговый", "патрон",
        "хвостовик", "расточной", "расточная", "кассета", "кассеты",
        "пружинный", "прецизионный", "инструмент", "вставка",
        "головка", "резцедержатель", "переходник", "вкладыш"
    ])


@dataclass
class Token:
    """Токен текста с метаинформацией"""
    text: str
    token_type: str  # 'word', 'numeric', 'technical', 'special', 'whitespace'
    position: int
    original: str
    correction_applied: Optional[str] = None  # Для debug


class OCRPostProcessor:
    """
    Класс для постобработки OCR-результатов с использованием
    специализированных библиотек.
    """
    
    # Базовая карта схожих символов латиницы и кириллицы (расширенная)
    BASIC_LATIN_TO_CYRILLIC = {
        'A': 'А', 'a': 'а', 'B': 'В', 'E': 'Е', 'e': 'е', 'K': 'К', 'M': 'М', 
        'H': 'Н', 'O': 'О', 'o': 'о', 'P': 'Р', 'p': 'р', 'C': 'С', 'c': 'с',
        'T': 'Т', 't': 'т', 'y': 'у', 'X': 'Х', 'x': 'х',
        # Дополнительные часто путаемые символы
        'r': 'г',  # r может быть г или т в контексте
        'u': 'и',  # u часто путается с и
        'n': 'п',  # n может быть п
        'i': 'і',  # i может быть і (украинская) или і
        'I': 'І',
        'N': 'П',
        'U': 'И',
        'Y': 'У',
    }
    
    # Обратная карта кириллицы в латиницу (для технических обозначений)
    BASIC_CYRILLIC_TO_LATIN = {v: k for k, v in BASIC_LATIN_TO_CYRILLIC.items()}
    
    # Конверсия OCR-ошибок (цифры в кириллицу)
    DIGIT_TO_CYRILLIC = {
        '3': 'З',
        '6': 'б',
    }
    
    # Замена цифр на похожие буквы в начале русских слов
    LEADING_DIGIT_TO_LETTER = {
        '1': 'Т',  # 1ехническое -> Техническое
        '3': 'З',  # 3амена -> Замена
    }
    
    # Замена специальных символов в единицах измерения
    SPECIAL_UNIT_CHARS = {
        'ⅲ': 'ш',  # римская тройка
        'Ⅲ': 'Ш',
        'ⅰ': 'i',
        'ⅱ': 'ii',
    }
    
    def __init__(self, config: Optional[OCRConfig] = None, dictionary_size: int = 100000):
        """
        Инициализация процессора
        
        Args:
            config: Объект конфигурации
            dictionary_size: Размер словаря для проверки слов
        """
        self.cfg = config or OCRConfig()
        
        # Инициализация морфоанализатора
        self.morph = pymorphy3.MorphAnalyzer()
        
        # Кэш для ускорения поиска
        self.dictionary_cache: Dict[str, str] = {}
        
        # Построение словарей
        self.russian_words = self._build_russian_dictionary()
        self.domain_dict = self._build_domain_dictionary()
        self.brand_dict = self._build_brand_dictionary()
        
        # Компиляция технических паттернов
        self.technical_patterns_compiled = [
            regex.compile(p) for p in self.cfg.technical_patterns
        ]
        
        # Инициализация логирования
        if self.cfg.debug:
            logger.setLevel(logging.DEBUG)
    
    def _build_russian_dictionary(self) -> set:
        """
        Построение словаря русских слов на основе pymorphy3
        
        Returns:
            Множество русских слов
        """
        base_words = set(self.cfg.domain_terms)
        
        # Добавление всех форм базовых слов
        extended_words = set()
        for word in base_words:
            parsed = self.morph.parse(word)
            if parsed:
                for p in parsed:
                    # Добавление нормальной формы
                    extended_words.add(p.normal_form)
                    # Добавление всех словоформ
                    try:
                        for lexeme in p.lexeme:
                            extended_words.add(lexeme.word)
                    except:
                        pass
        
        return extended_words | base_words
    
    def _build_domain_dictionary(self) -> Dict[str, str]:
        """
        Построение доменного словаря для быстрого поиска
        
        Returns:
            Словарь {нормализованное_слово: каноническое_слово}
        """
        domain_dict = {}
        for term in self.cfg.domain_terms:
            domain_dict[term.lower()] = term
            # Добавляем формы через морфологию
            parsed = self.morph.parse(term)
            if parsed:
                for p in parsed:
                    domain_dict[p.normal_form.lower()] = term
        
        return domain_dict
    
    def _build_brand_dictionary(self) -> Dict[str, str]:
        """
        Построение словаря брендов с вариантами написания
        
        Returns:
            Словарь {вариант_бренда: эталонный_бренд}
        """
        brand_dict = {}
        
        for brand in self.cfg.known_brands:
            # Эталонное название
            brand_dict[brand.lower()] = brand
            
            # Варианты без апострофов и кавычек
            normalized = regex.sub(r"[''`ʹ]", "", brand)
            brand_dict[normalized.lower()] = brand
            
            # Вариант с пробелом вместо апострофа
            spaced = regex.sub(r"[''`ʹ]", " ", brand)
            brand_dict[spaced.lower()] = brand
            
            # Генерация вариантов со смешанными символами для латинских брендов
            if regex.search(r'[A-Za-z]', brand):
                # Создаем варианты где латинские буквы заменены на похожие кириллические
                cyrillic_variant = brand
                for lat, cyr in self.BASIC_LATIN_TO_CYRILLIC.items():
                    cyrillic_variant = cyrillic_variant.replace(lat, cyr)
                brand_dict[cyrillic_variant.lower()] = brand
        
        return brand_dict
    
    def normalize_brand(self, text: str) -> str:
        """
        Универсальная нормализация брендов в тексте
        
        Args:
            text: Текст с возможно искаженными названиями брендов
            
        Returns:
            Текст с нормализованными брендами и пробелами вокруг них
        """
        if not text:
            return text
        
        # 1. Нормализация апострофов и кавычек во всех словах с заглавной буквы
        # Заменяем все варианты апострофов на стандартный
        text = regex.sub(r"([A-ZА-Я][a-zA-Zа-яА-Я]*)[''`ʹ]", r"\1'", text)
        
        # 2. Поиск и замена известных брендов через прямое совпадение
        words = regex.findall(r"[A-ZА-Я][a-zA-Zа-яА-Я''\-]*", text)
        
        for word in words:
            word_clean = regex.sub(r"[''`ʹ]", "", word).lower()
            
            # Проверка в словаре брендов
            if word_clean in self.brand_dict:
                canonical_brand = self.brand_dict[word_clean]
                # Заменяем в тексте с пробелами вокруг бренда
                text = regex.sub(
                    r'\b' + regex.escape(word) + r'\b',
                    f' {canonical_brand} ',
                    text,
                    flags=regex.IGNORECASE
                )
            else:
                # 3. Fuzzy поиск для неизвестных вариантов
                result = process.extractOne(
                    word_clean,
                    self.brand_dict.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=self.cfg.brand_fuzzy_threshold
                )
                
                if result:
                    matched_variant, score, _ = result
                    canonical_brand = self.brand_dict[matched_variant]
                    logger.debug(
                        f"Brand fuzzy match: '{word}' -> '{canonical_brand}' (score={score})"
                    )
                    # Заменяем в тексте с пробелами вокруг бренда
                    text = regex.sub(
                        r'\b' + regex.escape(word) + r'\b',
                        f' {canonical_brand} ',
                        text
                    )
        
        # 4. Убираем множественные пробелы
        text = regex.sub(r'\s+', ' ', text)
        
        # 5. Убираем пробелы перед знаками препинания
        text = regex.sub(r'\s+([.,;!?:])', r'\1', text)
        
        # 6. Убираем пробелы в начале и конце
        text = text.strip()
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """
        Базовая предобработка текста
        
        Args:
            text: Исходный текст
            
        Returns:
            Предобработанный текст
        """
        if not text or pd.isna(text):
            return ""
        
        # 1. Нормализация через ftfy (исправление кодировки и mojibake)
        text = ftfy.fix_text(text)
        
        # 2. Нормализация пробелов и кавычек
        text = regex.sub(r'[\u2018\u2019]', "'", text)  # Умные одинарные кавычки
        text = regex.sub(r'[\u201C\u201D]', '"', text)  # Умные двойные кавычки
        text = regex.sub(r'[\u2013\u2014]', '-', text)  # Em/en дэш
        
        # 3. Замена специальных символов (римские цифры и т.д.)
        for special_char, replacement in self.SPECIAL_UNIT_CHARS.items():
            text = text.replace(special_char, replacement)
        
        # 4. Удаление маркеров списка (тире в начале)
        text = regex.sub(r'^\s*[-–—]\s*', '', text)  # В начале строки
        
        # 5. Универсальная нормализация брендов
        text = self.normalize_brand(text)
        
        # 6. Нормализация единиц измерения (добавляем пробел между цифрой и единицей)
        # Сначала обрабатываем слитные варианты
        text = regex.sub(r'(\d+)([Шш][Тт]\.?|ШТ\.?)', r'\1 \2', text)
        # Затем унифицируем регистр и формат
        text = regex.sub(r'(\d+)\s+([Шш][Тт]\.?|шт\.?|ШТ\.?)', r'\1 шт.', text)
        
        # 7. Замена цифр в начале русских слов
        # 1ехническое -> Техническое
        for digit, letter in self.LEADING_DIGIT_TO_LETTER.items():
            text = regex.sub(
                rf'\b{digit}([а-яА-ЯёЁ]{{2,}})',
                rf'{letter}\1',
                text
            )
        
        return text
    
    def tokenize(self, text: str) -> List[Token]:
        """
        Токенизация текста с сохранением пробелов и знаков препинания
        
        Args:
            text: Текст для токенизации
            
        Returns:
            Список токенов
        """
        if self.cfg.use_razdel and RAZDEL_AVAILABLE:
            return self._tokenize_razdel(text)
        else:
            return self._tokenize_regex(text)
    
    def _tokenize_razdel(self, text: str) -> List[Token]:
        """Токенизация через razdel"""
        tokens = []
        position = 0
        
        # razdel возвращает только слова, нужно восстановить пробелы
        for token in razdel.tokenize(text):
            # Добавляем пробелы перед токеном если они были
            if token.start > position:
                whitespace = text[position:token.start]
                tokens.append(Token(
                    text=whitespace,
                    token_type='whitespace',
                    position=position,
                    original=whitespace
                ))
            
            # Добавляем сам токен
            token_text = token.text
            token_type = self.detect_token_type(token_text)
            tokens.append(Token(
                text=token_text,
                token_type=token_type,
                position=token.start,
                original=token_text
            ))
            
            position = token.stop
        
        # Добавляем оставшиеся символы
        if position < len(text):
            remaining = text[position:]
            tokens.append(Token(
                text=remaining,
                token_type='whitespace',
                position=position,
                original=remaining
            ))
        
        return tokens
    
    def _tokenize_regex(self, text: str) -> List[Token]:
        """Fallback токенизация через regex"""
        pattern = r'(\s+|[^\w\s])'  # Разделяем пробелы и знаки
        parts = regex.split(pattern, text)
        tokens = []
        position = 0
        
        for part in parts:
            if not part:
                continue
            
            if part.isspace():
                token_type = 'whitespace'
            else:
                token_type = self.detect_token_type(part)
            
            tokens.append(Token(
                text=part,
                token_type=token_type,
                position=position,
                original=part
            ))
            position += len(part)
        
        return tokens
    
    def convert_to_latin(self, word: str) -> str:
        """
        Конвертация смешанного технического обозначения в латиницу
        
        Args:
            word: Техническое обозначение со смешанными символами
            
        Returns:
            Обозначение в латинице
        """
        result = []
        for char in word:
            if char in self.BASIC_CYRILLIC_TO_LATIN:
                result.append(self.BASIC_CYRILLIC_TO_LATIN[char])
            else:
                result.append(char)
        return ''.join(result)
    
    def detect_token_type(self, token: str) -> str:
        """
        Определение типа токена
        
        Args:
            token: Токен для анализа
            
        Returns:
            Тип токена: 'word', 'numeric', 'technical', 'special', 'whitespace'
        """
        if not token:
            return 'special'
        
        # Охранная оговорка: если есть цифра И строчные латинские буквы,
        # это скорее всего искажённое русское слово (Pe3eu), НЕ техника
        has_digit = regex.search(r'\d', token)
        has_lowercase_latin = regex.search(r'[a-z]', token)
        
        if has_digit and has_lowercase_latin:
            # Это вероятно OCR-ошибка, не техническое обозначение
            return 'word'
        
        # Проверка на техническое обозначение
        if self.cfg.protect_technical:
            for pattern in self.technical_patterns_compiled:
                if pattern.fullmatch(token):
                    return 'technical'
        
        # Дополнительная проверка: если есть цифры и заглавные буквы - возможно техническое
        has_uppercase = regex.search(r'[A-ZА-Я]', token)
        if has_digit and has_uppercase and len(token) >= 3:
            # Проверяем: если все буквы заглавные или почти все - техническое
            letters = regex.findall(r'[A-Za-zА-Яа-я]', token)
            if letters:
                uppercase_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
                if uppercase_ratio >= 0.7:
                    return 'technical'
        
        # Проверка на числа
        if regex.fullmatch(r'[\d.,]+', token):
            return 'numeric'
        
        # Подсчет кириллических и латинских букв
        cyrillic_count = sum(1 for c in token if 'CYRILLIC' in unicodedata.name(c, ''))
        latin_count = sum(1 for c in token if 'LATIN' in unicodedata.name(c, ''))
        
        if cyrillic_count > 0 or latin_count > 0:
            return 'word'
        else:
            return 'special'
    
    def convert_to_cyrillic(self, word: str) -> str:
        """
        Конвертация смешанного слова в кириллицу с умным выбором вариантов
        
        Args:
            word: Слово со смешанными символами
            
        Returns:
            Слово в кириллице
        """
        result_word = word
        
        # 1. Используем confusable_homoglyphs если доступно
        if CONFUSABLES_AVAILABLE:
            try:
                # confusables.is_dangerous проверяет омографы
                # Но мы делаем простую конверсию через базовую карту
                pass
            except:
                pass
        
        # 2. Базовая конверсия через карту
        result = []
        ambiguous_positions = []  # Позиции с неоднозначными буквами
        
        for i, char in enumerate(result_word):
            if char in self.BASIC_LATIN_TO_CYRILLIC:
                result.append(self.BASIC_LATIN_TO_CYRILLIC[char])
                # Отмечаем 'r' как неоднозначную (может быть 'г' или 'т')
                if char in ['r', 'R']:
                    ambiguous_positions.append(i)
            else:
                result.append(char)
        
        result_word = ''.join(result)
        
        # 3. Конвертация цифр в кириллицу (только если есть кириллические буквы)
        if regex.search(r'[А-Яа-я]', result_word):
            result_word = ''.join(
                self.DIGIT_TO_CYRILLIC.get(ch, ch) for ch in result_word
            )
        
        # 4. Если есть неоднозначные позиции, пробуем альтернативные варианты
        if ambiguous_positions and regex.search(r'[А-Яа-я]', result_word):
            # Пробуем заменить 'г' на 'т' для 'r' и проверяем, какой вариант лучше
            alternatives = [result_word]
            for pos in ambiguous_positions:
                if pos < len(result_word) and result_word[pos] in ['г', 'Г']:
                    # Создаем альтернативный вариант
                    alt = list(result_word)
                    alt[pos] = 'т' if result_word[pos] == 'г' else 'Т'
                    alternatives.append(''.join(alt))
            
            # Выбираем лучший вариант через проверку в словаре
            best_variant = result_word
            for variant in alternatives:
                if self.is_valid_russian_word(variant):
                    best_variant = variant
                    break
            
            # Если ни один не валиден, пробуем через rapidfuzz
            if best_variant == result_word and len(alternatives) > 1:
                best_score = 0
                for variant in alternatives:
                    closest = self.find_closest_word(variant, use_domain=True)
                    if closest:
                        # Проверяем насколько близко
                        score = fuzz.ratio(variant.lower(), closest.lower())
                        if score > best_score:
                            best_score = score
                            best_variant = variant
            
            result_word = best_variant
        
        return result_word
    
    def is_valid_russian_word(self, word: str) -> bool:
        """
        Проверка корректности русского слова через pymorphy3
        
        Args:
            word: Слово для проверки
            
        Returns:
            True если слово корректное
        """
        if not word:
            return False
        
        word_lower = word.lower()
        
        # Проверка в собственном словаре
        if word_lower in self.russian_words:
            return True
        
        # Проверка через pymorphy3
        parsed = self.morph.parse(word)
        if parsed:
            for p in parsed:
                if 'UNKN' not in str(p.tag):
                    return True
        
        return False
    
    def find_closest_word(self, word: str, use_domain: bool = True) -> Optional[str]:
        """
        Поиск ближайшего корректного слова через rapidfuzz
        
        Args:
            word: Некорректное слово
            use_domain: Использовать ли доменный словарь первым
            
        Returns:
            Ближайшее корректное слово или None
        """
        if not word:
            return None
        
        # Проверка в кэше
        cache_key = f"{word}_{use_domain}"
        if cache_key in self.dictionary_cache:
            return self.dictionary_cache[cache_key]
        
        word_lower = word.lower()
        
        # Определяем max_distance
        if len(word_lower) <= self.cfg.short_word_threshold:
            max_distance = self.cfg.max_distance_short
        else:
            max_distance = self.cfg.max_distance_long
        
        best_match = None
        
        # 1. Сначала ищем в доменном словаре (приоритет)
        if use_domain and self.domain_dict:
            result = process.extractOne(
                word_lower,
                self.domain_dict.keys(),
                scorer=fuzz.ratio,
                score_cutoff=70
            )
            if result:
                matched_word, score, _ = result
                if score >= 70:  # Минимальный порог
                    best_match = self.domain_dict[matched_word]
                    if self.cfg.log_corrections:
                        logger.debug(f"Domain match: {word} -> {best_match} (score={score})")
        
        # 2. Если не нашли, ищем в общем словаре
        if not best_match and self.russian_words:
            result = process.extractOne(
                word_lower,
                self.russian_words,
                scorer=fuzz.ratio,
                score_cutoff=60
            )
            if result:
                matched_word, score, _ = result
                if score >= 60:
                    best_match = matched_word
                    if self.cfg.log_corrections:
                        logger.debug(f"Dictionary match: {word} -> {best_match} (score={score})")
        
        # Сохранение в кэш
        if best_match:
            self.dictionary_cache[cache_key] = best_match
        
        return best_match
    
    def normalize_case(self, word: str, original: str) -> str:
        """
        Восстановление регистра по оригиналу
        
        Args:
            word: Слово после коррекции
            original: Оригинальное слово
            
        Returns:
            Слово с восстановленным регистром
        """
        if not word or not original:
            return word
        
        if original.isupper():
            return word.upper()
        elif original[:1].isupper():
            return word.capitalize()
        else:
            return word.lower()
    
    def correct_token(self, token: Token, context_tokens: List[Token]) -> str:
        """
        Коррекция одного токена с учетом контекста
        
        Args:
            token: Токен для коррекции
            context_tokens: Окружающие токены
            
        Returns:
            Исправленное слово
        """
        word = token.text
        original_word = word
        
        # Технические обозначения со смешанными символами - конвертируем в латиницу
        if token.token_type == 'technical':
            # Если есть смешение кириллицы и латиницы - конвертируем в латиницу
            has_cyrillic = regex.search(r'[А-Яа-я]', word)
            has_latin = regex.search(r'[A-Za-z]', word)
            if has_cyrillic and has_latin:
                word = self.convert_to_latin(word)
                if self.cfg.log_corrections and word != original_word:
                    logger.info(f"Technical normalized: '{original_word}' -> '{word}'")
            return word
        
        # Пропускаем числа, спец символы, пробелы
        if token.token_type in ('numeric', 'special', 'whitespace'):
            return word
        
        # Проверка контекста: есть ли рядом числа
        neighbors = [t for t in context_tokens if t is not token]
        has_numeric_neighbor = any(t.token_type == 'numeric' for t in neighbors)
        
        # Конвертация смешанных символов в кириллицу
        if regex.search(r'[a-zA-Z]', word) and regex.search(r'[А-Яа-я]', word):
            # Смешанное слово - конвертируем
            word = self.convert_to_cyrillic(word)
        elif regex.search(r'[a-zA-Z]', word) and not has_numeric_neighbor:
            # Только латиница - возможно искажение кириллицы
            # Но проверяем защищенные бренды через fuzzy matching
            if self.cfg.protect_brands:
                word_clean = regex.sub(r"[''`ʹ]", "", word).lower()
                # Проверяем в словаре брендов
                if word_clean in self.brand_dict:
                    return self.brand_dict[word_clean]
                # Fuzzy поиск
                result = process.extractOne(
                    word_clean,
                    self.brand_dict.keys(),
                    scorer=fuzz.ratio,
                    score_cutoff=self.cfg.brand_fuzzy_threshold
                )
                if result:
                    matched_variant, score, _ = result
                    return self.brand_dict[matched_variant]
            # Конвертируем
            word = self.convert_to_cyrillic(word)
        
        # Проверка и коррекция кириллических слов
        if regex.search(r'[А-Яа-я]', word):
            word_lower = word.lower()
            
            # Проверка корректности
            if self.is_valid_russian_word(word):
                closest = word_lower
            else:
                # Поиск ближайшего слова
                closest = self.find_closest_word(word)
            
            if closest:
                word = closest
                if self.cfg.log_corrections and word != original_word:
                    logger.info(f"Corrected: '{original_word}' -> '{word}'")
        
        # Всегда нормализуем регистр для кириллических слов
        if regex.search(r'[А-Яа-я]', word):
            word = self.normalize_case(word, token.original)
        
        return word
    
    def correct_tokens(self, tokens: List[Token]) -> List[str]:
        """
        Коррекция всех токенов с учетом контекста
        
        Args:
            tokens: Список токенов
            
        Returns:
            Список исправленных слов
        """
        corrected = []
        
        for i, token in enumerate(tokens):
            # Получение контекста (2 токена до и после)
            context_start = max(0, i - 2)
            context_end = min(len(tokens), i + 3)
            context = tokens[context_start:context_end]
            
            corrected_word = self.correct_token(token, context)
            corrected.append(corrected_word)
        
        return corrected
    
    def post_process_result(self, text: str) -> str:
        """
        Финальная постобработка результата
        
        Args:
            text: Текст после основной обработки
            
        Returns:
            Финально обработанный текст
        """
        # 1. Убираем маркеры списка (тире в начале)
        text = regex.sub(r'^\s*[-–—]\s*', '', text)
        
        # 2. Убираем пробелы перед знаками препинания
        text = regex.sub(r'\s+([.,;!?])', r'\1', text)
        
        # 3. Повторная универсальная нормализация брендов (на случай если токенизация разбила)
        text = self.normalize_brand(text)
        
        # 4. Финальная нормализация единиц измерения (слитные + унификация)
        text = regex.sub(r'(\d+)([Шш][Тт]\.?|ШТ\.?)', r'\1 \2', text)  # Добавляем пробел
        text = regex.sub(r'(\d+)\s+([Шш][Тт]\.?|шт\.?|ШТ\.?)', r'\1 шт.', text)  # Унифицируем
        
        # 5. Нормализация пробелов (удаление множественных)
        text = regex.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def process_text(self, text: str) -> str:
        """
        Полная обработка текста: предобработка, токенизация, коррекция
        
        Args:
            text: Исходный текст
            
        Returns:
            Обработанный текст
        """
        # 1. Предобработка
        text = self.preprocess_text(text)
        
        # 2. Токенизация
        tokens = self.tokenize(text)
        
        # 3. Коррекция токенов
        corrected = self.correct_tokens(tokens)
        
        # 4. Сборка текста (сохраняя все пробелы и знаки)
        result = ''.join(corrected)
        
        # 5. Финальная постобработка
        result = self.post_process_result(result)
        
        return result
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Обработка DataFrame с OCR-результатами
        
        Args:
            df: DataFrame для обработки
            
        Returns:
            Обработанный DataFrame
        """
        logger.info(f"OCR post-processing DataFrame with shape: {df.shape}")
        
        # Обработка всех строковых столбцов
        for col in df.columns:
            if df[col].dtype == 'object':
                logger.debug(f"Processing column: {col}")
                df[col] = df[col].apply(
                    lambda x: self.process_text(str(x)) if pd.notna(x) and str(x).strip() else x
                )
                
                # Дополнительная прямая обработка слитных единиц (на случай если не сработало в токенизаторе)
                df[col] = df[col].apply(
                    lambda x: regex.sub(r'(\d+)([Шш][Тт]\.?|ШТ\.?)', r'\1 шт.', str(x)) 
                    if pd.notna(x) else x
                )
                
                # Дополнительная обработка тире в начале
                df[col] = df[col].apply(
                    lambda x: regex.sub(r'^\s*[-–—]\s*', '', str(x)) 
                    if pd.notna(x) else x
                )
                
                # Финальная универсальная нормализация брендов
                df[col] = df[col].apply(
                    lambda x: self.normalize_brand(str(x)) if pd.notna(x) else x
                )
        
        logger.info(f"OCR post-processing completed")
        return df

