PARSERS = {}


def register_parser(*suffixes):
    """Декоратор для регистрации парсера по расширению файла"""
    def wrapper(func):
        for suffix in suffixes:
            PARSERS[suffix.lower()] = func
        return func
    return wrapper