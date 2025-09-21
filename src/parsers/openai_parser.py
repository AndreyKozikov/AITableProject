import os
import re
import logging
from typing import List, Tuple
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from src.utils.config import OPENAI_API_KEY
from src.utils.registry import register_parser

# Настройка логгера
# создаём папку для логов, если её нет
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/openai_parser.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
    force=True
)

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """
Ты — ассистент по обработке табличных данных.
Формат выходных данных: **только таблица в Markdown**, без пояснений и заголовков.

Правила:
- Используй ровно эти столбцы: Обозначение,Наименование,Производитель,единица измерения,Количество,Техническое задание.
- Каждое входное значение помещается только в один столбец.
- Если подходит под 'Количество', 'Наименование' или 'Единица измерения' — ставь туда.
- Если данных нет для столбца — оставляй пустую ячейку.
- Не добавляй новых строк или столбцов.
- Артикул не является наименованием.

Входные данные представлены в приложенных файлах. 
Выведи результат в виде Markdown-таблицы.
"""

def is_retryable_exception(e: Exception) -> bool:
    text = str(e)
    # Не повторяем, если именно rate_limit_exceeded
    if "rate_limit_exceeded" in text:
        return False
    return True

@register_parser("openai")
def openai_parser(files: List[str]) -> Tuple[List[List[str]], List[str]]:
    if not files:
        raise ValueError("Список файлов пустой")

    # Проверка существования файлов
    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")

    client = OpenAI(api_key=OPENAI_API_KEY)

    uploaded_files = []
    thread = None
    try:
        # Загружаем файлы (они будут доступны ассистенту через file_search)
        for path in files:
            with open(path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="assistants"
                )
                uploaded_files.append(uploaded_file)
                logger.info(f"Загружен файл: {path} -> {uploaded_file.id}")

        # Создаём ассистента
        assistant = client.beta.assistants.create(
            name="Табличный ассистент",
            instructions=PROMPT_TEMPLATE,
            model="gpt-5",  # актуальная модель
            tools=[{"type": "file_search"}]
        )
        logger.info(f"Создан ассистент: {assistant.id}")

        # Создаём тред
        thread = client.beta.threads.create()
        logger.info(f"Создан тред: {thread.id}")

        # Добавляем сообщение
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Обработай приложенные файлы и верни таблицу в Markdown."
        )
        logger.info(f"Добавлено сообщение: {message.id}")

        # Запускаем
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        logger.info(f"Запущен run: {run.id}")

        # Ждём завершения
        @retry(stop=stop_after_attempt(25),
               wait=wait_exponential(multiplier=1, min=10, max=30),
               retry=retry_if_exception(is_retryable_exception),
               )
        def wait_for_run_completion():
            run_status = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run_status.status == "failed":
                raise ValueError(f"Задание завершилось с ошибкой: {run_status.last_error}")
            if run_status.status == "completed":
                return run_status
            raise ValueError("Run still in progress")

        try:
            wait_for_run_completion()
        except Exception as e:
            logger.error(f"Run не завершился за 25 попыток: {e}")
            raise

        logger.info("Обработка завершена успешно")

        # Получаем ответ
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        if not messages.data:
            raise ValueError("Ответ ассистента пустой")

        full_text = ""
        for m in messages.data:
            if m.role == "assistant":
                for c in m.content:
                    if c.type == "text":
                        full_text += c.text.value + "\n"

        if not full_text.strip():
            raise ValueError("Ассистент не вернул текстовый ответ")

        logger.debug("Полный ответ ассистента:\n%s", full_text)

        # Ищем первую таблицу
        table_match = re.search(
            r"(?m)^(\|.*\|\r?\n\|(?:[:-]+\s*\|)+\r?\n(?:\|.*\|\r?\n?)+)",
            full_text
        )
        if not table_match:
            raise ValueError("Таблица в ответе не найдена. Ответ ассистента:\n" + full_text)

        markdown_table = table_match.group(1)

        # Парсим таблицу
        rows = []
        header = []
        lines = markdown_table.splitlines()
        for i, line in enumerate(lines):
            if not line.startswith("|"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if i == 0:
                header = cells
                continue
            if all(re.match(r"^:?-+:?$", c) for c in cells):  # разделитель
                continue
            if cells and any(cell for cell in cells):
                rows.append(cells)

        if not header:
            raise ValueError("Заголовок таблицы не найден")

        logger.info(f"Извлечено {len(rows)} строк из таблицы")
        return rows, header

    except Exception as e:
        logger.error(f"Ошибка в openai_parser: {str(e)}", exc_info=True)
        raise
    finally:
        # Чистим загруженные файлы
        for file_obj in uploaded_files:
            try:
                client.files.delete(file_obj.id)
                logger.info(f"Удален файл: {file_obj.id}")
            except Exception as e:
                logger.warning(f"Не удалось удалить файл {file_obj.id}: {str(e)}")
