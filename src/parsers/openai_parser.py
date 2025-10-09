"""OpenAI Parser Module.

Модуль парсера OpenAI.

Parser that uses OpenAI API for processing files and extracting
tabular data with AI assistance.
"""

import os
import re
from typing import List, Tuple

from openai import OpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src.utils.config import OPENAI_API_KEY
from src.utils.logging_config import get_logger
from src.utils.registry import register_parser

# Получение настроенного логгера
logger = get_logger(__name__)

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
    """Check if exception is retryable.
    
    Args:
        e: Exception to check.
        
    Returns:
        True if exception should trigger retry, False otherwise.
    """
    text = str(e)
    # Don't retry if rate_limit_exceeded
    if "rate_limit_exceeded" in text:
        logger.warning(f"Rate limit exceeded, not retrying: {text}")
        return False
    logger.debug(f"Exception is retryable: {text}")
    return True


@register_parser("openai")
def openai_parser(files: List[str]) -> Tuple[List[List[str]], List[str]]:
    """Parse files using OpenAI API.
    
    Args:
        files: List of file paths to process.
        
    Returns:
        Tuple of (rows, headers) for table data.
        
    Raises:
        ValueError: If files list is empty.
        FileNotFoundError: If file not found.
        Exception: If OpenAI API call fails.
    """
    logger.info(f"Starting OpenAI parsing for {len(files)} files")
    
    if not files:
        raise ValueError("Список файлов пустой")

    for path in files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")

    client = OpenAI(api_key=OPENAI_API_KEY)

    uploaded_files = []
    thread = None
    try:
        # Upload files
        logger.info("Uploading files to OpenAI...")
        for path in files:
            with open(path, "rb") as f:
                uploaded_file = client.files.create(
                    file=f,
                    purpose="assistants"
                )
                uploaded_files.append(uploaded_file)
                logger.info(f"Uploaded file: {path} -> {uploaded_file.id}")

        # Create assistant
        logger.info("Creating OpenAI assistant...")
        assistant = client.beta.assistants.create(
            name="Табличный ассистент",
            instructions=PROMPT_TEMPLATE,
            model="gpt-4o",
            tools=[{"type": "file_search"}]
        )
        logger.info(f"Created assistant: {assistant.id}")

        # Create thread
        thread = client.beta.threads.create()
        logger.info(f"Created thread: {thread.id}")

        # Create message with file attachments
        attachments = [
            {"file_id": f.id, "tools": [{"type": "file_search"}]} 
            for f in uploaded_files
        ]
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content="Обработай приложенные файлы и верни таблицу в Markdown.",
            attachments=attachments
        )
        logger.info(f"Added message: {message.id}")

        # Run assistant with retry logic
        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=5, max=60),
            retry=retry_if_exception(is_retryable_exception)
        )
        def run_with_retry():
            logger.info("Running OpenAI assistant...")
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id,
                timeout=300
            )
            
            if run.status == 'completed':
                logger.info("Assistant run completed successfully")
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                )
                
                for msg in messages.data:
                    if msg.role == 'assistant':
                        content = msg.content[0].text.value
                        logger.info(f"Assistant response received: {len(content)} characters")
                        logger.debug(f"Assistant response preview: {content[:200]}...")
                        return content
                        
                raise Exception("Нет ответа от ассистента")
            else:
                raise Exception(f"Выполнение не завершено. Статус: {run.status}")

        content = run_with_retry()
        
        # Parse markdown table
        rows, headers = parse_markdown_table(content)
        logger.info(f"Parsed {len(rows)} rows with {len(headers)} headers")
        
        return rows, headers

    except Exception as e:
        logger.error(f"Error in OpenAI parser: {e}")
        raise
        
    finally:
        # Cleanup
        logger.info("Cleaning up OpenAI resources...")
        try:
            if thread:
                client.beta.threads.delete(thread.id)
                logger.info(f"Deleted thread: {thread.id}")
        except Exception as e:
            logger.warning(f"Failed to delete thread: {e}")
            
        for uploaded_file in uploaded_files:
            try:
                client.files.delete(uploaded_file.id)
                logger.debug(f"Deleted file: {uploaded_file.id}")
            except Exception as e:
                logger.warning(f"Failed to delete file {uploaded_file.id}: {e}")


def parse_markdown_table(content: str) -> Tuple[List[List[str]], List[str]]:
    """Parse markdown table from content.
    
    Args:
        content: Markdown content containing table.
        
    Returns:
        Tuple of (rows, headers).
    """
    logger.debug("Parsing markdown table from content")
    
    lines = content.strip().split('\n')
    
    # Find table lines (containing |)
    table_lines = [line for line in lines if '|' in line and line.strip()]
    
    if not table_lines:
        logger.warning("No table found in content")
        return [], []
    
    # Parse header
    header_line = table_lines[0]
    headers = [cell.strip() for cell in header_line.split('|')[1:-1]]
    logger.debug(f"Found headers: {headers}")
    
    # Skip separator line if present
    start_idx = 2 if len(table_lines) > 1 and '-' in table_lines[1] else 1
    
    # Parse data rows
    rows = []
    for line in table_lines[start_idx:]:
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if cells:  # Skip empty rows
            rows.append(cells)
    
    logger.info(f"Parsed markdown table: {len(headers)} headers, {len(rows)} rows")
    return rows, headers