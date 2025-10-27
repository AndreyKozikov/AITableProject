"""Data Mapping Module.

Модуль сопоставления данных.

Maps extracted data from various sources into unified table structure
using AI models for intelligent data processing.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Union

import pandas as pd

from src.mapper.ask_qwen2 import ask_qwen2
from src.mapper.ask_qwen3 import ask_qwen3
from src.mapper.ask_llama2 import ask_llama2
from src.mapper.ask_qwen3_so import ask_qwen3_structured, extract_rows_as_dicts
from src.mapper.ask_qwen3_cot import ask_qwen3_cot, extract_cot_rows_as_dicts
from src.utils.config import MODEL_DIR, PARSING_DIR, PROMPT_TEMPLATE, PROMPT_TEMPLATE_SO

# Настройка логирования
logger = logging.getLogger(__name__)


def load_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load JSON data from file.
    
    Args:
        path: Path to JSON file.
        
    Returns:
        List of dictionaries with loaded data.
        
    Raises:
        FileNotFoundError: If file not found.
        json.JSONDecodeError: If invalid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)  # список словарей
    logger.debug(f"Loaded JSON file: {path}, records: {len(data)}")
    return data


def chunk_data(data: List[Any], chunk_size: int = 20) -> Generator[List[Any], None, None]:
    """Split data into chunks of specified size.
    
    Args:
        data: Data to split into chunks.
        chunk_size: Size of each chunk.
        
    Yields:
        Data chunks of specified size.
    """
    for i in range(0, len(data), chunk_size):
        yield data[i:i + chunk_size]


def mapper_structured(
    files: List[Union[str, Path]], 
    extended: bool = False,
    enable_thinking: bool = False,
    use_cot: bool = False
) -> List[Dict[str, str]]:
    """Main data mapping function with structured output.
    
    Loads JSON data, chunks it, and processes through Qwen3 model
    with structured output for intelligent data mapping.
    
    Args:
        files: List of JSON files to process.
        extended: Extended processing mode flag.
        enable_thinking: Enable Chain of Thought reasoning.
        use_cot: Использовать модель с Chain-of-Thought reasoning.
        
    Returns:
        Tuple of (all_rows, headers) where all_rows is list of dicts.
    """
    max_new_tokens = 4000
    all_rows = []
    chunk_size = 1
    
    if extended:
        logger.info("Starting mapper_structured in extended mode")
        file_path_header = MODEL_DIR / "extended.csv"
    else:
        logger.info("Starting mapper_structured in simplified mode")
        file_path_header = MODEL_DIR / "simplified.csv"

    # Load target headers from CSV
    header = list(pd.read_csv(file_path_header, nrows=0, sep=";").columns)
    header_str = ", ".join(header)
    logger.info(f"Target headers: {header_str}")

    for file_idx, file in enumerate(files, 1):
        file_path = PARSING_DIR / file
        logger.info(f"Processing file {file_idx}/{len(files)}: {file}")
        
        try:
            # Load JSON data
            json_data = load_json(file_path)
            logger.info(f"Loaded {len(json_data)} records from {file}")
            
            if not json_data:
                logger.warning(f"File {file} is empty, skipping")
                continue
            
            # Convert JSON list of dicts to DataFrame for logging
            df = pd.DataFrame(json_data)
            logger.info(f"DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            
            # Split data into chunks
            chunk_count = 0
            for chunk in chunk_data(list(json_data), chunk_size=chunk_size):
                chunk_count += 1
                
                # Format chunk data as key-value pairs
                tables_text_parts = []
                for record_idx, record in enumerate(chunk, 1):
                    record_lines = [f"Запись {record_idx}:"]
                    for key, value in record.items():
                        record_lines.append(f"  {key}: {value}")
                    tables_text_parts.append("\n".join(record_lines))
                
                # Combine all records
                tables_text = "\n\n".join(tables_text_parts)
                
                # Log chunk info
                logger.info(f"Processing chunk {chunk_count} with {len(chunk)} records")
                logger.info(f"Chunk keys: {list(chunk[0].keys()) if chunk else 'empty'}")
                
                # Log prompt info
                logger.info(f"Prompt length: {len(tables_text)} characters")
                
                # Send to Qwen3 structured output model
                if use_cot:
                    logger.info(f"Sending chunk {chunk_count} to Qwen3 CoT model...")
                    mode = "extended" if extended else "simplified"
                    cot_result = ask_qwen3_cot(tables_text, mode=mode)
                    
                    logger.info(f"CoT model returned success={cot_result.get('success')}")
                    logger.info(f"CoT raw response:\n{cot_result.get('raw_response', '')}")
                    
                    # Для тестирования принимаем любой результат
                    structured_result = cot_result.get("result", {})
                    
                    # Создаем объект совместимый с extract_rows_as_dicts
                    if structured_result and "rows" in structured_result:
                        class SimpleResult:
                            def __init__(self, rows_data):
                                self.rows = rows_data
                        
                        structured_result = SimpleResult(structured_result["rows"])
                    
                    if cot_result.get("reasoning"):
                        logger.info(f"CoT reasoning:\n{cot_result['reasoning']}")
                    
                    # Не используем fallback для тестирования
                    if not cot_result.get("success"):
                        logger.warning(f"CoT model error: {cot_result.get('error')}")
                        # Возвращаем пустой результат для анализа
                        structured_result = {}
                else:
                    logger.info(f"Sending chunk {chunk_count} to Qwen3 structured model...")
                    structured_result = ask_qwen3_structured(
                        prompt=tables_text,
                        extended=extended,
                        max_new_tokens=max_new_tokens,
                        enable_thinking=enable_thinking
                    )
                
                # Log structured result info
                if structured_result:
                    logger.info(f"Received structured response from model")
                    logger.info(f"Structured result type: {type(structured_result)}")
                    logger.info(f"Structured result preview: {str(structured_result)}")
                else:
                    logger.warning(f"Empty structured response from model for chunk {chunk_count}")
                
                # Extract rows as dictionaries with Russian field names
                if use_cot:
                    chunk_rows = extract_cot_rows_as_dicts(cot_result, use_aliases=True)
                else:
                    chunk_rows = extract_rows_as_dicts(structured_result, use_aliases=True)
                
                logger.info(f"Received {len(chunk_rows)} structured rows from chunk {chunk_count}")
                if chunk_rows:
                    logger.info(f"Sample row preview: {chunk_rows[0]}")
                all_rows.extend(chunk_rows)
            
            logger.info(f"Completed processing file {file}: {chunk_count} chunks, {len(all_rows)} total rows")
                
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            continue
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file}: {e}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file}: {e}", exc_info=True)
            continue
    
    logger.info(f"Total structured rows collected: {len(all_rows)}")
    
    # Optionally save to JSON
    out_file = PARSING_DIR / "output_structured.json"
    try:
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        logger.info(f"Structured results saved to: {out_file}")
    except Exception as e:
        logger.error(f"Error saving results to {out_file}: {e}")

    return all_rows