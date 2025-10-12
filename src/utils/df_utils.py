."""DataFrame Utilities Module.

Модуль утилит для работы с DataFrame.

This module provides utilities for DataFrame processing, header detection,
data cleaning, and file operations for the AITableProject.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd

from src.utils.config import HEADER_ANCHORS
from src.utils.logging_config import get_logger

# Получение настроенного логгера
logger = get_logger(__name__)


def is_header_row_semantic(
    rows: Union[List[List[str]], Path, str],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    similarity_threshold: float = 0.75
) -> bool:
    """Determine if CSV contains header using semantic embeddings.
    
    Uses sentence-transformers to encode the first row and compare it with
    a reference set of known headers using cosine similarity.
    
    Args:
        rows: Either list of rows or path to CSV file.
        model_name: Name of the sentence-transformers model to use.
        similarity_threshold: Minimum similarity score to consider as header (0-1).
        
    Returns:
        True if first row is semantically similar to known headers, False otherwise.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        return False
    
    logger.debug(f"Starting semantic header detection with model: {model_name}")
    
    # If path provided, read the file first
    if isinstance(rows, (Path, str)):
        try:
            with open(rows, 'r', encoding='utf-8') as f:
                reader = csv.reader(f, delimiter=';')  # Используем разделитель ';'
                rows = list(reader)
            logger.debug(f"Read {len(rows)} rows from file")
        except Exception as e:
            logger.error(f"Error reading file for semantic header detection: {e}")
            return False
    
    if not isinstance(rows, list) or len(rows) < 1:
        logger.debug("Too few rows for semantic analysis")
        return False
    
    # Get first row cells
    first_row = [cell.strip() for cell in rows[0] if cell.strip() != ""]
    
    if not first_row:
        logger.debug("First row is empty")
        return False
    
    logger.debug(f"Analyzing first row with {len(first_row)} cells")
    
    # Проверяем, что первая строка не состоит в основном из чисел
    numeric_cells = sum(
        1 for cell in first_row 
        if cell.replace('.', '').replace(',', '').replace('-', '').isdigit()
    )
    numeric_ratio = numeric_cells / len(first_row) if first_row else 0
    
    if numeric_ratio > 0.3:
        logger.debug(
            f"First row contains too many numeric values ({numeric_ratio:.1%}), "
            f"likely data row, not header"
        )
        return False
    
    # Извлекаем reference headers из HEADER_ANCHORS
    reference_headers = []
    for _, header_variants in HEADER_ANCHORS:
        reference_headers.extend(header_variants)
    
    logger.debug(f"Using {len(reference_headers)} reference headers from HEADER_ANCHORS")
    
    try:
        # Load model (cached after first load)
        logger.debug("Loading sentence-transformers model...")
        model = SentenceTransformer(model_name)
        
        # Encode first row cells
        logger.debug(f"Encoding {len(first_row)} cells from first row")
        first_row_embeddings = model.encode(first_row, convert_to_numpy=True)
        
        # Encode reference headers
        logger.debug(f"Encoding {len(reference_headers)} reference headers")
        reference_embeddings = model.encode(reference_headers, convert_to_numpy=True)
        
        # Calculate similarities between first row and reference headers
        # For each cell in first row, find max similarity with any reference header
        max_similarities = []
        for cell_embedding in first_row_embeddings:
            # Cosine similarity with all reference headers
            similarities = model.similarity(
                cell_embedding.reshape(1, -1),
                reference_embeddings
            )
            max_sim = float(similarities.max())
            max_similarities.append(max_sim)
            logger.debug(
                f"Cell '{first_row[len(max_similarities)-1]}' "
                f"max similarity: {max_sim:.4f}"
            )
        
        # Calculate average similarity across all cells
        avg_similarity = float(np.mean(max_similarities))
        logger.info(
            f"Average semantic similarity: {avg_similarity:.4f} "
            f"(threshold: {similarity_threshold})"
        )
        
        # Also check how many cells exceed threshold
        cells_above_threshold = sum(
            1 for sim in max_similarities if sim >= similarity_threshold
        )
        ratio_above_threshold = cells_above_threshold / len(max_similarities)
        
        logger.info(
            f"Cells above threshold: {cells_above_threshold}/{len(max_similarities)} "
            f"({ratio_above_threshold:.2%})"
        )
        
        # Decision: high average AND majority of cells above threshold (stricter condition)
        is_header = (
            avg_similarity >= similarity_threshold and
            ratio_above_threshold >= 0.6
        )
        
        if is_header:
            logger.info(
                f"Header detected by semantic analysis "
                f"(avg_sim={avg_similarity:.4f}, "
                f"ratio={ratio_above_threshold:.2%})"
            )
        else:
            logger.debug(
                f"No header detected by semantic analysis "
                f"(avg_sim={avg_similarity:.4f}, "
                f"ratio={ratio_above_threshold:.2%})"
            )
        
        return is_header
        
    except Exception as e:
        logger.error(f"Error in semantic header detection: {e}")
        return False


def write_to_json(
    file_path: Union[str, Path],
    data: Any,
    detect_headers: bool = False,
    temp_dir: Union[str, Path, None] = None
) -> bool:
    """Write data to JSON file with optional header detection and generation.
    
    Args:
        file_path: Path to output JSON file.
        data: Data to write to JSON. Can be DataFrame, list of lists, or any JSON-serializable data.
        detect_headers: If True, attempts to detect headers in DataFrame data and generates them if not found.
        temp_dir: Directory for temporary files during header detection. Required if detect_headers=True.
        
    Returns:
        True if successful, False otherwise.
    """
    try:
        file_path = Path(file_path)
        logger.debug(f"Writing JSON to: {file_path}")
        
        # If data is DataFrame and header detection is requested
        if detect_headers and isinstance(data, pd.DataFrame):
            logger.debug("Header detection requested for DataFrame")
            
            if temp_dir is None:
                logger.warning("temp_dir not provided, using default directory")
                temp_dir = file_path.parent
            
            temp_dir = Path(temp_dir)
            temp_csv_path = temp_dir / f"temp_{file_path.stem}.csv"
            
            try:
                # Save DataFrame to temporary CSV without headers
                data.to_csv(temp_csv_path, index=False, sep=";", header=False)
                logger.debug(f"Created temporary CSV: {temp_csv_path}")
                
                # Check if header row exists
                has_header = is_header_row_semantic(temp_csv_path)
                logger.info(f"Header detection result: {has_header}")
                
                if has_header:
                    # Read CSV with detected header
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=0, dtype=str)
                    logger.info(f"Using detected headers: {list(df_with_header.columns)}")
                else:
                    # Read CSV without header and generate column names
                    df_with_header = pd.read_csv(temp_csv_path, sep=";", header=None, dtype=str)
                    num_columns = len(df_with_header.columns)
                    generated_headers = [f"Заголовок {j+1}" for j in range(num_columns)]
                    df_with_header.columns = generated_headers
                    logger.info(f"Generated headers: {generated_headers}")
                
                # Convert DataFrame to list of dictionaries
                data = df_with_header.to_dict('records')
                logger.debug(f"Converted DataFrame to {len(data)} records")
                
            finally:
                # Временный CSV НЕ удаляется для отладки
                if temp_csv_path.exists():
                    logger.info(f"Temporary CSV kept for debugging: {temp_csv_path}")
                    # temp_csv_path.unlink()  # Закомментировано для отладки
        
        elif isinstance(data, pd.DataFrame):
            # Convert DataFrame to records without header detection
            data = data.to_dict('records')
            logger.debug(f"Converted DataFrame to {len(data)} records")
        
        # Write data to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully wrote JSON file: {file_path}")
        logger.debug(f"File size: {file_path.stat().st_size} bytes")
        return True
        
    except Exception as e:
        logger.error(f"Error writing JSON file {file_path}: {e}")
        return False


def read_from_json(file_path: Union[str, Path]) -> Any:
    """Read data from JSON file.
    
    Args:
        file_path: Path to JSON file.
        
    Returns:
        Loaded data or None if failed.
    """
    try:
        file_path = Path(file_path)
        logger.debug(f"Reading JSON from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Successfully read JSON file: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"Error reading JSON file {file_path}: {e}")
        return None


def clean_dataframe(df: pd.DataFrame, use_languagetool: bool = False) -> pd.DataFrame:
    """Clean DataFrame by removing empty rows/columns and fixing data.
    
    Args:
        df: DataFrame to clean.
        use_languagetool: Whether to use language tool for text correction.
        
    Returns:
        Cleaned DataFrame.
    """
    logger.debug(f"Cleaning DataFrame with shape: {df.shape}")
    
    try:
        original_shape = df.shape
        
        # Remove completely empty rows and columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN with empty strings
        df = df.fillna('')
        
        # Strip whitespace from string columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        logger.info(f"DataFrame cleaned: {original_shape} -> {df.shape}")
        
        if use_languagetool:
            logger.info("Language tool correction requested but not implemented")
            # TODO: Implement language tool correction if needed
        
        return df
        
    except Exception as e:
        logger.error(f"Error cleaning DataFrame: {e}")
        return df



def validate_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate DataFrame and return quality metrics.
    
    Args:
        df: DataFrame to validate.
        
    Returns:
        Dictionary with validation results and metrics.
    """
    try:
        logger.debug(f"Validating DataFrame with shape: {df.shape}")
        
        validation_result = {
            'is_valid': True,
            'shape': df.shape,
            'empty_cells': df.isnull().sum().sum(),
            'duplicate_rows': df.duplicated().sum(),
            'column_types': df.dtypes.to_dict(),
            'warnings': [],
            'errors': []
        }
        
        # Check for empty DataFrame
        if df.empty:
            validation_result['errors'].append("DataFrame is empty")
            validation_result['is_valid'] = False
        
        # Check for too many empty cells
        total_cells = df.shape[0] * df.shape[1]
        if total_cells > 0:
            empty_ratio = validation_result['empty_cells'] / total_cells
            if empty_ratio > 0.5:
                validation_result['warnings'].append(
                    f"High empty cell ratio: {empty_ratio:.2%}"
                )
        
        # Check for duplicate rows
        if validation_result['duplicate_rows'] > 0:
            validation_result['warnings'].append(
                f"Found {validation_result['duplicate_rows']} duplicate rows"
            )
        
        logger.info(f"DataFrame validation completed. Valid: {validation_result['is_valid']}")
        logger.debug(f"Validation warnings: {len(validation_result['warnings'])}")
        logger.debug(f"Validation errors: {len(validation_result['errors'])}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating DataFrame: {e}")
        return {
            'is_valid': False,
            'errors': [f"Validation failed: {str(e)}"]
        }