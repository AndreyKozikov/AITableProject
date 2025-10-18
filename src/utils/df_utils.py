"""DataFrame Utilities Module.

Модуль утилит для работы с DataFrame.

This module provides utilities for DataFrame processing, header detection,
data cleaning, and file operations for the AITableProject.
"""

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
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


def _calculate_iou(box1: list, box2: list) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value (intersection area / union area), range [0, 1]
    """
    # Calculate intersection area
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Check if boxes intersect
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    # Avoid division by zero
    if union <= 0:
        return 0.0
    
    return intersection / union


def _merge_ocr_blocks_by_cells(json_data: dict, iou_threshold: float = 0.05) -> tuple:
    """Merge OCR blocks based on table cell coordinates from PPStructure V3.
    
    This private helper method aggregates OCR text blocks according to detected
    table cell boundaries using a two-stage strategy:
    1. Fast filtering: Check if OCR block center is inside cell
    2. IoU matching: For unmatched blocks, use Intersection over Union
    
    Args:
        json_data: Dictionary containing PPStructure V3 results with:
                  - 'table_res_list': List of table recognition results
                  - 'overall_ocr_res': OCR detection results with 'rec_boxes' and 'rec_texts'
        iou_threshold: Minimum IoU value to assign block to cell (default: 0.05)
                  
    Returns:
        Tuple of (merged_rec_boxes, merged_rec_texts) where each element corresponds
        to one table cell, or (None, None) if merging is not applicable.
        
    Algorithm:
        1. Extract cell coordinates from table_res_list
        2. Extract OCR boxes and texts from overall_ocr_res
        3. Stage 1: Match blocks by center-in-cell criterion (fast)
        4. Stage 2: Match remaining blocks by IoU (accurate)
        5. Sort blocks within each cell by position
        6. Smart text merging: space for same line, newline for different lines
        7. Use cell coordinates as merged block coordinates
    """
    try:
        # Check if table_res_list exists
        table_res_list = json_data.get('table_res_list', [])
        if len(table_res_list) == 0:
            logger.debug("No table_res_list found, skipping OCR block merging")
            return None, None
        
        # Get the first table (support for multiple tables can be added later)
        table_res = table_res_list[0]
        cell_box_list = table_res.get('cell_box_list', [])
        
        if len(cell_box_list) == 0:
            logger.debug("No cell_box_list found in table_res, skipping OCR block merging")
            return None, None
        
        # Extract OCR boxes and texts from overall_ocr_res
        overall_ocr_res = json_data.get('overall_ocr_res', {})
        rec_boxes = overall_ocr_res.get('rec_boxes', [])
        rec_texts = overall_ocr_res.get('rec_texts', [])
        
        if len(rec_boxes) == 0 or len(rec_texts) == 0:
            logger.debug("No rec_boxes or rec_texts in overall_ocr_res")
            return None, None
        
        logger.info(f"Merging {len(rec_boxes)} OCR blocks into {len(cell_box_list)} table cells")
        
        # Statistics counters
        matched_by_center = 0
        matched_by_iou = 0
        unmatched_blocks = 0
        
        # Track which OCR blocks have been assigned to cells
        assigned_blocks = set()
        
        # Dictionary to store blocks for each cell
        cell_blocks = {cell_idx: [] for cell_idx in range(len(cell_box_list))}
        
        # Stage 1: Match by center criterion (fast filtering)
        for box_idx, (ocr_box, ocr_text) in enumerate(zip(rec_boxes, rec_texts)):
            if len(ocr_box) != 4:
                continue
            
            ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
            center_x = (ocr_x1 + ocr_x2) / 2
            center_y = (ocr_y1 + ocr_y2) / 2
            
            # Check each cell
            for cell_idx, cell_box in enumerate(cell_box_list):
                cell_x1, cell_y1, cell_x2, cell_y2 = cell_box
                
                if (cell_x1 <= center_x <= cell_x2 and 
                    cell_y1 <= center_y <= cell_y2):
                    cell_blocks[cell_idx].append({
                        'index': box_idx,
                        'box': [ocr_x1, ocr_y1, ocr_x2, ocr_y2],
                        'text': ocr_text,
                        'center_y': center_y,
                        'center_x': center_x,
                        'method': 'center'
                    })
                    assigned_blocks.add(box_idx)
                    matched_by_center += 1
                    break  # Block assigned, move to next
        
        # Stage 2: Match remaining blocks by IoU
        for box_idx, (ocr_box, ocr_text) in enumerate(zip(rec_boxes, rec_texts)):
            if box_idx in assigned_blocks or len(ocr_box) != 4:
                continue
            
            ocr_x1, ocr_y1, ocr_x2, ocr_y2 = ocr_box
            center_x = (ocr_x1 + ocr_x2) / 2
            center_y = (ocr_y1 + ocr_y2) / 2
            
            # Calculate IoU with all cells
            best_iou = 0.0
            best_cell_idx = -1
            
            for cell_idx, cell_box in enumerate(cell_box_list):
                iou = _calculate_iou([ocr_x1, ocr_y1, ocr_x2, ocr_y2], cell_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cell_idx = cell_idx
            
            # Assign to cell if IoU exceeds threshold
            if best_iou >= iou_threshold and best_cell_idx >= 0:
                cell_blocks[best_cell_idx].append({
                    'index': box_idx,
                    'box': [ocr_x1, ocr_y1, ocr_x2, ocr_y2],
                    'text': ocr_text,
                    'center_y': center_y,
                    'center_x': center_x,
                    'method': 'iou',
                    'iou': best_iou
                })
                assigned_blocks.add(box_idx)
                matched_by_iou += 1
            else:
                unmatched_blocks += 1
        
        # Build merged results
        merged_boxes = []
        merged_texts = []
        
        for cell_idx, cell_box in enumerate(cell_box_list):
            blocks = cell_blocks[cell_idx]
            
            if not blocks:
                # Empty cell
                merged_texts.append('')
                merged_boxes.append(list(cell_box))
                continue
            
            # Sort blocks by vertical position, then horizontal
            blocks.sort(key=lambda b: (b['center_y'], b['center_x']))
            
            # Smart text merging: detect line breaks
            combined_parts = []
            prev_y = None
            
            for block in blocks:
                current_y = block['center_y']
                
                # Determine if this is a new line (significant vertical gap)
                if prev_y is not None:
                    y_gap = abs(current_y - prev_y)
                    # If gap is larger than ~10 pixels, consider it a new line
                    if y_gap > 10:
                        combined_parts.append('\n')
                    elif combined_parts:  # Same line, add space
                        combined_parts.append(' ')
                
                combined_parts.append(block['text'])
                prev_y = current_y
            
            combined_text = ''.join(combined_parts)
            merged_texts.append(combined_text)
            merged_boxes.append(list(cell_box))
        
        # Log statistics
        logger.info(
            f"OCR block merging completed: {len(rec_boxes)} blocks -> {len(merged_boxes)} cells"
        )
        logger.info(
            f"  Matched by center: {matched_by_center}, "
            f"by IoU: {matched_by_iou}, "
            f"unmatched: {unmatched_blocks}"
        )
        
        if matched_by_iou > 0:
            logger.debug(
                f"IoU threshold {iou_threshold} helped match {matched_by_iou} additional blocks"
            )
        
        return merged_boxes, merged_texts
        
    except Exception as e:
        logger.error(f"Error in _merge_ocr_blocks_by_cells: {e}", exc_info=True)
        return None, None


def _detect_table_columns(
    rec_boxes: list,
    image_width: int = 1200,
    min_gap_width: int = 10
) -> List[int]:
    """Detect table column boundaries using density map analysis.
    
    This private helper method analyzes the horizontal distribution of OCR boxes
    to identify vertical gaps that serve as column separators.
    
    Args:
        rec_boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
        image_width: Width of the analyzed image in pixels for density map.
        min_gap_width: Minimum width of empty zone to be considered a column boundary.
        
    Returns:
        List of x-coordinates representing column boundaries (vertical separators).
        Empty list if no boundaries found.
        
    Algorithm:
        1. Build horizontal density map across image width
        2. Identify continuous zero zones (gaps without text)
        3. Filter gaps by minimum width threshold
        4. Return midpoint of each valid gap as boundary
    """
    try:
        # Step 1: Build density map
        density = np.zeros(image_width, dtype=int)
        for box in rec_boxes:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                x1_int = int(max(0, x1))
                x2_int = int(min(image_width - 1, x2))
                density[x1_int:x2_int] += 1
        
        logger.debug(f"Density map built with max density: {density.max()}")
        
        # Step 2: Find column boundaries (continuous zero zones)
        boundaries = []
        in_gap = False
        start = None
        
        for i, val in enumerate(density):
            if val == 0 and not in_gap:
                in_gap = True
                start = i
            elif val != 0 and in_gap:
                end = i
                in_gap = False
                if end - start > min_gap_width:
                    middle = int((start + end) / 2)
                    boundaries.append(middle)
        
        # Check if last gap extends to image edge
        if in_gap and start is not None and (image_width - start) > min_gap_width:
            middle = int((start + image_width) / 2)
            boundaries.append(middle)
        
        logger.info(f"Detected {len(boundaries)} column boundaries at x-positions: {boundaries}")
        
        return boundaries
        
    except Exception as e:
        logger.error(f"Error detecting table columns: {e}", exc_info=True)
        return []


def reconstruct_table_from_ocr(
    json_data: dict,
    image_width: int = 1200,
    min_gap_width: int = 10
) -> pd.DataFrame:
    """Reconstruct table structure from OCR results based on spatial analysis.
    
    This function rebuilds tabular structure from OCR bounding boxes and texts
    by analyzing the spatial distribution of text blocks to determine column
    boundaries and row positions.
    
    Args:
        json_data: Dictionary containing OCR results with 'rec_boxes' and 'rec_texts'.
                  Expected structure: {'rec_boxes': [[x1,y1,x2,y2], ...], 'rec_texts': [...]}
                  Optionally contains 'table_res_list' for cell-based merging.
        image_width: Width of the analyzed image in pixels for density map.
        min_gap_width: Minimum width of empty zone to be considered a column boundary.
        
    Returns:
        DataFrame with reconstructed table structure.
        
    Algorithm:
        1. Detect column boundaries from original OCR data
        2. Merge OCR blocks by table cells (if available)
        3. Calculate block centers (x_center, y_center)
        4. Distribute blocks into columns based on detected boundaries
        5. Sort blocks within columns by y_center
        6. Align rows across columns by maximum row count
    """
    try:
        logger.debug(f"Starting table reconstruction from OCR data (image_width={image_width})")
        
        # Extract original boxes and texts for column detection
        original_boxes = json_data.get('rec_boxes', [])
        original_texts = json_data.get('rec_texts', [])
        
        if len(original_boxes) == 0 or len(original_texts) == 0:
            logger.warning("No OCR boxes or texts found in data")
            return pd.DataFrame()
        
        # Step 1: Detect column boundaries from original OCR data
        boundaries = _detect_table_columns(
            original_boxes,
            image_width=image_width,
            min_gap_width=min_gap_width
        )
        
        # Step 2: Try to merge OCR blocks by table cells (if available)
        merged_boxes, merged_texts = _merge_ocr_blocks_by_cells(json_data)
        
        # Determine which data to use for table reconstruction
        if merged_boxes is not None and merged_texts is not None:
            logger.info("Using cell-merged OCR data for table reconstruction")
            boxes = merged_boxes
            texts = merged_texts
        else:
            logger.debug("Using original OCR data for table reconstruction")
            boxes = original_boxes
            texts = original_texts
        
        if len(boxes) == 0 or len(texts) == 0:
            logger.warning("No boxes or texts after processing")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(boxes)} blocks for table reconstruction")
        
        # Step 3: Calculate block centers
        blocks = []
        for (x1, y1, x2, y2), text in zip(boxes, texts):
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            blocks.append((x_center, y_center, text.strip()))
        
        # Step 4: Distribute blocks into columns
        num_columns = len(boundaries) + 1
        columns = [[] for _ in range(num_columns)]
        
        for x_center, y_center, text in blocks:
            col_index = sum(x_center > b for b in boundaries)
            columns[col_index].append((y_center, text))
        
        logger.debug(f"Blocks distributed into {num_columns} columns")
        
        # Step 5: Sort blocks within each column by y_center
        for col in columns:
            col.sort(key=lambda x: x[0])
        
        # Step 6: Build table with row alignment
        max_rows = max(len(col) for col in columns) if columns else 0
        
        if max_rows == 0:
            logger.warning("No rows found in table")
            return pd.DataFrame()
        
        table = []
        for i in range(max_rows):
            row = []
            for col in columns:
                if i < len(col):
                    row.append(col[i][1])  # Append text
                else:
                    row.append("")  # Empty cell
            table.append(row)
        
        df = pd.DataFrame(table)
        logger.info(f"Table reconstructed with shape: {df.shape}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error reconstructing table from OCR data: {e}", exc_info=True)
        return pd.DataFrame()


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
                    temp_csv_path.unlink()  # Закомментировано для отладки
        
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
        
        # Remove completely empty rows and columns (NaN values)
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Fill NaN with empty strings
        df = df.fillna('')
        
        # Strip whitespace from string columns and replace semicolons
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.replace(';', ' ', regex=False)
        
        # Remove columns where all values are empty strings (after stripping)
        non_empty_cols = (df != '').any(axis=0)
        df = df.loc[:, non_empty_cols]
        
        # # Remove rows where more than 1/3 of cells are empty
        # if len(df.columns) > 0:
        #     empty_cells_per_row = (df == '').sum(axis=1)
        #     total_cols = len(df.columns)
        #     empty_threshold = total_cols / 3
        #     rows_to_keep = empty_cells_per_row <= empty_threshold
        #     rows_removed = (~rows_to_keep).sum()
        #     if rows_removed > 0:
        #         logger.debug(f"Removing {rows_removed} rows with >1/3 empty cells")
        #     df = df.loc[rows_to_keep, :]
        
        # Remove rows where all values are empty strings
        non_empty_rows = (df != '').any(axis=1)
        df = df.loc[non_empty_rows, :]
        
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