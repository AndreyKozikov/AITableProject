import cv2
import numpy as np
from pathlib import Path

from src.utils.config import (
    APPLY_BINARIZATION,
    APPLY_INVERSION,
    APPLY_SHARPENING,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DENOISE_STRENGTH,
    DESKEW_MAX_ANGLE,
    DESKEW_MIN_ANGLE_THRESHOLD,
    DET_LIMIT_SIDE_LEN,
    DET_LIMIT_TYPE,
    PARSING_DIR,
    PERSPECTIVE_ANGLE_THRESHOLD,
    SHARPEN_AMOUNT,
    SHARPEN_KERNEL_SIZE,
    SHARPEN_SIGMA,
    SHARPEN_THRESHOLD,
)


def _correct_perspective(image: np.ndarray, angle_threshold: float = PERSPECTIVE_ANGLE_THRESHOLD) -> np.ndarray:
    """
    Коррекция перспективы документа на изображении.

    Функция находит контур листа документа, определяет 4 угла,
    проверяет наличие искажения перспективы и применяет коррекцию
    только если искажение превышает пороговое значение.

    Args:
        image: Входное изображение в градациях серого (np.ndarray).
        angle_threshold: Порог отклонения углов от 90 градусов (в градусах).
                        Если все углы близки к 90°, коррекция не применяется.

    Returns:
        np.ndarray: Изображение с исправленной перспективой или исходное изображение.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Улучшенное выделение границ для документов со слабым контрастом
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    document_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * perimeter, True)

    # Резервный метод определения углов через minAreaRect
    if len(approx) != 4:
        rect = cv2.minAreaRect(document_contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)

    points = approx.reshape(4, 2).astype(np.float32)

    # Вычисление углов четырехугольника для проверки искажения перспективы
    def calculate_angle(p1, p2, p3):
        """Вычисляет угол между векторами p1->p2 и p2->p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle) * 180.0 / np.pi
        return angle

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

    # Упорядоченные точки для вычисления углов
    ordered_points = [top_left, top_right, bottom_right, bottom_left]
    
    # Проверка всех четырех углов
    angles = []
    for i in range(4):
        p1 = ordered_points[i]
        p2 = ordered_points[(i + 1) % 4]
        p3 = ordered_points[(i + 2) % 4]
        angle = calculate_angle(p1, p2, p3)
        angles.append(angle)
    
    # Если все углы близки к 90 градусам, перспектива не искажена
    angle_deviations = [abs(angle - 90.0) for angle in angles]
    max_deviation = max(angle_deviations)
    
    if max_deviation < angle_threshold:
        # Перспектива не искажена, возвращаем исходное изображение
        return image

    src_points = np.array(
        [top_left, top_right, bottom_right, bottom_left],
        dtype=np.float32
    )

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)

    max_width = int(max(width_a, width_b))
    max_height = int(max(height_a, height_b))

    dst_points = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype=np.float32)

    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(
        image,
        perspective_matrix,
        (max_width, max_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # Автоматическая обрезка пустых полей после исправления перспективы
    gray_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
    coords = cv2.findNonZero(cv2.bitwise_not(gray_warped))
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = warped[y:y+h, x:x+w]
        return cropped
    
    return warped


def _contrast_equalision(img, clip_limit=CLAHE_CLIP_LIMIT, tile_grid_size=CLAHE_TILE_GRID_SIZE):
    img_cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = img_cl.apply(img)
    return img


def _denoise_image(img, h=DENOISE_STRENGTH):
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def _deskew_image(img, max_angle=DESKEW_MAX_ANGLE, min_angle_threshold=DESKEW_MIN_ANGLE_THRESHOLD):
    """
    Выравнивает наклон изображения, если наклон превышает пороговое значение.
    
    Args:
        img: Входное изображение в градациях серого.
        max_angle: Максимальный угол наклона для поиска (в градусах).
        min_angle_threshold: Минимальный угол наклона для применения коррекции (в градусах).
                            Если угол меньше этого порога, изображение не изменяется.
    
    Returns:
        Tuple[np.ndarray, float]: (выровненное изображение или исходное, угол поворота).
    """
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Альтернативный метод оценки угла наклона через minAreaRect
    coords = np.column_stack(np.where(binary > 0))
    angle_alt = cv2.minAreaRect(coords)[-1]
    if angle_alt < -45:
        angle_alt = -(90 + angle_alt)
    else:
        angle_alt = -angle_alt

    k = max(15, img.shape[1] // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    bands = cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(bands, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

    if lines is None:
        # Линии не обнаружены, используем альтернативный метод
        if abs(angle_alt) >= min_angle_threshold:
            angle = angle_alt
        else:
            return img, 0.0
    else:
        lns = lines.reshape(-1, 2)

        angles = []
        for rho, theta in lns:
            ang = (theta * 180.0 / np.pi) - 90
            if -max_angle < ang < max_angle:
                angles.append(ang)

        if not angles:
            # Углы не найдены в допустимом диапазоне, используем альтернативный метод
            if abs(angle_alt) >= min_angle_threshold:
                angle = angle_alt
            else:
                return img, 0.0
        else:
            angle = float(np.median(angles))
            
            # Проверка порога наклона - применяем коррекцию только если наклон значительный
            if abs(angle) < min_angle_threshold:
                # Наклон незначительный, возвращаем исходное изображение
                return img, 0.0

    # Применяем коррекцию наклона
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def _resize_for_ocr(img: np.ndarray, limit_side_len: int = DET_LIMIT_SIDE_LEN, 
                    limit_type: str = DET_LIMIT_TYPE) -> np.ndarray:
    """
    Масштабирует изображение согласно требованиям PaddleOCR.
    
    Args:
        img: Входное изображение.
        limit_side_len: Целевая длина стороны изображения (по умолчанию 736 для table detection).
        limit_type: Тип ограничения - 'min' или 'max'.
                   'min' - минимальная сторона будет >= limit_side_len
                   'max' - максимальная сторона будет <= limit_side_len
    
    Returns:
        np.ndarray: Масштабированное изображение.
    """
    h, w = img.shape[:2]
    
    if limit_type == 'min':
        # Минимальная сторона не должна быть меньше limit_side_len
        min_side = min(h, w)
        if min_side < limit_side_len:
            scale = limit_side_len / min_side
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:  # 'max'
        # Максимальная сторона не должна быть больше limit_side_len
        max_side = max(h, w)
        if max_side > limit_side_len:
            scale = limit_side_len / max_side
            new_h = int(h * scale)
            new_w = int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    return img


def _binarize_image(img: np.ndarray) -> np.ndarray:
    """
    Применяет бинаризацию Оцу к изображению для улучшения контраста текста.
    
    Args:
        img: Входное изображение в градациях серого.
    
    Returns:
        np.ndarray: Бинаризованное изображение.
    """
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def _invert_image(img: np.ndarray) -> np.ndarray:
    """
    Инвертирует цвета изображения (полезно для негативных изображений).
    
    Args:
        img: Входное изображение.
    
    Returns:
        np.ndarray: Инвертированное изображение.
    """
    return cv2.bitwise_not(img)


def _sharpen_image(img: np.ndarray, kernel_size: tuple = SHARPEN_KERNEL_SIZE,
                   sigma: float = SHARPEN_SIGMA, amount: float = SHARPEN_AMOUNT,
                   threshold: int = SHARPEN_THRESHOLD) -> np.ndarray:
    """
    Применяет фильтр повышения резкости (unsharp mask) к изображению.
    
    Args:
        img: Входное изображение в градациях серого.
        kernel_size: Размер ядра Гауссова фильтра.
        sigma: Стандартное отклонение Гауссова фильтра.
        amount: Сила повышения резкости (обычно 1.0-2.0).
        threshold: Порог для применения эффекта (0 = применяется везде).
    
    Returns:
        np.ndarray: Изображение с повышенной резкостью.
    """
    # Создаем размытую версию изображения
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    
    # Вычисляем unsharp mask
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    
    # Применяем threshold если задан
    if threshold > 0:
        low_contrast_mask = np.abs(img - blurred) < threshold
        sharpened = np.where(low_contrast_mask, img, sharpened)
    
    return sharpened



def preprocess_image(image_path: Path, save_to_disk: bool = True):
    """Preprocess image for OCR with optimal pipeline order optimized for PaddleOCR.
    
    Applies preprocessing steps in the following order:
    1. Grayscale conversion
    2. Inversion (if enabled and needed for negative images)
    3. Perspective correction with enhanced edge detection (geometric transformation)
    4. Deskew/rotation correction with fallback method (geometric transformation)
    5. Image resizing according to PaddleOCR requirements (det_limit_side_len)
    6. Noise reduction (quality enhancement)
    7. Sharpening (if enabled, for better edge detection)
    8. Lighting correction to normalize illumination (quality enhancement)
    9. Contrast enhancement with CLAHE (quality enhancement)
    10. Binarization (if enabled, for high-contrast text extraction)
    
    Args:
        image_path: Path to input image file.
        save_to_disk: If True, save processed image to disk and return path.
                     If False, return processed image as np.array.
    
    Returns:
        Path to saved image (if save_to_disk=True) or np.array (if save_to_disk=False).
        
    Raises:
        FileNotFoundError: If image cannot be loaded.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение:{image_path}")
    
    # Apply preprocessing pipeline in optimal order for PaddleOCR
    # 1. Convert to grayscale first
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply inversion if enabled (for negative/inverted scans)
    if APPLY_INVERSION:
        img = _invert_image(img)
    
    # 3. Apply geometric transformations before quality enhancements
    # Perspective correction must be first to fix document distortion
    img = _correct_perspective(img)
    
    # 4. Deskew to fix rotation after perspective is corrected
    img, angle = _deskew_image(img)
    
    # 5. Resize image according to PaddleOCR requirements
    # This ensures optimal input size for table detection (736px min side)
    img = _resize_for_ocr(img, DET_LIMIT_SIDE_LEN, DET_LIMIT_TYPE)
    
    # 6. Apply quality enhancements after geometric corrections
    # Denoise before other enhancements for better results
    img = _denoise_image(img)
    
    # 7. Sharpen image to improve edge detection (if enabled)
    if APPLY_SHARPENING:
        img = _sharpen_image(img)
    
    # 8. Lighting correction to normalize illumination
    background = cv2.GaussianBlur(img, (55, 55), 0)
    img = cv2.divide(img, background, scale=255)
    
    # 9. Contrast enhancement to improve text visibility
    img = _contrast_equalision(img)
    
    # 10. Apply binarization if enabled (for maximum text contrast)
    if APPLY_BINARIZATION:
        img = _binarize_image(img)

    # Convert grayscale back to BGR for compatibility with OCR engines
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if save_to_disk:
        # Save to disk and return path
        out_path = PARSING_DIR / "preprocess.png"
        cv2.imwrite(str(out_path), img)

    return img


