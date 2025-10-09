import cv2
import numpy as np
from src.utils.config import PARSING_DIR


def deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Выравнивание наклона текста на изображении документа.

    Функция находит контуры текста, определяет основной блок текста,
    вычисляет угол наклона и выполняет аффинное преобразование для
    горизонтального выравнивания текста.

    Args:
        image: Входное изображение в градациях серого (np.ndarray).

    Returns:
        np.ndarray: Выровненное изображение.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)
    angle = rect[2]

    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    if abs(angle) < 0.5:
        return image

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Коррекция перспективы документа на изображении.

    Функция находит контур листа документа, определяет 4 угла,
    вычисляет матрицу перспективного преобразования и выравнивает
    изображение документа в прямоугольник.

    Args:
        image: Входное изображение в градациях серого (np.ndarray).

    Returns:
        np.ndarray: Изображение с исправленной перспективой.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    inverted = cv2.bitwise_not(binary)

    dilated = cv2.dilate(inverted, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return image

    document_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(document_contour, True)
    approx = cv2.approxPolyDP(document_contour, 0.02 * perimeter, True)

    if len(approx) != 4:
        return image

    points = approx.reshape(4, 2).astype(np.float32)

    s = points.sum(axis=1)
    diff = np.diff(points, axis=1).flatten()

    top_left = points[np.argmin(s)]
    bottom_right = points[np.argmax(s)]
    top_right = points[np.argmin(diff)]
    bottom_left = points[np.argmax(diff)]

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

    return warped


def _contrast_equalision(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = img_cl.apply(img)
    return img


def _denoise_image(img, h=10):
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def _deskew_image(img, max_angle=12):
    binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    k = max(15, img.shape[1] // 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
    bands = cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    edges = cv2.Canny(bands, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

    if lines is None:
        print("Can't detect lines")
        return img, False
    lns = lines.reshape(-1, 2)
    print(lines.shape)

    angles = []
    for rho, theta in lns:
        ang = (theta * 180.0 / np.pi) - 90
        if -max_angle < ang < max_angle:
            angles.append(ang)

    if not angles:
        print("Can't calculate angles")
        return img, False

    angle = float(np.median(angles))
    if abs(angle) < 0.2:
        return img, 0.0

    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


def _wrap_image(img):
    thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    invert = cv2.bitwise_not(thr)

    inv = cv2.dilate(invert, np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return img, False

    cnt = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    if len(approx) != 4:
        return img, False

    pts = approx.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    src = np.array([tl, tr, br, bl], dtype=np.float32)

    wA = np.linalg.norm(br - bl);
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br);
    hB = np.linalg.norm(tl - bl)
    W = int(max(wA, wB));
    H = int(max(hA, hB))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, True


def preprocess_image(image_path: str, save_to_disk: bool = True):
    """Preprocess image with various enhancement techniques.
    
    Args:
        image_path: Path to input image file.
        save_to_disk: If True, save processed image to disk and return path.
                     If False, return processed image as np.array.
    
    Returns:
        Path to saved image (if save_to_disk=True) or np.array (if save_to_disk=False).
        
    Raises:
        FileNotFoundError: If image cannot be loaded.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение:{image_path}")
    
    # Apply preprocessing pipeline
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = _contrast_equalision(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img = _denoise_image(img)
    img, angle = _deskew_image(img)
    img = correct_perspective(img)
    #img, wrap = wrap_image(img)
    
    # Convert grayscale back to BGR for compatibility with OCR engines
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if save_to_disk:
        # Save to disk and return path
        out_path = PARSING_DIR / "preprocess.png"
        cv2.imwrite(out_path, img)
        return out_path
    else:
        # Return image as np.array
        return img


