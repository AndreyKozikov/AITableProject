"""
Препроцессор изображений для OCR с 14-этапным конвейером обработки.

Image preprocessor for OCR with a 14-stage processing pipeline.
Provides class-based interface with YAML configuration and individual stage control.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Препроцессор изображений для OCR с 14-этапным конвейером обработки.
    
    Класс обеспечивает гибкую обработку изображений с возможностью:
    - Загрузки конфигурации из YAML
    - Включения/отключения отдельных этапов
    - Сохранения промежуточных результатов
    - Выполнения полного конвейера или отдельных этапов
    
    Attributes:
        out_dir: Директория для сохранения результатов
        base_name: Базовое имя для файлов (по умолчанию 'preprocess')
        save_to_disk: Сохранять ли итоговое изображение в run()
        apply_inversion: Применять ли инверсию цветов
        apply_sharpen: Применять ли повышение резкости
        apply_binarization: Применять ли бинаризацию
    """
    
    def __init__(
        self,
        # Stage toggles (from YAML or override)
        apply_inversion: Optional[bool] = None,
        apply_perspective_correction: Optional[bool] = None,
        apply_deskew: Optional[bool] = None,
        apply_resize: Optional[bool] = None,
        apply_denoise: Optional[bool] = None,
        apply_sharpen: Optional[bool] = None,
        apply_lighting_correction: Optional[bool] = None,
        apply_contrast_equalization: Optional[bool] = None,
        apply_binarization: Optional[bool] = None,
        apply_local_distortion_correction: Optional[bool] = None,
        # Defaults
        save_to_disk: Optional[bool] = None,
        out_dir: Optional[Path] = None,
        base_name: Optional[str] = None,
    ):
        """
        Инициализация препроцессора изображений.
        
        Args:
            apply_inversion: Применять ли инверсию цветов
            apply_perspective_correction: Применять ли коррекцию перспективы
            apply_deskew: Применять ли устранение наклона
            apply_resize: Применять ли масштабирование под OCR
            apply_denoise: Применять ли шумоподавление
            apply_sharpen: Применять ли повышение резкости
            apply_lighting_correction: Применять ли выравнивание освещения
            apply_contrast_equalization: Применять ли усиление контраста (CLAHE)
            apply_binarization: Применять ли бинаризацию
            apply_local_distortion_correction: Применять ли локальную коррекцию искажений
            save_to_disk: Сохранять ли итоговое изображение (по умолчанию из config)
            out_dir: Директория для сохранения (по умолчанию parsing_files/)
            base_name: Базовое имя для файлов (по умолчанию из config)
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
        """
        # Load YAML configuration
        # Используем parents[2] для доступа к корню проекта из src/utils/
        config_path = Path(__file__).resolve().parents[2] / "config" / "preprocess_image_config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
        
        # Build merged configuration with constructor overrides
        self._config = {
            'defaults': {
                'save_to_disk': save_to_disk if save_to_disk is not None else yaml_config['defaults']['save_to_disk'],
                'base_name': base_name if base_name is not None else yaml_config['defaults']['base_name'],
            },
            'stages': self._merge_stages(yaml_config['stages'], locals()),
            'parameters': yaml_config['parameters']  # Keep as-is from YAML
        }
        
        # Set output directory
        if out_dir is not None:
            self.out_dir = out_dir
        else:
            # Используем parents[2] для доступа к корню проекта
            self.out_dir = Path(__file__).resolve().parents[2] / "parsing_files"
        self.out_dir.mkdir(exist_ok=True)
        
        # Expose frequently-used toggles as attributes for convenience
        self.apply_inversion = self._config['stages']['apply_inversion']
        self.apply_sharpen = self._config['stages']['apply_sharpen']
        self.apply_binarization = self._config['stages']['apply_binarization']
        self.save_to_disk = self._config['defaults']['save_to_disk']
        self.base_name = self._config['defaults']['base_name']
    
    def _merge_stages(self, yaml_stages: dict, constructor_locals: dict) -> dict:
        """
        Объединяет настройки этапов из YAML с параметрами конструктора.
        
        Args:
            yaml_stages: Словарь настроек этапов из YAML
            constructor_locals: Локальные переменные конструктора
            
        Returns:
            Объединённый словарь настроек этапов
        """
        result = yaml_stages.copy()
        for key in result.keys():
            if key in constructor_locals and constructor_locals[key] is not None:
                result[key] = constructor_locals[key]
        return result
    
    def _save_stage_result(self, image: np.ndarray, stage_name: str) -> None:
        """
        Сохраняет промежуточный результат этапа с стандартным именованием.
        
        Args:
            image: Изображение для сохранения
            stage_name: Название этапа для формирования имени файла
        """
        filename = f"{self.base_name}_{stage_name}.png"
        path = self.out_dir / filename
        cv2.imwrite(str(path), image)
    
    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Загружает изображение с обработкой ошибок.
        
        Args:
            image_path: Путь к файлу изображения
            
        Returns:
            Загруженное изображение
            
        Raises:
            FileNotFoundError: Если изображение не удалось загрузить
        """
        img = cv2.imread(str(image_path))

        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        return img
    
    def invert_image(self, image: np.ndarray, save: bool = False) -> np.ndarray:
        """
        Инвертирует цвета изображения (полезно для негативных изображений).
        
        Args:
            image: Входное изображение
            save: Сохранить промежуточный результат на диск
            
        Returns:
            Инвертированное изображение
        """
        result = cv2.bitwise_not(image)
        
        if save:
            self._save_stage_result(result, "invert")
        
        return result
    
    def correct_perspective(
        self, 
        image: np.ndarray, 
        save: bool = False,
        angle_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Коррекция перспективы документа на изображении.
        
        Функция находит контур листа документа, определяет 4 угла,
        проверяет наличие искажения перспективы и применяет коррекцию
        только если искажение превышает пороговое значение.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            angle_threshold: Порог отклонения углов от 90° (None = из конфига)
            
        Returns:
            Изображение с исправленной перспективой или исходное изображение
        """
        threshold = angle_threshold if angle_threshold is not None else \
                    self._config['parameters']['perspective']['angle_threshold']
        
        # === ORIGINAL ALGORITHM (line-by-line port) ===
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Усиление контраста для улучшения видимости верхней границы документа
        gray = cv2.equalizeHist(gray)

        # Улучшенное выделение границ для документов со слабым контрастом
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        document_contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(document_contour, True)
        approx = cv2.approxPolyDP(document_contour, 0.02 * perimeter, True)

        # Если контур найден не полностью, используем выпуклую оболочку для восстановления формы
        if len(approx) < 4:
            hull = cv2.convexHull(document_contour)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            approx = box.astype(int)
        
        # Резервный метод определения углов через minAreaRect
        if len(approx) != 4:
            rect = cv2.minAreaRect(document_contour)
            box = cv2.boxPoints(rect)
            approx = box.astype(int)

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
        
        if max_deviation < threshold:
            # Перспектива не искажена, возвращаем исходное изображение
            result = image
        else:
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
                result = cropped
            else:
                result = warped
        
        if save:
            self._save_stage_result(result, "perspective")
        
        return result
    
    def deskew_image(
        self,
        image: np.ndarray,
        save: bool = False,
        max_angle: Optional[int] = None,
        min_angle_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Выравнивает наклон изображения, если наклон превышает пороговое значение.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            max_angle: Максимальный угол наклона для поиска (None = из конфига)
            min_angle_threshold: Минимальный угол для применения коррекции (None = из конфига)
        
        Returns:
            Tuple[np.ndarray, float]: (выровненное изображение или исходное, угол поворота)
        """
        max_ang = max_angle if max_angle is not None else \
                  self._config['parameters']['deskew']['max_angle']
        min_thresh = min_angle_threshold if min_angle_threshold is not None else \
                     self._config['parameters']['deskew']['min_angle_threshold']
        
        # === ORIGINAL ALGORITHM ===
        binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # Альтернативный метод оценки угла наклона через minAreaRect
        coords = np.column_stack(np.where(binary > 0))
        angle_alt = cv2.minAreaRect(coords)[-1]
        if angle_alt < -45:
            angle_alt = -(90 + angle_alt)
        else:
            angle_alt = -angle_alt

        k = max(15, image.shape[1] // 60)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))
        bands = cv2.morphologyEx(255 - binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.Canny(bands, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)

        if lines is None:
            # Линии не обнаружены, используем альтернативный метод
            if abs(angle_alt) >= min_thresh:
                angle = angle_alt
            else:
                result = image, 0.0
                if save:
                    self._save_stage_result(image, "deskew")
                return result
        else:
            lns = lines.reshape(-1, 2)

            angles = []
            for rho, theta in lns:
                ang = (theta * 180.0 / np.pi) - 90
                if -max_ang < ang < max_ang:
                    angles.append(ang)

            if not angles:
                # Углы не найдены в допустимом диапазоне, используем альтернативный метод
                if abs(angle_alt) >= min_thresh:
                    angle = angle_alt
                else:
                    result = image, 0.0
                    if save:
                        self._save_stage_result(image, "deskew")
                    return result
            else:
                angle = float(np.median(angles))
                
                # Проверка порога наклона - применяем коррекцию только если наклон значительный
                if abs(angle) < min_thresh:
                    # Наклон незначительный, возвращаем исходное изображение
                    result = image, 0.0
                    if save:
                        self._save_stage_result(image, "deskew")
                    return result

        # Применяем коррекцию наклона
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        if save:
            self._save_stage_result(rotated, "deskew")
        
        return rotated, angle
    
    def resize_for_ocr(
        self,
        image: np.ndarray,
        save: bool = False,
        limit_side_len: Optional[int] = None,
        limit_type: Optional[str] = None
    ) -> np.ndarray:
        """
        Масштабирует изображение согласно требованиям PaddleOCR.
        
        Args:
            image: Входное изображение
            save: Сохранить промежуточный результат на диск
            limit_side_len: Целевая длина стороны изображения (None = из конфига)
            limit_type: Тип ограничения - 'min' или 'max' (None = из конфига)
        
        Returns:
            Масштабированное изображение
        """
        lim_len = limit_side_len if limit_side_len is not None else \
                  self._config['parameters']['resize']['limit_side_len']
        lim_type = limit_type if limit_type is not None else \
                   self._config['parameters']['resize']['limit_type']
        
        # === ORIGINAL ALGORITHM ===
        h, w = image.shape[:2]
        
        if lim_type == 'min':
            # Минимальная сторона не должна быть меньше limit_side_len
            min_side = min(h, w)
            if min_side < lim_len:
                scale = lim_len / min_side
                new_h = int(h * scale)
                new_w = int(w * scale)
                result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                result = image
        else:  # 'max'
            # Максимальная сторона не должна быть больше limit_side_len
            max_side = max(h, w)
            if max_side > lim_len:
                scale = lim_len / max_side
                new_h = int(h * scale)
                new_w = int(w * scale)
                result = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                result = image

        # Толщина рамки (в пикселях)
        border_size = 50  # можно увеличить до 20–30, если текст совсем у края

        # Цвет рамки — белый (в формате BGR)
        color = [255, 255, 255]

        # Добавляем рамку
        result = cv2.copyMakeBorder(
            result,
            top=border_size,
            bottom=border_size,
            left=border_size,
            right=border_size,
            borderType=cv2.BORDER_CONSTANT,
            value=color
        )
        
        if save:
            self._save_stage_result(result, "resize")
        
        return result
    
    def denoise_image(
        self,
        image: np.ndarray,
        save: bool = False,
        h: Optional[int] = None
    ) -> np.ndarray:
        """
        Применяет шумоподавление к изображению.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            h: Параметр силы фильтрации (None = из конфига)
        
        Returns:
            Изображение с уменьшенным шумом
        """
        strength = h if h is not None else \
                   self._config['parameters']['denoise']['strength']
        
        # === ORIGINAL ALGORITHM ===
        img = image.copy()
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        result = cv2.fastNlMeansDenoising(img, None, strength, 7, 21)
        
        if save:
            self._save_stage_result(result, "denoise")
        
        return result
    
    def sharpen_image(
        self,
        image: np.ndarray,
        save: bool = False,
        kernel_size: Optional[Tuple[int, int]] = None,
        sigma: Optional[float] = None,
        amount: Optional[float] = None,
        threshold: Optional[int] = None
    ) -> np.ndarray:
        """
        Применяет фильтр повышения резкости (unsharp mask) к изображению.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            kernel_size: Размер ядра Гауссова фильтра (None = из конфига)
            sigma: Стандартное отклонение Гауссова фильтра (None = из конфига)
            amount: Сила повышения резкости (None = из конфига)
            threshold: Порог для применения эффекта (None = из конфига)
        
        Returns:
            Изображение с повышенной резкостью
        """
        k_size = tuple(kernel_size) if kernel_size is not None else \
                 tuple(self._config['parameters']['sharpen']['kernel_size'])
        sig = sigma if sigma is not None else \
              self._config['parameters']['sharpen']['sigma']
        amt = amount if amount is not None else \
              self._config['parameters']['sharpen']['amount']
        thresh = threshold if threshold is not None else \
                 self._config['parameters']['sharpen']['threshold']
        
        # === ORIGINAL ALGORITHM ===
        # Создаем размытую версию изображения
        blurred = cv2.GaussianBlur(image, k_size, sig)
        
        # Вычисляем unsharp mask
        sharpened = cv2.addWeighted(image, 1.0 + amt, blurred, -amt, 0)
        
        # Применяем threshold если задан
        if thresh > 0:
            low_contrast_mask = np.abs(image - blurred) < thresh
            result = np.where(low_contrast_mask, image, sharpened)
        else:
            result = sharpened
        
        if save:
            self._save_stage_result(result, "sharpen")
        
        return result
    
    def lighting_correction(
        self,
        image: np.ndarray,
        save: bool = False
    ) -> np.ndarray:
        """
        Выравнивание освещения изображения (деление на размытие).
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
        
        Returns:
            Изображение с нормализованным освещением
        """
        # === ORIGINAL ALGORITHM ===
        background = cv2.GaussianBlur(image, (55, 55), 0)
        result = cv2.divide(image, background, scale=255)
        
        if save:
            self._save_stage_result(result, "lighting")
        
        return result
    
    def contrast_equalization(
        self,
        image: np.ndarray,
        save: bool = False,
        clip_limit: Optional[float] = None,
        tile_grid_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Применяет адаптивное выравнивание гистограммы (CLAHE).
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            clip_limit: Предел обрезки для CLAHE (None = из конфига)
            tile_grid_size: Размер сетки для локальной обработки (None = из конфига)
        
        Returns:
            Изображение с улучшенным контрастом
        """
        clip_lim = clip_limit if clip_limit is not None else \
                   self._config['parameters']['clahe']['clip_limit']
        tile_grid = tuple(tile_grid_size) if tile_grid_size is not None else \
                    tuple(self._config['parameters']['clahe']['tile_grid_size'])
        
        # === ORIGINAL ALGORITHM ===
        img_cl = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=tile_grid)
        result = img_cl.apply(image)
        
        if save:
            self._save_stage_result(result, "clahe")
        
        return result
    
    def binarize_image(
        self,
        image: np.ndarray,
        save: bool = False
    ) -> np.ndarray:
        """
        Применяет бинаризацию Оцу к изображению для улучшения контраста текста.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
        
        Returns:
            Бинаризованное изображение
        """
        # === ORIGINAL ALGORITHM ===
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if save:
            self._save_stage_result(result, "binarize")
        
        return result
    
    def correct_local_distortions(
        self,
        image: np.ndarray,
        save: bool = False,
        angle_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Второй проход коррекции перспективы — локально для искажённых областей.
        
        Находит крупные контуры на изображении и проверяет их прямоугольность.
        Если углы отклоняются от 90° больше порогового значения,
        применяет локальную коррекцию перспективы к этой области.
        
        Args:
            image: Входное изображение в градациях серого
            save: Сохранить промежуточный результат на диск
            angle_threshold: Порог отклонения углов от 90° (None = из конфига)
        
        Returns:
            Изображение с исправленными локальными искажениями
        """
        threshold = angle_threshold if angle_threshold is not None else \
                    self._config['parameters']['perspective']['angle_threshold']
        debug = self._config['parameters']['local_distortions']['debug']
        
        # === ORIGINAL ALGORITHM ===
        # Создаем копию для работы
        result = image.copy()
        
        # Бинаризация для поиска контуров
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Добавляем выделение горизонтальных линий для улучшения коррекции верхней части таблицы
        horizontal = cv2.Sobel(binary, cv2.CV_8U, 1, 0, ksize=3)
        horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, np.ones((25, 1), np.uint8), iterations=2)
        binary = cv2.bitwise_or(binary, horizontal)
        
        # Морфологическая обработка для улучшения качества контуров
        # Замыкание пробелов и устранение шума
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if save:
                self._save_stage_result(result, "local_distortions")
            return result
        
        # Вычисляем минимальную площадь для фильтрации мелких контуров
        # Берем только контуры, занимающие хотя бы 5% от площади изображения
        img_area = image.shape[0] * image.shape[1]
        min_area = img_area * 0.05
        
        # Ускорение: если контуров больше 10, обрабатываем только 5 крупнейших
        if len(contours) > 10:
            # Сортируем контуры по площади в убывающем порядке
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        def calculate_angle(p1, p2, p3):
            """Вычисляет угол между векторами p1->p2 и p2->p3"""
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180.0 / np.pi
            return angle
        
        # Счетчик исправленных областей для отладки
        corrected_count = 0
        
        # Обрабатываем каждый крупный контур
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Фильтруем мелкие контуры
            if area < min_area:
                continue
            
            # Получаем ограничивающий прямоугольник для проверки соотношения сторон
            x_temp, y_temp, w_temp, h_temp = cv2.boundingRect(contour)
            
            # Фильтр по соотношению сторон (игнорирование слишком узких/вытянутых областей)
            aspect_ratio = w_temp / float(h_temp)
            if aspect_ratio < 0.3 or aspect_ratio > 3.5:
                continue
            
            # Аппроксимируем контур
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            
            # Пытаемся получить 4 угла
            if len(approx) != 4:
                # Используем minAreaRect как резервный метод
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                approx = box.astype(int)
            
            if len(approx) != 4:
                continue
            
            # Проверяем прямоугольность
            points = approx.reshape(4, 2).astype(np.float32)
            
            # Упорядочиваем точки
            s = points.sum(axis=1)
            diff = np.diff(points, axis=1).flatten()
            
            top_left = points[np.argmin(s)]
            bottom_right = points[np.argmax(s)]
            top_right = points[np.argmin(diff)]
            bottom_left = points[np.argmax(diff)]
            
            ordered_points = [top_left, top_right, bottom_right, bottom_left]
            
            # Вычисляем все углы
            angles = []
            for i in range(4):
                p1 = ordered_points[i]
                p2 = ordered_points[(i + 1) % 4]
                p3 = ordered_points[(i + 2) % 4]
                angle = calculate_angle(p1, p2, p3)
                angles.append(angle)
            
            # Проверяем отклонение от 90 градусов
            angle_deviations = [abs(angle - 90.0) for angle in angles]
            max_deviation = max(angle_deviations)
            
            # Если отклонение превышает порог, применяем локальную коррекцию
            if max_deviation >= threshold:
                # Получаем ограничивающий прямоугольник ROI
                x, y, w, h = cv2.boundingRect(contour)
                
                # Добавляем небольшой отступ для безопасности
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                # Сохраняем исходные размеры ROI
                original_roi_h, original_roi_w = h, w
                
                # Извлекаем ROI
                roi = result[y:y+h, x:x+w].copy()
                
                # Применяем коррекцию перспективы к ROI
                roi_fixed = self.correct_perspective(roi, save=False, angle_threshold=threshold)
                
                # Если размеры изменились после коррекции, масштабируем обратно
                if roi_fixed.shape != roi.shape:
                    roi_fixed = cv2.resize(roi_fixed, (original_roi_w, original_roi_h), 
                                          interpolation=cv2.INTER_LINEAR)
                
                # Вставляем исправленную область обратно
                result[y:y+h, x:x+w] = roi_fixed
                corrected_count += 1
                
                # Отладочная визуализация
                if debug:
                    cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 255), 2)
        
        if save:
            self._save_stage_result(result, "local_distortions")
        
        return result

    
    def run(self, image_path: Path) -> np.ndarray:
        """
        Выполняет весь конвейер предобработки в установленной последовательности.
        
        Применяет этапы обработки в следующем порядке:
        1. Загрузка изображения
        2. Перевод в оттенки серого
        3. Инвертирование (если включено)
        4. Коррекция перспективы (глобальная)
        5. Устранение наклона (deskew)
        6. Масштабирование под требования OCR
        7. Шумоподавление
        8. Повышение резкости (если включено)
        9. Выравнивание освещения (деление на размытие)
        10. Усиление контраста (CLAHE)
        11. Бинаризация (если включена)
        12. Локальная коррекция искажений (второй проход)
        13. Перевод в BGR для совместимости
        14. Сохранение файла (если save_to_disk=True)
        
        Args:
            image_path: Путь к входному изображению
            
        Returns:
            Обработанное изображение (np.ndarray)
            
        Raises:
            FileNotFoundError: Если изображение не найдено
        """
        # 1. Load image
        img = self.load_image(image_path)
        
        # 2. Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Inversion (if enabled)
        if self.apply_inversion:
            img = self.invert_image(img)
        
        # 4. Perspective correction (global)
        if self._config['stages']['apply_perspective_correction']:
            img = self.correct_perspective(img)
        
        # 5. Deskew (rotation correction)
        if self._config['stages']['apply_deskew']:
            img, angle = self.deskew_image(img)
        
        # 6. Resize for OCR
        if self._config['stages']['apply_resize']:
            img = self.resize_for_ocr(img)
        
        # 7. Noise reduction
        if self._config['stages']['apply_denoise']:
            img = self.denoise_image(img)
        
        # 8. Sharpening (if enabled)
        if self.apply_sharpen:
            img = self.sharpen_image(img)
        
        # 9. Lighting correction (divide by blur)
        if self._config['stages']['apply_lighting_correction']:
            img = self.lighting_correction(img)
        
        # 10. Contrast enhancement (CLAHE)
        if self._config['stages']['apply_contrast_equalization']:
            img = self.contrast_equalization(img)
        
        # 11. Binarization (if enabled)
        if self.apply_binarization:
            img = self.binarize_image(img)
        
        # 12. Local distortion correction (second pass)
        if self._config['stages']['apply_local_distortion_correction']:
            img = self.correct_local_distortions(img)
        
        # 13. Convert to BGR for compatibility
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # 14. Save if configured
        if self.save_to_disk:
            final_path = self.out_dir / f"{self.base_name}.png"
            cv2.imwrite(str(final_path), img)
        
        return img
