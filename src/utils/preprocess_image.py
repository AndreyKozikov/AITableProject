import cv2
import numpy as np
from src.utils.config import PARSING_DIR


def contrast_equalision(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img_cl = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img = img_cl.apply(img)
    return img


def denoise_image(img, h=10):
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.fastNlMeansDenoising(img, None, h, 7, 21)


def deskew_image(img, max_angle=12):
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


def wrap_image(img):
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


def preprocess_image(image_path: str):
    out_path = PARSING_DIR / "preprocess.png"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение:{image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = contrast_equalision(img, clip_limit=2.0, tile_grid_size=(8, 8))
    img = denoise_image(img)
    img, angle = deskew_image(img)
    #img, wrap = wrap_image(img)
    cv2.imwrite(out_path, img)
    return out_path


