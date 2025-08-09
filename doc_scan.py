import cv2
import numpy as np
from PIL import Image


def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect


def detect_document_edges(image: np.ndarray) -> np.ndarray:
    # 元の画像サイズで処理
    image_small = image.copy()
    gray = cv2.cvtColor(image_small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    doc_cnt = None
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            doc_cnt = approx.reshape(4, 2)
            break
    if doc_cnt is None:
        return None
    rect = order_points(doc_cnt)
    return rect


def perspective_transform(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    (tl, tr, br, bl) = points
    wA = np.linalg.norm(br - bl)
    wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br)
    hB = np.linalg.norm(tl - bl)
    maxWidth = int(max(wA, wB))
    maxHeight = int(max(hA, hB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def scan_document(pil_image: Image.Image) -> Image.Image:
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    points = detect_document_edges(image)
    if points is None:
        print("書類らしき四角形が見つかりませんでした")
        return pil_image
    warped = perspective_transform(image, points)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
    return Image.fromarray(warped_rgb)


def enhance_document_quality(
    image: Image.Image,
    blur: str = None,  # 'gaussian', 'median', None
    binarize: bool = True,  # 2値化するか
    clahe: bool = False,  # コントラスト強調するか
    adaptive_block_size: int = 25,  # adaptiveThresholdのblockSize
    adaptive_C: int = 40,  # adaptiveThresholdのC
    global_thresh: int = None,  # 大域的2値化の閾値（NoneならadaptiveThreshold）
) -> Image.Image:
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if blur == "gaussian":
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif blur == "median":
        gray = cv2.medianBlur(gray, 3)

    if clahe:
        clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe_obj.apply(gray)

    if binarize:
        if global_thresh is not None:
            _, gray = cv2.threshold(gray, global_thresh, 255, cv2.THRESH_BINARY)
        else:
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                adaptive_block_size,
                adaptive_C,
            )

    return Image.fromarray(gray)
