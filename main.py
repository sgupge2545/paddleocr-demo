from paddleocr import PaddleOCR
import cv2
import numpy as np


def resize_image(image_path, max_size=1200):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")

    height, width = img.shape[:2]

    if max(height, width) <= max_size:
        return img

    scale = max_size / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_img


ocr = PaddleOCR(
    lang="japan",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# 画像をリサイズ
resized_image = resize_image("IMG_0564.JPG", max_size=1200)

results = ocr.predict(resized_image)

for i, res in enumerate(results):
    res.print()  # 文字列とスコアをコンソール表示
    res.save_to_img(f"out_{i}.png")  # 枠描画済み画像を保存
    res.save_to_json(f"out_{i}.json")  # JSONで保存
