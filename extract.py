import cv2
import numpy as np
from paddleocr import PaddleOCR
import unicodedata
import re
import json
import os
from band_processor import find_horizontal_bands

ocr = PaddleOCR(
    lang="japan",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", "", s)


TARGET = normalize("自動車登録番号又は車両番号")


def extract_label_value(
    img_path, save_debug=False, save_json=False, save_bands=False, debug_dir="debug"
):
    img = cv2.imread(img_path)
    bands = find_horizontal_bands(img, save_debug=save_debug, debug_dir=debug_dir)

    # bandsディレクトリを作成
    bands_dir = f"{debug_dir}/bands"
    if save_bands:
        os.makedirs(bands_dir, exist_ok=True)

    label_band_idx = None
    texts_by_band = []
    all_ocr_results = []

    for i, (y1, y2) in enumerate(bands):
        crop = img[y1:y2, :]

        # バンド画像を保存
        if save_bands:
            band_filename = f"{bands_dir}/band_{i:02d}_y{y1}-{y2}.png"
            cv2.imwrite(band_filename, crop)

        # OCR処理前にバンドを最大サイズ1500にリサイズ
        h, w = crop.shape[:2]
        max_size = 1500
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            crop_resized = cv2.resize(crop, (new_w, new_h))
        else:
            crop_resized = crop

        res = ocr.predict(crop_resized)[0]
        texts = res["rec_texts"]
        boxes = res["rec_polys"]
        confidences = res["rec_scores"]

        # バンド内のOCR結果を記録
        band_results = []
        for j, (text, box, conf) in enumerate(zip(texts, boxes, confidences)):
            # 元画像座標に変換
            original_box = box.copy()
            original_box[:, 1] += y1

            band_results.append(
                {
                    "text": text,
                    "confidence": float(conf),
                    "box": original_box.tolist(),
                    "band_index": i,
                    "band_position": (y1, y2),
                }
            )

        all_ocr_results.extend(band_results)

        joined = normalize("".join(texts))
        texts_by_band.append((i, texts, joined, (y1, y2)))
        if TARGET in joined:
            label_band_idx = i

    # JSONファイルに保存
    if save_json:
        json_data = {
            "image_path": img_path,
            "total_bands": len(bands),
            "ocr_results": all_ocr_results,
            "bands": [
                {"index": i, "position": (y1, y2)} for i, (y1, y2) in enumerate(bands)
            ],
            "target_band_index": label_band_idx,
        }

        with open(f"{debug_dir}/ocr_results.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    if label_band_idx is None:
        return None

    # 右側テキスト抽出（同じバンド内でやる例）
    i, texts, _, (y1, y2) = texts_by_band[label_band_idx]
    return "".join(texts)


val = extract_label_value(
    "IMG_0564.JPG", save_debug=True, save_json=True, save_bands=True, debug_dir="debug"
)
print(val)
