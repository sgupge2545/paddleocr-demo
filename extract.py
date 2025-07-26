import cv2
import numpy as np
from paddleocr import PaddleOCR
import unicodedata
import re
import json
import os

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


def find_horizontal_bands(
    img,
    min_len_ratio=0.5,
    angle_thresh=5,
    merge_gap=5,
    save_debug=False,
    debug_dir="debug",
):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if save_debug:
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(f"{debug_dir}/debug_1_gray.png", gray)

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 15
    )

    if save_debug:
        cv2.imwrite(f"{debug_dir}/debug_2_binary.png", bw)

    edges = cv2.Canny(bw, 50, 150, apertureSize=3)

    if save_debug:
        cv2.imwrite(f"{debug_dir}/debug_3_edges.png", edges)

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=int(w * min_len_ratio),
        maxLineGap=10,
    )

    # 線検出結果を可視化
    if save_debug:
        line_img = img.copy()
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < angle_thresh:
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imwrite(f"{debug_dir}/debug_4_lines.png", line_img)

    ys = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < angle_thresh:
                ys.append((min(y1, y2), max(y1, y2)))

    ys.sort()
    # マージ
    bands = []
    for y1, y2 in ys:
        if not bands or y1 - bands[-1][1] > merge_gap:
            bands.append([y1, y2])
        else:
            bands[-1][1] = max(bands[-1][1], y2)
    # 画像の上下端もバンドとして追加して区切る
    cuts = [0] + [int((a + b) / 2) for a, b in bands] + [h]
    strips = [
        (cuts[i], cuts[i + 1])
        for i in range(len(cuts) - 1)
        if cuts[i + 1] - cuts[i] > 10
    ]

    # 最終的なバンド区切りを可視化
    if save_debug:
        band_img = img.copy()
        for y1, y2 in strips:
            cv2.line(band_img, (0, y1), (w, y1), (255, 0, 0), 2)
            cv2.line(band_img, (0, y2), (w, y2), (255, 0, 0), 2)
        cv2.imwrite(f"{debug_dir}/debug_5_bands.png", band_img)

    return strips  # [(y_top, y_bottom), ...]


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

        res = ocr.predict(crop)[0]
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
    "test.jpg", save_debug=True, save_json=True, save_bands=True, debug_dir="debug"
)
print(val)
